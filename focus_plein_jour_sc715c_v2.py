#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
ASTRO CAM — FOCUS PLEIN JOUR V2

Serveur autonome minimal pour régler mécaniquement la caméra en journée.

Fonctions :
- démarre automatiquement INDI + indi_svbonycam_ccd ;
- détecte dynamiquement la SC715C en USB2 ou USB3 ;
- connexion INDI persistante ;
- RAW 1920x1080 bin2 ;
- réglage Gain + Exposition depuis le navigateur ;
- aucun stretch / gamma / histogramme / plate solving ;
- aperçu RAW linéaire 12 bits -> 8 bits uniquement pour affichage JPEG.

Lancement :
    python3 focus_plein_jour_sc715c.py

Puis :
    http://IP_DE_L_ARDUINO:5051/
'''

import base64
import html
import os
import signal
import socket
import subprocess
import threading
import time
import xml.parsers.expat
import zlib

import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit(
        "OpenCV manque. Installe-le avec : sudo apt install -y python3-opencv"
    )

try:
    from flask import Flask, Response, jsonify, request
except ImportError:
    raise SystemExit(
        "Flask manque. Installe-le avec : sudo apt install -y python3-flask"
    )


HOST = "127.0.0.1"
INDI_PORT = 7624
DEVICE_MATCH = "SVBONY_SC715C"

WEB_HOST = "0.0.0.0"
WEB_PORT = 5051

WIDTH = 1920
HEIGHT = 1080
BINNING = 2

GAIN_DEFAULT = 100
GAIN_MIN = 100
GAIN_MAX = 500

EXPOSURE_DEFAULT_MS = 10.0
EXPOSURE_MIN_MS = 1.0
EXPOSURE_MAX_MS = 2000.0

JPEG_QUALITY = 90
FRAME_TIMEOUT_S = 10.0


stop_event = threading.Event()

state_lock = threading.Lock()
latest_jpeg = None
frame_seq = 0

gain_value = GAIN_DEFAULT
exposure_ms_value = EXPOSURE_DEFAULT_MS

camera_connected = False
camera_name = None
last_error = ""
last_min = 0
last_max = 0
last_mean = 0.0


def tcp_port_listening_linux(port):
    port_hex = f"{int(port):04X}"

    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, "r", encoding="ascii") as handle:
                next(handle, None)

                for line in handle:
                    fields = line.split()
                    if len(fields) < 4:
                        continue

                    local_address = fields[1]
                    tcp_state = fields[3]

                    if tcp_state != "0A":
                        continue

                    try:
                        _addr, local_port = local_address.rsplit(":", 1)
                    except ValueError:
                        continue

                    if local_port.upper() == port_hex:
                        return True

        except (FileNotFoundError, PermissionError, OSError):
            pass

    return False


def start_indiserver_if_needed():
    if tcp_port_listening_linux(INDI_PORT):
        print(
            f"[INDI] serveur déjà actif sur {HOST}:{INDI_PORT} — réutilisation.",
            flush=True
        )
        return None, False

    indiserver_bin = "/usr/bin/indiserver"
    driver_bin = "/usr/bin/indi_svbonycam_ccd"

    if not os.path.exists(indiserver_bin):
        raise RuntimeError(f"indiserver introuvable : {indiserver_bin}")

    if not os.path.exists(driver_bin):
        raise RuntimeError(f"driver introuvable : {driver_bin}")

    cmd = [
        indiserver_bin,
        "-v",
        "-r", "5",
        "-p", str(INDI_PORT),
        driver_bin,
    ]

    print("[INDI] démarrage automatique...", flush=True)
    print("[INDI] " + " ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        start_new_session=True,
        close_fds=True,
    )

    deadline = time.monotonic() + 10.0

    while time.monotonic() < deadline:
        rc = proc.poll()

        if rc is not None:
            raise RuntimeError(
                f"indiserver s'est arrêté pendant le démarrage (code {rc})."
            )

        if tcp_port_listening_linux(INDI_PORT):
            print(
                f"[INDI] prêt sur {HOST}:{INDI_PORT} (pid={proc.pid}).",
                flush=True
            )
            return proc, True

        time.sleep(0.10)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass

    raise RuntimeError("indiserver n'écoute pas après 10 secondes.")


def stop_managed_indiserver(proc):
    if proc is None or proc.poll() is not None:
        return

    print("[INDI] arrêt du serveur lancé par Focus plein jour...", flush=True)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def parse_fits_uint16(blob):
    if len(blob) < 2880:
        raise RuntimeError("FITS trop court.")

    header = {}
    pos = 0
    found_end = False

    while pos + 2880 <= len(blob):
        block = blob[pos:pos + 2880]
        pos += 2880

        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode("ascii", errors="ignore")
            key = card[:8].strip()

            if key == "END":
                found_end = True
                break

            if len(card) >= 10 and card[8] == "=":
                raw = card[10:80].split("/", 1)[0].strip()

                if raw.startswith("'") and raw.endswith("'"):
                    value = raw[1:-1].strip()
                elif raw in ("T", "F"):
                    value = (raw == "T")
                else:
                    try:
                        if "." in raw or "E" in raw.upper():
                            value = float(raw)
                        else:
                            value = int(raw)
                    except ValueError:
                        value = raw

                header[key] = value

        if found_end:
            break

    if not found_end:
        raise RuntimeError("END FITS introuvable.")

    bitpix = int(header.get("BITPIX", 0))
    w = int(header.get("NAXIS1", 0))
    h = int(header.get("NAXIS2", 0))

    if bitpix != 16:
        raise RuntimeError(f"BITPIX={bitpix}, attendu 16.")

    if (w, h) != (WIDTH, HEIGHT):
        raise RuntimeError(
            f"Dimensions FITS {w}x{h}, attendu {WIDTH}x{HEIGHT}."
        )

    count = w * h
    need = count * 2

    if len(blob) < pos + need:
        raise RuntimeError("Données FITS incomplètes.")

    raw = np.frombuffer(
        blob,
        dtype=np.dtype(">i2"),
        count=count,
        offset=pos
    ).reshape((h, w))

    bzero = float(header.get("BZERO", 0.0))
    bscale = float(header.get("BSCALE", 1.0))

    if bzero == 32768.0 and bscale == 1.0:
        img = (raw.astype(np.int32) + 32768).astype(np.uint16)
    elif bzero == 0.0 and bscale == 1.0:
        img = raw.astype(np.uint16)
    else:
        img = np.clip(
            raw.astype(np.float32) * bscale + bzero,
            0, 65535
        ).astype(np.uint16)

    return img


class IndiClient:
    def __init__(self):
        self.sock = None

        self.device = None
        self.properties = set()

        self.in_blob_vector = False
        self.collect_blob = False
        self.blob_chunks = []
        self.blob_attrs = {}
        self.ready_blobs = []
        self.last_message = None

        self.parser = xml.parsers.expat.ParserCreate()
        self.parser.StartElementHandler = self._start
        self.parser.EndElementHandler = self._end
        self.parser.CharacterDataHandler = self._chars
        self.parser.Parse("<indi_stream>", False)

    def connect(self):
        self.sock = socket.create_connection(
            (HOST, INDI_PORT),
            timeout=5.0
        )
        self.sock.settimeout(0.5)

        self.request_properties()
        self.pump_for(0.2)

    def close(self):
        if self.sock is None:
            return

        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            self.sock.close()
        except OSError:
            pass

        self.sock = None

    def send(self, xml):
        if self.sock is None:
            raise RuntimeError("Socket INDI fermé.")

        self.sock.sendall((xml + "\n").encode("utf-8"))

    def request_properties(self):
        if self.device:
            self.send(
                f'<getProperties version="1.7" '
                f'device="{html.escape(self.device, quote=True)}"/>'
            )
        else:
            self.send('<getProperties version="1.7"/>')

    def _device_name(self):
        if not self.device:
            raise RuntimeError("SC715C non encore découverte.")
        return self.device

    def enable_blob(self):
        dev = self._device_name()

        self.send(
            f'<enableBLOB device="{html.escape(dev, quote=True)}" '
            f'name="CCD1">Also</enableBLOB>'
        )

    def switch(self, prop, values):
        dev = self._device_name()

        parts = [
            f'<newSwitchVector device="{html.escape(dev, quote=True)}" '
            f'name="{html.escape(prop, quote=True)}">'
        ]

        for name, val in values.items():
            parts.append(
                f'<oneSwitch name="{html.escape(name, quote=True)}">'
                f'{val}</oneSwitch>'
            )

        parts.append("</newSwitchVector>")
        self.send("".join(parts))

    def number(self, prop, values):
        dev = self._device_name()

        parts = [
            f'<newNumberVector device="{html.escape(dev, quote=True)}" '
            f'name="{html.escape(prop, quote=True)}">'
        ]

        for name, val in values.items():
            parts.append(
                f'<oneNumber name="{html.escape(name, quote=True)}">'
                f'{val}</oneNumber>'
            )

        parts.append("</newNumberVector>")
        self.send("".join(parts))

    def wait_for_properties(
        self,
        required,
        timeout=20.0,
        refresh_interval=0.8
    ):
        required = set(required)
        deadline = time.monotonic() + timeout
        next_refresh = 0.0

        while not stop_event.is_set() and time.monotonic() < deadline:
            missing = required.difference(self.properties)

            if not missing:
                return

            now = time.monotonic()

            if now >= next_refresh:
                self.request_properties()
                next_refresh = now + refresh_interval

            self.pump_once()

        missing = sorted(required.difference(self.properties))

        raise TimeoutError(
            "Timeout propriétés INDI : " + ", ".join(missing)
        )

    def pump_once(self):
        try:
            chunk = self.sock.recv(65536)
        except socket.timeout:
            return

        if not chunk:
            raise ConnectionError("indiserver a fermé la connexion.")

        self.parser.Parse(chunk, False)

    def pump_for(self, seconds):
        deadline = time.monotonic() + seconds

        while not stop_event.is_set() and time.monotonic() < deadline:
            self.pump_once()

    def wait_blob(self, timeout):
        deadline = time.monotonic() + timeout

        while not stop_event.is_set() and time.monotonic() < deadline:
            if self.ready_blobs:
                obj = self.ready_blobs.pop(0)

                if isinstance(obj, Exception):
                    raise obj

                return obj

            self.pump_once()

        if stop_event.is_set():
            return None

        raise TimeoutError(
            f"Aucun BLOB CCD1 après {timeout:.1f}s. "
            f"Dernier message={self.last_message!r}"
        )

    def _start(self, name, attrs):
        announced = attrs.get("device")

        if (
            announced
            and DEVICE_MATCH in announced
            and self.device != announced
        ):
            self.device = announced
            print(
                f"[INDI CLIENT] SC715C découverte : {self.device}",
                flush=True
            )

        is_ours = (
            self.device is not None
            and announced == self.device
        )

        if (
            is_ours
            and attrs.get("name")
            and name.endswith("Vector")
            and (
                name.startswith("def")
                or name.startswith("set")
            )
        ):
            self.properties.add(attrs["name"])

        if name == "setBLOBVector":
            self.in_blob_vector = (
                is_ours
                and attrs.get("name") == "CCD1"
            )

        elif name == "oneBLOB" and self.in_blob_vector:
            self.collect_blob = True
            self.blob_chunks = []
            self.blob_attrs = dict(attrs)

        elif name == "message":
            if announced in (None, self.device):
                self.last_message = attrs.get("message")

    def _chars(self, data):
        if self.collect_blob:
            self.blob_chunks.append(data)

    def _end(self, name):
        if name == "oneBLOB" and self.collect_blob:
            encoded = "".join(self.blob_chunks)

            self.collect_blob = False
            self.blob_chunks = []

            try:
                payload = base64.b64decode(
                    encoded,
                    validate=False
                )

                fmt = self.blob_attrs.get("format", "")

                if fmt.endswith(".z"):
                    payload = zlib.decompress(payload)

                self.ready_blobs.append(payload)

            except Exception as exc:
                self.ready_blobs.append(exc)

        elif name == "setBLOBVector":
            self.in_blob_vector = False


def camera_worker():
    global latest_jpeg
    global frame_seq
    global camera_connected
    global camera_name
    global last_error
    global last_min
    global last_max
    global last_mean

    client = None
    applied_gain = None

    try:
        client = IndiClient()
        client.connect()

        print("[CAM] attente SC715C...", flush=True)

        client.wait_for_properties(
            {"CONNECTION"},
            timeout=25.0
        )

        camera_name = client.device

        print(
            f"[CAM] connexion : {client.device}",
            flush=True
        )

        client.switch(
            "CONNECTION",
            {
                "CONNECT": "On",
                "DISCONNECT": "Off"
            }
        )

        required = {
            "UPLOAD_MODE",
            "CCD_CAPTURE_FORMAT",
            "CCD_BINNING",
            "CCD_BINNING_MODE",
            "CCD_CONTROLS",
            "CCD_EXPOSURE",
        }

        client.wait_for_properties(
            required,
            timeout=25.0
        )

        client.enable_blob()
        client.pump_for(0.3)

        client.switch(
            "UPLOAD_MODE",
            {
                "UPLOAD_CLIENT": "On",
                "UPLOAD_LOCAL": "Off",
                "UPLOAD_BOTH": "Off",
            }
        )

        client.switch(
            "CCD_CAPTURE_FORMAT",
            {
                "INDI_RGB": "Off",
                "INDI_RAW": "On"
            }
        )

        client.number(
            "CCD_BINNING",
            {
                "HOR_BIN": BINNING,
                "VER_BIN": BINNING
            }
        )

        client.switch(
            "CCD_BINNING_MODE",
            {
                "TC_BINNING_AVG": "Off",
                "TC_BINNING_ADD": "On"
            }
        )

        camera_connected = True
        last_error = ""

        print(
            "[CAM] prête — Bayer GRBG dématricé, dynamique linéaire.",
            flush=True
        )

        while not stop_event.is_set():
            with state_lock:
                gain = int(round(gain_value))
                exposure_ms = float(exposure_ms_value)

            gain = max(GAIN_MIN, min(GAIN_MAX, gain))
            exposure_ms = max(
                EXPOSURE_MIN_MS,
                min(EXPOSURE_MAX_MS, exposure_ms)
            )

            if applied_gain != gain:
                client.number(
                    "CCD_CONTROLS",
                    {"Gain": gain}
                )
                applied_gain = gain

            client.ready_blobs.clear()

            client.number(
                "CCD_EXPOSURE",
                {
                    "CCD_EXPOSURE_VALUE":
                        exposure_ms / 1000.0
                }
            )

            blob = client.wait_blob(
                max(
                    FRAME_TIMEOUT_S,
                    exposure_ms / 1000.0 + 5.0
                )
            )

            if blob is None:
                break

            raw16 = parse_fits_uint16(blob)

            # Valeurs réelles 12 bits stockées dans uint16 :
            # raw12 = raw16 >> 4.
            raw12 = np.right_shift(raw16, 4).astype(
                np.uint16,
                copy=False
            )

            # -------------------------------------------------------
            # DÉMATRICAGE BAYER UNIQUEMENT
            # -------------------------------------------------------
            # La SC715C est un capteur couleur à matrice Bayer GRBG.
            #
            # Afficher le RAW directement en niveaux de gris fait
            # apparaître la mosaïque R/V/B sous forme de grille,
            # particulièrement visible en plein jour.
            #
            # cvtColor reconstruit simplement les 3 canaux couleur.
            # Ce n'est PAS un traitement tonal :
            #   - aucun stretch
            #   - aucun gamma
            #   - aucun contraste automatique
            #   - aucune normalisation
            #   - aucune égalisation d'histogramme
            #   - aucune netteté
            #
            # La dynamique reste strictement linéaire.
            bgr12 = cv2.cvtColor(
                raw12,
                cv2.COLOR_BayerGR2BGR
            )

            # Conversion linéaire 12 bits -> 8 bits pour JPEG.
            preview8 = np.right_shift(
                bgr12,
                4
            ).astype(
                np.uint8,
                copy=False
            )

            ok, encoded = cv2.imencode(
                ".jpg",
                preview8,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    JPEG_QUALITY
                ]
            )

            if not ok:
                raise RuntimeError("Encodage JPEG impossible.")

            jpeg = encoded.tobytes()

            with state_lock:
                latest_jpeg = jpeg
                frame_seq += 1
                last_min = int(raw12.min())
                last_max = int(raw12.max())
                last_mean = float(raw12.mean())

    except Exception as exc:
        camera_connected = False
        last_error = f"{type(exc).__name__}: {exc}"

        print(
            f"[CAM ERREUR] {last_error}",
            flush=True
        )

    finally:
        camera_connected = False

        if client is not None:
            try:
                if client.device and client.sock is not None:
                    client.switch(
                        "CONNECTION",
                        {
                            "CONNECT": "Off",
                            "DISCONNECT": "On"
                        }
                    )
                    client.pump_for(1.0)
            except Exception:
                pass

            try:
                client.close()
            except Exception:
                pass

        print("[CAM] arrêt.", flush=True)


app = Flask(__name__)

PAGE = r'''<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Astro Cam — Focus plein jour V2</title>
<style>
*{box-sizing:border-box}
html,body{
    margin:0;
    width:100%;
    height:100%;
    background:#05080b;
    color:#dce7ef;
    font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
}
body{
    display:flex;
    flex-direction:column;
    overflow:hidden;
}
header{
    height:44px;
    min-height:44px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 12px;
    border-bottom:1px solid #1b2a34;
    background:#081018;
}
.title{
    font-size:14px;
    font-weight:700;
}
.status{
    font-size:11px;
    color:#91a8b8;
}
main{
    flex:1;
    min-height:0;
    display:grid;
    grid-template-columns:minmax(0,1fr) 260px;
    gap:8px;
    padding:8px;
}
.viewer{
    min-width:0;
    min-height:0;
    display:flex;
    align-items:center;
    justify-content:center;
    background:#000;
    border:1px solid #1d2a33;
    border-radius:10px;
    overflow:hidden;
}
.viewer img{
    display:block;
    width:100%;
    height:100%;
    object-fit:contain;
    background:#000;
}
.panel{
    min-width:0;
    min-height:0;
    padding:10px;
    border:1px solid #1d2a33;
    border-radius:10px;
    background:#091119;
}
.panel h2{
    margin:0 0 12px;
    font-size:14px;
}
.control{
    margin-bottom:18px;
}
.control-head{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    margin-bottom:7px;
    font-size:12px;
}
.value{
    font-variant-numeric:tabular-nums;
    font-weight:700;
}
input[type=range]{
    width:100%;
    cursor:pointer;
}
.note{
    margin-top:12px;
    padding:8px;
    border-radius:7px;
    background:#0c171f;
    color:#93a9b7;
    font-size:10px;
    line-height:1.4;
}
.stats{
    margin-top:12px;
    font-size:10px;
    color:#8da4b4;
    line-height:1.55;
    font-variant-numeric:tabular-nums;
}
label,.tick{
    user-select:none;
}
@media(max-width:800px){
    body{overflow:auto}
    main{
        display:flex;
        flex-direction:column;
    }
    .viewer{
        width:100%;
        aspect-ratio:16/9;
        min-height:auto;
    }
    .panel{
        width:100%;
    }
}
</style>
</head>
<body>
<header>
    <div class="title">☀ Focus plein jour V2 — SC715C</div>
    <div id="status" class="status">Connexion…</div>
</header>

<main>
    <div class="viewer">
        <img id="live" src="/stream.mjpg" alt="Flux caméra">
    </div>

    <aside class="panel">
        <h2>Caméra</h2>

        <div class="control">
            <div class="control-head">
                <label for="gain">Gain</label>
                <span id="gainValue" class="value">100</span>
            </div>
            <input id="gain" type="range"
                   min="100" max="500" step="1" value="100">
        </div>

        <div class="control">
            <div class="control-head">
                <label for="exposure">Exposition</label>
                <span id="expValue" class="value">10 ms</span>
            </div>
            <input id="exposure" type="range"
                   min="1" max="2000" step="1" value="10">
        </div>

        <div class="note">
            Image Bayer GRBG dématricée, dynamique linéaire.<br>
            Aucun stretch, gamma, contraste automatique,
            histogramme, plate solving ou traitement astro.
        </div>

        <div id="stats" class="stats"></div>
    </aside>
</main>

<script>
const gain=document.getElementById("gain");
const exposure=document.getElementById("exposure");
const gainValue=document.getElementById("gainValue");
const expValue=document.getElementById("expValue");
const statusEl=document.getElementById("status");
const stats=document.getElementById("stats");

let timer=null;

function sendSettings(){
    clearTimeout(timer);

    timer=setTimeout(async()=>{
        const body={
            gain:Number(gain.value),
            exposure_ms:Number(exposure.value)
        };

        try{
            await fetch("/api/settings",{
                method:"POST",
                headers:{"Content-Type":"application/json"},
                body:JSON.stringify(body)
            });
        }catch(e){}
    },80);
}

gain.addEventListener("input",()=>{
    gainValue.textContent=gain.value;
    sendSettings();
});

exposure.addEventListener("input",()=>{
    expValue.textContent=`${exposure.value} ms`;
    sendSettings();
});

async function refresh(){
    try{
        const r=await fetch("/api/status",{cache:"no-store"});
        const s=await r.json();

        gain.value=s.gain;
        exposure.value=s.exposure_ms;

        gainValue.textContent=s.gain;
        expValue.textContent=`${s.exposure_ms.toFixed(0)} ms`;

        if(s.camera_connected){
            statusEl.textContent=
                `Caméra OK · ${s.camera_name || "SC715C"} · frame #${s.frame_seq}`;
        }else{
            statusEl.textContent=
                s.error || "Caméra non connectée";
        }

        stats.innerHTML=
            `RAW12 min : ${s.raw_min}<br>`+
            `RAW12 max : ${s.raw_max}<br>`+
            `RAW12 moyenne : ${s.raw_mean.toFixed(1)}<br>`+
            `Résolution : 1920 × 1080`;
    }catch(e){
        statusEl.textContent="Serveur indisponible";
    }
}

setInterval(refresh,1000);
refresh();
</script>
</body>
</html>
'''


@app.route("/")
def index():
    return PAGE


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify({
            "camera_connected": bool(camera_connected),
            "camera_name": camera_name,
            "gain": int(gain_value),
            "exposure_ms": float(exposure_ms_value),
            "frame_seq": int(frame_seq),
            "raw_min": int(last_min),
            "raw_max": int(last_max),
            "raw_mean": float(last_mean),
            "error": last_error,
        })


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global gain_value
    global exposure_ms_value

    data = request.get_json(silent=True) or {}

    with state_lock:
        if "gain" in data:
            gain_value = max(
                GAIN_MIN,
                min(GAIN_MAX, int(round(float(data["gain"]))))
            )

        if "exposure_ms" in data:
            exposure_ms_value = max(
                EXPOSURE_MIN_MS,
                min(
                    EXPOSURE_MAX_MS,
                    float(data["exposure_ms"])
                )
            )

        return jsonify({
            "ok": True,
            "gain": int(gain_value),
            "exposure_ms": float(exposure_ms_value),
        })


def mjpeg_generator():
    last_seq = -1

    while not stop_event.is_set():
        with state_lock:
            seq = frame_seq
            jpeg = latest_jpeg

        if jpeg is None or seq == last_seq:
            time.sleep(0.03)
            continue

        last_seq = seq

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n"
            b"\r\n"
            + jpeg
            + b"\r\n"
        )


@app.route("/stream.mjpg")
def stream_mjpg():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    managed_indi = None
    camera_thread = None

    print("=" * 78)
    print("ASTRO CAM — FOCUS PLEIN JOUR V2")
    print("=" * 78)
    print("Caméra       : SC715C dynamique USB2/USB3")
    print("RAW          : 1920x1080 uint16 / bin2")
    print("Affichage    : Bayer GRBG -> couleur linéaire 12 -> 8 bits")
    print("Traitement   : dématriçage Bayer uniquement")
    print("Gain départ  :", GAIN_DEFAULT)
    print("Expo départ  :", f"{EXPOSURE_DEFAULT_MS:.0f} ms")
    print("Serveur Web  :", f"http://0.0.0.0:{WEB_PORT}/")
    print("=" * 78)

    try:
        managed_indi, _started = start_indiserver_if_needed()

        camera_thread = threading.Thread(
            target=camera_worker,
            name="camera-worker",
            daemon=True
        )
        camera_thread.start()

        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=True,
            debug=False,
            use_reloader=False
        )

    except KeyboardInterrupt:
        pass

    finally:
        print("\n[MAIN] arrêt demandé...", flush=True)
        stop_event.set()

        if camera_thread is not None:
            try:
                camera_thread.join(timeout=4.0)
            except Exception:
                pass

        stop_managed_indiserver(managed_indi)

        print("[MAIN] Focus plein jour arrêté.", flush=True)


if __name__ == "__main__":
    main()
