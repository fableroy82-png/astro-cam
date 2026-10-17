#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import ctypes
import html
import os
import signal
import socket
import time
import xml.parsers.expat
import zlib
import multiprocessing as mp

import numpy as np


HOST = "127.0.0.1"
INDI_PORT = 7624
DEVICE = "SVBONY2 SVBONY_SC715C"
DEVICE_MATCH = "SVBONY_SC715C"

WEB_HOST = "0.0.0.0"
WEB_PORT = 5050

WIDTH = 1920
HEIGHT = 1080
PIXELS = WIDTH * HEIGHT

FOCAL_LENGTH_MM = 184.0
PIXEL_SIZE_UM = 1.45
BINNING = 2

GAIN_DEFAULT = 180
EXPOSURE_DEFAULT_MS = 800.0

GAIN_MIN = 100
GAIN_MAX = 500

EXPOSURE_MIN_MS = 10.0
EXPOSURE_MAX_MS = 10000.0


# 1 = ADD, 0 = AVG
BINNING_MODE_DEFAULT = 1

FRAME_TIMEOUT_S = 15.0

# Récupération automatique si la caméra reste connectée mais ne renvoie
# plus de BLOB après une exposition.
CAMERA_RECONNECT_DELAY_S = 3.0
CAMERA_REARM_SETTLE_S = 0.6
CAMERA_DISCONNECT_SETTLE_S = 1.0

# Réglages du stretch d'affichage.
# Deux rendus : Technique (V9) et Ciel noir.
WHITE_PERCENTILE = 99.9
MAD_FLOOR = 1.0

PREVIEW_TECHNIQUE = 0
PREVIEW_DARK_SKY = 1
PREVIEW_MODE_DEFAULT = PREVIEW_DARK_SKY

# Mode Technique : priorité à la visibilité maximale du signal.
TECH_BLACK_SIGMA_OFFSET = 0.50
TECH_WHITE_SIGMA_SPAN = 12.0
TECH_ASINH_STRENGTH = 8.0
TECH_GAMMA = 1.20

# Mode Ciel noir : vrai fond noir sur la PREVIEW uniquement.
# Le point noir est placé AU-DESSUS du fond mesuré.
DARK_BLACK_SIGMA_ABOVE = 2.50
DARK_WHITE_SIGMA_SPAN = 8.0
DARK_ASINH_STRENGTH = 12.0
DARK_GAMMA = 1.18

RAW12_MIN = 0
SAT_THRESHOLD = 4094

JPEG_QUALITY = 88
MAX_JPEG_BYTES = 4 * 1024 * 1024

# Plate solving
SOLVE_TIMEOUT_S = 45
SOLVE_DOWNSAMPLE = 2
SOLVE_SCALE_LOW_FACTOR = 0.72
SOLVE_SCALE_HIGH_FACTOR = 1.32

SOLVE_IDLE = 0
SOLVE_RUNNING = 1
SOLVE_OK = 2
SOLVE_ERROR = 3

# Mise en station polaire — hémisphère Nord.
POLAR_IDLE = 0
POLAR_COLLECTING = 1
POLAR_CALIBRATED = 2
POLAR_ERROR = 3



def set_shared_text(shared_buffer, value):
    data = str(value).encode("utf-8", errors="replace")
    max_len = len(shared_buffer) - 1
    with shared_buffer.get_lock():
        shared_buffer.value = data[:max_len]


def get_shared_text(shared_buffer):
    with shared_buffer.get_lock():
        return bytes(shared_buffer.value).decode("utf-8", errors="replace")


def format_ra_hms(ra_deg):
    ra_deg = float(ra_deg) % 360.0
    hours_total = ra_deg / 15.0
    h = int(hours_total)
    minutes_total = (hours_total - h) * 60.0
    m = int(minutes_total)
    s = (minutes_total - m) * 60.0
    return f"{h:02d}h {m:02d}m {s:05.2f}s"


def format_dec_dms(dec_deg):
    dec_deg = float(dec_deg)
    sign = "+" if dec_deg >= 0 else "-"
    value = abs(dec_deg)
    d = int(value)
    minutes_total = (value - d) * 60.0
    m = int(minutes_total)
    s = (minutes_total - m) * 60.0
    return f"{sign}{d:02d}° {m:02d}′ {s:04.1f}″"



def radec_to_vector(ra_deg, dec_deg):
    ra = np.deg2rad(float(ra_deg))
    dec = np.deg2rad(float(dec_deg))
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ], dtype=np.float64)


def vector_to_radec(vector):
    v = np.asarray(vector, dtype=np.float64)
    v /= np.linalg.norm(v)
    ra = np.rad2deg(np.arctan2(v[1], v[0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return float(ra), float(dec)


def rotation_matrix_from_to(v_from, v_to):
    a = np.asarray(v_from, dtype=np.float64)
    b = np.asarray(v_to, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    cross = np.cross(a, b)
    s = float(np.linalg.norm(cross))
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)

        # Cas opposé, théoriquement impossible ici.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.8:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, helper)
        axis /= np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = cross / s
        angle = np.arctan2(s, c)

    kx, ky, kz = axis
    K = np.array([
        [0.0, -kz, ky],
        [kz, 0.0, -kx],
        [-ky, kx, 0.0]
    ], dtype=np.float64)

    I = np.eye(3, dtype=np.float64)

    return (
        I * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * K
    )


def compute_polar_alignment_solution(ra_values, dec_values, jacobian):
    """
    Détermine l'axe AD réel à partir de trois pointages obtenus
    en gardant la DEC fixe et en tournant uniquement l'axe AD.

    Retourne aussi le déplacement en pixels que subirait le champ
    si l'axe AD était amené sur le pôle Nord céleste.
    """
    points = [
        radec_to_vector(ra_values[i], dec_values[i])
        for i in range(3)
    ]

    v1, v2, v3 = points

    # Les trois pointages appartiennent à un même petit cercle dont
    # la normale est l'axe de rotation AD.
    normal = np.cross(v2 - v1, v3 - v1)
    norm = float(np.linalg.norm(normal))

    if norm < 1e-5:
        raise RuntimeError(
            "Rotation AD insuffisante ou géométrie dégénérée. "
            "Écarte davantage les trois positions AD."
        )

    axis = normal / norm

    # En France / hémisphère Nord on choisit l'axe dirigé vers le NCP.
    if axis[2] < 0.0:
        axis = -axis

    north_pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    dot_axis = float(np.clip(np.dot(axis, north_pole), -1.0, 1.0))
    polar_error_rad = float(np.arccos(dot_axis))
    polar_error_arcmin = float(np.rad2deg(polar_error_rad) * 60.0)

    axis_ra, axis_dec = vector_to_radec(axis)

    # Rotation physique nécessaire pour amener l'axe AD sur le NCP.
    correction = rotation_matrix_from_to(axis, north_pole)

    # On utilise le troisième pointage comme repère de la frame finale.
    center = v3 / np.linalg.norm(v3)

    # Quand on tourne physiquement la monture de R, le ciel fixe apparaît
    # dans la caméra comme R^-1. On applique donc la transposée.
    apparent_target = correction.T @ center
    apparent_target /= np.linalg.norm(apparent_target)

    ra0 = np.deg2rad(float(ra_values[2]))
    dec0 = np.deg2rad(float(dec_values[2]))

    east = np.array([
        -np.sin(ra0),
        np.cos(ra0),
        0.0
    ], dtype=np.float64)

    north = np.array([
        -np.sin(dec0) * np.cos(ra0),
        -np.sin(dec0) * np.sin(ra0),
        np.cos(dec0)
    ], dtype=np.float64)

    denom = float(np.dot(apparent_target, center))

    east_deg = float(np.rad2deg(
        np.arctan2(np.dot(apparent_target, east), denom)
    ))
    north_deg = float(np.rad2deg(
        np.arctan2(np.dot(apparent_target, north), denom)
    ))

    # Jacobien WCS local :
    # [ east_deg  ]   [j00 j01] [dx]
    # [ north_deg ] = [j10 j11] [dy]
    j = np.asarray(jacobian, dtype=np.float64).reshape((2, 2))

    det = float(np.linalg.det(j))
    if abs(det) < 1e-12:
        raise RuntimeError("Orientation WCS invalide pour la mise en station.")

    dx, dy = np.linalg.solve(
        j,
        np.array([east_deg, north_deg], dtype=np.float64)
    )

    return {
        "axis_ra_deg": axis_ra,
        "axis_dec_deg": axis_dec,
        "error_arcmin": polar_error_arcmin,
        "shift_x_px": float(dx),
        "shift_y_px": float(dy),
        "east_deg": east_deg,
        "north_deg": north_deg,
    }




# ---------------------------------------------------------------------------
# CATALOGUE OPENNGC
# ---------------------------------------------------------------------------

OPENNGC_MAIN_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/"
    "master/database_files/NGC.csv"
)
OPENNGC_ADDENDUM_URL = (
    "https://raw.githubusercontent.com/mattiaverga/OpenNGC/"
    "master/database_files/addendum.csv"
)

OPENNGC_TYPE_LABELS = {
    "*": "Étoile",
    "**": "Étoile double",
    "*Ass": "Association stellaire",
    "OCl": "Amas ouvert",
    "GCl": "Amas globulaire",
    "Cl+N": "Amas + nébuleuse",
    "G": "Galaxie",
    "GPair": "Paire de galaxies",
    "GTrpl": "Triplet de galaxies",
    "GGroup": "Groupe de galaxies",
    "PN": "Nébuleuse planétaire",
    "HII": "Région HII",
    "DrkN": "Nébuleuse sombre",
    "EmN": "Nébuleuse en émission",
    "Neb": "Nébuleuse",
    "RfN": "Nébuleuse par réflexion",
    "SNR": "Rémanent de supernova",
    "Nova": "Nova",
    "Other": "Objet",
}


def _catalog_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent / "catalogs"


def _download_file_atomic(url, path, timeout=15.0):
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AstroCam-UNOQ/1.0"}
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    if len(data) < 100:
        raise RuntimeError(
            f"Téléchargement anormalement petit : {len(data)} octets"
        )

    temp.write_bytes(data)
    temp.replace(path)


def ensure_openngc_files():
    catalog_dir = _catalog_dir()
    main_path = catalog_dir / "OpenNGC_NGC.csv"
    addendum_path = catalog_dir / "OpenNGC_addendum.csv"

    for url, path in (
        (OPENNGC_MAIN_URL, main_path),
        (OPENNGC_ADDENDUM_URL, addendum_path),
    ):
        if path.is_file() and path.stat().st_size > 100:
            continue

        print(
            f"[CATALOG] téléchargement OpenNGC : {path.name}...",
            flush=True
        )

        try:
            _download_file_atomic(url, path)
            print(
                f"[CATALOG] installé : {path}",
                flush=True
            )
        except Exception as exc:
            print(
                f"[CATALOG] téléchargement impossible pour "
                f"{path.name}: {exc}",
                flush=True
            )

    return main_path, addendum_path


def _sexagesimal_ra_to_deg(value):
    value = str(value or "").strip()
    if not value:
        raise ValueError("RA vide")

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"RA invalide : {value!r}")

    h, m, s = map(float, parts)
    return (h + m / 60.0 + s / 3600.0) * 15.0


def _sexagesimal_dec_to_deg(value):
    value = str(value or "").strip()
    if not value:
        raise ValueError("DEC vide")

    sign = -1.0 if value.startswith("-") else 1.0
    value = value.lstrip("+-")
    parts = value.split(":")

    if len(parts) != 3:
        raise ValueError(f"DEC invalide : {value!r}")

    d, m, s = map(float, parts)
    return sign * (d + m / 60.0 + s / 3600.0)


def _pretty_catalog_name(value):
    import re

    value = str(value or "").strip()
    if not value:
        return ""

    match = re.match(
        r"^(NGC|IC)0*([0-9]+)(.*)$",
        value,
        flags=re.IGNORECASE
    )

    if not match:
        return value

    prefix = match.group(1).upper()
    number = int(match.group(2))
    suffix = match.group(3).strip()

    result = f"{prefix} {number}"

    if suffix:
        result += " " + suffix

    return result


def _split_field(value):
    value = str(value or "").strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _messier_aliases(value):
    aliases = []

    for part in _split_field(value):
        raw = part.upper().replace("M", "").strip()

        try:
            number = int(raw)
        except ValueError:
            continue

        if 1 <= number <= 110:
            aliases.append(f"M{number}")

    return aliases


def _cross_catalog_aliases(prefix, value):
    import re

    aliases = []

    for part in _split_field(value):
        raw = part.upper().replace(prefix, "").strip()
        match = re.match(r"^0*([0-9]+)(.*)$", raw)

        if not match:
            continue

        number = int(match.group(1))
        suffix = match.group(2).strip()

        alias = f"{prefix} {number}"

        if suffix:
            alias += " " + suffix

        aliases.append(alias)

    return aliases


def _catalog_numeric_sort(value):
    import re

    match = re.search(r"([0-9]+)", str(value or ""))

    if not match:
        return 999999

    return int(match.group(1))


class OpenNGCCatalog:
    def __init__(self, paths):
        self.records = []

        for path in paths:
            self._load_file(path)

        if not self.records:
            raise RuntimeError("Aucun objet OpenNGC valide chargé.")

        print(
            f"[CATALOG] OpenNGC : {len(self.records)} objets chargés.",
            flush=True
        )

    @staticmethod
    def _detect_delimiter(header):
        return ";" if header.count(";") >= header.count(",") else ","

    def _load_file(self, path):
        import csv

        if not path.is_file():
            return

        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        ) as handle:
            header = handle.readline()

            if not header:
                return

            delimiter = self._detect_delimiter(header)
            handle.seek(0)

            reader = csv.DictReader(handle, delimiter=delimiter)

            for row in reader:
                record = self._normalize_row(row)

                if record is not None:
                    self.records.append(record)

    def _normalize_row(self, row):
        raw_name = str(row.get("Name") or "").strip()
        obj_type = str(row.get("Type") or "").strip()

        if obj_type in {"NonEx", "Dup"}:
            return None

        try:
            ra_deg = _sexagesimal_ra_to_deg(row.get("RA"))
            dec_deg = _sexagesimal_dec_to_deg(row.get("Dec"))
        except (TypeError, ValueError):
            return None

        canonical = _pretty_catalog_name(raw_name)

        messier = _messier_aliases(row.get("M"))

        ngc_aliases = []
        if raw_name.upper().startswith("NGC"):
            ngc_aliases.append(canonical)
        ngc_aliases.extend(
            _cross_catalog_aliases("NGC", row.get("NGC"))
        )

        ic_aliases = []
        if raw_name.upper().startswith("IC"):
            ic_aliases.append(canonical)
        ic_aliases.extend(
            _cross_catalog_aliases("IC", row.get("IC"))
        )

        common_names = _split_field(row.get("Common names"))
        identifiers = _split_field(row.get("Identifiers"))

        catalogs = []

        if messier:
            catalogs.append("M")
        if ngc_aliases:
            catalogs.append("NGC")
        if ic_aliases:
            catalogs.append("IC")
        if not catalogs:
            catalogs.append("OTHER")

        type_label = OPENNGC_TYPE_LABELS.get(
            obj_type,
            obj_type or "Objet"
        )

        magnitude = None
        magnitude_band = None

        def optional_float(field):
            raw = str(row.get(field) or "").strip()

            if not raw:
                return None

            try:
                value = float(raw)
            except ValueError:
                return None

            if not np.isfinite(value):
                return None

            return value

        # OpenNGC : dimensions apparentes en minutes d'arc.
        # PosAng : angle de position en degrés, Est depuis le Nord.
        major_arcmin = optional_float("MajAx")
        minor_arcmin = optional_float("MinAx")
        position_angle_deg = optional_float("PosAng")

        for field, band in (
            ("V-Mag", "V"),
            ("B-Mag", "B"),
            ("J-Mag", "J"),
        ):
            raw = str(row.get(field) or "").strip()

            if not raw:
                continue

            try:
                magnitude = float(raw)
                magnitude_band = band
                break
            except ValueError:
                pass

        aliases = []

        for alias in (
            [canonical]
            + messier
            + ngc_aliases
            + ic_aliases
            + common_names
            + identifiers
        ):
            alias = str(alias or "").strip()

            if alias and alias not in aliases:
                aliases.append(alias)

        search_text = " ".join(
            aliases
            + [
                obj_type,
                type_label,
                str(row.get("Const") or "")
            ]
        ).casefold()

        compact_text = (
            search_text
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

        return {
            "canonical": canonical or raw_name,
            "messier": messier,
            "ngc": ngc_aliases,
            "ic": ic_aliases,
            "catalogs": catalogs,
            "common_name": common_names[0] if common_names else "",
            "search_text": search_text,
            "compact_text": compact_text,
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
            "const": str(row.get("Const") or "").strip(),
            "type": obj_type,
            "type_label": type_label,
            "magnitude": magnitude,
            "magnitude_band": magnitude_band,
            "major_arcmin": major_arcmin,
            "minor_arcmin": minor_arcmin,
            "position_angle_deg": position_angle_deg,
        }

    @staticmethod
    def _display_id(record, kind, query):
        q = (
            str(query or "")
            .upper()
            .replace(" ", "")
            .replace("-", "")
        )

        if q.startswith("M") and record["messier"]:
            return record["messier"][0]

        if q.startswith("NGC") and record["ngc"]:
            return record["ngc"][0]

        if q.startswith("IC") and record["ic"]:
            return record["ic"][0]

        if kind == "M" and record["messier"]:
            return record["messier"][0]

        if kind == "NGC" and record["ngc"]:
            return record["ngc"][0]

        if kind == "IC" and record["ic"]:
            return record["ic"][0]

        if record["messier"]:
            return record["messier"][0]

        return record["canonical"]

    def search(self, query="", kind="ALL", limit=120):
        query = str(query or "").strip()
        q = query.casefold()

        q_compact = (
            q
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
        )

        rows = []

        for record in self.records:
            if kind != "ALL" and kind not in record["catalogs"]:
                continue

            score = 10

            if q:
                ids = (
                    record["messier"]
                    + record["ngc"]
                    + record["ic"]
                    + [record["canonical"]]
                )

                compact_ids = [
                    value.casefold()
                    .replace(" ", "")
                    .replace("-", "")
                    .replace("_", "")
                    for value in ids
                ]

                if q_compact in compact_ids:
                    score = 0
                elif any(
                    value.startswith(q_compact)
                    for value in compact_ids
                ):
                    score = 1
                elif q in record["search_text"]:
                    score = 3
                elif q_compact in record["compact_text"]:
                    score = 4
                else:
                    continue

            display_id = self._display_id(
                record,
                kind,
                query
            )

            label = (
                record["common_name"]
                or record["type_label"]
            )

            item = {
                "id": display_id,
                "name": label,
                "ra_deg": record["ra_deg"],
                "dec_deg": record["dec_deg"],
                "family": (
                    "Messier"
                    if display_id.upper().startswith("M")
                    and not display_id.upper().startswith("MEL")
                    else (
                        "NGC"
                        if display_id.upper().startswith("NGC")
                        else (
                            "IC"
                            if display_id.upper().startswith("IC")
                            else "Autre"
                        )
                    )
                ),
                "type": record["type"],
                "type_label": record["type_label"],
                "const": record["const"],
                "magnitude": record["magnitude"],
                "magnitude_band": record["magnitude_band"],
            }

            rows.append((
                score,
                _catalog_numeric_sort(display_id),
                display_id,
                item
            ))

        rows.sort(
            key=lambda value: (
                value[0],
                value[1],
                value[2]
            )
        )

        seen = set()
        output = []

        for _score, _number, _name, item in rows:
            key = (
                item["id"].upper(),
                round(item["ra_deg"], 7),
                round(item["dec_deg"], 7),
            )

            if key in seen:
                continue

            seen.add(key)

            item["ra_hms"] = format_ra_hms(item["ra_deg"])
            item["dec_dms"] = format_dec_dms(item["dec_deg"])

            output.append(item)

            if len(output) >= int(limit):
                break

        return output



    def overlay_items(
        self,
        center_ra_deg,
        center_dec_deg,
        jacobian,
        pixel_scale_arcsec,
        limit=80
    ):
        """
        Retourne uniquement les objets OpenNGC réellement présents
        dans le champ du dernier plate solve.

        Un préfiltre RA/DEC très léger évite de projeter les ~14k objets.
        La projection finale utilise exactement le même repère WCS local
        que le Push-To.
        """
        center_ra_deg = float(center_ra_deg)
        center_dec_deg = float(center_dec_deg)
        pixel_scale_arcsec = max(1e-6, float(pixel_scale_arcsec))

        half_w_deg = WIDTH * pixel_scale_arcsec / 7200.0
        half_h_deg = HEIGHT * pixel_scale_arcsec / 7200.0

        # Marge pour les objets étendus près du bord.
        margin_deg = 0.40

        cos_dec = max(
            0.08,
            abs(float(np.cos(np.deg2rad(center_dec_deg))))
        )

        rows = []

        for record in self.records:
            ra = float(record["ra_deg"])
            dec = float(record["dec_deg"])

            dra = ((ra - center_ra_deg + 180.0) % 360.0) - 180.0
            east_approx = dra * cos_dec
            north_approx = dec - center_dec_deg

            if abs(east_approx) > half_w_deg + margin_deg:
                continue

            if abs(north_approx) > half_h_deg + margin_deg:
                continue

            try:
                guidance = target_guidance_from_solution(
                    center_ra_deg,
                    center_dec_deg,
                    ra,
                    dec,
                    jacobian
                )
            except Exception:
                continue

            if not guidance["in_frame"]:
                continue

            if record["messier"]:
                display_id = record["messier"][0]
                family = "Messier"
                family_rank = 0
            elif record["ngc"]:
                display_id = record["ngc"][0]
                family = "NGC"
                family_rank = 1
            elif record["ic"]:
                display_id = record["ic"][0]
                family = "IC"
                family_rank = 2
            else:
                # La couche graphique demandée reste volontairement
                # centrée sur Messier / NGC / IC.
                continue

            magnitude = record["magnitude"]
            major = record.get("major_arcmin")

            # Classement : Messier d'abord, puis objets les plus
            # brillants / étendus. Cela limite la surcharge graphique
            # dans les champs très riches.
            mag_rank = (
                float(magnitude)
                if magnitude is not None
                else 99.0
            )

            size_rank = -(
                float(major)
                if major is not None
                else 0.0
            )

            rows.append((
                family_rank,
                mag_rank,
                size_rank,
                {
                    "id": display_id,
                    "name": record["common_name"],
                    "family": family,
                    "type_label": record["type_label"],
                    "magnitude": magnitude,
                    "magnitude_band": record["magnitude_band"],
                    "x": float(guidance["target_x"]),
                    "y": float(guidance["target_y"]),
                    "major_arcmin": record.get("major_arcmin"),
                    "minor_arcmin": record.get("minor_arcmin"),
                    "position_angle_deg": record.get("position_angle_deg"),
                }
            ))

        rows.sort(key=lambda item: (item[0], item[1], item[2]))

        return [
            item[3]
            for item in rows[:max(1, int(limit))]
        ]



def load_openngc_catalog():
    try:
        main_path, addendum_path = ensure_openngc_files()

        paths = [
            path
            for path in (main_path, addendum_path)
            if path.is_file() and path.stat().st_size > 100
        ]

        if not paths:
            return None, None, (
                "OpenNGC absent et téléchargement impossible"
            )

        catalog = OpenNGCCatalog(paths)

        source = (
            f"OpenNGC local ({len(catalog.records)} objets) — "
            "Mattia Verga / CC BY-SA 4.0"
        )

        return catalog, source, None

    except Exception as exc:
        return None, None, str(exc)



def load_legacy_catalog_search():
    """
    Réutilise le catalogue de l'ancien projet Astro Cam.

    Recherche, dans l'ordre :
    1) catalog.py à côté d'astro_cam.py ;
    2) un ancien catalog.py extrait dans astro_archive ;
    3) catalog.py directement dans l'archive tar.gz créée lors du ménage.

    La fonction historique attendue est :
        search_objects(query) -> list[dict]
    """
    import importlib.util
    import tarfile
    import types
    from pathlib import Path

    candidates = [
        Path(__file__).resolve().parent / "catalog.py",
        Path("/home/arduino/astro_archive/legacy_files/catalog.py"),
        Path("/home/arduino/astro_archive/legacy_files/astro_core/catalog.py"),
    ]

    archive_root = Path("/home/arduino/astro_archive")

    if archive_root.exists():
        try:
            candidates.extend(archive_root.rglob("catalog.py"))
        except OSError:
            pass

    seen = set()

    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue

        if path in seen or not path.is_file():
            continue
        seen.add(path)

        try:
            spec = importlib.util.spec_from_file_location(
                "astrocam_legacy_catalog",
                str(path)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            function = getattr(module, "search_objects", None)

            if callable(function):
                return function, f"fichier {path}"

        except Exception as exc:
            print(
                f"[CATALOG] impossible de charger {path}: {exc}",
                flush=True
            )

    # L'archive que nous avons créée pendant le nettoyage contient
    # normalement astro_unoq/astro_core/catalog.py.
    tar_candidates = []

    if archive_root.exists():
        tar_candidates.extend(
            sorted(archive_root.glob("*.tar.gz"))
        )

        legacy_dir = archive_root / "legacy_files"
        if legacy_dir.exists():
            tar_candidates.extend(
                sorted(legacy_dir.rglob("*.tar.gz"))
            )

    for tar_path in tar_candidates:
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                member_names = [
                    name for name in tar.getnames()
                    if name.endswith("/astro_core/catalog.py")
                    or name.endswith("/catalog.py")
                ]

                for member_name in member_names:
                    extracted = tar.extractfile(member_name)
                    if extracted is None:
                        continue

                    source = extracted.read().decode(
                        "utf-8",
                        errors="replace"
                    )

                    module = types.ModuleType(
                        "astrocam_legacy_catalog_from_archive"
                    )
                    module.__file__ = (
                        f"{tar_path}!/{member_name}"
                    )

                    exec(
                        compile(
                            source,
                            module.__file__,
                            "exec"
                        ),
                        module.__dict__
                    )

                    function = getattr(
                        module,
                        "search_objects",
                        None
                    )

                    if callable(function):
                        return (
                            function,
                            f"archive {tar_path.name}:{member_name}"
                        )

        except Exception as exc:
            print(
                f"[CATALOG] lecture archive {tar_path}: {exc}",
                flush=True
            )

    return None, (
        "catalogue historique introuvable. "
        "Le fichier astro_core/catalog.py de l'ancien projet est requis."
    )


def normalize_catalog_item(item):
    """
    Normalise les formats possibles du vieux catalogue.
    """
    if not isinstance(item, dict):
        return None

    identifier = (
        item.get("id")
        or item.get("catalog")
        or item.get("designation")
        or item.get("name")
        or ""
    )

    name = (
        item.get("name")
        or item.get("nom")
        or item.get("common_name")
        or identifier
    )

    ra = (
        item.get("ra_deg")
        if item.get("ra_deg") is not None
        else item.get("ra")
    )

    dec = (
        item.get("dec_deg")
        if item.get("dec_deg") is not None
        else item.get("dec")
    )

    try:
        ra = float(ra)
        dec = float(dec)
    except (TypeError, ValueError):
        return None

    identifier = str(identifier).strip()
    name = str(name).strip()

    if not identifier:
        identifier = name or "Objet"

    prefix = identifier.upper().replace(" ", "")

    if prefix.startswith("M") and not prefix.startswith("MEL"):
        family = "Messier"
    elif prefix.startswith("NGC"):
        family = "NGC"
    elif prefix.startswith("IC"):
        family = "IC"
    else:
        family = str(item.get("family") or item.get("catalogue") or "Autre")

    return {
        "id": identifier,
        "name": name or identifier,
        "ra_deg": ra,
        "dec_deg": dec,
        "family": family,
    }


def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(float(ra1_deg))
    dec1 = np.deg2rad(float(dec1_deg))
    ra2 = np.deg2rad(float(ra2_deg))
    dec2 = np.deg2rad(float(dec2_deg))

    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra2 - ra1)
    )

    return float(
        np.rad2deg(
            np.arccos(np.clip(cos_sep, -1.0, 1.0))
        )
    )


def target_guidance_from_solution(
    center_ra_deg,
    center_dec_deg,
    target_ra_deg,
    target_dec_deg,
    jacobian
):
    """
    Convertit la direction céleste vers la cible en direction/pixel caméra.

    Le WCS complet disparaît après le solve, mais le Jacobien tangent local
    [Est/Nord par pixel] suffit pour :
    - placer précisément une cible proche/dans le champ ;
    - donner la bonne direction vers une cible hors champ.
    """
    ra0 = np.deg2rad(float(center_ra_deg))
    dec0 = np.deg2rad(float(center_dec_deg))
    ra = np.deg2rad(float(target_ra_deg))
    dec = np.deg2rad(float(target_dec_deg))

    dra = np.arctan2(
        np.sin(ra - ra0),
        np.cos(ra - ra0)
    )

    # Composantes tangentes / bearing vers la cible.
    east_dir = np.cos(dec) * np.sin(dra)
    north_dir = (
        np.cos(dec0) * np.sin(dec)
        - np.sin(dec0) * np.cos(dec) * np.cos(dra)
    )

    direction_norm = float(np.hypot(east_dir, north_dir))

    if direction_norm < 1e-12:
        east_unit = 0.0
        north_unit = 0.0
    else:
        east_unit = float(east_dir / direction_norm)
        north_unit = float(north_dir / direction_norm)

    separation = angular_separation_deg(
        center_ra_deg,
        center_dec_deg,
        target_ra_deg,
        target_dec_deg
    )

    # Décomposition de la correction angulaire sur les deux axes célestes.
    # C'est cette information qu'on utilisera pour donner deux commandes
    # indépendantes à l'utilisateur :
    #   - AD : Est / Ouest
    #   - DEC : Nord / Sud
    #
    # Pour notre Push-To manuel, cette décomposition est beaucoup plus
    # utile qu'une unique flèche diagonale.
    east_component_deg = float(separation * east_unit)
    north_component_deg = float(separation * north_unit)

    j = np.asarray(jacobian, dtype=np.float64).reshape((2, 2))
    det = float(np.linalg.det(j))

    if abs(det) < 1e-12:
        raise RuntimeError("Jacobien WCS absent ou invalide.")

    inv_j = np.linalg.inv(j)

    # Axes célestes dans les coordonnées écran/pixel.
    # Comme ils passent par le même Jacobien WCS que la cible, ils permettent
    # de vérifier visuellement l'orientation réelle de la caméra.
    east_pixel = inv_j @ np.array([1.0, 0.0], dtype=np.float64)
    north_pixel = inv_j @ np.array([0.0, 1.0], dtype=np.float64)

    east_norm = float(np.hypot(east_pixel[0], east_pixel[1]))
    north_norm = float(np.hypot(north_pixel[0], north_pixel[1]))

    if east_norm > 1e-12:
        east_screen_x = float(east_pixel[0] / east_norm)
        east_screen_y = float(east_pixel[1] / east_norm)
    else:
        east_screen_x = 1.0
        east_screen_y = 0.0

    if north_norm > 1e-12:
        north_screen_x = float(north_pixel[0] / north_norm)
        north_screen_y = float(north_pixel[1] / north_norm)
    else:
        north_screen_x = 0.0
        north_screen_y = -1.0

    # Direction pixel, toujours utilisable même si la cible est très loin.
    pixel_direction = inv_j @ np.array(
        [east_unit, north_unit],
        dtype=np.float64
    )

    dx_dir = float(pixel_direction[0])
    dy_dir = float(pixel_direction[1])

    norm_px = float(np.hypot(dx_dir, dy_dir))

    if norm_px > 1e-12:
        dx_dir /= norm_px
        dy_dir /= norm_px

    # IMPORTANT :
    # dx_dir/dy_dir indiquent où se trouve la cible DANS L'IMAGE.
    #
    # Mais lorsqu'on déplace physiquement la lunette, le champ projeté
    # sur le capteur se déplace dans le sens opposé. Pour guider la main
    # de l'utilisateur ("déplace le télescope dans cette direction"),
    # il faut donc inverser LES DEUX axes.
    move_dx_dir = -dx_dir
    move_dy_dir = -dy_dir

    # Projection gnomonique exacte si la cible se trouve dans l'hémisphère
    # visible du plan tangent.
    denom = float(
        np.sin(dec0) * np.sin(dec)
        + np.cos(dec0) * np.cos(dec) * np.cos(dra)
    )

    target_x = None
    target_y = None
    in_front = denom > 1e-8

    if in_front:
        xi = np.cos(dec) * np.sin(dra) / denom
        eta = (
            np.cos(dec0) * np.sin(dec)
            - np.sin(dec0) * np.cos(dec) * np.cos(dra)
        ) / denom

        east_deg = float(np.rad2deg(xi))
        north_deg = float(np.rad2deg(eta))

        dx, dy = inv_j @ np.array(
            [east_deg, north_deg],
            dtype=np.float64
        )

        target_x = float(WIDTH / 2.0 + dx)
        target_y = float(HEIGHT / 2.0 + dy)

    in_frame = (
        target_x is not None
        and target_y is not None
        and 0.0 <= target_x < WIDTH
        and 0.0 <= target_y < HEIGHT
    )

    return {
        "separation_deg": separation,
        # direction_x/y = position céleste de la cible dans l'image.
        "direction_x": dx_dir,
        "direction_y": dy_dir,

        # move_direction_x/y = sens PHYSIQUE dans lequel déplacer
        # la lunette / le télescope.
        "move_direction_x": move_dx_dir,
        "move_direction_y": move_dy_dir,

        "target_x": target_x,
        "target_y": target_y,
        "in_frame": bool(in_frame),
        "east_screen_x": east_screen_x,
        "east_screen_y": east_screen_y,
        "north_screen_x": north_screen_x,
        "north_screen_y": north_screen_y,
        "east_component_deg": east_component_deg,
        "north_component_deg": north_component_deg,
    }


def rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def parse_fits_to_uint16(blob):
    pos = 0
    header = {}
    found_end = False

    while not found_end:
        block = blob[pos:pos + 2880]
        if len(block) != 2880:
            raise RuntimeError("En-tête FITS incomplet.")
        pos += 2880

        for i in range(0, 2880, 80):
            card = block[i:i + 80].decode("ascii", errors="replace")
            key = card[:8].strip()

            if key == "END":
                found_end = True
                break

            if len(card) >= 10 and card[8] == "=":
                raw = card[10:80].split("/", 1)[0].strip()

                if raw.startswith("'"):
                    val = raw.strip().strip("'").strip()
                elif raw in ("T", "F"):
                    val = raw == "T"
                else:
                    try:
                        if any(c in raw.upper() for c in (".", "E")):
                            val = float(raw)
                        else:
                            val = int(raw)
                    except ValueError:
                        val = raw

                header[key] = val

    bitpix = int(header.get("BITPIX", 0))
    w = int(header.get("NAXIS1", 0))
    h = int(header.get("NAXIS2", 0))

    if bitpix != 16:
        raise RuntimeError(f"BITPIX={bitpix}, attendu 16.")
    if (w, h) != (WIDTH, HEIGHT):
        raise RuntimeError(f"Dimensions {w}x{h}, attendu {WIDTH}x{HEIGHT}.")

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
    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.sock = None

        # IMPORTANT :
        # Tous les attributs que les callbacks XML peuvent lire doivent
        # exister AVANT le premier parser.Parse().
        #
        # V33 initialisait self.device après parser.Parse("<indi_stream>"),
        # ce qui déclenchait _start() trop tôt et provoquait :
        # AttributeError: 'IndiClient' object has no attribute 'device'
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
        self.sock = socket.create_connection((HOST, INDI_PORT), timeout=5.0)
        self.sock.settimeout(0.5)

        # Le port INDI peut être prêt avant le HotPlug de la caméra.
        # On lance seulement la découverte ici.
        self.request_properties()
        self.pump_for(0.2)

    def request_properties(self):
        if self.device:
            self.send(
                f'<getProperties version="1.7" '
                f'device="{html.escape(self.device, quote=True)}"/>'
            )
        else:
            # Découverte générique tant que le vrai nom INDI de la SC715C
            # n'est pas connu. Nécessaire car USB2/USB3 changent le libellé.
            self.send('<getProperties version="1.7"/>')

    def _device_name(self):
        if not self.device:
            raise RuntimeError(
                "Nom INDI de la SC715C pas encore découvert."
            )
        return self.device

    def enable_blob(self):
        device = self._device_name()
        self.send(
            f'<enableBLOB device="{html.escape(device, quote=True)}" '
            f'name="CCD1">Also</enableBLOB>'
        )

    def wait_for_properties(
        self,
        required,
        timeout=20.0,
        refresh_interval=0.8,
        label="propriétés INDI"
    ):
        required = set(required)
        deadline = time.monotonic() + timeout
        next_refresh = 0.0

        while (
            not self.stop_event.is_set()
            and time.monotonic() < deadline
        ):
            missing = required.difference(self.properties)

            if not missing:
                return

            now = time.monotonic()

            if now >= next_refresh:
                self.request_properties()
                next_refresh = now + refresh_interval

            self.pump_once()

        if self.stop_event.is_set():
            return

        missing = sorted(required.difference(self.properties))

        raise TimeoutError(
            f"Timeout en attente de {label}. "
            f"Manquantes: {', '.join(missing)}"
        )

    def close(self):
        if self.sock:
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
        self.sock.sendall((xml + "\n").encode("utf-8"))

    def switch(self, prop, values):
        device = self._device_name()
        parts = [
            f'<newSwitchVector device="{html.escape(device, quote=True)}" '
            f'name="{html.escape(prop, quote=True)}">'
        ]
        for name, val in values.items():
            parts.append(
                f'<oneSwitch name="{html.escape(name, quote=True)}">{val}</oneSwitch>'
            )
        parts.append("</newSwitchVector>")
        self.send("".join(parts))

    def number(self, prop, values):
        device = self._device_name()
        parts = [
            f'<newNumberVector device="{html.escape(device, quote=True)}" '
            f'name="{html.escape(prop, quote=True)}">'
        ]
        for name, val in values.items():
            parts.append(
                f'<oneNumber name="{html.escape(name, quote=True)}">{val}</oneNumber>'
            )
        parts.append("</newNumberVector>")
        self.send("".join(parts))

    def _start(self, name, attrs):
        announced_device = attrs.get("device")

        # Découverte dynamique du vrai nom INDI.
        #
        # Le nom peut changer suivant l'énumération USB :
        # USB3 : SVBONY2 SVBONY_SC715C
        # USB2 : SVBONY_SC715C(USB2.0)
        #
        # Tant que le modèle SC715C est présent dans le nom, on l'accepte.
        if (
            announced_device
            and DEVICE_MATCH in announced_device
            and self.device != announced_device
        ):
            previous = self.device
            self.device = announced_device

            if previous is None:
                print(
                    f"[INDI CLIENT] SC715C découverte : {self.device}",
                    flush=True
                )
            else:
                print(
                    f"[INDI CLIENT] nom SC715C mis à jour : "
                    f"{previous} -> {self.device}",
                    flush=True
                )

        is_our_camera = (
            self.device is not None
            and announced_device == self.device
        )

        if (
            is_our_camera
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
                is_our_camera
                and attrs.get("name") == "CCD1"
            )
        elif name == "oneBLOB" and self.in_blob_vector:
            self.collect_blob = True
            self.blob_chunks = []
            self.blob_attrs = dict(attrs)
        elif name == "message":
            if announced_device in (None, self.device):
                self.last_message = attrs.get("message")

    def _chars(self, data):
        if self.collect_blob:
            self.blob_chunks.append(data)

    def _end(self, name):
        if name == "oneBLOB" and self.collect_blob:
            encoded = "".join(self.blob_chunks)
            self.blob_chunks = []
            self.collect_blob = False

            try:
                payload = base64.b64decode(encoded, validate=False)
                fmt = self.blob_attrs.get("format", "")
                if fmt.endswith(".z"):
                    payload = zlib.decompress(payload)
                self.ready_blobs.append(payload)
            except Exception as exc:
                self.ready_blobs.append(exc)

        elif name == "setBLOBVector":
            self.in_blob_vector = False

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
        while (
            not self.stop_event.is_set()
            and time.monotonic() < deadline
        ):
            self.pump_once()

    def wait_blob(self, timeout):
        deadline = time.monotonic() + timeout

        while (
            not self.stop_event.is_set()
            and time.monotonic() < deadline
        ):
            if self.ready_blobs:
                obj = self.ready_blobs.pop(0)
                if isinstance(obj, Exception):
                    raise obj
                return obj
            self.pump_once()

        if self.stop_event.is_set():
            return None

        raise TimeoutError(
            f"Aucun BLOB CCD1 après {timeout:.1f}s. "
            f"Dernier message={self.last_message!r}"
        )


def acquisition_process(
    shared_raw,
    raw_lock,
    raw_sequence,
    stop_event,
    acq_count,
    last_acq_ms,
    exposure_ms_value,
    gain_value,
    binning_mode_value,
    heartbeat_acq,
    indi_connected
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shared = np.frombuffer(
        shared_raw,
        dtype=np.uint16,
        count=PIXELS
    ).reshape((HEIGHT, WIDTH))

    client = None
    indi_connected.value = 0

    print(f"[ACQ pid={os.getpid()}] démarrage", flush=True)

    required_camera_properties = {
        "UPLOAD_MODE",
        "CCD_CAPTURE_FORMAT",
        "CCD_BINNING",
        "CCD_BINNING_MODE",
        "CCD_CONTROLS",
        "CCD_EXPOSURE",
    }

    def desired_binning_mode():
        return 1 if int(binning_mode_value.value) else 0

    def desired_gain():
        value = int(round(float(gain_value.value)))
        return max(GAIN_MIN, min(GAIN_MAX, value))

    def configure_binning_mode(active_client, mode_value):
        active_client.wait_for_properties(
            {"CCD_BINNING_MODE"},
            timeout=5.0,
            label="CCD_BINNING_MODE"
        )

        if mode_value:
            active_client.switch(
                "CCD_BINNING_MODE",
                {
                    "TC_BINNING_AVG": "Off",
                    "TC_BINNING_ADD": "On"
                }
            )
            mode_name = "ADD"
        else:
            active_client.switch(
                "CCD_BINNING_MODE",
                {
                    "TC_BINNING_AVG": "On",
                    "TC_BINNING_ADD": "Off"
                }
            )
            mode_name = "AVG"

        print(
            f"[ACQ] mode binning appliqué : {mode_name}",
            flush=True
        )

    def configure_gain(active_client, gain):
        active_client.wait_for_properties(
            {"CCD_CONTROLS"},
            timeout=5.0,
            label="CCD_CONTROLS"
        )
        active_client.number("CCD_CONTROLS", {"Gain": gain})
        print(
            f"[ACQ] nouveau gain appliqué : {gain}",
            flush=True
        )

    def connect_and_configure_camera(
        active_client,
        gain,
        binning_mode,
        reconnect=False
    ):
        """
        Ouvre un client INDI, attend le HotPlug réel de la SC715C,
        connecte la caméra puis réapplique toute la configuration nécessaire.
        """
        active_client.connect()

        if reconnect:
            print(
                "[ACQ RECUP] attente réapparition SC715C...",
                flush=True
            )
        else:
            print(
                "[ACQ] attente apparition SC715C dans INDI...",
                flush=True
            )

        active_client.wait_for_properties(
            {"CONNECTION"},
            timeout=20.0,
            label="la caméra SC715C / CONNECTION"
        )

        if reconnect:
            print(
                f"[ACQ RECUP] SC715C détectée sous le nom "
                f"{active_client.device!r} — reconnexion caméra...",
                flush=True
            )
        else:
            print(
                f"[ACQ] SC715C détectée par INDI sous le nom "
                f"{active_client.device!r} — connexion caméra...",
                flush=True
            )

        active_client.switch(
            "CONNECTION",
            {"CONNECT": "On", "DISCONNECT": "Off"}
        )

        active_client.wait_for_properties(
            required_camera_properties,
            timeout=25.0,
            label="les propriétés caméra après connexion"
        )

        # Le flux BLOB n'est activé qu'une fois les propriétés disponibles.
        active_client.enable_blob()
        active_client.pump_for(0.3)

        active_client.switch(
            "UPLOAD_MODE",
            {
                "UPLOAD_CLIENT": "On",
                "UPLOAD_LOCAL": "Off",
                "UPLOAD_BOTH": "Off",
            }
        )

        active_client.switch(
            "CCD_CAPTURE_FORMAT",
            {"INDI_RGB": "Off", "INDI_RAW": "On"}
        )

        active_client.number(
            "CCD_BINNING",
            {"HOR_BIN": BINNING, "VER_BIN": BINNING}
        )

        configure_binning_mode(active_client, binning_mode)
        configure_gain(active_client, gain)

        active_client.pump_for(0.5)

        indi_connected.value = 1

        if reconnect:
            print(
                "[ACQ RECUP] caméra reconnectée et reconfigurée.",
                flush=True
            )
        else:
            print(
                "[ACQ] connexion INDI persistante établie — "
                "propriétés caméra prêtes",
                flush=True
            )

    def clean_disconnect_camera(active_client, label="[ACQ]"):
        """
        Demande une déconnexion INDI propre avant de fermer le socket.
        Cela réduit les redémarrages brutaux du SDK caméra.
        """
        if active_client is None:
            return

        try:
            if active_client.sock is not None:
                active_client.switch(
                    "CONNECTION",
                    {"CONNECT": "Off", "DISCONNECT": "On"}
                )
                active_client.pump_for(CAMERA_DISCONNECT_SETTLE_S)
                print(
                    f"{label} déconnexion caméra demandée proprement.",
                    flush=True
                )
        except Exception as exc:
            print(
                f"{label} déconnexion propre impossible : {exc}",
                flush=True
            )

    def rearm_blob_and_retry(active_client, exposure_s):
        """
        Niveau 1 de récupération :
        - vide les anciens BLOB éventuels ;
        - réactive le flux CCD1 ;
        - réaffirme UPLOAD_CLIENT ;
        - relance exactement la même exposition.
        """
        print(
            "[ACQ RECUP] niveau 1/2 : réarmement du flux CCD1/BLOB...",
            flush=True
        )

        active_client.ready_blobs.clear()
        active_client.enable_blob()

        active_client.switch(
            "UPLOAD_MODE",
            {
                "UPLOAD_CLIENT": "On",
                "UPLOAD_LOCAL": "Off",
                "UPLOAD_BOTH": "Off",
            }
        )

        active_client.pump_for(CAMERA_REARM_SETTLE_S)

        active_client.number(
            "CCD_EXPOSURE",
            {"CCD_EXPOSURE_VALUE": exposure_s}
        )

        blob = active_client.wait_blob(FRAME_TIMEOUT_S)

        print(
            "[ACQ RECUP] niveau 1 réussi : flux BLOB revenu.",
            flush=True
        )

        return blob

    def reconnect_camera_and_retry(
        old_client,
        exposure_s,
        gain,
        binning_mode
    ):
        """
        Niveau 2 de récupération :
        déconnexion caméra + fermeture du client INDI + pause volontaire,
        puis nouveau client, reconnexion/reconfiguration et nouvelle exposition.

        On ne redémarre pas indiserver ici : le serveur reste vivant.
        """
        nonlocal client

        print(
            "[ACQ RECUP] niveau 2/2 : reconnexion complète de la caméra...",
            flush=True
        )

        indi_connected.value = 0

        clean_disconnect_camera(
            old_client,
            label="[ACQ RECUP]"
        )

        try:
            old_client.close()
        except Exception:
            pass

        if stop_event.wait(CAMERA_RECONNECT_DELAY_S):
            return None, old_client

        new_client = IndiClient(stop_event)
        client = new_client

        connect_and_configure_camera(
            new_client,
            gain,
            binning_mode,
            reconnect=True
        )

        new_client.ready_blobs.clear()

        new_client.number(
            "CCD_EXPOSURE",
            {"CCD_EXPOSURE_VALUE": exposure_s}
        )

        blob = new_client.wait_blob(FRAME_TIMEOUT_S)

        print(
            "[ACQ RECUP] niveau 2 réussi : première image reçue.",
            flush=True
        )

        return blob, new_client

    try:
        client = IndiClient(stop_event)

        applied_gain = desired_gain()
        applied_binning_mode = desired_binning_mode()

        connect_and_configure_camera(
            client,
            applied_gain,
            applied_binning_mode,
            reconnect=False
        )

        while not stop_event.is_set():
            t0 = time.monotonic()

            # Les réglages Web sont appliqués au début de la frame suivante.
            current_binning_mode = desired_binning_mode()

            if applied_binning_mode != current_binning_mode:
                configure_binning_mode(
                    client,
                    current_binning_mode
                )
                applied_binning_mode = current_binning_mode

            current_gain = desired_gain()

            if applied_gain != current_gain:
                configure_gain(client, current_gain)
                applied_gain = current_gain

            current_exp_ms = float(exposure_ms_value.value)
            current_exp_ms = max(
                EXPOSURE_MIN_MS,
                min(EXPOSURE_MAX_MS, current_exp_ms)
            )
            current_exp_s = current_exp_ms / 1000.0

            client.wait_for_properties(
                {"CCD_EXPOSURE"},
                timeout=5.0,
                label="CCD_EXPOSURE"
            )

            client.ready_blobs.clear()

            client.number(
                "CCD_EXPOSURE",
                {"CCD_EXPOSURE_VALUE": current_exp_s}
            )

            try:
                blob = client.wait_blob(FRAME_TIMEOUT_S)

            except TimeoutError as first_timeout:
                print(
                    f"[ACQ RECUP] aucun BLOB : {first_timeout}",
                    flush=True
                )

                # Niveau 1 : simple réarmement du flux.
                try:
                    blob = rearm_blob_and_retry(
                        client,
                        current_exp_s
                    )

                except TimeoutError as rearm_timeout:
                    print(
                        "[ACQ RECUP] réarmement BLOB insuffisant : "
                        f"{rearm_timeout}",
                        flush=True
                    )

                    # Niveau 2 : reconnexion complète, une seule fois.
                    try:
                        blob, client = reconnect_camera_and_retry(
                            client,
                            current_exp_s,
                            current_gain,
                            current_binning_mode
                        )

                        if blob is None:
                            break

                        # Toute la configuration vient d'être réappliquée.
                        applied_gain = current_gain
                        applied_binning_mode = current_binning_mode

                    except Exception as reconnect_exc:
                        raise TimeoutError(
                            "La SC715C est détectée mais aucun BLOB ne revient "
                            "après réarmement puis reconnexion automatique. "
                            "Un reset électrique USB (débrancher/rebrancher la "
                            "caméra) peut être nécessaire. "
                            f"Dernière erreur: {reconnect_exc}"
                        ) from reconnect_exc

            if blob is None:
                break

            img = parse_fits_to_uint16(blob)

            with raw_lock:
                np.copyto(shared, img, casting="no")
                raw_sequence.value += 1
                seq = raw_sequence.value

            cycle_ms = (time.monotonic() - t0) * 1000.0
            last_acq_ms.value = cycle_ms
            heartbeat_acq.value = time.time()

            with acq_count.get_lock():
                acq_count.value += 1

            if seq <= 5 or seq % 20 == 0:
                print(
                    f"[ACQ] frame={seq} "
                    f"exp={current_exp_ms:.0f}ms "
                    f"gain={applied_gain} "
                    f"bin={'ADD' if applied_binning_mode else 'AVG'} "
                    f"cycle={cycle_ms:.1f}ms "
                    f"RSS={rss_mb():.1f} MiB",
                    flush=True
                )

            del img
            del blob

    except Exception as exc:
        print(
            f"[ACQ ERREUR] {type(exc).__name__}: {exc}",
            flush=True
        )
        stop_event.set()

    finally:
        indi_connected.value = 0

        # Fermeture plus douce qu'avant : on demande au driver de libérer
        # explicitement la caméra avant que le main coupe indiserver.
        clean_disconnect_camera(client)

        if client is not None:
            client.close()

        print("[ACQ] arrêt.", flush=True)


def build_preview(
    raw16,
    raw12,
    gray12,
    work_float,
    preview8,
    cv2,
    preview_mode
):
    np.right_shift(raw16, 4, out=raw12)

    cv2.cvtColor(
        raw12,
        cv2.COLOR_BayerGR2GRAY,
        dst=gray12
    )

    sample = gray12[::4, ::4]
    valid = (sample > RAW12_MIN) & (sample < SAT_THRESHOLD)
    vals = sample[valid]

    # Une frame noire, saturée ou encore instable au démarrage ne doit
    # JAMAIS arrêter toute l'application. Auparavant on levait une
    # RuntimeError ici, ce qui stoppait PROC puis le main complet.
    #
    # Si trop peu de pixels sont exploitables, on publie simplement une
    # preview de secours 12 bits -> 8 bits. Le RAW scientifique reste
    # strictement inchangé. Dès qu'une frame normale arrive, le stretch
    # Technique/Ciel noir reprend automatiquement.
    if vals.size < 1000:
        np.copyto(work_float, gray12, casting="unsafe")
        work_float /= 16.0
        np.clip(work_float, 0.0, 255.0, out=work_float)
        np.copyto(preview8, work_float, casting="unsafe")

        background = float(np.median(sample))
        mad = float(np.median(np.abs(sample.astype(np.float32) - background)))
        sigma = max(1.4826 * mad, MAD_FLOOR)

        black = float(RAW12_MIN)
        white = max(
            16.0,
            float(np.percentile(sample, 99.9))
        )

        gr = float(raw12[0::2, 0::2].mean())
        r  = float(raw12[0::2, 1::2].mean())
        b  = float(raw12[1::2, 0::2].mean())
        gb = float(raw12[1::2, 1::2].mean())

        return (
            black,
            white,
            background,
            sigma,
            gr,
            r,
            b,
            gb,
            "ATTENTE_SIGNAL"
        )

    background = float(np.median(vals))
    mad = float(np.median(np.abs(vals - background)))
    sigma = max(1.4826 * mad, MAD_FLOOR)

    if int(preview_mode) == PREVIEW_DARK_SKY:
        mode_name = "CIEL_NOIR"
        white_sigma_span = DARK_WHITE_SIGMA_SPAN
        asinh_strength = DARK_ASINH_STRENGTH
        gamma = DARK_GAMMA

        # Ici on coupe réellement le fond :
        # les pixels du bruit situés sous fond + 2.5 sigma deviennent noirs.
        black = max(
            float(RAW12_MIN),
            background + DARK_BLACK_SIGMA_ABOVE * sigma
        )
    else:
        mode_name = "TECHNIQUE"
        white_sigma_span = TECH_WHITE_SIGMA_SPAN
        asinh_strength = TECH_ASINH_STRENGTH
        gamma = TECH_GAMMA

        # Mode technique V9 : conserve volontairement davantage de fond/bruit
        # pour maximiser la visibilité des signaux faibles.
        black = max(
            float(RAW12_MIN),
            background - TECH_BLACK_SIGMA_OFFSET * sigma
        )

    white_pct = float(np.percentile(vals, WHITE_PERCENTILE))
    white = max(
        white_pct,
        background + white_sigma_span * sigma,
        black + 16.0
    )

    np.copyto(work_float, gray12, casting="unsafe")
    work_float -= black
    work_float /= (white - black)
    np.clip(work_float, 0.0, 1.0, out=work_float)

    work_float *= asinh_strength
    np.arcsinh(work_float, out=work_float)
    work_float /= np.arcsinh(asinh_strength)

    if gamma != 1.0:
        np.power(work_float, gamma, out=work_float)

    work_float *= 255.0
    np.clip(work_float, 0.0, 255.0, out=work_float)
    np.copyto(preview8, work_float, casting="unsafe")

    preview8[gray12 >= SAT_THRESHOLD] = 255
    preview8[gray12 <= RAW12_MIN] = 0

    gr = float(raw12[0::2, 0::2].mean())
    r  = float(raw12[0::2, 1::2].mean())
    b  = float(raw12[1::2, 0::2].mean())
    gb = float(raw12[1::2, 1::2].mean())

    return black, white, background, sigma, gr, r, b, gb, mode_name


def processing_process(
    shared_raw,
    raw_lock,
    raw_sequence,
    shared_preview,
    preview_lock,
    preview_sequence,
    stop_event,
    proc_count,
    skipped_count,
    last_proc_ms,
    background_value,
    black_value,
    white_value,
    heartbeat_proc,
    preview_mode_value
):
    try:
        import cv2
    except ImportError:
        print(
            "[PROC ERREUR] OpenCV manque : sudo apt install -y python3-opencv",
            flush=True
        )
        stop_event.set()
        return

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    raw_shared = np.frombuffer(
        shared_raw,
        dtype=np.uint16,
        count=PIXELS
    ).reshape((HEIGHT, WIDTH))

    preview_shared = np.frombuffer(
        shared_preview,
        dtype=np.uint8,
        count=PIXELS
    ).reshape((HEIGHT, WIDTH))

    raw_local = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    raw12 = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    gray12 = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    work_float = np.empty((HEIGHT, WIDTH), dtype=np.float32)
    preview_local = np.empty((HEIGHT, WIDTH), dtype=np.uint8)

    last_seq = 0

    print(f"[PROC pid={os.getpid()}] démarrage", flush=True)

    try:
        while not stop_event.is_set():
            seq = raw_sequence.value

            if seq == last_seq:
                time.sleep(0.005)
                continue

            with raw_lock:
                seq = raw_sequence.value
                np.copyto(raw_local, raw_shared, casting="no")

            skipped = (
                max(0, seq - last_seq - 1)
                if last_seq
                else 0
            )

            t0 = time.monotonic()

            (
                black, white, background, sigma,
                cfa_gr, cfa_r, cfa_b, cfa_gb, preview_name
            ) = build_preview(
                raw_local,
                raw12,
                gray12,
                work_float,
                preview_local,
                cv2,
                preview_mode_value.value
            )

            with preview_lock:
                np.copyto(preview_shared, preview_local, casting="no")
                preview_sequence.value = seq

            proc_ms = (time.monotonic() - t0) * 1000.0

            last_proc_ms.value = proc_ms
            heartbeat_proc.value = time.time()
            background_value.value = background
            black_value.value = black
            white_value.value = white

            with proc_count.get_lock():
                proc_count.value += 1

            if skipped:
                with skipped_count.get_lock():
                    skipped_count.value += skipped

            if seq <= 5 or seq % 10 == 0:
                print(
                    f"[PROC] seq={seq} skip={skipped} "
                    f"mode={preview_name} "
                    f"black={black:.1f} fond={background:.1f} "
                    f"sigma={sigma:.1f} "
                    f"white={white:.0f} "
                    f"CFA(GR/R/B/GB)="
                    f"{cfa_gr:.0f}/{cfa_r:.0f}/{cfa_b:.0f}/{cfa_gb:.0f} "
                    f"trait={proc_ms:.1f}ms "
                    f"RSS={rss_mb():.1f} MiB",
                    flush=True
                )

            last_seq = seq

    except Exception as exc:
        print(f"[PROC ERREUR] {type(exc).__name__}: {exc}", flush=True)
        stop_event.set()

    finally:
        print("[PROC] arrêt.", flush=True)


def jpeg_process(
    shared_preview,
    preview_lock,
    preview_sequence,
    shared_jpeg,
    jpeg_lock,
    jpeg_size,
    jpeg_sequence,
    stop_event,
    jpeg_count,
    last_jpeg_ms,
    heartbeat_jpeg
):
    try:
        import cv2
    except ImportError:
        print(
            "[JPEG ERREUR] OpenCV manque : sudo apt install -y python3-opencv",
            flush=True
        )
        stop_event.set()
        return

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    preview_shared = np.frombuffer(
        shared_preview,
        dtype=np.uint8,
        count=PIXELS
    ).reshape((HEIGHT, WIDTH))

    jpeg_shared = np.frombuffer(
        shared_jpeg,
        dtype=np.uint8,
        count=MAX_JPEG_BYTES
    )

    local = np.empty((HEIGHT, WIDTH), dtype=np.uint8)
    last_seq = 0

    print(
        f"[JPEG pid={os.getpid()}] démarrage qualité={JPEG_QUALITY}",
        flush=True
    )

    try:
        while not stop_event.is_set():
            seq = preview_sequence.value

            if seq == 0 or seq == last_seq:
                time.sleep(0.005)
                continue

            with preview_lock:
                seq = preview_sequence.value
                np.copyto(local, preview_shared, casting="no")

            t0 = time.monotonic()

            ok, encoded = cv2.imencode(
                ".jpg",
                local,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not ok:
                raise RuntimeError("cv2.imencode() a échoué.")

            data = encoded.reshape(-1)
            size = data.size

            if size > MAX_JPEG_BYTES:
                raise RuntimeError(f"JPEG trop gros : {size} octets.")

            with jpeg_lock:
                jpeg_shared[:size] = data
                jpeg_size.value = size
                jpeg_sequence.value = seq

            jpeg_ms = (time.monotonic() - t0) * 1000.0
            last_jpeg_ms.value = jpeg_ms
            heartbeat_jpeg.value = time.time()

            with jpeg_count.get_lock():
                jpeg_count.value += 1

            if seq <= 5 or seq % 10 == 0:
                print(
                    f"[JPEG] seq={seq} "
                    f"size={size/1024:.1f} KiB "
                    f"enc={jpeg_ms:.1f}ms "
                    f"RSS={rss_mb():.1f} MiB",
                    flush=True
                )

            last_seq = seq

    except Exception as exc:
        print(f"[JPEG ERREUR] {type(exc).__name__}: {exc}", flush=True)
        stop_event.set()

    finally:
        print("[JPEG] arrêt.", flush=True)



def plate_solver_process(
    shared_raw,
    raw_lock,
    raw_sequence,
    stop_event,
    solve_request,
    solve_status,
    solve_available,
    solve_source_seq,
    solve_ra,
    solve_dec,
    solve_scale,
    solve_rotation,
    solve_elapsed,
    solve_message,
    solve_jacobian,
    solve_context,
    polar_valid,
    polar_ra_values,
    polar_dec_values,
    polar_jacobian,
    polar_state,
    polar_message
):
    """
    Process indépendant de plate solving.
    Copie la dernière RAW rapidement, puis travaille hors lock.
    Le FITS temporaire est écrit dans /dev/shm (RAM).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        import shutil
        import subprocess
        import tempfile
        import cv2
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        solve_available.value = 0
        solve_status.value = SOLVE_ERROR
        set_shared_text(
            solve_message,
            f"Dépendance plate solving absente : {exc}"
        )
        print(f"[SOLVE ERREUR] {get_shared_text(solve_message)}", flush=True)
        while not stop_event.is_set():
            time.sleep(0.5)
        return

    command = shutil.which("solve-field")
    if not command:
        solve_available.value = 0
        solve_status.value = SOLVE_ERROR
        set_shared_text(
            solve_message,
            "solve-field introuvable. Astrometry.net doit être installé."
        )
        print(f"[SOLVE ERREUR] {get_shared_text(solve_message)}", flush=True)
        while not stop_event.is_set():
            time.sleep(0.5)
        return

    solve_available.value = 1
    solve_status.value = SOLVE_IDLE
    set_shared_text(solve_message, "Prêt")

    # Priorité CPU plus faible que l'acquisition / traitement live.
    try:
        os.nice(10)
    except OSError:
        pass

    raw_shared = np.frombuffer(
        shared_raw,
        dtype=np.uint16,
        count=PIXELS
    ).reshape((HEIGHT, WIDTH))

    raw_local = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    raw12 = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
    gray12 = np.empty((HEIGHT, WIDTH), dtype=np.uint16)

    expected_scale = (
        206.265
        * (PIXEL_SIZE_UM * BINNING)
        / FOCAL_LENGTH_MM
    )

    print(
        f"[SOLVE pid={os.getpid()}] prêt | "
        f"scale attendue ~{expected_scale:.3f} arcsec/pixel | "
        f"downsample={SOLVE_DOWNSAMPLE}",
        flush=True
    )

    last_request = 0

    while not stop_event.is_set():
        request_id = solve_request.value

        if request_id == last_request:
            time.sleep(0.05)
            continue

        last_request = request_id
        request_context = int(solve_context.value)
        solve_status.value = SOLVE_RUNNING
        solve_elapsed.value = 0.0
        set_shared_text(solve_message, "Résolution en cours...")

        seq = raw_sequence.value
        if seq <= 0:
            solve_status.value = SOLVE_ERROR
            set_shared_text(solve_message, "Aucune frame RAW disponible.")
            continue

        with raw_lock:
            seq = raw_sequence.value
            np.copyto(raw_local, raw_shared, casting="no")

        # solve_source_seq représente uniquement la dernière solution
        # WCS VALIDEE. On ne le change donc pas avant la réussite du solve.
        t0 = time.monotonic()

        try:
            np.right_shift(raw_local, 4, out=raw12)

            cv2.cvtColor(
                raw12,
                cv2.COLOR_BayerGR2GRAY,
                dst=gray12
            )

            ram_tmp = "/dev/shm" if os.path.isdir("/dev/shm") else None

            with tempfile.TemporaryDirectory(
                prefix="astrocam-solve-",
                dir=ram_tmp
            ) as directory:

                input_path = os.path.join(directory, "capture.fits")
                wcs_path = os.path.join(directory, "capture.wcs")

                fits.writeto(
                    input_path,
                    gray12,
                    overwrite=True
                )

                args = [
                    command,
                    input_path,
                    "--overwrite",
                    "--no-plots",
                    "--dir", directory,
                    "--downsample", str(SOLVE_DOWNSAMPLE),
                    "--scale-units", "arcsecperpix",
                    "--scale-low",
                    f"{expected_scale * SOLVE_SCALE_LOW_FACTOR:.4f}",
                    "--scale-high",
                    f"{expected_scale * SOLVE_SCALE_HIGH_FACTOR:.4f}",
                    "--cpulimit", str(SOLVE_TIMEOUT_S),
                ]

                print(
                    f"[SOLVE] demande={request_id} frame={seq} | "
                    f"scale={expected_scale:.3f} arcsec/px | lancement solve-field",
                    flush=True
                )

                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=SOLVE_TIMEOUT_S + 8,
                    check=False
                )

                if result.returncode != 0 or not os.path.exists(wcs_path):
                    tail = (
                        (result.stderr or "") + "\n" + (result.stdout or "")
                    ).strip()[-700:]

                    raise RuntimeError(
                        "Image non résolue"
                        + (f" : {tail}" if tail else "")
                    )

                with fits.open(wcs_path, memmap=False) as hdul:
                    header = hdul[0].header.copy()
                    wcs = WCS(header)

                center_x = WIDTH / 2.0
                center_y = HEIGHT / 2.0

                center = wcs.pixel_to_world(
                    center_x,
                    center_y
                )

                # Jacobien tangent local réel du WCS :
                # déplacement Est/Nord pour +1 pixel X/Y.
                world_x = wcs.pixel_to_world(
                    center_x + 1.0,
                    center_y
                )
                world_y = wcs.pixel_to_world(
                    center_x,
                    center_y + 1.0
                )

                east_x, north_x = center.spherical_offsets_to(world_x)
                east_y, north_y = center.spherical_offsets_to(world_y)

                local_jacobian = (
                    float(east_x.deg),
                    float(east_y.deg),
                    float(north_x.deg),
                    float(north_y.deg),
                )

                matrix = wcs.pixel_scale_matrix

                pixel_scale = float(
                    np.sqrt(abs(np.linalg.det(matrix))) * 3600.0
                )

                rotation = float(
                    np.degrees(
                        np.arctan2(matrix[0, 1], matrix[0, 0])
                    )
                )

                ra_deg = float(center.ra.deg)
                dec_deg = float(center.dec.deg)

            elapsed = time.monotonic() - t0

            solve_ra.value = ra_deg
            solve_dec.value = dec_deg
            solve_scale.value = pixel_scale
            solve_rotation.value = rotation
            solve_elapsed.value = elapsed

            # Le dernier solve, qu'il soit manuel, Push-To ou polaire,
            # publie toujours son orientation locale de champ.
            with solve_jacobian.get_lock():
                for jidx, value in enumerate(local_jacobian):
                    solve_jacobian[jidx] = value

            # Commit en dernier : à partir d'ici le dernier WCS est complet.
            solve_source_seq.value = seq
            solve_status.value = SOLVE_OK

            # Une demande de contexte 1/2/3 appartient à la procédure
            # de mise en station. On mémorise alors ce pointage.
            if request_context in (1, 2, 3):
                idx = request_context - 1

                with polar_ra_values.get_lock():
                    polar_ra_values[idx] = ra_deg

                with polar_dec_values.get_lock():
                    polar_dec_values[idx] = dec_deg

                with polar_valid.get_lock():
                    polar_valid[idx] = 1

                # Le WCS de la troisième position sert de repère
                # pour calculer la cible affichée sur le live.
                if request_context == 3:
                    with polar_jacobian.get_lock():
                        for jidx, value in enumerate(local_jacobian):
                            polar_jacobian[jidx] = value

                polar_state.value = POLAR_COLLECTING
                set_shared_text(
                    polar_message,
                    f"Position {request_context}/3 résolue et mémorisée."
                )

                print(
                    f"[POLAR] position {request_context}/3 mémorisée | "
                    f"RA={ra_deg:.6f}° DEC={dec_deg:.6f}°",
                    flush=True
                )

            solve_context.value = 0

            msg = (
                f"OK — RA {format_ra_hms(ra_deg)} | "
                f"DEC {format_dec_dms(dec_deg)} | "
                f"{pixel_scale:.3f} arcsec/px | "
                f"rotation {rotation:.1f}° | "
                f"{elapsed:.1f}s"
            )
            set_shared_text(solve_message, msg)

            print(
                f"[SOLVE OK] frame={seq} | "
                f"RA={ra_deg:.6f}° DEC={dec_deg:.6f}° | "
                f"scale={pixel_scale:.3f} arcsec/px | "
                f"rot={rotation:.1f}° | "
                f"{elapsed:.2f}s",
                flush=True
            )

        except subprocess.TimeoutExpired:
            if request_context in (1, 2, 3):
                polar_state.value = POLAR_ERROR
                set_shared_text(
                    polar_message,
                    f"Échec du solve pour la position {request_context}/3."
                )
            solve_context.value = 0

            elapsed = time.monotonic() - t0
            solve_elapsed.value = elapsed
            solve_status.value = SOLVE_ERROR
            set_shared_text(
                solve_message,
                f"Timeout plate solving après {elapsed:.1f}s"
            )
            print(
                f"[SOLVE ECHEC] timeout après {elapsed:.1f}s",
                flush=True
            )

        except Exception as exc:
            if request_context in (1, 2, 3):
                polar_state.value = POLAR_ERROR
                set_shared_text(
                    polar_message,
                    f"Erreur position {request_context}/3 : {exc}"
                )
            solve_context.value = 0

            elapsed = time.monotonic() - t0
            solve_elapsed.value = elapsed
            solve_status.value = SOLVE_ERROR

            message = str(exc).replace("\n", " ")
            if len(message) > 450:
                message = message[-450:]

            set_shared_text(solve_message, message)

            print(
                f"[SOLVE ECHEC] frame={seq} | {message}",
                flush=True
            )

    print("[SOLVE] arrêt.", flush=True)


def web_process(
    shared_jpeg,
    jpeg_lock,
    jpeg_size,
    jpeg_sequence,
    stop_event,
    acq_count,
    proc_count,
    jpeg_count,
    skipped_count,
    last_acq_ms,
    last_proc_ms,
    last_jpeg_ms,
    background_value,
    black_value,
    white_value,
    exposure_ms_value,
    gain_value,
    binning_mode_value,
    heartbeat_acq,
    heartbeat_proc,
    heartbeat_jpeg,
    indi_connected,
    solve_request,
    solve_status,
    solve_available,
    solve_source_seq,
    solve_ra,
    solve_dec,
    solve_scale,
    solve_rotation,
    solve_elapsed,
    solve_message,
    solve_jacobian,
    preview_mode_value,
    solve_context,
    polar_valid,
    polar_ra_values,
    polar_dec_values,
    polar_jacobian,
    polar_state,
    polar_error_arcmin,
    polar_axis_ra,
    polar_axis_dec,
    polar_shift_x,
    polar_shift_y,
    polar_ref_x,
    polar_ref_y,
    polar_target_x,
    polar_target_y,
    polar_guidance_active,
    polar_message,
    target_active,
    target_ra,
    target_dec,
    target_id,
    target_name
):
    try:
        import logging
        from flask import Flask, Response, jsonify, request
    except ImportError:
        print(
            "[WEB ERREUR] Flask manque : sudo apt install -y python3-flask",
            flush=True
        )
        stop_event.set()
        return

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Les requêtes /api/status arrivent régulièrement :
    # on garde le terminal pour les vrais événements utiles.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    jpeg_shared = np.frombuffer(
        shared_jpeg,
        dtype=np.uint8,
        count=MAX_JPEG_BYTES
    )

    openngc_catalog, openngc_source, openngc_error = load_openngc_catalog()

    catalog_mode = "OPENNGC" if openngc_catalog is not None else "LEGACY"

    if openngc_catalog is not None:
        catalog_search = None
        catalog_source = openngc_source

        print(
            f"[CATALOG] {catalog_source}",
            flush=True
        )

    else:
        catalog_search, catalog_source = load_legacy_catalog_search()

        if catalog_search is not None:
            print(
                f"[CATALOG] OpenNGC indisponible ({openngc_error}) — "
                f"secours ancien catalogue : {catalog_source}",
                flush=True
            )
        else:
            catalog_source = (
                f"OpenNGC: {openngc_error} ; "
                f"ancien catalogue: {catalog_source}"
            )

            print(
                f"[CATALOG ERREUR] {catalog_source}",
                flush=True
            )

    app = Flask(__name__)

    # Cache de la couche céleste : recalcul uniquement après un nouveau
    # plate solve validé, jamais à chaque rafraîchissement HTTP.
    sky_overlay_cache_seq = -1
    sky_overlay_cache = []

    # Pendant le développement on veut toujours la toute dernière interface,
    # jamais une ancienne page gardée en cache par le navigateur.
    @app.after_request
    def disable_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    PAGE = r"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Astro Cam</title>
<style>
:root{
    color-scheme:dark;
    --bg:#06090d;
    --panel:#0b1118;
    --panel2:#101923;
    --line:#223142;
    --soft:#8ea3b7;
    --text:#eef5fb;
    --accent:#4d8df7;
    --ok:#36c96d;
    --bad:#ff5f62;
    --warn:#e5b545;
}
*{box-sizing:border-box}
html,body{
    margin:0;
    background:#05080c;
    color:var(--text);
    font-family:"Segoe UI",Inter,Roboto,Arial,sans-serif;
}
.topbar{
    position:sticky;
    top:0;
    z-index:50;
    background:rgba(6,10,15,.94);
    border-bottom:1px solid rgba(86,118,151,.22);
    backdrop-filter:blur(10px);
}
header{
    padding:7px 11px;
    display:flex;
    gap:7px;
    align-items:center;
    flex-wrap:wrap;
}
.title{font-weight:750}
.badge,.status,.metric{
    border:1px solid rgba(119,148,179,.16);
    background:linear-gradient(180deg,#17222d,#101820);
    border-radius:999px;
}
.badge{
    padding:4px 9px;
    font-size:11px;
}
.toolbar-spacer{flex:1}
.ghost-btn,.main-btn,.nav-btn{
    border:1px solid rgba(122,156,196,.25);
    color:white;
    cursor:pointer;
    font-weight:700;
    transition:.15s ease;
}
.ghost-btn{
    border-radius:999px;
    padding:6px 11px;
    background:linear-gradient(180deg,#17222d,#101821);
}
.main-btn{
    border-radius:10px;
    padding:7px 12px;
    background:linear-gradient(180deg,#397ceb,#2764c9);
}
.main-btn.secondary{
    background:linear-gradient(180deg,#1a2734,#111a23);
}
.main-btn:disabled{
    opacity:.45;
    cursor:default;
}
.statusbar{
    padding:0 11px 7px;
    display:flex;
    flex-wrap:wrap;
    gap:5px;
}
.status{
    padding:4px 8px;
    display:flex;
    align-items:center;
    gap:5px;
    font-size:10px;
}
.dot{
    width:7px;height:7px;border-radius:50%;background:#73879b;
}
.dot.ok{background:var(--ok)}
.dot.bad{background:var(--bad)}
.dot.wait,.dot.running{background:var(--warn)}

.mode-nav{
    display:flex;
    gap:7px;
    padding:0 10px 8px;
}
.nav-btn{
    border-radius:12px;
    padding:8px 15px;
    background:linear-gradient(180deg,#151f29,#0e161e);
    color:#b9c7d5;
}
.nav-btn.active{
    background:linear-gradient(180deg,#427fe8,#2d68cc);
    border-color:rgba(132,174,255,.55);
    color:white;
    box-shadow:0 5px 16px rgba(48,113,225,.22);
}
.panel-wrap{padding:0 9px}
.controls{
    display:grid;
    grid-template-columns:minmax(260px,1fr) minmax(260px,1fr) auto auto;
    gap:8px;
    overflow:hidden;
    max-height:220px;
    opacity:1;
    transition:.2s ease;
}
body.controls-hidden .controls{
    max-height:0;
    opacity:0;
}
.card,.mode-panel{
    border:1px solid rgba(96,128,163,.18);
    background:linear-gradient(180deg,#0f1821,#0a1118);
    border-radius:13px;
}
.card{
    padding:8px 10px 7px;
}
.control-head{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:8px;
    margin-bottom:4px;
}
.control-title{font-size:12px;font-weight:750}
.unit{font-size:10px;color:var(--soft)}
.exact{
    width:75px;
    background:#121b25;
    border:1px solid rgba(123,156,192,.23);
    color:white;
    border-radius:9px;
    padding:4px 7px;
    font-size:11px;
}
.range-wrap{position:relative;padding-bottom:17px}
input[type=range]{
    width:100%;
    height:20px;
    margin:0;
    accent-color:#6ea8ff;
}
.scale{
    position:absolute;
    left:5px;right:5px;top:20px;
    font-size:9px;
    color:#8fa5ba;
}
.tick{
    position:absolute;
    transform:translateX(-50%);
    white-space:nowrap;
}
.segmented{
    display:flex;
    gap:5px;
}
.segmented button{
    border:1px solid rgba(121,154,189,.2);
    background:#121c26;
    color:#dce8f3;
    border-radius:10px;
    padding:6px 11px;
    font-size:11px;
    font-weight:700;
    cursor:pointer;
}
.segmented button.active{
    background:linear-gradient(180deg,#447fe9,#336fe0);
    color:white;
}
.mode-panel{
    margin:8px 9px;
    padding:9px 11px;
    display:none;
}
.mode-panel.active{display:block}
.mode-title{
    font-size:14px;
    font-weight:800;
    margin-bottom:5px;
}
.mode-help{
    color:#a9bac9;
    font-size:11px;
    line-height:1.45;
}
.polar-actions{
    display:flex;
    flex-wrap:wrap;
    gap:7px;
    margin-top:8px;
}
.polar-grid{
    display:grid;
    grid-template-columns:repeat(3,minmax(160px,1fr));
    gap:7px;
    margin-top:8px;
}
.polar-step{
    padding:7px 9px;
    border-radius:10px;
    background:#0c141c;
    border:1px solid rgba(111,143,177,.13);
    font-size:11px;
}
.polar-step strong{display:block;margin-bottom:3px}
.polar-good{color:#64dc8a}
.polar-warn{color:#f2c85b}
.polar-error{color:#ff7b7d}
.metrics{
    display:flex;
    gap:7px;
    flex-wrap:wrap;
    padding:0 10px 8px;
}
.metric{
    padding:3px 8px;
    font-size:10px;
    color:#a9bbca;
}
.metric strong{color:#eff6fd}

main{
    background:#000;
    width:100%;
    display:flex;
    justify-content:center;
}
.viewer{
    width:100%;
    display:flex;
    justify-content:center;
}
.viewer-frame{
    position:relative;
    display:inline-block;
    max-width:100%;
    line-height:0;
}
#liveImage{
    display:block;
    width:auto;
    height:auto;
    max-width:100%;
    max-height:calc(100vh - 230px);
    object-fit:contain;
}
body.controls-hidden #liveImage{
    max-height:calc(100vh - 155px);
}
#overlayCanvas{
    position:absolute;
    inset:0;
    width:100%;
    height:100%;
    cursor:default;
}
#overlayCanvas.select-star{cursor:crosshair}
.overlay-label{
    position:absolute;
    left:12px;
    bottom:12px;
    max-width:min(620px,80%);
    padding:7px 10px;
    border-radius:10px;
    background:rgba(5,9,13,.76);
    border:1px solid rgba(132,174,255,.22);
    color:#eef5fb;
    line-height:1.3;
    font-size:12px;
    pointer-events:none;
}
.overlay-label:empty{display:none}

@media(max-width:980px){
    .controls{grid-template-columns:1fr}
    .polar-grid{grid-template-columns:1fr}
    #liveImage{max-height:none}
}


.mount-mode-row{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:5px;
    margin-bottom:6px;
}
.mount-mode-row button{
    min-height:29px;
}
.site-altaz{
    display:grid;
    grid-template-columns:1fr 1fr auto;
    gap:5px;
    align-items:center;
    margin-bottom:6px;
}
.site-altaz input{
    width:100%;
    min-width:0;
    height:29px;
    padding:4px 7px;
    border-radius:7px;
    border:1px solid #263846;
    background:#0b141c;
    color:#dceaf4;
    font-size:11px;
}
.site-altaz button{
    height:29px;
    padding:0 9px;
    border-radius:7px;
}
.site-note{
    margin:-1px 0 7px;
    font-size:10px;
    color:#8fa6b7;
    line-height:1.25;
}
.site-note.good{color:#73dfa0}
.site-note.bad{color:#ffb36b}
.site-altaz.hidden,
.site-note.hidden{
    display:none;
}

.catalog-toolbar{
    display:grid;
    grid-template-columns:auto minmax(220px,1fr);
    gap:7px;
    margin-top:8px;
}
.catalog-filter{
    display:flex;
    gap:5px;
}
.catalog-filter button{
    border:1px solid rgba(121,154,189,.2);
    background:#121c26;
    color:#cbd8e4;
    border-radius:10px;
    padding:7px 10px;
    font-size:11px;
    font-weight:700;
    cursor:pointer;
}
.catalog-filter button.active{
    background:linear-gradient(180deg,#447fe9,#336fe0);
    color:white;
}
.catalog-search{
    width:100%;
    border:1px solid rgba(123,156,192,.23);
    background:#0d151d;
    color:white;
    border-radius:10px;
    padding:7px 10px;
    outline:none;
}
.sky-overlay-row{
    display:flex;
    align-items:center;
    gap:8px;
    margin:5px 0 2px;
}
.sky-overlay-btn{
    flex:0 0 auto;
    border:1px solid rgba(121,154,189,.22);
    background:#101923;
    color:#b9c8d6;
    border-radius:8px;
    padding:6px 9px;
    font-size:10px;
    font-weight:700;
    cursor:pointer;
}
.sky-overlay-btn.active{
    border-color:rgba(89,205,195,.55);
    background:rgba(33,117,112,.22);
    color:#8de4dc;
}
.sky-overlay-info{
    min-width:0;
    color:#71889b;
    font-size:9px;
    white-space:nowrap;
    overflow:hidden;
    text-overflow:ellipsis;
}
.catalog-credit{
    margin:4px 2px 7px;
    font-size:9px;
    line-height:1.2;
    color:#617888;
    user-select:none;
}

.catalog-results{
    margin-top:7px;
    max-height:180px;
    overflow:auto;
    display:grid;
    gap:5px;
}
.catalog-item{
    width:100%;
    text-align:left;
    border:1px solid rgba(111,143,177,.13);
    background:#0c141c;
    color:#e7f0f8;
    border-radius:10px;
    padding:7px 9px;
    cursor:pointer;
}
.catalog-item:hover{
    border-color:rgba(110,168,255,.45);
}
.catalog-item strong{
    display:block;
    font-size:12px;
}
.catalog-item small{
    display:block;
    margin-top:2px;
    color:#92a6b9;
}
.target-card{
    margin-top:8px;
    padding:8px 10px;
    border-radius:11px;
    border:1px solid rgba(93,139,202,.25);
    background:linear-gradient(180deg,#101c28,#0b131b);
    font-size:11px;
}
.target-card strong{
    font-size:13px;
}
.push-actions{
    display:flex;
    flex-wrap:wrap;
    gap:7px;
    margin-top:8px;
}
.push-state{
    margin-top:7px;
    color:#b8c7d4;
    font-size:11px;
    line-height:1.4;
}


/* ============================================================
   V18 — LIVE FIRST
   Le LiveView est le coeur permanent de l'interface.
   Le panneau latéral est le seul élément scrollable sur desktop.
   ============================================================ */

html,
body{
    width:100%;
    height:100%;
    overflow:hidden;
}

body{
    margin:0;
}

.app-shell{
    width:100%;
    height:100vh;
    min-height:0;
    display:flex;
    flex-direction:column;
    overflow:hidden;
    background:#05080c;
}

.topbar{
    position:static;
    flex:0 0 auto;
    z-index:20;
    background:#060a0f;
    border-bottom:1px solid rgba(86,118,151,.20);
    backdrop-filter:none;
}

.app-header{
    min-height:42px;
    padding:5px 8px 3px;
    display:flex;
    align-items:center;
    gap:8px;
}

.brand-line{
    min-width:0;
    display:flex;
    align-items:center;
    gap:5px;
    flex-wrap:wrap;
}

.title{
    margin-right:3px;
    white-space:nowrap;
    font-size:13px;
}

.badge{
    padding:3px 7px;
    font-size:9px;
}

.header-right{
    margin-left:auto;
    display:flex;
    align-items:center;
    gap:7px;
}

.statusbar{
    padding:0;
    display:flex;
    gap:3px;
    flex-wrap:nowrap;
}

.status{
    padding:3px 6px;
    font-size:9px;
    white-space:nowrap;
}

.dot{
    width:6px;
    height:6px;
}

.compact-hide-btn{
    padding:5px 9px;
    font-size:10px;
    white-space:nowrap;
}

.mode-nav{
    height:38px;
    padding:2px 8px 5px;
    display:flex;
    align-items:center;
    gap:5px;
}

.nav-btn{
    padding:6px 12px;
    border-radius:9px;
    font-size:11px;
}

.workspace{
    flex:1 1 auto;
    min-height:0;
    min-width:0;
    display:grid;
    grid-template-columns:minmax(0,1fr) 370px;
    gap:0;
    overflow:hidden;
}

.viewer-pane{
    min-width:0;
    min-height:0;
    width:auto;
    background:#000;
    display:flex;
    align-items:center;
    justify-content:center;
    padding:8px;
    overflow:hidden;
}

.viewer{
    width:100%;
    height:100%;
    min-width:0;
    min-height:0;
    display:flex;
    align-items:center;
    justify-content:center;
}

.viewer-frame{
    position:relative;
    overflow:hidden;
    width:min(
        100%,
        calc((100vh - 92px) * 16 / 9)
    );
    max-width:100%;
    aspect-ratio:16 / 9;
    display:block;
    line-height:0;
}

#liveImage{
    position:absolute;
    inset:0;
    display:block;
    width:100%;
    height:100%;
    max-width:none;
    max-height:none;
    object-fit:contain;
    background:#000;
    transform-origin:50% 50%;
    will-change:transform;
}

body.controls-hidden #liveImage{
    max-height:none;
}

#overlayCanvas{
    position:absolute;
    inset:0;
    width:100%;
    height:100%;
    transform-origin:50% 50%;
    will-change:transform;
}

.orientation-badge{
    position:absolute;
    top:10px;
    right:10px;
    z-index:6;
    padding:5px 8px;
    border-radius:999px;
    background:rgba(5,9,13,.78);
    border:1px solid rgba(125,211,252,.28);
    color:#d8edf8;
    font-size:10px;
    line-height:1;
    pointer-events:none;
}

.side-panel{
    min-width:0;
    min-height:0;
    padding:7px;
    overflow-x:hidden;
    overflow-y:auto;
    overscroll-behavior:contain;
    border-left:1px solid rgba(86,118,151,.20);
    background:#080d13;
    scrollbar-width:thin;
}

.sidebar-section-title{
    margin:0 0 5px 2px;
    color:#92a6b8;
    font-size:9px;
    font-weight:800;
    letter-spacing:.08em;
    text-transform:uppercase;
}

.panel-wrap{
    padding:0;
}

.camera-panel{
    margin-bottom:7px;
}

.controls{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:5px;
    max-height:260px;
    opacity:1;
    overflow:hidden;
    transition:.18s ease;
}

body.controls-hidden .controls{
    max-height:0;
    opacity:0;
    margin:0;
}

.controls .exposure-card,
.controls .gain-card{
    grid-column:1 / -1;
}

.card{
    border-radius:9px;
    padding:6px 7px 5px;
}

.control-head{
    margin-bottom:1px;
}

.control-title{
    font-size:10px;
}

.exact{
    width:64px;
    padding:3px 6px;
    border-radius:7px;
    font-size:10px;
}

.unit{
    font-size:9px;
}

.range-wrap{
    padding-bottom:14px;
}

input[type=range]{
    height:15px;
}

.scale{
    top:16px;
    left:4px;
    right:4px;
    font-size:7px;
}

.segmented{
    gap:4px;
}

.segmented button{
    min-width:0;
    flex:1;
    padding:5px 5px;
    border-radius:8px;
    font-size:9px;
}

.mode-panel{
    margin:0 0 7px;
    padding:8px;
    border-radius:10px;
}

.mode-title{
    margin-bottom:4px;
    font-size:12px;
}

.compact-help{
    font-size:9px;
    line-height:1.3;
}

.compact-actions,
.push-actions,
.polar-actions{
    gap:5px;
    margin-top:6px;
}

.main-btn{
    padding:6px 8px;
    border-radius:8px;
    font-size:9px;
}

.compact-polar-grid,
.polar-grid{
    grid-template-columns:repeat(3,1fr);
    gap:4px;
    margin-top:6px;
}

.polar-step{
    padding:5px 6px;
    border-radius:8px;
    font-size:8px;
    text-align:center;
}

.polar-step strong{
    margin-bottom:2px;
    font-size:10px;
}

.result-box{
    margin-top:6px !important;
    padding:5px 6px;
    border-radius:8px;
    background:#0a1118;
    border:1px solid rgba(111,143,177,.12);
}

.compact-catalog-toolbar,
.catalog-toolbar{
    display:block;
    margin-top:5px;
}

.catalog-filter{
    margin-bottom:5px;
}

.catalog-filter button{
    flex:1;
    padding:5px 7px;
    border-radius:8px;
    font-size:9px;
}

.catalog-search{
    padding:6px 8px;
    border-radius:8px;
    font-size:10px;
}

.catalog-results{
    max-height:235px;
    margin-top:5px;
    gap:3px;
}

.catalog-item{
    border-radius:7px;
    padding:5px 7px;
}

.catalog-item strong{
    font-size:10px;
}

.catalog-item small{
    margin-top:1px;
    font-size:8px;
}

.target-card{
    margin-top:5px;
    padding:6px 7px;
    border-radius:8px;
    font-size:9px;
}

.target-card strong{
    font-size:10px;
}

.push-state,
.mode-help{
    font-size:9px;
}

.compact-metrics,
.metrics{
    padding:0;
    gap:3px;
}

.metric{
    padding:3px 5px;
    font-size:8px;
}

/* Le panneau de mode prend la place utile sans gonfler inutilement. */
#solvePanel .push-actions{
    display:grid;
    grid-template-columns:1fr 1fr;
}

#solvePanel #solveBtn{
    grid-column:1 / -1;
}

#solvePanel #clearTargetBtn{
    min-width:0;
}

.overlay-label{
    left:9px;
    bottom:9px;
    max-width:min(560px,82%);
    padding:6px 8px;
    border-radius:8px;
    font-size:10px;
}

/* Large desktop : un peu plus de place au panneau catalogue. */
@media(min-width:1500px){
    .workspace{
        grid-template-columns:minmax(0,1fr) 390px;
    }
}

/* Ecrans intermédiaires : panneau plus étroit, live toujours prioritaire. */
@media(max-width:1200px) and (min-width:821px){
    .workspace{
        grid-template-columns:minmax(0,1fr) 330px;
    }

    .badge:nth-of-type(1),
    .badge:nth-of-type(2){
        display:none;
    }

    .status{
        padding:3px 4px;
    }
}

/* Téléphone/tablette :
   le LiveView passe EN PREMIER, les commandes viennent dessous.
   On rend alors le document scrollable normalement. */
@media(max-width:820px){
    html,
    body{
        height:auto;
        min-height:100%;
        overflow:auto;
    }

    .app-shell{
        height:auto;
        min-height:100vh;
        overflow:visible;
    }

    .app-header{
        align-items:flex-start;
        flex-wrap:wrap;
    }

    .header-right{
        width:100%;
        margin-left:0;
        justify-content:space-between;
    }

    .statusbar{
        flex-wrap:wrap;
    }

    .workspace{
        display:flex;
        flex-direction:column;
        overflow:visible;
    }

    .viewer-pane{
        order:1;
        width:100%;
        min-height:0;
        padding:4px;
        overflow:visible;
    }

    .viewer-frame{
        width:100%;
        aspect-ratio:16 / 9;
    }

    .side-panel{
        order:2;
        width:100%;
        overflow:visible;
        border-left:0;
        border-top:1px solid rgba(86,118,151,.20);
    }

    .catalog-results{
        max-height:210px;
    }
}

</style>
</head>
<body>
<div class="app-shell">

  <div class="topbar">
    <header class="app-header">
      <div class="brand-line">
        <strong class="title">Astro Cam — SC715C · V37</strong>
        <span class="badge">1920×1080</span>
        <span class="badge">Bin 2×2</span>
        <span class="badge">184 mm</span>
        <span class="badge" id="badgeGain">Gain 180</span>
        <span class="badge" id="badgeExp">Exp 0,8 s</span>
        <span class="badge" id="badgePreview">Ciel noir</span>
      </div>

      <div class="header-right">
        <section class="statusbar">
          <span class="status"><i id="stIndi" class="dot wait"></i>INDI</span>
          <span class="status"><i id="stCam" class="dot wait"></i>Acq.</span>
          <span class="status"><i id="stProc" class="dot wait"></i>Trait.</span>
          <span class="status"><i id="stJpeg" class="dot wait"></i>JPEG</span>
          <span class="status"><i id="stSolve" class="dot wait"></i>Solve</span>
          <span class="status"><i class="dot ok"></i>Web</span>
        </section>

        <button id="toggleControlsBtn" class="ghost-btn compact-hide-btn" type="button">
          Masquer réglages
        </button>
      </div>
    </header>

    <nav class="mode-nav">
      <button id="navPolar" class="nav-btn active" type="button">Mise en station</button>
      <button id="navFocus" class="nav-btn" type="button">Focus</button>
      <button id="navSolve" class="nav-btn" type="button">Plate solving</button>
    </nav>
  </div>

  <div class="workspace">

    <main class="viewer-pane">
      <div class="viewer">
        <div class="viewer-frame" id="viewerFrame">
          <img id="liveImage" src="/stream" alt="Astro Cam Live">
          <canvas id="overlayCanvas" width="1920" height="1080"></canvas>
          <div id="orientationBadge" class="orientation-badge">Orientation capteur</div>
          <div id="overlayLabel" class="overlay-label"></div>
        </div>
      </div>
    </main>

    <aside class="side-panel">

      <div class="panel-wrap camera-panel">
        <div class="sidebar-section-title">Caméra</div>

        <section class="controls">
          <div class="card exposure-card">
            <div class="control-head">
              <span class="control-title">Exposition</span>
              <span>
                <input id="expExact" class="exact" type="number" min="10" max="10000" step="1">
                <span class="unit">ms</span>
              </span>
            </div>
            <div class="range-wrap">
              <input id="expSlider" type="range" min="0" max="1000" step="1">
              <div class="scale" id="expScale"></div>
            </div>
          </div>

          <div class="card gain-card">
            <div class="control-head">
              <span class="control-title">Gain</span>
              <input id="gainExact" class="exact" type="number" min="100" max="500" step="1">
            </div>
            <div class="range-wrap">
              <input id="gainSlider" type="range" min="100" max="500" step="1">
              <div class="scale" id="gainScale"></div>
            </div>
          </div>

          <div class="card mini-card">
            <div class="control-head">
              <span class="control-title">Binning</span>
              <span class="unit">2×2</span>
            </div>
            <div class="segmented">
              <button id="binAdd" type="button">ADD</button>
              <button id="binAvg" type="button">AVG</button>
            </div>
          </div>

          <div class="card mini-card">
            <div class="control-head">
              <span class="control-title">Affichage</span>
            </div>
            <div class="segmented">
              <button id="previewTech" type="button">Technique</button>
              <button id="previewDark" type="button">Ciel noir</button>
            </div>
          </div>
        </section>
      </div>

      <section id="polarPanel" class="mode-panel active">
        <div class="mode-title">Mise en station</div>
        <div class="mode-help compact-help">
          DEC fixe · tourne uniquement l’AD entre A, B et C.
        </div>

        <div class="polar-actions compact-actions">
          <button id="polarA" class="main-btn" type="button">A · Solve</button>
          <button id="polarB" class="main-btn" type="button">B · Solve</button>
          <button id="polarC" class="main-btn" type="button">C · Solve</button>
          <button id="polarReset" class="main-btn secondary" type="button">Reset</button>
        </div>

        <div class="polar-grid compact-polar-grid">
          <div class="polar-step">
            <strong>A</strong>
            <span id="polarAState">En attente</span>
          </div>
          <div class="polar-step">
            <strong>B</strong>
            <span id="polarBState">En attente</span>
          </div>
          <div class="polar-step">
            <strong>C</strong>
            <span id="polarCState">En attente</span>
          </div>
        </div>

        <div class="mode-help result-box" id="polarResult">
          Calibration non commencée.
        </div>
      </section>

      <section id="focusPanel" class="mode-panel">
        <div class="mode-title">Focus</div>
        <div class="mode-help compact-help">
          LiveView prioritaire. HFD/FWHM sera ajouté ici.
        </div>
      </section>

      <section id="solvePanel" class="mode-panel">
        <div class="mode-title">Plate solving / Push-To</div>

        <div class="mount-mode-row">
          <button id="mountAltAz" class="active" type="button">Monture Alt-Az</button>
          <button id="mountEq" type="button">Équatoriale</button>
        </div>

        <div id="siteAltAz" class="site-altaz">
          <input id="siteLat" type="number" step="0.0001" min="-90" max="90"
                 placeholder="Latitude °">
          <input id="siteLon" type="number" step="0.0001" min="-180" max="180"
                 placeholder="Longitude °">
          <button id="siteSave" type="button">Site</button>
        </div>
        <div id="siteNote" class="site-note bad">
          Alt-Az : renseigne latitude / longitude une seule fois.
        </div>

        <div class="catalog-toolbar compact-catalog-toolbar">
          <div class="catalog-filter">
            <button id="filterMessier" class="active" type="button">Messier</button>
            <button id="filterNgc" type="button">NGC</button>
            <button id="filterIc" type="button">IC</button>
            <button id="filterAll" type="button">Tous</button>
          </div>

          <input
            id="catalogSearch"
            class="catalog-search"
            type="search"
            placeholder="M31, NGC 7000, IC 5070, Andromeda…"
            autocomplete="off"
          >
        </div>

        <div class="sky-overlay-row">
          <button id="skyOverlayBtn" class="sky-overlay-btn active" type="button">
            Objets du ciel · ON
          </button>
          <span id="skyOverlayInfo" class="sky-overlay-info">
            Recalage après plate solve
          </span>
        </div>

        <div id="catalogResults" class="catalog-results"></div>
        <div class="catalog-credit">
          OpenNGC — Mattia Verga — CC BY-SA 4.0
        </div>

        <div id="targetCard" class="target-card">
          <strong>Aucune cible</strong>
          <div>Sélectionne un objet.</div>
        </div>

        <div class="push-actions compact-actions">
          <button id="solveBtn" class="main-btn" type="button">Plate solve</button>
          <button id="pushAutoBtn" class="main-btn secondary" type="button">
            Guidage cible
          </button>
          <button id="clearTargetBtn" class="main-btn secondary" type="button">
            Effacer
          </button>
        </div>

        <div class="push-state result-box" id="solveResult">Plate solve : prêt</div>
        <div class="push-state" id="pushResult"></div>
      </section>

      <section class="metrics compact-metrics">
        <span class="metric">RAW <strong id="mRaw">0</strong></span>
        <span class="metric">Trait <strong id="mProc">0</strong></span>
        <span class="metric">JPEG <strong id="mJpeg">0</strong></span>
        <span class="metric">Skip <strong id="mSkip">0</strong></span>
        <span class="metric">Acq <strong id="mAcqMs">0 ms</strong></span>
        <span class="metric">Trait <strong id="mProcMs">0 ms</strong></span>
        <span class="metric">JPEG <strong id="mJpegMs">0 ms</strong></span>
      </section>

    </aside>
  </div>
</div>

<script>
const WIDTH=1920, HEIGHT=1080;
const EXP_MIN=10, EXP_MAX=10000;
const GAIN_MIN=100, GAIN_MAX=500;

const expSlider=document.getElementById("expSlider");
const expExact=document.getElementById("expExact");
const gainSlider=document.getElementById("gainSlider");
const gainExact=document.getElementById("gainExact");
const binAdd=document.getElementById("binAdd");
const binAvg=document.getElementById("binAvg");
const previewTech=document.getElementById("previewTech");
const previewDark=document.getElementById("previewDark");
const badgeGain=document.getElementById("badgeGain");
const badgeExp=document.getElementById("badgeExp");
const badgePreview=document.getElementById("badgePreview");
const toggleControlsBtn=document.getElementById("toggleControlsBtn");
const solveBtn=document.getElementById("solveBtn");
const solveResult=document.getElementById("solveResult");
const pushResult=document.getElementById("pushResult");
const pushAutoBtn=document.getElementById("pushAutoBtn");
const clearTargetBtn=document.getElementById("clearTargetBtn");

const mountAltAz=document.getElementById("mountAltAz");
const mountEq=document.getElementById("mountEq");
const siteAltAz=document.getElementById("siteAltAz");
const siteLatInput=document.getElementById("siteLat");
const siteLonInput=document.getElementById("siteLon");
const siteSave=document.getElementById("siteSave");
const siteNote=document.getElementById("siteNote");
const catalogSearch=document.getElementById("catalogSearch");
const catalogResults=document.getElementById("catalogResults");
const targetCard=document.getElementById("targetCard");
const filterMessier=document.getElementById("filterMessier");
const filterNgc=document.getElementById("filterNgc");
const filterIc=document.getElementById("filterIc");
const filterAll=document.getElementById("filterAll");
const skyOverlayBtn=document.getElementById("skyOverlayBtn");
const skyOverlayInfo=document.getElementById("skyOverlayInfo");

const navPolar=document.getElementById("navPolar");
const navFocus=document.getElementById("navFocus");
const navSolve=document.getElementById("navSolve");
const polarPanel=document.getElementById("polarPanel");
const focusPanel=document.getElementById("focusPanel");
const solvePanel=document.getElementById("solvePanel");

const polarA=document.getElementById("polarA");
const polarB=document.getElementById("polarB");
const polarC=document.getElementById("polarC");
const polarReset=document.getElementById("polarReset");
const polarResult=document.getElementById("polarResult");
const polarAState=document.getElementById("polarAState");
const polarBState=document.getElementById("polarBState");
const polarCState=document.getElementById("polarCState");

const liveImage=document.getElementById("liveImage");
const viewerFrame=document.getElementById("viewerFrame");
const overlayCanvas=document.getElementById("overlayCanvas");
const overlayLabel=document.getElementById("overlayLabel");
const orientationBadge=document.getElementById("orientationBadge");
const ctx=overlayCanvas.getContext("2d");

let postTimer=null;
let currentBinningMode="ADD";
let currentPreviewMode="DARK";
let controlsHidden=false;
let currentPage="polar";
let lastPolar=null;
let lastTarget=null;
let catalogKind="M";
let catalogTimer=null;
let pushAuto=false;
let autoSolveCooldownUntil=0;

// Couche graphique céleste OpenNGC.
// Activée par défaut, mémorisée dans le navigateur.
let skyOverlayEnabled=
    localStorage.getItem("astrocam_sky_overlay")!=="0";
let skyOverlayItems=[];
let skyOverlaySeq=0;
let skyOverlayScale=NaN;
let skyOverlayLoading=false;
let lastStatusSolveSeq=0;

// Derniers solves Push-To réellement terminés.
// Sert à afficher si le mouvement effectué nous rapproche ou nous éloigne.
let guidanceLastSeq=null;
let guidanceLastSeparation=null;
let guidanceTrend="";
let currentSolveRunning=false;

// Mode de monture Push-To.
// On garde les deux possibilités : Alt-Az maintenant, équatorial plus tard.
let mountMode=localStorage.getItem("astrocam_mount_mode") || "ALTAZ";

let siteLatDeg=Number(localStorage.getItem("astrocam_site_lat"));
let siteLonDeg=Number(localStorage.getItem("astrocam_site_lon"));

if(!Number.isFinite(siteLatDeg))siteLatDeg=NaN;
if(!Number.isFinite(siteLonDeg))siteLonDeg=NaN;

// Orientation d'affichage issue du dernier WCS valide.
// On ne transforme que le JPEG et son overlay : jamais le RAW scientifique.
let lastWcsOrientation=null;
let displayRotationRad=0;
let displayCropScale=1;

function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}

function expPosToMs(pos){
    const t=Number(pos)/1000;
    const ms=EXP_MIN*Math.pow(EXP_MAX/EXP_MIN,t);
    if(ms<100)return Math.round(ms);
    if(ms<1000)return Math.round(ms/5)*5;
    return Math.round(ms/10)*10;
}
function expMsToPos(ms){
    ms=clamp(Number(ms),EXP_MIN,EXP_MAX);
    return Math.round(
        Math.log(ms/EXP_MIN)/Math.log(EXP_MAX/EXP_MIN)*1000
    );
}
function formatExposure(ms){
    if(ms<1000)return `${Math.round(ms)} ms`;
    return `${(ms/1000).toLocaleString("fr-FR",{maximumFractionDigits:2})} s`;
}
function applyCollapsedState(){
    document.body.classList.toggle("controls-hidden",controlsHidden);
    toggleControlsBtn.textContent=controlsHidden?"Afficher réglages":"Masquer réglages";
    localStorage.setItem("astrocam_controls_hidden",controlsHidden?"1":"0");
}
function refreshLabels(){
    badgeExp.textContent=`Exp ${formatExposure(Number(expExact.value))}`;
    badgeGain.textContent=`Gain ${Math.round(Number(gainExact.value))}`;
    binAdd.classList.toggle("active",currentBinningMode==="ADD");
    binAvg.classList.toggle("active",currentBinningMode==="AVG");
    previewTech.classList.toggle("active",currentPreviewMode==="TECHNIQUE");
    previewDark.classList.toggle("active",currentPreviewMode==="DARK");
    badgePreview.textContent=currentPreviewMode==="DARK"?"Ciel noir":"Technique";
}
async function postSettings(){
    postTimer=null;
    const payload={
        exposure_ms:clamp(Number(expExact.value),EXP_MIN,EXP_MAX),
        gain:clamp(Math.round(Number(gainExact.value)),GAIN_MIN,GAIN_MAX),
        binning_mode:currentBinningMode,
        preview_mode:currentPreviewMode
    };
    refreshLabels();
    try{
        await fetch("/api/settings",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(payload)
        });
    }catch(e){console.error(e)}
}
function schedulePost(){
    clearTimeout(postTimer);
    postTimer=setTimeout(postSettings,120);
}
function setExposure(ms,send=true){
    ms=clamp(Number(ms),EXP_MIN,EXP_MAX);
    expExact.value=ms;
    expSlider.value=expMsToPos(ms);
    refreshLabels();
    if(send)schedulePost();
}
function setGain(v,send=true){
    v=clamp(Math.round(Number(v)),GAIN_MIN,GAIN_MAX);
    gainExact.value=v;
    gainSlider.value=v;
    refreshLabels();
    if(send)schedulePost();
}
function setBinningMode(mode,send=true){
    currentBinningMode=mode==="AVG"?"AVG":"ADD";
    refreshLabels();
    if(send)schedulePost();
}
function setPreviewMode(mode,send=true){
    currentPreviewMode=mode==="TECHNIQUE"?"TECHNIQUE":"DARK";
    refreshLabels();
    if(send)schedulePost();
}

function normalizeAngleRad(a){
    while(a>Math.PI)a-=Math.PI*2;
    while(a<=-Math.PI)a+=Math.PI*2;
    return a;
}

function resetDisplayOrientation(){
    displayRotationRad=0;
    displayCropScale=1;

    // FULL FRAME NATIF : aucune rotation, aucun zoom, aucun recadrage.
    liveImage.style.transform="";
    overlayCanvas.style.transform="";
}

function applyDisplayOrientation(){
    // On garde toujours les pixels caméra tels quels.
    resetDisplayOrientation();

    if(
        lastWcsOrientation
        && lastWcsOrientation.available
        && Number.isFinite(lastWcsOrientation.rotation_deg)
    ){
        orientationBadge.textContent=
            `Full frame · rotation WCS ${lastWcsOrientation.rotation_deg.toFixed(1)}°`;
    }else{
        orientationBadge.textContent="Full frame · orientation après solve";
    }
}


function updateSkyOverlayButton(){
    skyOverlayBtn.classList.toggle("active",skyOverlayEnabled);
    skyOverlayBtn.textContent=
        skyOverlayEnabled
        ? "Objets du ciel · ON"
        : "Objets du ciel · OFF";

    if(!skyOverlayEnabled){
        skyOverlayInfo.textContent="Couche masquée";
    }else if(lastStatusSolveSeq<=0){
        skyOverlayInfo.textContent="Disponible après un plate solve";
    }
}

async function refreshSkyOverlay(force=false){
    if(
        !skyOverlayEnabled
        || currentPage!=="solve"
        || lastStatusSolveSeq<=0
        || skyOverlayLoading
    ){
        return;
    }

    if(!force && skyOverlaySeq===lastStatusSolveSeq){
        return;
    }

    skyOverlayLoading=true;
    skyOverlayInfo.textContent="Calcul des objets du champ…";

    try{
        const r=await fetch("/api/sky-overlay",{cache:"no-store"});
        const d=await r.json();

        if(!r.ok || !d.ok){
            skyOverlayInfo.textContent=d.error||"Couche céleste indisponible";
            return;
        }

        skyOverlayItems=Array.isArray(d.items) ? d.items : [];
        skyOverlaySeq=Number(d.source_seq)||0;
        skyOverlayScale=Number(d.scale_arcsec_px);

        skyOverlayInfo.textContent=
            `${skyOverlayItems.length} objet${skyOverlayItems.length>1?"s":""} dans le champ`;

        drawMainOverlay();
    }catch(e){
        skyOverlayInfo.textContent="Erreur couche céleste";
    }finally{
        skyOverlayLoading=false;
    }
}

function skyObjectStroke(item){
    if(item.family==="Messier"){
        return "rgba(255,198,92,.82)";
    }

    if(item.family==="IC"){
        return "rgba(195,139,255,.66)";
    }

    return "rgba(113,198,242,.68)";
}

function drawSkyObjectOverlay(){
    if(
        currentPage!=="solve"
        || !skyOverlayEnabled
        || !Array.isArray(skyOverlayItems)
        || skyOverlayItems.length===0
        || !Number.isFinite(skyOverlayScale)
        || skyOverlayScale<=0
    ){
        return;
    }

    ctx.save();

    // Pendant un nouveau solve, le dernier WCS est potentiellement périmé :
    // on conserve l'information mais très atténuée jusqu'au recalage.
    ctx.globalAlpha=currentSolveRunning ? 0.24 : 1.0;
    ctx.lineCap="round";
    ctx.lineJoin="round";

    for(const item of skyOverlayItems){
        const x=Number(item.x);
        const y=Number(item.y);

        if(!Number.isFinite(x) || !Number.isFinite(y)){
            continue;
        }

        const majorArcmin=Number(item.major_arcmin);
        const minorArcmin=Number(item.minor_arcmin);

        let rx=11;
        let ry=11;

        if(Number.isFinite(majorArcmin) && majorArcmin>0){
            rx=majorArcmin*60/skyOverlayScale/2;
        }

        if(Number.isFinite(minorArcmin) && minorArcmin>0){
            ry=minorArcmin*60/skyOverlayScale/2;
        }else{
            ry=rx;
        }

        // Les objets minuscules restent visibles ; les objets gigantesques
        // restent dessinables mais ne prennent pas tout le canvas.
        rx=Math.max(9,Math.min(260,rx));
        ry=Math.max(7,Math.min(220,ry));

        let rotation=0;
        const pa=Number(item.position_angle_deg);

        if(
            Number.isFinite(pa)
            && lastWcsOrientation
            && lastWcsOrientation.available
        ){
            const p=deg2rad(pa);

            // OpenNGC : PA mesuré depuis le Nord vers l'Est.
            const vx=
                Math.cos(p)*lastWcsOrientation.north_x
                +Math.sin(p)*lastWcsOrientation.east_x;

            const vy=
                Math.cos(p)*lastWcsOrientation.north_y
                +Math.sin(p)*lastWcsOrientation.east_y;

            rotation=Math.atan2(vy,vx);
        }

        const color=skyObjectStroke(item);

        ctx.strokeStyle=color;
        ctx.lineWidth=item.family==="Messier" ? 1.8 : 1.25;

        ctx.beginPath();
        ctx.ellipse(x,y,rx,ry,rotation,0,Math.PI*2);
        ctx.stroke();

        const label=String(item.id||"").trim();

        if(label){
            ctx.font=item.family==="Messier"
                ? "600 18px Segoe UI"
                : "500 16px Segoe UI";

            ctx.fillStyle=color;
            ctx.textBaseline="middle";

            let lx=x+Math.max(13,Math.min(rx+7,38));
            let ly=y-Math.max(10,Math.min(ry+5,25));

            const tw=ctx.measureText(label).width;

            if(lx+tw>WIDTH-8){
                lx=x-Math.max(13,Math.min(rx+7,38))-tw;
            }

            if(ly<16){
                ly=y+Math.max(16,Math.min(ry+12,30));
            }

            ctx.fillText(label,lx,ly);
        }
    }

    ctx.restore();
}

skyOverlayBtn.addEventListener("click",()=>{
    skyOverlayEnabled=!skyOverlayEnabled;

    localStorage.setItem(
        "astrocam_sky_overlay",
        skyOverlayEnabled ? "1" : "0"
    );

    updateSkyOverlayButton();

    if(skyOverlayEnabled){
        refreshSkyOverlay(true);
    }

    drawMainOverlay();
});


function setPage(page){
    currentPage=page;
    navPolar.classList.toggle("active",page==="polar");
    navFocus.classList.toggle("active",page==="focus");
    navSolve.classList.toggle("active",page==="solve");
    polarPanel.classList.toggle("active",page==="polar");
    focusPanel.classList.toggle("active",page==="focus");
    solvePanel.classList.toggle("active",page==="solve");
    applyDisplayOrientation();
    drawMainOverlay();

    if(page==="solve" && skyOverlayEnabled){
        refreshSkyOverlay();
    }
}

function addTicks(container,ticks,percentFn,setter){
    container.innerHTML="";
    for(const tick of ticks){
        const span=document.createElement("span");
        span.className="tick";
        span.style.left=`${percentFn(tick.value)}%`;
        span.textContent=tick.label;
        span.addEventListener("click",()=>setter(tick.value));
        container.appendChild(span);
    }
}

addTicks(
    document.getElementById("expScale"),
    [
      {value:10,label:"10"},{value:25,label:"25"},{value:50,label:"50"},
      {value:100,label:"100"},{value:250,label:"250"},{value:500,label:"500"},
      {value:1000,label:"1s"},{value:2000,label:"2s"},
      {value:5000,label:"5s"},{value:10000,label:"10s"}
    ],
    v=>100*Math.log(v/EXP_MIN)/Math.log(EXP_MAX/EXP_MIN),
    v=>setExposure(v)
);
addTicks(
    document.getElementById("gainScale"),
    [100,150,200,250,300,350,400,450,500].map(v=>({value:v,label:String(v)})),
    v=>100*(v-GAIN_MIN)/(GAIN_MAX-GAIN_MIN),
    v=>setGain(v)
);

expSlider.addEventListener("input",()=>setExposure(expPosToMs(expSlider.value)));
gainSlider.addEventListener("input",()=>setGain(gainSlider.value));
expExact.addEventListener("change",()=>setExposure(expExact.value));
gainExact.addEventListener("change",()=>setGain(gainExact.value));
binAdd.addEventListener("click",()=>setBinningMode("ADD"));
binAvg.addEventListener("click",()=>setBinningMode("AVG"));
previewTech.addEventListener("click",()=>setPreviewMode("TECHNIQUE"));
previewDark.addEventListener("click",()=>setPreviewMode("DARK"));
toggleControlsBtn.addEventListener("click",()=>{
    controlsHidden=!controlsHidden;
    applyCollapsedState();
});
navPolar.addEventListener("click",()=>setPage("polar"));
navFocus.addEventListener("click",()=>setPage("focus"));
navSolve.addEventListener("click",()=>setPage("solve"));

function setDot(id,state){
    const e=document.getElementById(id);
    e.className="dot "+(state?"ok":"bad");
}
function updateSolveUi(s){
    const dot=document.getElementById("stSolve");
    if(!s.solve_available){
        dot.className="dot bad";
        solveBtn.disabled=true;
        solveBtn.textContent="Plate solve indisponible";
    }else if(s.solve_status==="running"){
        dot.className="dot running";
        solveBtn.disabled=true;
        solveBtn.textContent="Plate solving…";
    }else if(s.solve_status==="ok"){
        dot.className="dot ok";
        solveBtn.disabled=false;
        solveBtn.textContent="Plate solve";
    }else if(s.solve_status==="error"){
        dot.className="dot bad";
        solveBtn.disabled=false;
        solveBtn.textContent="Réessayer plate solve";
    }else{
        dot.className="dot wait";
        solveBtn.disabled=false;
        solveBtn.textContent="Plate solve";
    }
    if(s.solve_message){
        solveResult.textContent="Plate solve : "+s.solve_message;
    }
}

solveBtn.addEventListener("click",async()=>{
    solveBtn.disabled=true;
    solveResult.textContent="Plate solve : résolution en cours…";
    try{
        const r=await fetch("/api/solve",{method:"POST",cache:"no-store"});
        const d=await r.json();
        if(!r.ok)solveResult.textContent="Plate solve : "+(d.error||"erreur");
    }catch(e){
        solveResult.textContent="Plate solve : erreur réseau";
    }
});


function setCatalogKind(kind){
    catalogKind=kind;
    filterMessier.classList.toggle("active",kind==="M");
    filterNgc.classList.toggle("active",kind==="NGC");
    filterIc.classList.toggle("active",kind==="IC");
    filterAll.classList.toggle("active",kind==="ALL");
    loadCatalog();
}

filterMessier.addEventListener("click",()=>setCatalogKind("M"));
filterNgc.addEventListener("click",()=>setCatalogKind("NGC"));
filterIc.addEventListener("click",()=>setCatalogKind("IC"));
filterAll.addEventListener("click",()=>setCatalogKind("ALL"));

catalogSearch.addEventListener("input",()=>{
    clearTimeout(catalogTimer);
    catalogTimer=setTimeout(loadCatalog,180);
});

function renderCatalog(items){
    catalogResults.innerHTML="";

    if(!items.length){
        const empty=document.createElement("div");
        empty.className="mode-help";
        empty.textContent="Aucun objet trouvé.";
        catalogResults.appendChild(empty);
        return;
    }

    for(const item of items){
        const button=document.createElement("button");
        button.type="button";
        button.className="catalog-item";

        const title=document.createElement("strong");
        title.textContent=
            item.id+(item.name && item.name!==item.id ? ` — ${item.name}` : "");

        const details=document.createElement("small");
        const parts=[];

        if(item.type_label){
            parts.push(item.type_label);
        }

        if(
            item.magnitude!==null
            && item.magnitude!==undefined
            && Number.isFinite(Number(item.magnitude))
        ){
            const band=item.magnitude_band ? ` ${item.magnitude_band}` : "";
            parts.push(`mag${band} ${Number(item.magnitude).toFixed(1)}`);
        }

        if(item.const){
            parts.push(item.const);
        }

        const coords=document.createElement("small");
        coords.textContent=
            `RA ${item.ra_hms} · DEC ${item.dec_dms}`;

        button.appendChild(title);

        if(parts.length){
            details.textContent=parts.join(" · ");
            button.appendChild(details);
        }

        button.appendChild(coords);

        button.addEventListener("click",()=>selectTarget(item));
        catalogResults.appendChild(button);
    }
}

async function loadCatalog(){
    try{
        const q=encodeURIComponent(catalogSearch.value.trim());
        const kind=encodeURIComponent(catalogKind);
        const r=await fetch(`/api/catalog?q=${q}&kind=${kind}`,{
            cache:"no-store"
        });
        const data=await r.json();

        if(!r.ok){
            catalogResults.innerHTML="";
            const msg=document.createElement("div");
            msg.className="polar-error";
            msg.textContent=data.error||"Catalogue indisponible.";
            catalogResults.appendChild(msg);
            return;
        }

        renderCatalog(data.items||[]);
    }catch(e){
        catalogResults.innerHTML=
            '<div class="polar-error">Erreur de chargement du catalogue.</div>';
    }
}

async function selectTarget(item){
    try{
        const r=await fetch("/api/target",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(item)
        });
        const data=await r.json();

        if(!r.ok){
            pushResult.textContent=data.error||"Impossible de sélectionner la cible.";
            return;
        }

        pushAuto=false;
        pushAutoBtn.textContent="Démarrer guidage vers cible";
        guidanceLastSeq=null;
        guidanceLastSeparation=null;
        guidanceTrend="";
        await refreshTarget();
    }catch(e){
        pushResult.textContent="Erreur réseau lors de la sélection.";
    }
}

mountAltAz.addEventListener("click",()=>{
    setMountMode("ALTAZ");
    refreshTarget();
});

mountEq.addEventListener("click",()=>{
    setMountMode("EQ");
    refreshTarget();
});

siteSave.addEventListener("click",()=>{
    saveSite();
    refreshTarget();
});

siteLatInput.addEventListener("keydown",e=>{
    if(e.key==="Enter"){
        saveSite();
        refreshTarget();
    }
});
siteLonInput.addEventListener("keydown",e=>{
    if(e.key==="Enter"){
        saveSite();
        refreshTarget();
    }
});

if(Number.isFinite(siteLatDeg))siteLatInput.value=siteLatDeg;
if(Number.isFinite(siteLonDeg))siteLonInput.value=siteLonDeg;
setMountMode(mountMode);


clearTargetBtn.addEventListener("click",async()=>{
    pushAuto=false;
    guidanceLastSeq=null;
    guidanceLastSeparation=null;
    guidanceTrend="";
    pushAutoBtn.textContent="Démarrer guidage vers cible";
    await fetch("/api/target/clear",{method:"POST",cache:"no-store"});
    await refreshTarget();
});

pushAutoBtn.addEventListener("click",async()=>{
    if(!lastTarget || !lastTarget.active){
        pushResult.textContent="Choisis d'abord un objet Messier ou NGC.";
        return;
    }

    pushAuto=!pushAuto;
    pushAutoBtn.textContent=
        pushAuto ? "Arrêter guidage auto" : "Démarrer guidage vers cible";

    if(pushAuto){
        // Autorise immédiatement le premier solve automatique.
        autoSolveCooldownUntil=0;
        pushResult.textContent=
            "Guidage auto actif — lancement du prochain plate solve…";
    }else{
        pushResult.textContent=
            "Guidage auto arrêté — la dernière direction reste affichée.";
    }
});

async function refreshTarget(){
    try{
        const r=await fetch("/api/target/status",{cache:"no-store"});
        const t=await r.json();

        // En mode Alt-Az, conversion à l'heure réelle du navigateur.
        // Le WCS reste en RA/DEC côté solve : on ne touche pas au solveur.
        if(mountMode==="ALTAZ"){
            applyAltAzGuidance(t);
        }

        lastTarget=t;

        if(!t.active){
            targetCard.innerHTML=
                "<strong>Aucune cible sélectionnée</strong>"+
                "<div>Choisis un objet dans le catalogue.</div>";
            pushResult.textContent="";
            drawMainOverlay();
            return;
        }

        if(
            mountMode==="ALTAZ"
            && siteConfigured()
            && Number.isFinite(t.target_az_deg)
            && Number.isFinite(t.target_alt_deg)
        ){
            targetCard.innerHTML=
                `<strong>${t.id}${t.name && t.name!==t.id ? " — "+t.name : ""}</strong>`+
                `<div>AZ ${t.target_az_deg.toFixed(1)}° · ALT ${t.target_alt_deg.toFixed(1)}°</div>`+
                `<div style="opacity:.62">RA ${t.ra_hms} · DEC ${t.dec_dms}</div>`;
        }else{
            targetCard.innerHTML=
                `<strong>${t.id}${t.name && t.name!==t.id ? " — "+t.name : ""}</strong>`+
                `<div>RA ${t.ra_hms} · DEC ${t.dec_dms}</div>`;
        }

        if(t.solution_available){
            // Une variation de distance n'est calculée qu'une fois par
            // nouvelle solution WCS réussie.
            if(
                guidanceLastSeq!==null
                && t.solve_source_seq!==guidanceLastSeq
                && guidanceLastSeparation!==null
            ){
                const delta=t.separation_deg-guidanceLastSeparation;

                // 0,003° ≈ 11 arcsec : on ignore les petites variations
                // comparables au bruit de résolution.
                if(Math.abs(delta)>=0.003){
                    guidanceTrend=
                        delta<0
                        ? `RAPPROCHEMENT ${Math.abs(delta).toFixed(3)}°`
                        : `ÉLOIGNEMENT +${delta.toFixed(3)}°`;
                }else{
                    guidanceTrend="DISTANCE STABLE";
                }
            }

            if(
                guidanceLastSeq===null
                || t.solve_source_seq!==guidanceLastSeq
            ){
                guidanceLastSeq=t.solve_source_seq;
                guidanceLastSeparation=t.separation_deg;
            }

            const prefix=pushAuto ? "AUTO · " : "";
            const trend=guidanceTrend ? ` · ${guidanceTrend}` : "";
            const running=t.solve_running ? " · solve suivant en cours…" : "";

            const axes=formatAxisGuidance(t);

            if(mountMode==="ALTAZ"){
                pushResult.textContent=
                    t.in_frame
                    ? `${t.id} dans le champ · ${axes}`
                    : `${t.id} · ${t.separation_deg.toFixed(1)}° restant · ${axes}`;
            }else{
                pushResult.textContent=
                    t.in_frame
                    ? `${prefix}${t.id} dans le champ · écart total ${t.separation_deg.toFixed(3)}° · ${axes} · solve RAW #${t.solve_source_seq}${trend}${running}`
                    : `${prefix}${t.id} · écart total ${t.separation_deg.toFixed(2)}° · ${axes} · solve RAW #${t.solve_source_seq}${trend}${running}`;
            }
        }else{
            pushResult.textContent=
                pushAuto
                ? "AUTO · attente du premier plate solve…"
                : "Cible prête — lance un plate solve.";
        }

        drawMainOverlay();
    }catch(e){
        console.error(e);
    }
}


async function polarCapture(slot){
    try{
        const r=await fetch(`/api/polar/capture/${slot}`,{
            method:"POST",
            cache:"no-store"
        });
        const d=await r.json();
        if(!r.ok){
            polarResult.textContent=d.error||"Erreur de capture polaire.";
        }else{
            polarResult.textContent=`Résolution de la position ${slot}/3 en cours…`;
        }
    }catch(e){
        polarResult.textContent="Erreur réseau pendant la calibration.";
    }
}
polarA.addEventListener("click",()=>polarCapture(1));
polarB.addEventListener("click",()=>polarCapture(2));
polarC.addEventListener("click",()=>polarCapture(3));
polarReset.addEventListener("click",async()=>{
    await fetch("/api/polar/reset",{method:"POST",cache:"no-store"});
    lastPolar=null;
    drawMainOverlay();
    await refreshPolar();
});

function polarStateText(valid,index){
    return valid[index] ? "✓ Résolue" : "En attente";
}

async function refreshPolar(){
    try{
        const r=await fetch("/api/polar/status",{cache:"no-store"});
        const p=await r.json();
        lastPolar=p;

        polarAState.textContent=polarStateText(p.valid,0);
        polarBState.textContent=polarStateText(p.valid,1);
        polarCState.textContent=polarStateText(p.valid,2);

        polarAState.className=p.valid[0]?"polar-good":"";
        polarBState.className=p.valid[1]?"polar-good":"";
        polarCState.className=p.valid[2]?"polar-good":"";

        polarA.disabled=p.solve_running;
        polarB.disabled=p.solve_running || !p.valid[0];
        polarC.disabled=p.solve_running || !p.valid[1];

        if(p.calibrated){
            polarResult.innerHTML=
                `<span class="polar-good">Calibration OK</span> · `+
                `erreur polaire ≈ <strong>${p.error_arcmin.toFixed(1)}′</strong> · `+
                `axe AD : RA ${p.axis_ra_hms}, DEC ${p.axis_dec_dms}. `+
                (p.guidance_active
                    ? "Amène l’étoile cerclée vers la cible rouge avec les vis ALT/AZ."
                    : "Clique maintenant une étoile brillante dans le live.");
        }else if(p.message){
            polarResult.textContent=p.message;
        }

        overlayCanvas.classList.toggle(
            "select-star",
            currentPage==="polar" && p.calibrated && !p.guidance_active
        );

        drawMainOverlay();
    }catch(e){
        console.error(e);
    }
}

function drawArrow(x1,y1,x2,y2,color){
    const angle=Math.atan2(y2-y1,x2-x1);
    const head=15;

    ctx.save();
    ctx.strokeStyle=color;
    ctx.lineWidth=2.25;
    ctx.lineCap="round";
    ctx.lineJoin="round";

    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.stroke();

    // Pointe ouverte : beaucoup plus légère que l'ancien triangle plein.
    ctx.beginPath();
    ctx.moveTo(
        x2-head*Math.cos(angle-Math.PI/6),
        y2-head*Math.sin(angle-Math.PI/6)
    );
    ctx.lineTo(x2,y2);
    ctx.lineTo(
        x2-head*Math.cos(angle+Math.PI/6),
        y2-head*Math.sin(angle+Math.PI/6)
    );
    ctx.stroke();

    ctx.restore();
}

function edgePoint(x1,y1,x2,y2){
    const dx=x2-x1, dy=y2-y1;
    let t=1;

    const tests=[];
    if(dx>0)tests.push((WIDTH-12-x1)/dx);
    if(dx<0)tests.push((12-x1)/dx);
    if(dy>0)tests.push((HEIGHT-12-y1)/dy);
    if(dy<0)tests.push((12-y1)/dy);

    for(const value of tests){
        if(value>0 && value<t)t=value;
    }

    return [x1+dx*t,y1+dy*t];
}

function drawPolarOnly(){
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    overlayLabel.textContent="";

    if(currentPage!=="polar" || !lastPolar || !lastPolar.calibrated){
        return;
    }

    if(!lastPolar.guidance_active){
        overlayLabel.textContent=
            `Erreur polaire ≈ ${lastPolar.error_arcmin.toFixed(1)}′ — `+
            `clique une étoile brillante pour créer la cible de correction.`;
        return;
    }

    const rx=lastPolar.ref_x;
    const ry=lastPolar.ref_y;
    const tx=lastPolar.target_x;
    const ty=lastPolar.target_y;

    // Cercle vert = étoile choisie au départ.
    ctx.strokeStyle="#45e27b";
    ctx.lineWidth=4;
    ctx.beginPath();
    ctx.arc(rx,ry,24,0,Math.PI*2);
    ctx.stroke();

    const targetInside=tx>=10 && tx<=WIDTH-10 && ty>=10 && ty<=HEIGHT-10;

    if(targetInside){
        ctx.strokeStyle="#ff4d58";
        ctx.lineWidth=5;
        ctx.beginPath();
        ctx.arc(tx,ty,30,0,Math.PI*2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(tx-38,ty);
        ctx.lineTo(tx+38,ty);
        ctx.moveTo(tx,ty-38);
        ctx.lineTo(tx,ty+38);
        ctx.stroke();

        drawArrow(rx,ry,tx,ty,"#ffd34f");
        overlayLabel.textContent=
            `Déplace l’étoile verte vers la cible rouge uniquement avec ALT/AZ · `+
            `erreur polaire initiale ≈ ${lastPolar.error_arcmin.toFixed(1)}′`;
    }else{
        const [ex,ey]=edgePoint(rx,ry,tx,ty);
        drawArrow(rx,ry,ex,ey,"#ffd34f");

        ctx.fillStyle="#ff5c66";
        ctx.font="bold 30px Segoe UI";
        ctx.fillText("CIBLE HORS CHAMP",Math.max(20,ex-150),Math.max(40,ey-20));

        overlayLabel.textContent=
            `La correction dépasse le champ : suis la flèche avec les vis ALT/AZ, `+
            `puis recommence une calibration 3 points lorsque la cible se rapproche.`;
    }
}


function drawWcsCompass(t){
    if(
        !t
        || !t.solution_available
        || typeof t.north_screen_x!=="number"
        || typeof t.east_screen_x!=="number"
    ){
        return;
    }

    const ox=105;
    const oy=105;
    const len=48;

    // Fond discret.
    ctx.fillStyle="rgba(4,8,12,.42)";
    ctx.beginPath();
    ctx.arc(ox,oy,60,0,Math.PI*2);
    ctx.fill();

    // Nord WCS.
    const nx=ox+t.north_screen_x*len;
    const ny=oy+t.north_screen_y*len;
    drawArrow(ox,oy,nx,ny,"#7dd3fc");
    ctx.fillStyle="#7dd3fc";
    ctx.font="600 18px Segoe UI";
    ctx.fillText("N",nx+7,ny-5);

    // Est WCS.
    const ex=ox+t.east_screen_x*len;
    const ey=oy+t.east_screen_y*len;
    drawArrow(ox,oy,ex,ey,"#8df0a8");
    ctx.fillStyle="#8df0a8";
    ctx.font="600 18px Segoe UI";
    ctx.fillText("E",ex+7,ey-5);
}



function drawTargetName(x,y,label){
    if(!label)return;

    ctx.save();
    ctx.font="600 20px Segoe UI";

    const paddingX=8;
    const boxH=29;
    const textW=ctx.measureText(label).width;
    const boxW=textW+paddingX*2;

    // Par défaut : à droite et légèrement au-dessus de la cible.
    let bx=x+40;
    let by=y-46;

    // Empêche le cartouche de sortir de l'image.
    if(bx+boxW>WIDTH-8){
        bx=x-boxW-40;
    }
    if(by<8){
        by=y+18;
    }
    if(by+boxH>HEIGHT-8){
        by=HEIGHT-boxH-8;
    }

    ctx.fillStyle="rgba(20,5,8,.62)";
    ctx.strokeStyle="#ff4d58";
    ctx.lineWidth=1;

    ctx.beginPath();
    if(typeof ctx.roundRect==="function"){
        ctx.roundRect(bx,by,boxW,boxH,8);
    }else{
        ctx.rect(bx,by,boxW,boxH);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle="#ff7b82";
    ctx.textBaseline="middle";
    ctx.fillText(label,bx+paddingX,by+boxH/2+1);

    ctx.restore();
}


function deg2rad(v){return Number(v)*Math.PI/180;}
function rad2deg(v){return Number(v)*180/Math.PI;}

function norm360(v){
    v=Number(v)%360;
    return v<0 ? v+360 : v;
}
function norm180(v){
    v=norm360(v);
    return v>180 ? v-360 : v;
}
function clampNumber(v,a,b){
    return Math.max(a,Math.min(b,v));
}

function siteConfigured(){
    return (
        Number.isFinite(siteLatDeg)
        && Number.isFinite(siteLonDeg)
        && siteLatDeg>=-90 && siteLatDeg<=90
        && siteLonDeg>=-180 && siteLonDeg<=180
    );
}

function julianDate(date){
    return date.getTime()/86400000 + 2440587.5;
}

function localSiderealDeg(date,lonDeg){
    const jd=julianDate(date);
    const t=(jd-2451545.0)/36525.0;

    const gmst=
        280.46061837
        +360.98564736629*(jd-2451545.0)
        +0.000387933*t*t
        -(t*t*t)/38710000.0;

    return norm360(gmst+lonDeg);
}

function radecToAltAz(raDeg,decDeg,latDeg,lonDeg,date){
    const lst=localSiderealDeg(date,lonDeg);
    const h=deg2rad(norm180(lst-raDeg));
    const dec=deg2rad(decDeg);
    const lat=deg2rad(latDeg);

    const sinAlt=
        Math.sin(dec)*Math.sin(lat)
        +Math.cos(dec)*Math.cos(lat)*Math.cos(h);

    const alt=Math.asin(clampNumber(sinAlt,-1,1));

    // Azimut : 0° = Nord, 90° = Est.
    const y=-Math.sin(h)*Math.cos(dec);
    const x=
        Math.sin(dec)*Math.cos(lat)
        -Math.cos(dec)*Math.sin(lat)*Math.cos(h);

    const az=Math.atan2(y,x);

    return {
        az_deg:norm360(rad2deg(az)),
        alt_deg:rad2deg(alt)
    };
}

function altAzToRadec(azDeg,altDeg,latDeg,lonDeg,date){
    const az=deg2rad(norm360(azDeg));
    const alt=deg2rad(altDeg);
    const lat=deg2rad(latDeg);

    const sinDec=
        Math.sin(alt)*Math.sin(lat)
        +Math.cos(alt)*Math.cos(lat)*Math.cos(az);

    const dec=Math.asin(clampNumber(sinDec,-1,1));

    const cosDec=Math.max(1e-12,Math.cos(dec));

    const sinH=
        -Math.sin(az)*Math.cos(alt)/cosDec;

    const cosH=
        (Math.sin(alt)-Math.sin(lat)*Math.sin(dec))
        /Math.max(1e-12,Math.cos(lat)*cosDec);

    const h=Math.atan2(sinH,cosH);
    const lst=localSiderealDeg(date,lonDeg);
    const ra=norm360(lst-rad2deg(h));

    return {
        ra_deg:ra,
        dec_deg:rad2deg(dec)
    };
}

function radecOffsetScreenVector(
    centerRaDeg,centerDecDeg,
    raDeg,decDeg,
    t
){
    const ra0=deg2rad(centerRaDeg);
    const dec0=deg2rad(centerDecDeg);
    const ra=deg2rad(raDeg);
    const dec=deg2rad(decDeg);

    const dra=Math.atan2(
        Math.sin(ra-ra0),
        Math.cos(ra-ra0)
    );

    const east=
        Math.cos(dec)*Math.sin(dra);

    const north=
        Math.cos(dec0)*Math.sin(dec)
        -Math.sin(dec0)*Math.cos(dec)*Math.cos(dra);

    let sx=
        east*t.east_screen_x
        +north*t.north_screen_x;

    let sy=
        east*t.east_screen_y
        +north*t.north_screen_y;

    const n=Math.hypot(sx,sy);

    if(n<1e-12){
        return {x:0,y:0};
    }

    return {x:sx/n,y:sy/n};
}

function applyAltAzGuidance(t){
    if(!t || !t.active || !siteConfigured()){
        return;
    }

    const now=new Date();

    const targetHorizontal=
        radecToAltAz(
            Number(t.ra_deg),
            Number(t.dec_deg),
            siteLatDeg,
            siteLonDeg,
            now
        );

    t.target_az_deg=targetHorizontal.az_deg;
    t.target_alt_deg=targetHorizontal.alt_deg;

    if(!t.solution_available){
        return;
    }

    const centerHorizontal=
        radecToAltAz(
            Number(t.center_ra_deg),
            Number(t.center_dec_deg),
            siteLatDeg,
            siteLonDeg,
            now
        );

    t.center_az_deg=centerHorizontal.az_deg;
    t.center_alt_deg=centerHorizontal.alt_deg;

    // Commandes réelles de la monture Alt-Az.
    t.az_delta_deg=
        norm180(targetHorizontal.az_deg-centerHorizontal.az_deg);

    t.alt_delta_deg=
        targetHorizontal.alt_deg-centerHorizontal.alt_deg;

    // Orientation sur le capteur de l'axe +AZ et de l'axe +ALT.
    // Petit déplacement numérique suffisant pour obtenir le vecteur local.
    const step=0.05;

    const azPlus=
        altAzToRadec(
            centerHorizontal.az_deg+step,
            centerHorizontal.alt_deg,
            siteLatDeg,
            siteLonDeg,
            now
        );

    const altPlus=
        altAzToRadec(
            centerHorizontal.az_deg,
            clampNumber(centerHorizontal.alt_deg+step,-89.8,89.8),
            siteLatDeg,
            siteLonDeg,
            now
        );

    const azVec=radecOffsetScreenVector(
        Number(t.center_ra_deg),
        Number(t.center_dec_deg),
        azPlus.ra_deg,
        azPlus.dec_deg,
        t
    );

    const altVec=radecOffsetScreenVector(
        Number(t.center_ra_deg),
        Number(t.center_dec_deg),
        altPlus.ra_deg,
        altPlus.dec_deg,
        t
    );

    t.az_screen_x=azVec.x;
    t.az_screen_y=azVec.y;
    t.alt_screen_x=altVec.x;
    t.alt_screen_y=altVec.y;
}

function setMountMode(mode){
    mountMode=mode==="EQ" ? "EQ" : "ALTAZ";
    localStorage.setItem("astrocam_mount_mode",mountMode);

    mountAltAz.classList.toggle("active",mountMode==="ALTAZ");
    mountEq.classList.toggle("active",mountMode==="EQ");

    siteAltAz.classList.toggle("hidden",mountMode!=="ALTAZ");
    siteNote.classList.toggle("hidden",mountMode!=="ALTAZ");

    if(mountMode==="ALTAZ"){
        if(siteConfigured()){
            siteNote.className="site-note good";
            siteNote.textContent=
                `Site Alt-Az : ${siteLatDeg.toFixed(4)}°, ${siteLonDeg.toFixed(4)}°`;
        }else{
            siteNote.className="site-note bad";
            siteNote.textContent=
                "Alt-Az : renseigne latitude / longitude une seule fois.";
        }
    }

    if(lastTarget){
        applyAltAzGuidance(lastTarget);
        drawMainOverlay();
    }
}

function saveSite(){
    const lat=Number(siteLatInput.value);
    const lon=Number(siteLonInput.value);

    if(
        !Number.isFinite(lat)
        || !Number.isFinite(lon)
        || lat<-90 || lat>90
        || lon<-180 || lon>180
    ){
        siteNote.className="site-note bad";
        siteNote.textContent="Coordonnées invalides.";
        return;
    }

    siteLatDeg=lat;
    siteLonDeg=lon;

    localStorage.setItem("astrocam_site_lat",String(lat));
    localStorage.setItem("astrocam_site_lon",String(lon));

    siteNote.className="site-note good";
    siteNote.textContent=
        `Site Alt-Az : ${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;

    if(lastTarget){
        applyAltAzGuidance(lastTarget);
    }

    drawMainOverlay();
}


function axisArrowLength(componentDeg){
    const a=Math.abs(Number(componentDeg)||0);

    if(a<0.02){
        return 0;
    }

    // Courbe comprimée : on perçoit encore la différence loin/proche,
    // sans jamais traverser tout le LiveView.
    const len=65+115*Math.log10(1+a*4.0);
    return Math.max(70,Math.min(300,len));
}

function drawAxisInstruction(
    cx,cy,
    screenX,screenY,
    componentDeg,
    positiveLabel,
    negativeLabel,
    color
){
    const value=Number(componentDeg)||0;
    const len=axisArrowLength(value);

    if(len<=0){
        return;
    }

    // IMPORTANT :
    // north_screen/east_screen décrivent le déplacement d'une étoile
    // vers N/E dans l'image.
    //
    // Quand on déplace physiquement le TELESCOPE vers N/E, le champ
    // d'étoiles part dans le sens opposé. C'est exactement le mouvement
    // visuel que l'utilisateur constatera à l'écran.
    const sign=value>=0 ? 1 : -1;

    const vx=-screenX*sign;
    const vy=-screenY*sign;

    const x2=cx+vx*len;
    const y2=cy+vy*len;

    drawArrow(cx,cy,x2,y2,color);

    const direction=value>=0 ? positiveLabel : negativeLabel;
    const magnitude=Math.abs(value);

    ctx.save();
    ctx.font="600 18px Segoe UI";
    ctx.fillStyle=color;
    ctx.textBaseline="middle";

    const label=
        `${direction}  ${magnitude.toFixed(magnitude<1 ? 2 : 1)}°`;

    let lx=x2+12;
    let ly=y2-12;

    const w=ctx.measureText(label).width;

    if(lx+w>WIDTH-12){
        lx=x2-w-12;
    }
    if(ly<20){
        ly=y2+26;
    }
    if(ly>HEIGHT-18){
        ly=HEIGHT-18;
    }

    ctx.fillText(label,lx,ly);
    ctx.restore();
}

function resultantArrowLength(separationDeg){
    const sep=Math.max(0,Number(separationDeg)||0);

    // Direction globale visible sans traverser tout le LiveView.
    // L'échelle reste volontairement comprimée.
    const len=120+110*Math.log10(1+sep*2.5);
    return Math.max(125,Math.min(360,len));
}

function drawYellowResultant(t,compact=false){
    if(!t){
        return;
    }

    const cx=WIDTH/2;
    const cy=HEIGHT/2;

    let vx=null;
    let vy=null;

    if(mountMode==="ALTAZ"){
        const v=altAzCommandVector(t);

        if(!v){
            return;
        }

        vx=v.x;
        vy=v.y;
    }else{
        if(
            typeof t.move_direction_x!=="number"
            || typeof t.move_direction_y!=="number"
        ){
            return;
        }

        vx=t.move_direction_x;
        vy=t.move_direction_y;
    }

    let len;

    if(compact){
        // Dans le champ : la jaune reste, mais devient courte.
        const sep=Math.max(0,Number(t.separation_deg)||0);

        if(sep<0.050){
            return;
        }

        len=Math.max(
            55,
            Math.min(115,45+170*sep)
        );
    }else{
        len=resultantArrowLength(t.separation_deg);
    }

    const x2=cx+vx*len;
    const y2=cy+vy*len;

    drawArrow(cx,cy,x2,y2,"#ffd34f");
}


// ============================================================
// CALIBRATION ALT-AZ APRÈS RÉALIGNEMENT PHYSIQUE DE LA CAMÉRA
//
// La caméra a été tournée mécaniquement pour rendre l'affichage intuitif.
//
// Commande physique      Affichage souhaité
// ------------------------------------------------------------
// MONTE / +ALT           ↑
// DESCENDS / -ALT        ↓
// DROITE / EST / +AZ     →
// GAUCHE / OUEST / -AZ   ←
//
// IMPORTANT :
// le signe AZ a aussi été corrigé d'après le test réel précédent :
// quand V32 demandait "GAUCHE / EST", il fallait en réalité tourner
// à DROITE pour réduire l'écart cible.
// ============================================================
const ALTAZ_SCREEN={
    ALT_UP:    {x: 0, y:-1},
    ALT_DOWN:  {x: 0, y: 1},
    AZ_EAST:   {x: 1, y: 0},
    AZ_WEST:   {x:-1, y: 0}
};

function altAzCommandVector(t){
    if(
        !t
        || typeof t.az_delta_deg!=="number"
        || typeof t.alt_delta_deg!=="number"
    ){
        return null;
    }

    const altNow=
        Number.isFinite(Number(t.center_alt_deg))
        ? Number(t.center_alt_deg)
        : 0;

    // L'azimut se projette avec cos(altitude) dans le plan tangent.
    const azWeight=
        Number(t.az_delta_deg)
        *Math.max(0.15,Math.cos(deg2rad(altNow)));

    const altWeight=Number(t.alt_delta_deg);

    // +AZ = EST = rotation physique vers la DROITE sur la monture actuelle.
    const azVec=
        azWeight>=0
        ? ALTAZ_SCREEN.AZ_EAST
        : ALTAZ_SCREEN.AZ_WEST;

    const altVec=
        altWeight>=0
        ? ALTAZ_SCREEN.ALT_UP
        : ALTAZ_SCREEN.ALT_DOWN;

    let vx=
        Math.abs(azWeight)*azVec.x
        +Math.abs(altWeight)*altVec.x;

    let vy=
        Math.abs(azWeight)*azVec.y
        +Math.abs(altWeight)*altVec.y;

    const n=Math.hypot(vx,vy);

    if(n<1e-12){
        return null;
    }

    return {
        x:vx/n,
        y:vy/n
    };
}


function drawSimpleAltAzArrow(
    cx,cy,
    screenX,screenY,
    label,
    magnitude,
    color
){
    const a=Math.abs(Number(magnitude)||0);

    if(a<0.02){
        return;
    }

    const len=Math.max(
        90,
        Math.min(245,95+90*Math.log10(1+a*3.0))
    );

    const x2=cx+screenX*len;
    const y2=cy+screenY*len;

    drawArrow(cx,cy,x2,y2,color);

    ctx.save();
    ctx.font="600 20px Segoe UI";
    ctx.fillStyle=color;
    ctx.textBaseline="middle";

    const val=
        `${label}  ${a.toFixed(a<1 ? 2 : 1)}°`;

    const tw=ctx.measureText(val).width;

    let tx=x2+14;
    let ty=y2;

    if(screenX<0){
        tx=x2-tw-14;
    }

    if(screenY<0){
        ty=y2-18;
    }else{
        ty=y2+22;
    }

    tx=Math.max(12,Math.min(WIDTH-tw-12,tx));
    ty=Math.max(24,Math.min(HEIGHT-24,ty));

    ctx.fillText(val,tx,ty);
    ctx.restore();
}


function drawMountAxisGuidance(t){
    if(!t || !t.solution_available){
        return;
    }

    const cx=WIDTH/2;
    const cy=HEIGHT/2;

    if(mountMode==="ALTAZ"){
        if(
            !siteConfigured()
            || typeof t.az_delta_deg!=="number"
            || typeof t.alt_delta_deg!=="number"
        ){
            return;
        }

        // --------------------------------------------------------
        // AZIMUT : horizontal, puisque la caméra est maintenant
        // physiquement alignée.
        //
        // +AZ = EST = DROITE
        // -AZ = OUEST = GAUCHE
        // --------------------------------------------------------
        const azEast=t.az_delta_deg>=0;
        const azVec=
            azEast
            ? ALTAZ_SCREEN.AZ_EAST
            : ALTAZ_SCREEN.AZ_WEST;

        drawSimpleAltAzArrow(
            cx,cy,
            azVec.x,
            azVec.y,
            azEast ? "TOURNE À DROITE" : "TOURNE À GAUCHE",
            t.az_delta_deg,
            "#8df0a8"
        );

        // --------------------------------------------------------
        // ALTITUDE : verticale.
        //
        // +ALT = MONTE
        // -ALT = DESCENDS
        // --------------------------------------------------------
        const altUp=t.alt_delta_deg>=0;
        const altVec=
            altUp
            ? ALTAZ_SCREEN.ALT_UP
            : ALTAZ_SCREEN.ALT_DOWN;

        drawSimpleAltAzArrow(
            cx,cy,
            altVec.x,
            altVec.y,
            altUp ? "MONTE" : "DESCENDS",
            t.alt_delta_deg,
            "#7dd3fc"
        );

        return;
    }

    // Mode équatorial conservé pour plus tard sur l'EQ6.
    drawAxisInstruction(
        cx,cy,
        t.east_screen_x,
        t.east_screen_y,
        t.east_component_deg,
        "AD → EST",
        "AD → OUEST",
        "#8df0a8"
    );

    drawAxisInstruction(
        cx,cy,
        t.north_screen_x,
        t.north_screen_y,
        t.north_component_deg,
        "DEC → NORD",
        "DEC → SUD",
        "#7dd3fc"
    );
}


function formatAxisGuidance(t){
    if(!t){
        return "";
    }

    if(mountMode==="ALTAZ"){
        if(!siteConfigured()){
            return "ALT-AZ · site non renseigné";
        }

        if(
            typeof t.az_delta_deg!=="number"
            || typeof t.alt_delta_deg!=="number"
        ){
            return "ALT-AZ · attente du solve";
        }

        const azValue=Math.abs(t.az_delta_deg);
        const altValue=Math.abs(t.alt_delta_deg);

        const turn=
            t.az_delta_deg>=0
            ? "→ DROITE"
            : "← GAUCHE";

        const vertical=
            t.alt_delta_deg>=0
            ? "↑ MONTE"
            : "↓ DESCENDS";

        return (
            `${turn} ${azValue.toFixed(2)}°`+
            `   ${vertical} ${altValue.toFixed(2)}°`
        );
    }

    if(
        typeof t.east_component_deg!=="number"
        || typeof t.north_component_deg!=="number"
    ){
        return "";
    }

    const adValue=Math.abs(t.east_component_deg);
    const decValue=Math.abs(t.north_component_deg);

    const adDir=t.east_component_deg>=0 ? "EST" : "OUEST";
    const decDir=t.north_component_deg>=0 ? "NORD" : "SUD";

    return (
        `AD → ${adDir} ${adValue.toFixed(2)}°`+
        ` · DEC → ${decDir} ${decValue.toFixed(2)}°`
    );
}


function formatPushSummary(t){
    if(!t || !t.solution_available){
        return "";
    }

    return (
        `${t.id} · écart total ${t.separation_deg.toFixed(2)}° · `+
        formatAxisGuidance(t)
    );
}


function drawTargetOnly(){
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    overlayLabel.textContent="";

    if(currentPage!=="solve"){
        return;
    }

    // Couche OpenNGC en arrière-plan : la cible Push-To reste toujours
    // dessinée par-dessus et reste donc parfaitement identifiable.
    drawSkyObjectOverlay();

    if(!lastTarget || !lastTarget.active){
        return;
    }

    if(!lastTarget.solution_available){
        overlayLabel.textContent=
            `Cible ${lastTarget.id} sélectionnée — lance un plate solve.`;
        return;
    }

    const cx=WIDTH/2;
    const cy=HEIGHT/2;

    ctx.save();
    ctx.strokeStyle="rgba(110,168,255,.92)";
    ctx.lineWidth=1.6;
    ctx.beginPath();
    ctx.arc(cx,cy,14,0,Math.PI*2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx-20,cy);
    ctx.lineTo(cx-8,cy);
    ctx.moveTo(cx+8,cy);
    ctx.lineTo(cx+20,cy);
    ctx.moveTo(cx,cy-20);
    ctx.lineTo(cx,cy-8);
    ctx.moveTo(cx,cy+8);
    ctx.lineTo(cx,cy+20);
    ctx.stroke();
    ctx.restore();

    // Repère WCS permanent : permet de vérifier immédiatement le haut/bas
    // et la rotation réelle de la caméra sur le ciel.
    drawWcsCompass(lastTarget);

    if(lastTarget.in_frame){
        const tx=lastTarget.target_x;
        const ty=lastTarget.target_y;

        ctx.save();
        ctx.strokeStyle="#ff5964";
        ctx.lineWidth=2.2;
        ctx.beginPath();
        ctx.arc(tx,ty,25,0,Math.PI*2);
        ctx.stroke();

        ctx.lineWidth=1.5;
        ctx.beginPath();
        ctx.moveTo(tx-33,ty);
        ctx.lineTo(tx-17,ty);
        ctx.moveTo(tx+17,ty);
        ctx.lineTo(tx+33,ty);
        ctx.moveTo(tx,ty-33);
        ctx.lineTo(tx,ty-17);
        ctx.moveTo(tx,ty+17);
        ctx.lineTo(tx,ty+33);
        ctx.stroke();
        ctx.restore();

        // Nom directement attaché à la cible WCS.
        drawTargetName(tx,ty,lastTarget.id);

        // Une fois l'objet DANS LE CHAMP, la grosse flèche jaune ne sert plus.
        // Le guidage devient beaucoup plus lisible :
        //   - cercle bleu = centre optique actuel
        //   - cible rouge + nom = position réelle de l'objet
        //
        // L'utilisateur n'a plus qu'à centrer la cible rouge.
        const centered=lastTarget.separation_deg<0.050;

        if(!centered){
            drawYellowResultant(lastTarget,true);
            drawMountAxisGuidance(lastTarget);
        }

        if(mountMode==="ALTAZ"){
            overlayLabel.innerHTML=
                centered
                ? `<strong>${lastTarget.id} — CENTRÉ / ACQUIS</strong>`
                : `<strong>${lastTarget.id} — dans le champ · ${lastTarget.separation_deg.toFixed(3)}°</strong><br>`+
                  `${formatAxisGuidance(lastTarget)}`;
        }else{
            overlayLabel.textContent=
                `${lastTarget.id} DANS LE CHAMP · `+
                `écart total ${lastTarget.separation_deg.toFixed(3)}° · `+
                formatAxisGuidance(lastTarget)+
                (guidanceTrend ? ` · ${guidanceTrend}` : "")+
                (lastTarget.solve_running ? " · recalage en cours…" : "")+
                (centered
                    ? ` · ${lastTarget.id} CENTRÉ / ACQUIS.`
                    : ` · centre la cible rouge sur le rond bleu.`);
        }
    }else{
        // Hors champ :
        // - JAUNE = direction globale immédiate à suivre ;
        // - VERT = composante AD Est/Ouest ;
        // - BLEU = composante DEC Nord/Sud.
        //
        // Rien n'est retiré : les axes détaillent la flèche jaune.
        drawYellowResultant(lastTarget);
        drawMountAxisGuidance(lastTarget);

        if(mountMode==="ALTAZ"){
            overlayLabel.innerHTML=
                `<strong>${lastTarget.id} — ${lastTarget.separation_deg.toFixed(1)}° restant</strong><br>`+
                `${formatAxisGuidance(lastTarget)}`;
        }else{
            overlayLabel.textContent=
                formatPushSummary(lastTarget)+
                (guidanceTrend ? ` · ${guidanceTrend}` : "")+
                (lastTarget.solve_running ? " · recalage en cours…" : "")+
                ` · jaune = direction globale · vert = AD · bleu = DEC.`;
        }
    }
}

function drawMainOverlay(){
    if(currentPage==="polar"){
        drawPolarOnly();
    }else if(currentPage==="solve"){
        drawTargetOnly();
    }else{
        ctx.clearRect(0,0,WIDTH,HEIGHT);
        overlayLabel.textContent="";
    }
}

overlayCanvas.addEventListener("click",async(ev)=>{
    if(
        currentPage!=="polar" ||
        !lastPolar ||
        !lastPolar.calibrated ||
        lastPolar.solve_running
    ){
        return;
    }

    const rect=overlayCanvas.getBoundingClientRect();
    const x=(ev.clientX-rect.left)/rect.width*WIDTH;
    const y=(ev.clientY-rect.top)/rect.height*HEIGHT;

    try{
        const r=await fetch("/api/polar/select-star",{
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify({x,y})
        });
        const d=await r.json();
        if(!r.ok){
            polarResult.textContent=d.error||"Impossible de sélectionner l’étoile.";
        }
        await refreshPolar();
    }catch(e){
        polarResult.textContent="Erreur réseau lors de la sélection.";
    }
});

async function loadSettings(){
    try{
        const r=await fetch("/api/settings");
        const s=await r.json();
        setExposure(s.exposure_ms,false);
        setGain(s.gain,false);
        setBinningMode(s.binning_mode,false);
        setPreviewMode(s.preview_mode,false);
        refreshLabels();
    }catch(e){console.error(e)}
}

async function refreshStatus(){
    try{
        const r=await fetch("/api/status",{cache:"no-store"});
        const s=await r.json();

        setDot("stIndi",s.indi);
        setDot("stCam",s.acquisition);
        setDot("stProc",s.processing);
        setDot("stJpeg",s.jpeg);
        updateSolveUi(s);
        currentSolveRunning=(s.solve_status==="running");

        lastWcsOrientation={
            available:Boolean(s.wcs_orientation_available),
            north_x:Number(s.wcs_north_x),
            north_y:Number(s.wcs_north_y),
            east_x:Number(s.wcs_east_x),
            east_y:Number(s.wcs_east_y),
            rotation_deg:Number(s.solve_rotation_deg)
        };

        lastStatusSolveSeq=Number(s.solve_source_seq)||0;

        if(Number.isFinite(Number(s.solve_scale_arcsec_px))){
            skyOverlayScale=Number(s.solve_scale_arcsec_px);
        }

        applyDisplayOrientation();

        if(
            skyOverlayEnabled
            && currentPage==="solve"
            && lastStatusSolveSeq>0
            && skyOverlaySeq!==lastStatusSolveSeq
        ){
            refreshSkyOverlay();
        }

        document.getElementById("mRaw").textContent=s.raw_frames;
        document.getElementById("mProc").textContent=s.processed_frames;
        document.getElementById("mJpeg").textContent=s.jpeg_frames;
        document.getElementById("mSkip").textContent=s.skipped_frames;
        document.getElementById("mAcqMs").textContent=`${s.acquisition_ms.toFixed(0)} ms`;
        document.getElementById("mProcMs").textContent=`${s.processing_ms.toFixed(0)} ms`;
        document.getElementById("mJpegMs").textContent=`${s.jpeg_ms.toFixed(0)} ms`;

        await refreshPolar();
        await refreshTarget();

        // ------------------------------------------------------------
        // PUSH-TO AUTOMATIQUE
        // ------------------------------------------------------------
        // Tant que le bouton "Guidage cible" est actif :
        // 1. on attend la fin du solve courant ;
        // 2. on lance un nouveau solve sur la RAW la plus récente ;
        // 3. refreshTarget() récupère le nouveau WCS au cycle suivant ;
        // 4. la flèche est donc recalculée après chaque solve.
        //
        // Ce n'est PAS un tracking vidéo continu : avec solve-field à
        // ~8 secondes, la flèche se recale environ toutes les 8-10 s.
        if(
            pushAuto
            && currentPage==="solve"
            && lastTarget
            && lastTarget.active
            && s.solve_available
            && s.solve_status!=="running"
            && Date.now()>=autoSolveCooldownUntil
        ){
            // Évite une double requête entre deux refresh HTTP.
            autoSolveCooldownUntil=Date.now()+2500;

            pushResult.textContent=
                "AUTO · nouveau plate solve en cours — garde la lunette immobile quelques secondes…";

            try{
                const rr=await fetch("/api/solve",{
                    method:"POST",
                    cache:"no-store"
                });

                const data=await rr.json().catch(()=>({}));

                if(!rr.ok){
                    pushResult.textContent=
                        "AUTO · solve non lancé : "+(data.error||"erreur");
                    autoSolveCooldownUntil=Date.now()+4000;
                }
            }catch(e){
                pushResult.textContent="AUTO · erreur réseau pendant le solve.";
                autoSolveCooldownUntil=Date.now()+4000;
            }
        }else if(
            pushAuto
            && currentPage==="solve"
            && s.solve_status==="running"
        ){
            pushResult.textContent=
                "AUTO · plate solving en cours — la flèche sera recalculée à la fin…";
        }

    }catch(e){
        ["stIndi","stCam","stProc","stJpeg","stSolve"].forEach(id=>setDot(id,false));
    }
}

controlsHidden=localStorage.getItem("astrocam_controls_hidden")==="1";
applyCollapsedState();
updateSkyOverlayButton();

// La page par défaut est TOUJOURS Mise en station.
setPage("polar");
loadSettings();
loadCatalog();
refreshStatus();
setInterval(refreshStatus,1000);
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return PAGE

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        if request.method == "POST":
            data = request.get_json(silent=True) or {}

            if "exposure_ms" in data:
                try:
                    value = float(data["exposure_ms"])
                    exposure_ms_value.value = max(
                        EXPOSURE_MIN_MS,
                        min(EXPOSURE_MAX_MS, value)
                    )
                except (TypeError, ValueError):
                    pass

            if "gain" in data:
                try:
                    value = int(round(float(data["gain"])))
                    gain_value.value = max(
                        GAIN_MIN,
                        min(GAIN_MAX, value)
                    )
                except (TypeError, ValueError):
                    pass

            if "binning_mode" in data:
                mode = str(data["binning_mode"]).upper()
                if mode in ("ADD", "AVG"):
                    binning_mode_value.value = 1 if mode == "ADD" else 0

            if "preview_mode" in data:
                mode = str(data["preview_mode"]).upper()
                if mode in ("TECHNIQUE", "TECH"):
                    preview_mode_value.value = PREVIEW_TECHNIQUE
                elif mode in ("DARK", "CIEL_NOIR", "CIEL NOIR"):
                    preview_mode_value.value = PREVIEW_DARK_SKY

        return jsonify({
            "exposure_ms": round(float(exposure_ms_value.value), 1),
            "gain": int(round(float(gain_value.value))),
            "binning_mode": "ADD" if int(binning_mode_value.value) else "AVG",
            "preview_mode": (
                "DARK"
                if int(preview_mode_value.value) == PREVIEW_DARK_SKY
                else "TECHNIQUE"
            ),
            "exposure_min_ms": EXPOSURE_MIN_MS,
            "exposure_max_ms": EXPOSURE_MAX_MS,
            "gain_min": GAIN_MIN,
            "gain_max": GAIN_MAX,
        })

    @app.route("/api/solve", methods=["POST"])
    def api_solve():
        if not bool(solve_available.value):
            return jsonify({
                "ok": False,
                "error": get_shared_text(solve_message) or "Plate solver indisponible"
            }), 503

        if solve_status.value == SOLVE_RUNNING:
            return jsonify({
                "ok": False,
                "error": "Un plate solving est déjà en cours."
            }), 409

        solve_context.value = 0
        solve_status.value = SOLVE_RUNNING
        set_shared_text(solve_message, "Résolution en cours...")

        with solve_request.get_lock():
            solve_request.value += 1
            request_id = int(solve_request.value)

        return jsonify({
            "ok": True,
            "request_id": request_id
        })

    def ensure_polar_solution():
        valid = [int(polar_valid[i]) for i in range(3)]

        if valid != [1, 1, 1]:
            return

        if polar_state.value == POLAR_CALIBRATED:
            return

        try:
            ra_values = [float(polar_ra_values[i]) for i in range(3)]
            dec_values = [float(polar_dec_values[i]) for i in range(3)]
            jac = [float(polar_jacobian[i]) for i in range(4)]

            solution = compute_polar_alignment_solution(
                ra_values,
                dec_values,
                jac
            )

            polar_error_arcmin.value = solution["error_arcmin"]
            polar_axis_ra.value = solution["axis_ra_deg"]
            polar_axis_dec.value = solution["axis_dec_deg"]
            polar_shift_x.value = solution["shift_x_px"]
            polar_shift_y.value = solution["shift_y_px"]
            polar_guidance_active.value = 0
            polar_state.value = POLAR_CALIBRATED

            set_shared_text(
                polar_message,
                "Calibration 3 points terminée. "
                "Clique une étoile brillante dans le live."
            )

            print(
                f"[POLAR OK] erreur={solution['error_arcmin']:.2f}' | "
                f"axe RA={solution['axis_ra_deg']:.5f}° "
                f"DEC={solution['axis_dec_deg']:.5f}° | "
                f"shift={solution['shift_x_px']:.1f},"
                f"{solution['shift_y_px']:.1f}px",
                flush=True
            )

        except Exception as exc:
            polar_state.value = POLAR_ERROR
            set_shared_text(polar_message, str(exc))
            print(f"[POLAR ERREUR] {exc}", flush=True)


    @app.route("/api/polar/reset", methods=["POST"])
    def api_polar_reset():
        if solve_status.value == SOLVE_RUNNING:
            return jsonify({
                "ok": False,
                "error": "Un plate solving est en cours."
            }), 409

        with polar_valid.get_lock():
            for i in range(3):
                polar_valid[i] = 0

        polar_state.value = POLAR_IDLE
        polar_guidance_active.value = 0
        polar_error_arcmin.value = 0.0
        polar_shift_x.value = 0.0
        polar_shift_y.value = 0.0
        polar_ref_x.value = 0.0
        polar_ref_y.value = 0.0
        polar_target_x.value = 0.0
        polar_target_y.value = 0.0
        solve_context.value = 0

        set_shared_text(
            polar_message,
            "Prêt pour la position A."
        )

        return jsonify({"ok": True})


    @app.route("/api/polar/capture/<int:slot>", methods=["POST"])
    def api_polar_capture(slot):
        if slot not in (1, 2, 3):
            return jsonify({"ok": False, "error": "Position invalide."}), 400

        if not bool(solve_available.value):
            return jsonify({
                "ok": False,
                "error": get_shared_text(solve_message) or "Solver indisponible."
            }), 503

        if solve_status.value == SOLVE_RUNNING:
            return jsonify({
                "ok": False,
                "error": "Un plate solving est déjà en cours."
            }), 409

        # La position A démarre toujours une nouvelle calibration.
        if slot == 1:
            with polar_valid.get_lock():
                for i in range(3):
                    polar_valid[i] = 0
            polar_guidance_active.value = 0
            polar_state.value = POLAR_COLLECTING

        if slot == 2 and not bool(polar_valid[0]):
            return jsonify({
                "ok": False,
                "error": "Capture d'abord la position A."
            }), 409

        if slot == 3 and not bool(polar_valid[1]):
            return jsonify({
                "ok": False,
                "error": "Capture d'abord la position B."
            }), 409

        solve_context.value = slot
        solve_status.value = SOLVE_RUNNING
        set_shared_text(
            solve_message,
            f"Mise en station : résolution position {slot}/3..."
        )
        set_shared_text(
            polar_message,
            f"Résolution position {slot}/3 en cours..."
        )

        with solve_request.get_lock():
            solve_request.value += 1
            request_id = int(solve_request.value)

        return jsonify({
            "ok": True,
            "request_id": request_id,
            "slot": slot
        })


    @app.route("/api/polar/select-star", methods=["POST"])
    def api_polar_select_star():
        ensure_polar_solution()

        if polar_state.value != POLAR_CALIBRATED:
            return jsonify({
                "ok": False,
                "error": "Calibration polaire non terminée."
            }), 409

        data = request.get_json(silent=True) or {}

        try:
            x = float(data.get("x"))
            y = float(data.get("y"))
        except (TypeError, ValueError):
            return jsonify({
                "ok": False,
                "error": "Coordonnées étoile invalides."
            }), 400

        if not (0.0 <= x < WIDTH and 0.0 <= y < HEIGHT):
            return jsonify({
                "ok": False,
                "error": "Clique à l'intérieur de l'image."
            }), 400

        polar_ref_x.value = x
        polar_ref_y.value = y
        polar_target_x.value = x + polar_shift_x.value
        polar_target_y.value = y + polar_shift_y.value
        polar_guidance_active.value = 1

        set_shared_text(
            polar_message,
            "Guidage polaire actif : déplace l'étoile vers la cible rouge "
            "uniquement avec les vis ALT/AZ."
        )

        return jsonify({
            "ok": True,
            "ref_x": x,
            "ref_y": y,
            "target_x": float(polar_target_x.value),
            "target_y": float(polar_target_y.value)
        })


    @app.route("/api/polar/status")
    def api_polar_status():
        ensure_polar_solution()

        valid = [bool(polar_valid[i]) for i in range(3)]

        calibrated = polar_state.value == POLAR_CALIBRATED

        return jsonify({
            "valid": valid,
            "state": int(polar_state.value),
            "calibrated": calibrated,
            "solve_running": solve_status.value == SOLVE_RUNNING,
            "error_arcmin": float(polar_error_arcmin.value),
            "axis_ra_deg": float(polar_axis_ra.value),
            "axis_dec_deg": float(polar_axis_dec.value),
            "axis_ra_hms": format_ra_hms(float(polar_axis_ra.value)),
            "axis_dec_dms": format_dec_dms(float(polar_axis_dec.value)),
            "shift_x": float(polar_shift_x.value),
            "shift_y": float(polar_shift_y.value),
            "guidance_active": bool(polar_guidance_active.value),
            "ref_x": float(polar_ref_x.value),
            "ref_y": float(polar_ref_y.value),
            "target_x": float(polar_target_x.value),
            "target_y": float(polar_target_y.value),
            "message": get_shared_text(polar_message),
        })


    @app.route("/api/sky-overlay")
    def api_sky_overlay():
        nonlocal sky_overlay_cache_seq, sky_overlay_cache

        if catalog_mode != "OPENNGC" or openngc_catalog is None:
            return jsonify({
                "ok": False,
                "error": "OpenNGC indisponible."
            }), 503

        seq = int(solve_source_seq.value)

        if seq <= 0:
            return jsonify({
                "ok": True,
                "source_seq": 0,
                "scale_arcsec_px": float(solve_scale.value),
                "items": [],
            })

        if sky_overlay_cache_seq != seq:
            jac = [
                float(solve_jacobian[i])
                for i in range(4)
            ]

            try:
                sky_overlay_cache = openngc_catalog.overlay_items(
                    float(solve_ra.value),
                    float(solve_dec.value),
                    jac,
                    float(solve_scale.value),
                    limit=80
                )
                sky_overlay_cache_seq = seq
            except Exception as exc:
                return jsonify({
                    "ok": False,
                    "error": f"Couche céleste : {exc}"
                }), 500

        return jsonify({
            "ok": True,
            "source_seq": seq,
            "scale_arcsec_px": float(solve_scale.value),
            "solve_running": solve_status.value == SOLVE_RUNNING,
            "items": sky_overlay_cache,
        })


    @app.route("/api/catalog")
    def api_catalog():
        query = str(request.args.get("q", "")).strip()
        kind = str(request.args.get("kind", "M")).upper()

        if kind not in {"M", "NGC", "IC", "ALL"}:
            kind = "ALL"

        if catalog_mode == "OPENNGC" and openngc_catalog is not None:
            try:
                items = openngc_catalog.search(
                    query=query,
                    kind=kind,
                    limit=120
                )
            except Exception as exc:
                return jsonify({
                    "ok": False,
                    "error": f"Erreur OpenNGC : {exc}"
                }), 500

            return jsonify({
                "ok": True,
                "source": catalog_source,
                "catalog_mode": "OPENNGC",
                "total_objects": len(openngc_catalog.records),
                "items": items,
            })

        if catalog_search is None:
            return jsonify({
                "ok": False,
                "error": catalog_source
            }), 503

        search_query = query

        if not search_query:
            if kind == "M":
                search_query = "M"
            elif kind == "NGC":
                search_query = "NGC"
            elif kind == "IC":
                search_query = "IC"

        try:
            raw_items = catalog_search(search_query)
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": f"Erreur catalogue : {exc}"
            }), 500

        items = []

        for raw in raw_items or []:
            item = normalize_catalog_item(raw)

            if item is None:
                continue

            if kind == "M" and item["family"] != "Messier":
                continue

            if kind == "NGC" and item["family"] != "NGC":
                continue

            if kind == "IC" and item["family"] != "IC":
                continue

            item["ra_hms"] = format_ra_hms(item["ra_deg"])
            item["dec_dms"] = format_dec_dms(item["dec_deg"])
            items.append(item)

            if len(items) >= 80:
                break

        return jsonify({
            "ok": True,
            "source": catalog_source,
            "catalog_mode": "LEGACY",
            "items": items,
        })


    @app.route("/api/target", methods=["POST"])
    def api_target():
        data = request.get_json(silent=True) or {}

        try:
            ra = float(data["ra_deg"])
            dec = float(data["dec_deg"])
        except (KeyError, TypeError, ValueError):
            return jsonify({
                "ok": False,
                "error": "Coordonnées de cible invalides."
            }), 400

        if not (0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0):
            return jsonify({
                "ok": False,
                "error": "Coordonnées hors limites."
            }), 400

        identifier = str(data.get("id") or "Cible").strip()
        name = str(data.get("name") or identifier).strip()

        target_ra.value = ra
        target_dec.value = dec
        set_shared_text(target_id, identifier)
        set_shared_text(target_name, name)
        target_active.value = 1

        print(
            f"[TARGET] {identifier} — {name} | "
            f"RA={ra:.6f}° DEC={dec:.6f}°",
            flush=True
        )

        return jsonify({"ok": True})


    @app.route("/api/target/clear", methods=["POST"])
    def api_target_clear():
        target_active.value = 0
        set_shared_text(target_id, "")
        set_shared_text(target_name, "")
        return jsonify({"ok": True})


    @app.route("/api/target/status")
    def api_target_status():
        if not bool(target_active.value):
            return jsonify({
                "active": False,
                "solution_available": False,
            })

        ra = float(target_ra.value)
        dec = float(target_dec.value)

        result = {
            "active": True,
            "id": get_shared_text(target_id),
            "name": get_shared_text(target_name),
            "ra_deg": ra,
            "dec_deg": dec,
            "ra_hms": format_ra_hms(ra),
            "dec_dms": format_dec_dms(dec),
            "solution_available": False,
        }

        # Tant qu'au moins un solve a déjà réussi, on conserve sa
        # direction même si le solve suivant est actuellement RUNNING
        # ou vient d'échouer. La flèche ne disparaît donc jamais entre
        # deux recalages.
        last_success_seq = int(solve_source_seq.value)

        if last_success_seq <= 0:
            result["solve_running"] = (
                solve_status.value == SOLVE_RUNNING
            )
            return jsonify(result)

        jac = [float(solve_jacobian[i]) for i in range(4)]

        try:
            guidance = target_guidance_from_solution(
                float(solve_ra.value),
                float(solve_dec.value),
                ra,
                dec,
                jac
            )
        except Exception as exc:
            result["guidance_error"] = str(exc)
            return jsonify(result)

        result.update({
            "solution_available": True,
            "center_ra_deg": float(solve_ra.value),
            "center_dec_deg": float(solve_dec.value),
            "separation_deg": guidance["separation_deg"],
            "direction_x": guidance["direction_x"],
            "direction_y": guidance["direction_y"],
            "move_direction_x": guidance["move_direction_x"],
            "move_direction_y": guidance["move_direction_y"],
            "target_x": guidance["target_x"],
            "target_y": guidance["target_y"],
            "in_frame": guidance["in_frame"],
            "solve_source_seq": last_success_seq,
            "solve_running": solve_status.value == SOLVE_RUNNING,
            "east_screen_x": guidance["east_screen_x"],
            "east_screen_y": guidance["east_screen_y"],
            "north_screen_x": guidance["north_screen_x"],
            "north_screen_y": guidance["north_screen_y"],
            "east_component_deg": guidance["east_component_deg"],
            "north_component_deg": guidance["north_component_deg"],
        })

        return jsonify(result)


    @app.route("/api/status")
    def api_status():
        now = time.time()

        # L'exposition peut aller à 10 s : la tolérance de heartbeat
        # s'adapte pour ne pas déclarer à tort une panne pendant une pose longue.
        exp_s = float(exposure_ms_value.value) / 1000.0
        stale_after = max(3.0, exp_s + 2.5)

        acq_ok = (
            heartbeat_acq.value > 0
            and (now - heartbeat_acq.value) < stale_after
        )
        proc_ok = (
            heartbeat_proc.value > 0
            and (now - heartbeat_proc.value) < stale_after
        )
        jpeg_ok = (
            heartbeat_jpeg.value > 0
            and (now - heartbeat_jpeg.value) < stale_after
        )

        solve_code = int(solve_status.value)
        solve_name = {
            SOLVE_IDLE: "idle",
            SOLVE_RUNNING: "running",
            SOLVE_OK: "ok",
            SOLVE_ERROR: "error",
        }.get(solve_code, "error")

        # Orientation du dernier WCS valide, indépendamment de la cible
        # Push-To. Elle sert uniquement à remettre le LiveView "Nord en haut".
        wcs_orientation_available = False
        wcs_north_x = 0.0
        wcs_north_y = -1.0
        wcs_east_x = 1.0
        wcs_east_y = 0.0

        if int(solve_source_seq.value) > 0:
            try:
                j = np.asarray(
                    [float(solve_jacobian[i]) for i in range(4)],
                    dtype=np.float64
                ).reshape((2, 2))

                if abs(float(np.linalg.det(j))) > 1e-12:
                    inv_j = np.linalg.inv(j)

                    east_px = inv_j @ np.array([1.0, 0.0], dtype=np.float64)
                    north_px = inv_j @ np.array([0.0, 1.0], dtype=np.float64)

                    east_norm = float(np.hypot(east_px[0], east_px[1]))
                    north_norm = float(np.hypot(north_px[0], north_px[1]))

                    if east_norm > 1e-12 and north_norm > 1e-12:
                        wcs_east_x = float(east_px[0] / east_norm)
                        wcs_east_y = float(east_px[1] / east_norm)
                        wcs_north_x = float(north_px[0] / north_norm)
                        wcs_north_y = float(north_px[1] / north_norm)
                        wcs_orientation_available = True

            except Exception:
                pass

        return jsonify({
            "indi": bool(indi_connected.value),
            "acquisition": acq_ok,
            "processing": proc_ok,
            "jpeg": jpeg_ok,
            "raw_frames": int(acq_count.value),
            "processed_frames": int(proc_count.value),
            "jpeg_frames": int(jpeg_count.value),
            "skipped_frames": int(skipped_count.value),
            "acquisition_ms": float(last_acq_ms.value),
            "processing_ms": float(last_proc_ms.value),
            "jpeg_ms": float(last_jpeg_ms.value),
            "solve_available": bool(solve_available.value),
            "solve_status": solve_name,
            "solve_message": get_shared_text(solve_message),
            "solve_source_seq": int(solve_source_seq.value),
            "solve_ra_deg": float(solve_ra.value),
            "solve_dec_deg": float(solve_dec.value),
            "solve_scale_arcsec_px": float(solve_scale.value),
            "solve_rotation_deg": float(solve_rotation.value),
            "solve_elapsed_s": float(solve_elapsed.value),
            "wcs_orientation_available": wcs_orientation_available,
            "wcs_north_x": wcs_north_x,
            "wcs_north_y": wcs_north_y,
            "wcs_east_x": wcs_east_x,
            "wcs_east_y": wcs_east_y,
        })

    @app.route("/health")
    def health():
        return jsonify({
            "raw_frames": int(acq_count.value),
            "processed_frames": int(proc_count.value),
            "jpeg_frames": int(jpeg_count.value),
            "skipped_frames": int(skipped_count.value),
            "jpeg_sequence": int(jpeg_sequence.value),
            "jpeg_size": int(jpeg_size.value),
            "acquisition_ms": round(float(last_acq_ms.value), 1),
            "processing_ms": round(float(last_proc_ms.value), 1),
            "jpeg_ms": round(float(last_jpeg_ms.value), 1),
            "background_raw12": round(float(background_value.value), 1),
            "black_raw12": round(float(black_value.value), 1),
            "white_raw12": round(float(white_value.value), 1),
            "exposure_ms": round(float(exposure_ms_value.value), 1),
            "gain": int(round(float(gain_value.value))),
            "binning_mode": "ADD" if int(binning_mode_value.value) else "AVG",
        })

    def mjpeg_generator():
        last_seq = 0

        while not stop_event.is_set():
            seq = jpeg_sequence.value
            size = jpeg_size.value

            if seq == 0 or size <= 0 or seq == last_seq:
                time.sleep(0.01)
                continue

            with jpeg_lock:
                seq = jpeg_sequence.value
                size = jpeg_size.value
                frame = bytes(jpeg_shared[:size])

            last_seq = seq

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame)).encode("ascii")
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )

    @app.route("/stream")
    def stream():
        return Response(
            mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    print(
        f"[WEB pid={os.getpid()}] http://0.0.0.0:{WEB_PORT}/",
        flush=True
    )

    app.run(
        host=WEB_HOST,
        port=WEB_PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )



def tcp_port_listening_linux(port, host="127.0.0.1"):
    """
    Vérifie qu'un port TCP est en LISTEN sans ouvrir de connexion.
    On évite ainsi les New arrival / EOF dans indiserver uniquement
    pour faire du monitoring.
    """
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
                    state = fields[3]

                    # 0A = TCP_LISTEN
                    if state != "0A":
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
    """
    Retourne (process, started_here).

    - Si le port 7624 écoute déjà, on réutilise l'INDI existant.
    - Sinon Astro Cam lance indiserver + le driver SC715C.
    - Le process est placé dans son propre groupe pour pouvoir arrêter
      proprement indiserver ET son driver à la fermeture d'Astro Cam.
    """
    import shutil
    import subprocess

    if tcp_port_listening_linux(INDI_PORT):
        print(
            f"[INDI] serveur déjà actif sur {HOST}:{INDI_PORT} — réutilisation.",
            flush=True
        )
        return None, False

    indiserver_bin = shutil.which("indiserver")
    driver_bin = shutil.which("indi_svbonycam_ccd")

    if not indiserver_bin:
        raise RuntimeError("Commande indiserver introuvable.")

    if not driver_bin:
        raise RuntimeError(
            "Driver indi_svbonycam_ccd introuvable dans le PATH."
        )

    command = [
        indiserver_bin,
        "-v",
        "-r", "5",
        "-p", str(INDI_PORT),
        driver_bin,
    ]

    print("[INDI] démarrage automatique...", flush=True)
    print(
        f"[INDI] {indiserver_bin} -v -r 5 -p {INDI_PORT} {driver_bin}",
        flush=True
    )

    # stdout/stderr hérités : les logs INDI restent visibles dans
    # le même terminal que l'application.
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=None,
        stderr=None,
        start_new_session=True,
        close_fds=True,
    )

    deadline = time.monotonic() + 10.0

    while time.monotonic() < deadline:
        return_code = process.poll()

        if return_code is not None:
            raise RuntimeError(
                f"indiserver s'est arrêté pendant le démarrage "
                f"(code {return_code})."
            )

        if tcp_port_listening_linux(INDI_PORT):
            print(
                f"[INDI] prêt sur {HOST}:{INDI_PORT} "
                f"(pid={process.pid}).",
                flush=True
            )
            return process, True

        time.sleep(0.10)

    # Démarrage incomplet : on ne laisse rien traîner.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        process.wait(timeout=2.0)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    raise RuntimeError(
        f"indiserver n'écoute toujours pas sur le port {INDI_PORT} "
        "après 10 secondes."
    )


def stop_managed_indiserver(process):
    """
    Arrête uniquement l'INDI lancé par Astro Cam.
    Un serveur INDI déjà présent avant le lancement n'est jamais tué.
    """
    if process is None:
        return

    if process.poll() is not None:
        return

    print("[INDI] arrêt du serveur lancé par Astro Cam...", flush=True)

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        process.wait(timeout=4.0)
        print("[INDI] arrêté proprement.", flush=True)
        return
    except Exception:
        pass

    print("[INDI] arrêt forcé...", flush=True)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    try:
        process.wait(timeout=2.0)
    except Exception:
        pass


def main():
    sampling = (
        206.265
        * (PIXEL_SIZE_UM * BINNING)
        / FOCAL_LENGTH_MM
    )

    print("=" * 82)
    print("ASTRO CAM - SKY OVERLAY V37")
    print("=" * 82)
    print(f"Caméra          : {DEVICE} (détection dynamique USB2/USB3)")
    print(f"RAW             : {WIDTH}x{HEIGHT} uint16")
    print(f"Preview         : {WIDTH}x{HEIGHT} uint8")
    print(f"Exposition      : {EXPOSURE_DEFAULT_MS:.0f} ms au démarrage")
    print(f"Gain            : {GAIN_DEFAULT} au démarrage")
    print(f"Binning         : {BINNING}x{BINNING} ADD")
    print(f"Focale          : {FOCAL_LENGTH_MM:.1f} mm")
    print(f"Échantillonnage : ~{sampling:.2f} arcsec/pixel")
    print(f"JPEG qualité    : {JPEG_QUALITY}")
    print(f"Serveur Web     : http://192.168.1.53:{WEB_PORT}/")
    print("INDI            : auto-start + synchro HotPlug/propriétés")
    print("Plate solver    : solve-field + Push-To Messier/NGC")
    print("Catalogue       : OpenNGC complet local + secours ancien catalogue")
    print("Overlay ciel    : Messier / NGC / IC WCS, ON/OFF, tailles OpenNGC")
    print("Push-To visuel  : traits fins + flèches ouvertes + cible allégée")
    print("Mise en station : 3 solves AD + cible live ALT/AZ")
    print("Solve temporaire: /dev/shm (RAM)")
    print("Queue images    : AUCUNE")
    print("Preview robuste : frames noires/saturées non fatales")
    print("Récup caméra    : réarmement BLOB + reconnexion auto")
    print("Push-To         : flèche persistante + tendance + repère N/E")
    print("Sens Push-To    : mouvement physique X/Y inversé vs image WCS")
    print("Push-To axes    : Alt-Az ou équatorial, sélectionnable dans le Web")
    print("Flèche jaune    : direction globale conservée")
    print("LiveView        : Full frame natif, aucune rotation / aucun zoom")
    print("Indications     : axes Alt-Az calibrés sur le mouvement réel écran")
    print("Calibration     : MONTE ↑ · DESCENDS ↓ · DROITE/EST → · GAUCHE/OUEST ←")
    print("Affichage WCS   : orientation mesurée conservée comme information")
    print("Arrêt caméra    : déconnexion INDI propre avant arrêt serveur")
    print("Détection USB   : nom INDI dynamique + init parser corrigée")
    print("Repère Alt-Az   : caméra physiquement remise droite")
    print("=" * 82)

    indi_process = None
    indi_started_here = False

    try:
        indi_process, indi_started_here = start_indiserver_if_needed()
    except Exception as exc:
        print(
            f"\n[ERREUR INDI] {type(exc).__name__}: {exc}",
            flush=True
        )
        return 2

    print(
        "INDI            : "
        + (
            "démarré automatiquement par Astro Cam"
            if indi_started_here
            else "serveur existant réutilisé"
        )
    )

    shared_raw = mp.RawArray(ctypes.c_uint16, PIXELS)
    raw_lock = mp.Lock()
    raw_sequence = mp.Value(ctypes.c_ulonglong, 0, lock=False)

    shared_preview = mp.RawArray(ctypes.c_uint8, PIXELS)
    preview_lock = mp.Lock()
    preview_sequence = mp.Value(ctypes.c_ulonglong, 0, lock=False)

    shared_jpeg = mp.RawArray(ctypes.c_uint8, MAX_JPEG_BYTES)
    jpeg_lock = mp.Lock()
    jpeg_size = mp.Value(ctypes.c_uint32, 0, lock=False)
    jpeg_sequence = mp.Value(ctypes.c_ulonglong, 0, lock=False)

    stop_event = mp.Event()

    # Réglages caméra partagés entre le Web et le process acquisition.
    exposure_ms_value = mp.Value(
        ctypes.c_double,
        EXPOSURE_DEFAULT_MS
    )
    gain_value = mp.Value(
        ctypes.c_double,
        GAIN_DEFAULT
    )
    binning_mode_value = mp.Value(
        ctypes.c_int,
        BINNING_MODE_DEFAULT
    )
    preview_mode_value = mp.Value(
        ctypes.c_int,
        PREVIEW_MODE_DEFAULT
    )

    # Heartbeats indépendants : permettent au Web de voir si chaque brique vit.
    heartbeat_acq = mp.Value(ctypes.c_double, 0.0, lock=False)
    heartbeat_proc = mp.Value(ctypes.c_double, 0.0, lock=False)
    heartbeat_jpeg = mp.Value(ctypes.c_double, 0.0, lock=False)

    # État de la connexion INDI persistante.
    # Le Web lit cette valeur sans ouvrir de nouvelle socket vers indiserver.
    indi_connected = mp.Value(ctypes.c_int, 0, lock=False)

    acq_count = mp.Value(ctypes.c_ulonglong, 0)
    proc_count = mp.Value(ctypes.c_ulonglong, 0)
    jpeg_count = mp.Value(ctypes.c_ulonglong, 0)
    skipped_count = mp.Value(ctypes.c_ulonglong, 0)

    last_acq_ms = mp.Value(ctypes.c_double, 0.0, lock=False)
    last_proc_ms = mp.Value(ctypes.c_double, 0.0, lock=False)
    last_jpeg_ms = mp.Value(ctypes.c_double, 0.0, lock=False)

    background_value = mp.Value(ctypes.c_double, 0.0, lock=False)
    black_value = mp.Value(ctypes.c_double, 0.0, lock=False)
    white_value = mp.Value(ctypes.c_double, 0.0, lock=False)

    # Plate solver : état / requêtes / résultat partagés.
    solve_request = mp.Value(ctypes.c_ulonglong, 0)
    solve_status = mp.Value(ctypes.c_int, SOLVE_IDLE, lock=False)
    solve_available = mp.Value(ctypes.c_int, 0, lock=False)
    solve_source_seq = mp.Value(ctypes.c_ulonglong, 0, lock=False)
    solve_ra = mp.Value(ctypes.c_double, 0.0, lock=False)
    solve_dec = mp.Value(ctypes.c_double, 0.0, lock=False)
    solve_scale = mp.Value(ctypes.c_double, 0.0, lock=False)
    solve_rotation = mp.Value(ctypes.c_double, 0.0, lock=False)
    solve_elapsed = mp.Value(ctypes.c_double, 0.0, lock=False)
    solve_message = mp.Array(ctypes.c_char, 768)

    # Jacobien tangent du dernier plate solve, quel que soit son contexte.
    solve_jacobian = mp.Array(ctypes.c_double, 4)

    # Cible Push-To Messier / NGC.
    target_active = mp.Value(ctypes.c_int, 0, lock=False)
    target_ra = mp.Value(ctypes.c_double, 0.0, lock=False)
    target_dec = mp.Value(ctypes.c_double, 0.0, lock=False)
    target_id = mp.Array(ctypes.c_char, 96)
    target_name = mp.Array(ctypes.c_char, 192)

    # Contexte du solve : 0=générique, 1/2/3=positions polaires A/B/C.
    solve_context = mp.Value(ctypes.c_int, 0, lock=False)

    # Mise en station polaire.
    polar_valid = mp.Array(ctypes.c_int, 3)
    polar_ra_values = mp.Array(ctypes.c_double, 3)
    polar_dec_values = mp.Array(ctypes.c_double, 3)

    # Jacobien tangent local du WCS de la troisième position :
    # [east/dx, east/dy, north/dx, north/dy].
    polar_jacobian = mp.Array(ctypes.c_double, 4)

    polar_state = mp.Value(ctypes.c_int, POLAR_IDLE, lock=False)
    polar_error_arcmin = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_axis_ra = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_axis_dec = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_shift_x = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_shift_y = mp.Value(ctypes.c_double, 0.0, lock=False)

    polar_ref_x = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_ref_y = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_target_x = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_target_y = mp.Value(ctypes.c_double, 0.0, lock=False)
    polar_guidance_active = mp.Value(ctypes.c_int, 0, lock=False)
    polar_message = mp.Array(ctypes.c_char, 768)
    set_shared_text(polar_message, "Prêt pour la position A.")

    acq = mp.Process(
        target=acquisition_process,
        name="astrocam-acquisition",
        args=(
            shared_raw,
            raw_lock,
            raw_sequence,
            stop_event,
            acq_count,
            last_acq_ms,
            exposure_ms_value,
            gain_value,
            binning_mode_value,
            heartbeat_acq,
            indi_connected
        )
    )

    proc = mp.Process(
        target=processing_process,
        name="astrocam-processing",
        args=(
            shared_raw,
            raw_lock,
            raw_sequence,
            shared_preview,
            preview_lock,
            preview_sequence,
            stop_event,
            proc_count,
            skipped_count,
            last_proc_ms,
            background_value,
            black_value,
            white_value,
            heartbeat_proc,
            preview_mode_value
        )
    )

    jpeg = mp.Process(
        target=jpeg_process,
        name="astrocam-jpeg",
        args=(
            shared_preview,
            preview_lock,
            preview_sequence,
            shared_jpeg,
            jpeg_lock,
            jpeg_size,
            jpeg_sequence,
            stop_event,
            jpeg_count,
            last_jpeg_ms,
            heartbeat_jpeg
        )
    )

    solver = mp.Process(
        target=plate_solver_process,
        name="astrocam-plate-solver",
        args=(
            shared_raw,
            raw_lock,
            raw_sequence,
            stop_event,
            solve_request,
            solve_status,
            solve_available,
            solve_source_seq,
            solve_ra,
            solve_dec,
            solve_scale,
            solve_rotation,
            solve_elapsed,
            solve_message,
            solve_jacobian,
            solve_context,
            polar_valid,
            polar_ra_values,
            polar_dec_values,
            polar_jacobian,
            polar_state,
            polar_message
        )
    )

    web = mp.Process(
        target=web_process,
        name="astrocam-web",
        args=(
            shared_jpeg,
            jpeg_lock,
            jpeg_size,
            jpeg_sequence,
            stop_event,
            acq_count,
            proc_count,
            jpeg_count,
            skipped_count,
            last_acq_ms,
            last_proc_ms,
            last_jpeg_ms,
            background_value,
            black_value,
            white_value,
            exposure_ms_value,
            gain_value,
            binning_mode_value,
            heartbeat_acq,
            heartbeat_proc,
            heartbeat_jpeg,
            indi_connected,
            solve_request,
            solve_status,
            solve_available,
            solve_source_seq,
            solve_ra,
            solve_dec,
            solve_scale,
            solve_rotation,
            solve_elapsed,
            solve_message,
            solve_jacobian,
            preview_mode_value,
            solve_context,
            polar_valid,
            polar_ra_values,
            polar_dec_values,
            polar_jacobian,
            polar_state,
            polar_error_arcmin,
            polar_axis_ra,
            polar_axis_dec,
            polar_shift_x,
            polar_shift_y,
            polar_ref_x,
            polar_ref_y,
            polar_target_x,
            polar_target_y,
            polar_guidance_active,
            polar_message,
            target_active,
            target_ra,
            target_dec,
            target_id,
            target_name
        )
    )

    processes = [acq, proc, jpeg, solver, web]

    for p in processes:
        p.start()

    print(
        f"\n[MAIN pid={os.getpid()}] "
        f"ACQ={acq.pid} PROC={proc.pid} JPEG={jpeg.pid} "
        f"SOLVE={solver.pid} WEB={web.pid}"
    )

    print(f"\nOuvre : http://192.168.1.53:{WEB_PORT}/")
    print("Ctrl+C pour arrêter proprement.\n")

    def request_stop(signum=None, frame=None):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stop_event.is_set():
            if (
                indi_started_here
                and indi_process is not None
                and indi_process.poll() is not None
            ):
                print(
                    f"[MAIN] indiserver s'est arrêté "
                    f"(code {indi_process.returncode}).",
                    flush=True
                )
                stop_event.set()
                break

            for p in processes:
                if not p.is_alive():
                    print(f"[MAIN] {p.name} s'est arrêté.", flush=True)
                    stop_event.set()
                    break

            time.sleep(0.25)

    finally:
        stop_event.set()

        for p in processes:
            p.join(timeout=5)

        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # On ne coupe que le serveur INDI que CETTE instance
        # d'Astro Cam a elle-même démarré.
        if indi_started_here:
            stop_managed_indiserver(indi_process)

    print("\n" + "=" * 82)
    print("ARRÊT ASTRO CAM")
    print("=" * 82)
    print(f"Frames RAW      : {acq_count.value}")
    print(f"Frames traitées : {proc_count.value}")
    print(f"Frames JPEG     : {jpeg_count.value}")
    print(f"Frames sautées  : {skipped_count.value}")
    print(f"RAW seq         : {raw_sequence.value}")
    print(f"PREVIEW seq     : {preview_sequence.value}")
    print(f"JPEG seq        : {jpeg_sequence.value}")
    print("=" * 82)

    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    raise SystemExit(main())
