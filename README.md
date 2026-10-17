# Astro cam — Arduino UNO Q / SC715C

Interface web d'acquisition astronomique pour une caméra SVBONY SC715C et un Arduino UNO Q sous Linux : aperçu en direct, exposition/gain, plate solving Astrometry.net, mise en station polaire et aide au cadrage push-to avec catalogue OpenNGC.

## Version publiée

Copie du programme actif sur l'UNO Q, récupérée le 17 octobre 2026. `astro_cam.py` est identique à la v37 (`astro_cam_v37_sky_overlay_refined_push_to.py`). La v38 de test du Bridge n'est pas intégrée. La sélection repose sur le programme installé ; une nouvelle validation sous le ciel n'a pas été effectuée lors de la préparation du dépôt.

Aucun dépôt Git Astro cam n'a été retrouvé dans le répertoire actuel ou l'archive historique inspectée. Les tags `focus-stable-v1` et `astro-system-stable-v1` ne sont donc pas disponibles et ne sont pas recréés artificiellement.

## Fichiers

- `astro_cam.py` : application principale et interface web embarquée dans le fichier Python.
- `focus_plein_jour_sc715c_v2.py` : outil séparé de mise au point de jour.
- `start_indi.sh` : lancement manuel facultatif du serveur INDI.
- `firmware/st4_bridge/st4_bridge.ino` : firmware ST4 conservé depuis l'installation.
- `catalogs/` : catalogues OpenNGC locaux pour utilisation hors ligne et attribution.
- `requirements.txt` : dépendances Python ; `requirements-unooq.txt` : versions relevées sur la machine.

## Installation

Cible : Linux sur UNO Q. Le programme principal utilise le mode multiprocessing `fork`, et n'est pas prévu pour être lancé directement sous Windows.

Installer Python 3, NumPy, Flask, OpenCV, Astropy, Astrometry.net, INDI et le pilote `indi_svbonycam_ccd`. Sur Debian, les dépendances Python et le solveur peuvent être installés avec :

```bash
sudo apt update
sudo apt install python3 python3-venv python3-numpy python3-flask python3-opencv python3-astropy astrometry.net
```

Le serveur INDI et le pilote caméra sont des prérequis distincts : vérifier `command -v indiserver indi_svbonycam_ccd solve-field`. Sur l'UNO Q inspecté, les trois commandes sont dans `/usr/bin` ; le checkout conservé de `indi-3rdparty` est propre et correspond au tag `v2.2.4`. Les bibliothèques et pilotes compilés ne sont pas inclus dans ce dépôt.

Pour une installation Python isolée, `python3 -m venv .venv`, puis `.venv/bin/python -m pip install -r requirements.txt`. La disponibilité des paquets dépend de la plateforme ; la configuration Debian observée est décrite dans `requirements-unooq.txt`.

Installer également les index Astrometry.net adaptés au champ de la caméra. Ils ne sont pas inclus dans le dépôt. Vérifier les règles udev et les permissions USB du pilote.

## Lancement

```bash
python3 astro_cam.py
```

Ouvrir `http://ADRESSE_IP_UNO_Q:5050`. L'application démarre INDI si le port 7624 n'écoute pas déjà, puis arrête uniquement le serveur qu'elle a démarré. Arrêt avec Ctrl+C.

Pour la mise au point de jour, arrêter l'application principale puis lancer :

```bash
python3 focus_plein_jour_sc715c_v2.py
```

Ouvrir `http://ADRESSE_IP_UNO_Q:5051`. Ne pas lancer les deux applications simultanément sur la même caméra.

## Configuration actuelle

Les paramètres sont des constantes au début des scripts, pas un fichier `config.toml` : caméra 1920 × 1080, binning 2, focale 184 mm, pixels 1,45 µm, exposition initiale 800 ms, gain 180. INDI est contacté sur `127.0.0.1:7624`. Les serveurs web écoutent sur toutes les interfaces du réseau local, sans authentification : ne pas les exposer directement sur Internet.

Les CSV OpenNGC sont fournis pour le fonctionnement hors ligne. Le code possède également une récupération en ligne si nécessaire. Les archives anciennes ne sont pas nécessaires lorsque les CSV locaux sont présents.

## Matériel et limites

Le firmware ST4 fourni définit N=D2, S=D3, E=D4, W=D5, sorties actives à HIGH, impulsion maximale 1500 ms. Il est conservé comme source matérielle, sans prétendre que l'autoguidage ST4 est intégré ou validé dans cette application v37. Vérifier le câblage réel : ce brochage diffère des évolutions D3–D6 envisagées. Utiliser quatre optocoupleurs et vérifier les sorties au repos avant connexion à l'EQ6.

Le test Bridge v38, le commutateur hotspot, le bouton LED et le pilotage Bulb du Canon EOS 600D restent hors de cette publication.

## Vérification de syntaxe

```bash
python3 -m py_compile astro_cam.py focus_plein_jour_sc715c_v2.py
bash -n start_indi.sh
```

Ces vérifications ne remplacent pas un essai avec caméra, monture et ciel réel. Voir `PROVENANCE.md` pour le périmètre de préparation.

## Droits

Les catalogues OpenNGC sont sous CC BY-SA 4.0 ; voir `catalogs/README.md`. Aucune nouvelle licence n'est attribuée au code du projet par cette publication.
