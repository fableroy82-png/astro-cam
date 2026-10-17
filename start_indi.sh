#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${INDI_PORT:-7624}"
DRIVER="${INDI_DRIVER:-indi_svbonycam_ccd}"

echo "Démarrage INDI pour la SC715C"
echo "Driver : $DRIVER"
echo "Port   : $PORT"
echo "Arrêt  : Ctrl+C"
echo

if ! command -v "$DRIVER" >/dev/null 2>&1; then
    echo "ERREUR : driver INDI introuvable : $DRIVER"
    echo "Le driver SC715C doit être installé sur le système."
    exit 1
fi

exec indiserver -v -r 5 -p "$PORT" "$DRIVER"
