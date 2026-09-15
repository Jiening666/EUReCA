#!/usr/bin/env bash
# Download the MICCAI 2022 MELA dataset (https://mela.grand-challenge.org/Dataset/)
# to a cluster. Resumable: safe to re-run after interruption or job timeout.
#
# The MELA data is public on Zenodo (no login). This script reads a list of direct
# HTTPS URLs from mela_urls.txt and fetches each one, resuming partial files and
# skipping anything already complete.
#
# Usage:
#   bash download_mela.sh <destination_dir> [url_list_file]
#   default url_list_file: ./mela_urls.txt  (next to this script)
#
# Requirements (any one resumable HTTP tool):
#   aria2c   (preferred, parallel + resumable)   OR   wget   OR   curl
#
# Unzipping is left to you (see hint printed at the end) so the download job
# stays I/O-light and you control where volumes land.

set -euo pipefail

DEST="${1:?Usage: $0 <destination_dir> [url_list_file]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL_FILE="${2:-$SCRIPT_DIR/mela_urls.txt}"

[ -f "$URL_FILE" ] || { echo "URL list not found: $URL_FILE" >&2; exit 1; }
mkdir -p "$DEST"
LOG="$DEST/download.log"
echo "==== MELA download started $(date) ====" | tee -a "$LOG"
echo "dest=$DEST  urls=$URL_FILE" | tee -a "$LOG"

# ---- pick an HTTP downloader (resumable) -----------------------------------
http_get() {  # http_get <url> <out_path>
    local url="$1" out="$2"
    if command -v aria2c >/dev/null 2>&1; then
        # -c resume, -x/-s multi-connection, --auto-file-renaming=false keeps name
        aria2c -c -x 8 -s 8 --auto-file-renaming=false --allow-overwrite=true \
               -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - -o "$out" "$url"
    else
        echo "No aria2c/wget/curl found" >&2; return 1
    fi
}

# ---- main loop -------------------------------------------------------------
fail=0
while IFS= read -r line || [ -n "$line" ]; do
    # strip comments / blanks
    line="${line%%#*}"
    line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "$line" ] && continue

    # split into <name> <url> (or just <url>)
    name="$(echo "$line" | awk '{print $1}')"
    url="$(echo "$line"  | awk '{print $2}')"
    if [ -z "$url" ]; then
        url="$name"
        name="$(basename "${url%%\?*}")"
    fi
    out="$DEST/$name"

    echo "---- $name" | tee -a "$LOG"
    case "$url" in
        http://*|https://*) : ;;
        *) echo "  skip: not an http(s) url '$url'" | tee -a "$LOG"; fail=1; continue ;;
    esac

    if http_get "$url" "$out" >>"$LOG" 2>&1; then
        echo "  ok -> $out" | tee -a "$LOG"
    else
        echo "  FAILED ($url) — see $LOG" | tee -a "$LOG"
        fail=1
    fi
done < "$URL_FILE"

echo "==== finished $(date) (fail=$fail) ====" | tee -a "$LOG"

# optional checksum verification if a sha256 manifest sits next to the urls
if [ -f "$URL_FILE.sha256" ]; then
    echo "Verifying sha256..." | tee -a "$LOG"
    ( cd "$DEST" && sha256sum -c "$URL_FILE.sha256" ) | tee -a "$LOG" || fail=1
fi

echo
echo "Files in $DEST:"; ls -lh "$DEST"
echo
echo "To unzip archives afterwards (parallel, 4 at a time):"
echo "  ls $DEST/*.zip | xargs -P 4 -I{} unzip -n {} -d $DEST/"

exit $fail
