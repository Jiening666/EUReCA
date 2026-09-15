#!/usr/bin/env bash
# General-purpose file downloader for clusters.
# Reads a list of URLs and downloads each into <dest>. Resumable: safe to re-run
# after an interruption or job timeout — completed files are skipped, partial
# ones resume.
#
# Usage:
#   bash download_urls.sh <dest_dir> <url_list_file>
#   bash download_urls.sh <dest_dir> <url1> [url2] [url3] ...
#
# URL list file format (one entry per line):
#   https://host/path/file.zip              # name taken from the URL
#   myname.zip   https://host/path?x=1      # explicit output name (TAB/space sep)
#   # lines starting with # and blank lines are ignored
#
# Requires one of: aria2c (preferred, parallel+resumable), wget, or curl.
#
# Parallelism:
#   JOBS  = how many files to download concurrently   (default 3)
#   CONN  = connections aria2c opens per file          (default 4)
#   e.g.  JOBS=4 CONN=8 bash download_urls.sh dest urls.txt
#   Total open connections ~= JOBS*CONN; keep it modest so the host doesn't
#   rate-limit you (Zenodo does). JOBS=1 restores fully sequential behaviour.

set -euo pipefail

DEST="${1:?Usage: $0 <dest_dir> <url_list_file | url...>}"
shift
[ "$#" -ge 1 ] || { echo "Usage: $0 <dest_dir> <url_list_file | url...>" >&2; exit 1; }

JOBS="${JOBS:-3}"
CONN="${CONN:-4}"

mkdir -p "$DEST"
LOG="$DEST/download.log"
FAILMARK="$DEST/.dl_failures.$$"; : >"$FAILMARK"
echo "==== download started $(date) -> $DEST (JOBS=$JOBS CONN=$CONN) ====" | tee -a "$LOG"

# ---- resumable HTTP fetch --------------------------------------------------
http_get() {  # http_get <url> <out_path>
    local url="$1" out="$2"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -c -x "$CONN" -s "$CONN" --auto-file-renaming=false --allow-overwrite=true \
               -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -fL -C - -o "$out" "$url"
    else
        echo "No aria2c/wget/curl found" >&2; return 1
    fi
}

# ---- download one "<name> <url>" or "<url>" entry --------------------------
fetch_entry() {
    local name url out
    name="$(echo "$1" | awk '{print $1}')"
    url="$(echo "$1"  | awk '{print $2}')"
    if [ -z "$url" ]; then            # only a URL given
        url="$name"
        name="$(basename "${url%%\?*}")"
    fi
    [ -n "$name" ] || name="download_$RANDOM"
    out="$DEST/$name"

    echo "---- $name" | tee -a "$LOG"
    case "$url" in
        http://*|https://*|ftp://*) : ;;
        *) echo "  skip: not a url '$url'" | tee -a "$LOG"; return 1 ;;
    esac

    if http_get "$url" "$out" >>"$LOG" 2>&1; then
        echo "  ok -> $out" | tee -a "$LOG"; return 0
    else
        echo "  FAILED ($url) — see $LOG" | tee -a "$LOG"
        echo "$url" >>"$FAILMARK"; return 1
    fi
}

# ---- worker pool: run up to JOBS fetch_entry calls concurrently ------------
schedule() {  # schedule <entry>
    if [ "$JOBS" -le 1 ]; then
        fetch_entry "$1"                                     # sequential
        return
    fi
    # block until a slot frees up, then launch in background
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do
        wait -n 2>/dev/null || wait
    done
    fetch_entry "$1" &
}

# ---- dispatch: a single file arg = url list; otherwise treat args as URLs --
if [ "$#" -eq 1 ] && [ -f "$1" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"                                   # strip comments
        line="$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        [ -z "$line" ] && continue
        schedule "$line"
    done < "$1"
else
    for u in "$@"; do
        schedule "$u"
    done
fi

wait                                                          # let backgrounded jobs finish

fail=0
[ -s "$FAILMARK" ] && fail=1
echo "==== finished $(date) (fail=$fail) ====" | tee -a "$LOG"
rm -f "$FAILMARK"

# optional sha256 verification: place "<sha256>  <filename>" lines in DEST/SHA256SUMS
if [ -f "$DEST/SHA256SUMS" ]; then
    echo "Verifying sha256..." | tee -a "$LOG"
    ( cd "$DEST" && sha256sum -c SHA256SUMS ) | tee -a "$LOG" || fail=1
fi

echo; echo "Files in $DEST:"; ls -lh "$DEST"
exit $fail
