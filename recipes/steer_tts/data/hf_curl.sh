#!/bin/bash
# Parallel authenticated, resumable HF downloads; loops until every file in the list exists.
# Usage: hf_curl.sh <listfile>   (lines: repo<TAB>path<TAB>dest, dest relative to $STEER_TTS_ROOT/data).
# Token is read from $STEER_TTS_ROOT/data/.hf_curlrc (one line: header = "Authorization: Bearer hf_..."), never argv.
LIST="$(realpath "$1")"
cd "${STEER_TTS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/data"
dl() {
  IFS=$'\t' read -r repo path dest <<< "$1"
  [ -s "$dest" ] && exit 0
  mkdir -p "$(dirname "$dest")"
  curl -sfL --retry 3 --retry-delay 10 -C - -K .hf_curlrc "https://huggingface.co/datasets/$repo/resolve/main/$path" -o "$dest.part" && mv "$dest.part" "$dest" || echo "RETRY $path"
}
export -f dl
for round in $(seq 1 50); do
  missing=$(while IFS=$'\t' read -r r p d; do [ -s "$d" ] || echo "$r	$p	$d"; done < "$LIST")
  [ -z "$missing" ] && break
  echo "round $round: $(echo "$missing" | wc -l) files left"
  echo "$missing" | xargs -P 4 -I{} bash -c 'dl "$@"' _ {}
  sleep 20
done
echo "ALL DONE $LIST"
