#!/usr/bin/env bash
# Crab Rescue 3D: one command from a fresh checkout to a finished run.
#
#   ./run_crab_rescue.sh                 real Nebius/NVIDIA models + video   (asks for the key if it is not set)
#   ./run_crab_rescue.sh --no-video      same, without the slow video render
#   ./run_crab_rescue.sh --offline       no network and no key; stand-in decisions, clearly labelled
#   ./run_crab_rescue.sh --seed=7        another room layout
#
# The key is read from NEBIUS_API_KEY only. It is passed to Python through the environment, never through
# command-line arguments, and is never printed or written to a file.
set -euo pipefail
cd "$(dirname "$0")"

MODE=live; VIDEO=1; SEED=5; MAXUSD=0.05; MAXVLM=4
for a in "$@"; do
  case "$a" in
    --offline) MODE=offline ;;
    --no-video) VIDEO=0 ;;
    --seed=*) SEED="${a#*=}" ;;
    --max-usd=*) MAXUSD="${a#*=}" ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $a (see --help)"; exit 2 ;;
  esac
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# ---- 1. Python >= 3.10
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10+ is required. macOS: brew install python"; exit 1; }

# ---- 2. small private environment (does not touch your .venv or system Python)
VENV=.venv_rescue
DEPS="pillow numpy pydantic pyyaml playwright imageio-ffmpeg"
if [ ! -f "$VENV/.ready" ] || [ "$(cat "$VENV/.ready")" != "$DEPS" ]; then
  say "Installing dependencies (once, ~1-2 minutes)"
  "$PY" -m venv "$VENV"
  "$VENV/bin/python" -m pip install -q --upgrade pip
  # shellcheck disable=SC2086
  "$VENV/bin/python" -m pip install -q $DEPS
  echo "$DEPS" > "$VENV/.ready"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

if [ ! -f "$VENV/.chromium" ]; then
  say "Downloading headless Chromium to render the scene (once, ~150 MB)"
  python -m playwright install chromium
  touch "$VENV/.chromium"
fi

# ffmpeg is needed only for the video; reuse the system one, else the one bundled with imageio-ffmpeg
if [ "$VIDEO" = 1 ] && ! command -v ffmpeg >/dev/null 2>&1; then
  ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$VENV/bin/ffmpeg"
fi

# ---- 3. key (live mode only)
if [ "$MODE" = live ]; then
  if [ -z "${NEBIUS_API_KEY:-}" ]; then
    for f in ./nebius_api_key.txt ../nebius_api_key.txt; do
      if [ -f "$f" ]; then
        NEBIUS_API_KEY="$(tr -d '[:space:]' < "$f")"
        echo "Key taken from file $f (the value is not shown)."
        break
      fi
    done
  fi
  if [ -z "${NEBIUS_API_KEY:-}" ]; then
    printf 'Paste NEBIUS_API_KEY (input hidden) and press Enter: '
    read -rs NEBIUS_API_KEY; echo
    NEBIUS_API_KEY="$(printf '%s' "$NEBIUS_API_KEY" | tr -d '[:space:]')"
  fi
  [ -n "$NEBIUS_API_KEY" ] || { echo "No key set. To run without a key: ./run_crab_rescue.sh --offline"; exit 1; }
  export NEBIUS_API_KEY
fi

# ---- 4. run
ARGS=(--seed "$SEED" --out crab_rescue/runs3d --max-usd "$MAXUSD" --max-vlm-calls "$MAXVLM" --vlm-period 25)
[ "$MODE" = live ] && ARGS+=(--live)
[ "$VIDEO" = 1 ] && ARGS+=(--video)

if [ "$MODE" = live ]; then
  say "Live run: Nebius Token Factory, NVIDIA models (cap \$$MAXUSD, up to $MAXVLM vision calls)"
else
  say "OFFLINE run: a local stand-in makes the decisions, NOT an NVIDIA neural network"
fi
[ "$VIDEO" = 1 ] && echo "The video renders frame by frame and can take 10-40 minutes. Without it: --no-video"

set +e
python -m crab_rescue.mission3d "${ARGS[@]}"
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo
  echo "Run stopped (code $RC). The reason is above. Nothing is reported as a success."
  [ "$MODE" = live ] && echo "Check the scene without network: ./run_crab_rescue.sh --offline"
  exit $RC
fi

# ---- 5. show the result
RUN="$(ls -dt crab_rescue/runs3d/*/ | head -1)"; RUN="${RUN%/}"
say "Done"
echo "Run folder    : $RUN"
echo "Call log      : $RUN/ledger.jsonl   (input, raw reply, parse, validation, time, errors)"
echo "Mission (JSON): $RUN/mission.json"
[ -f "$RUN/mission.mp4" ] && echo "Video         : $RUN/mission.mp4"

PORT=8765
echo "Demo          : http://127.0.0.1:$PORT/index.html   (Ctrl+C stops the server)"
OPEN=""; command -v open >/dev/null 2>&1 && OPEN=open; [ -z "$OPEN" ] && command -v xdg-open >/dev/null 2>&1 && OPEN=xdg-open
[ -n "$OPEN" ] && { [ -f "$RUN/mission.mp4" ] && "$OPEN" "$RUN/mission.mp4" || true; ( sleep 1; "$OPEN" "http://127.0.0.1:$PORT/index.html" ) & }
python -m http.server "$PORT" --bind 127.0.0.1 --directory "$RUN" >/dev/null 2>&1
