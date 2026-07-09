#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-3000}"
API_INTERNAL_URL="${FASTAPI_BASE_URL:-http://127.0.0.1:${API_PORT}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/.run}"
API_LOG="${LOG_DIR}/bigbook-api.log"
WEB_LOG="${LOG_DIR}/bigbook-web.log"

API_PID=""
WEB_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  echo
  echo "Stopping BigBook app..."
  if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
    kill "${WEB_PID}" 2>/dev/null || true
  fi
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  wait "${WEB_PID}" 2>/dev/null || true
  wait "${API_PID}" 2>/dev/null || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    exit 1
  fi
}

check_port_free() {
  local port="$1"
  "$ROOT_DIR/env/bin/python" - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        print(f"Port {port} is already in use.", file=sys.stderr)
        sys.exit(1)
PY
}

wait_for_api() {
  "$ROOT_DIR/env/bin/python" - "$API_INTERNAL_URL/health" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request

url = sys.argv[1]
deadline = time.time() + 180
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
        if body.get("ready") is True:
            print(f"API ready: {body.get('books', 0):,} books, {body.get('users', 0):,} users")
            sys.exit(0)
        last_error = body.get("error") or "API is not ready yet"
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        last_error = str(exc)
    time.sleep(2)

print(f"API did not become ready in time: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

external_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

require_command npm

if [[ ! -x "$ROOT_DIR/env/bin/python" ]]; then
  echo "Missing Python virtualenv at env/bin/python"
  exit 1
fi

"$ROOT_DIR/env/bin/python" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("fastapi", "uvicorn") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("Run: env/bin/python -m pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)
PY

check_port_free "$API_PORT"
check_port_free "$WEB_PORT"

mkdir -p "$LOG_DIR"

if [[ ! -d "$ROOT_DIR/apps/web/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix apps/web install
fi

echo "Starting BigBook API on ${HOST}:${API_PORT}..."
"$ROOT_DIR/env/bin/python" -m uvicorn src.api.main:app \
  --host "$HOST" \
  --port "$API_PORT" \
  >"$API_LOG" 2>&1 &
API_PID=$!

echo "Waiting for recommender artifacts to load..."
if ! wait_for_api; then
  echo "API log:"
  tail -n 80 "$API_LOG" || true
  exit 1
fi

echo "Starting Next.js app on ${HOST}:${WEB_PORT}..."
FASTAPI_BASE_URL="$API_INTERNAL_URL" npm --prefix apps/web run dev -- \
  --hostname "$HOST" \
  --port "$WEB_PORT" \
  >"$WEB_LOG" 2>&1 &
WEB_PID=$!

sleep 3
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  echo "Next.js failed to start. Web log:"
  tail -n 80 "$WEB_LOG" || true
  exit 1
fi

IP_ADDRESS="$(external_ip || true)"

echo
echo "BigBook is running."
echo "Local web:   http://127.0.0.1:${WEB_PORT}"
if [[ -n "$IP_ADDRESS" ]]; then
  echo "LAN web:     http://${IP_ADDRESS}:${WEB_PORT}"
fi
echo "API health:  ${API_INTERNAL_URL}/health"
echo
echo "Logs:"
echo "  API: ${API_LOG}"
echo "  Web: ${WEB_LOG}"
echo
echo "Press Ctrl+C to stop both servers."

wait -n "$API_PID" "$WEB_PID"
