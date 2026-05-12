#!/usr/bin/env bash
# Download full BioASQ Task 10a allMeSH JSON via participant login (CSRF + Referer).
#
# Usage (interactive — prompts for username/password on a TTY):
#   bash download_bioasq_mesh.sh
#
# Usage (non-interactive — required when stdin is not a terminal, e.g. nohup):
#   export BIOASQ_USERNAME='your_user'
#   export BIOASQ_PASSWORD='your_password'
#   bash download_bioasq_mesh.sh
#
# Run detached (SSH-safe); log to file (must pass credentials via env — no TTY):
#   nohup env BIOASQ_USERNAME='…' BIOASQ_PASSWORD='…' ./download_bioasq_mesh.sh >> download_bioasq.log 2>&1 &
#   disown   # optional: drop job from shell so it is not SIGHUP’d on some setups
# Use BIOASQ_QUIET=1 for less noisy logs when not on a TTY.
#
# Optional:
#   BIOASQ_OUTPUT   output path (default: data/allMeSH_2022.json)
#   BIOASQ_COOKIE   cookie jar path (default: data/.bioasq_cookies.txt)
#   BIOASQ_DEBUG    set to 1 to print HTTP status line and snippet of login response on failures
#   BIOASQ_QUIET    set to 1 to silence the dataset download progress bar (default: progress on stderr)

set -euo pipefail

BASE_URL="${BIOASQ_BASE_URL:-https://participants-area.bioasq.org}"
LOGIN_URL="${BASE_URL}/accounts/login/"
DOWNLOAD_URL="${BASE_URL}/Tasks/10a/trainingDataset/raw/allMeSH/"
OUT="${BIOASQ_OUTPUT:-data/allMeSH_2022.json}"
COOKIE_JAR="${BIOASQ_COOKIE:-data/.bioasq_cookies.txt}"

UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

if [[ -z "${BIOASQ_USERNAME:-}" || -z "${BIOASQ_PASSWORD:-}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    read -r -p "BioASQ username: " BIOASQ_USERNAME
    read -r -s -p "BioASQ password: " BIOASQ_PASSWORD
    echo
  else
    echo "Set BIOASQ_USERNAME and BIOASQ_PASSWORD (no TTY for interactive login)." >&2
    exit 1
  fi
fi

if [[ -z "${BIOASQ_USERNAME:-}" || -z "${BIOASQ_PASSWORD:-}" ]]; then
  echo "BioASQ username and password cannot be empty." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$COOKIE_JAR")"

LOGIN_HTML="$(mktemp)"
trap 'rm -f "$LOGIN_HTML"' EXIT

# Netscape jar: columns are domain, subdomain-flag, path, secure, expiry, name, value
csrf_from_cookie_jar() {
  awk -F'\t' 'NF>=7 && ($6=="csrftoken") { print $7; exit }' "$COOKIE_JAR"
}

csrf_from_html() {
  local f="$1" tok
  [[ -s "$f" ]] || return 1

  tok="$(grep -oE 'name="csrfmiddlewaretoken"[^>]*value="[^"]+"' "$f" 2>/dev/null | head -1 | sed -n 's/.*value="\([^"]*\)".*/\1/p')"
  if [[ -n "$tok" ]]; then echo "$tok"; return 0; fi
  tok="$(grep -oE 'value="[^"]*"[^>]*name="csrfmiddlewaretoken"' "$f" 2>/dev/null | head -1 | sed -n 's/.*value="\([^"]*\)".*/\1/p')"
  if [[ -n "$tok" ]]; then echo "$tok"; return 0; fi
  tok="$(grep -oE "name='csrfmiddlewaretoken'[^>]*value='[^']+'" "$f" 2>/dev/null | head -1 | sed -n "s/.*value='\([^']*\)'.*/\1/p")"
  if [[ -n "$tok" ]]; then echo "$tok"; return 0; fi
  tok="$(grep -oE "value='[^']*'[^>]*name='csrfmiddlewaretoken'" "$f" 2>/dev/null | head -1 | sed -n "s/.*value='\([^']*\)'.*/\1/p")"
  if [[ -n "$tok" ]]; then echo "$tok"; return 0; fi

  # Flexible whitespace / minified Django markup
  if command -v python3 >/dev/null 2>&1; then
    tok="$(
      TOKEN_HTML_FILE="$f" python3 - <<'PY'
import os, re

html = open(os.environ["TOKEN_HTML_FILE"], encoding="utf-8", errors="ignore").read()
patterns = [
    r"name=[\"']csrfmiddlewaretoken[\"'][^>]*value=[\"']([^\"']+)",
    r"value=[\"']([^\"']+)[\"'][^>]*name=[\"']csrfmiddlewaretoken[\"']",
]
for pat in patterns:
    m = re.search(pat, html, flags=re.I | re.DOTALL)
    if m:
        print(m.group(1))
        raise SystemExit(0)
raise SystemExit(1)
PY
    )" || true
    if [[ -n "$tok" ]]; then echo "$tok"; return 0; fi
  fi

  return 1
}

echo "Fetching login page..."
LOGIN_CODE="$(curl -sS -o "$LOGIN_HTML" -w '%{http_code}' \
  -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -H "User-Agent: ${UA}" \
  -H "Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.8" \
  "$LOGIN_URL")" || true

if [[ "${BIOASQ_DEBUG:-0}" == "1" ]]; then
  echo "Login page GET HTTP status: ${LOGIN_CODE}" >&2
  echo "Saved body size: $(wc -c <"$LOGIN_HTML") bytes" >&2
fi

CSRF=""
# Django often sets csrftoken on the first GET; it matches the hidden field value.
if csrf="$(csrf_from_cookie_jar)" && [[ -n "$csrf" ]]; then
  CSRF="$csrf"
fi
if [[ -z "$CSRF" ]] && csrf="$(csrf_from_html "$LOGIN_HTML" 2>/dev/null)" && [[ -n "$csrf" ]]; then
  CSRF="$csrf"
fi

if [[ -z "$CSRF" ]]; then
  echo "Could not determine CSRF token (no csrftoken cookie and no csrfmiddlewaretoken in HTML)." >&2
  echo "Hint: BIOASQ_DEBUG=1 rerun. If GET is not 200, fix URL/network; if HTML is tiny, server may block non-browser requests." >&2
  if [[ "${BIOASQ_DEBUG:-0}" == "1" ]]; then
    echo "--- login response (first 2000 chars) ---" >&2
    head -c 2000 "$LOGIN_HTML" >&2 || true
    echo >&2
    echo "--- cookie jar (csrftoken lines) ---" >&2
    grep csrftoken "$COOKIE_JAR" >&2 || true
  fi
  exit 1
fi

echo "Logging in..."
curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -e "$LOGIN_URL" \
  -H "Referer: ${LOGIN_URL}" \
  -H "User-Agent: ${UA}" \
  --data-urlencode "username=${BIOASQ_USERNAME}" \
  --data-urlencode "password=${BIOASQ_PASSWORD}" \
  --data-urlencode "csrfmiddlewaretoken=${CSRF}" \
  "$LOGIN_URL" -o /dev/null

echo "Downloading allMeSH dataset..."
echo "  URL: ${DOWNLOAD_URL}"
echo "  -> ${OUT}"
if [[ "${BIOASQ_QUIET:-0}" == "1" ]]; then
  curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -e "${BASE_URL}/" \
    -H "Referer: ${BASE_URL}/" \
    -H "User-Agent: ${UA}" \
    "$DOWNLOAD_URL" \
    -o "$OUT"
else
  # --progress-bar: single-line transfer meter on stderr (body still goes only to OUT)
  curl --progress-bar -S -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
    -e "${BASE_URL}/" \
    -H "Referer: ${BASE_URL}/" \
    -H "User-Agent: ${UA}" \
    "$DOWNLOAD_URL" \
    -o "$OUT"
  echo >&2
fi

SAMPLE="$(head -c 500 "$OUT")"
if [[ "$SAMPLE" == *"<!DOCTYPE"* ]] || [[ "$SAMPLE" == *"<html"* ]]; then
  echo "Download looks like HTML, not JSON (wrong credentials, CSRF, or URL)." >&2
  echo "First bytes: ${SAMPLE//$'\n'/ }" >&2
  exit 1
fi

echo "Wrote: $OUT"
