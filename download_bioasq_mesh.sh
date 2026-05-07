#!/usr/bin/env bash
# Log in to BioASQ participants area (Django CSRF + Referer) and download Task 10a allMeSH JSON.
#
# Usage:
#   export BIOASQ_USERNAME='your_user'
#   export BIOASQ_PASSWORD='your_password'
#   bash download_bioasq_mesh.sh
#
# Optional:
#   BIOASQ_OUTPUT   output path (default: data/allMeSH_2022.json)
#   BIOASQ_COOKIE   cookie jar path (default: data/.bioasq_cookies.txt)

set -euo pipefail

BASE_URL="${BIOASQ_BASE_URL:-https://participants-area.bioasq.org}"
LOGIN_URL="${BASE_URL}/accounts/login/"
DOWNLOAD_URL="${BASE_URL}/Tasks/10a/trainingDataset/raw/allMeSH/"
OUT="${BIOASQ_OUTPUT:-data/allMeSH_2022.json}"
COOKIE_JAR="${BIOASQ_COOKIE:-data/.bioasq_cookies.txt}"

if [[ -z "${BIOASQ_USERNAME:-}" || -z "${BIOASQ_PASSWORD:-}" ]]; then
  echo "Set BIOASQ_USERNAME and BIOASQ_PASSWORD (non-interactive login)." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
mkdir -p "$(dirname "$COOKIE_JAR")"

LOGIN_HTML="$(mktemp)"
trap 'rm -f "$LOGIN_HTML"' EXIT

echo "Fetching login page..."
curl -sS -c "$COOKIE_JAR" -b "$COOKIE_JAR" "$LOGIN_URL" -o "$LOGIN_HTML"

CSRF="$(sed -n 's/.*name="csrfmiddlewaretoken" value="\([^"]*\)".*/\1/p' "$LOGIN_HTML" | head -1)"
if [[ -z "$CSRF" ]]; then
  echo "Could not find csrfmiddlewaretoken in login HTML. Check BIOASQ_BASE_URL or site changes." >&2
  exit 1
fi

echo "Logging in..."
curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -e "$LOGIN_URL" \
  -H "Referer: ${LOGIN_URL}" \
  --data-urlencode "username=${BIOASQ_USERNAME}" \
  --data-urlencode "password=${BIOASQ_PASSWORD}" \
  --data-urlencode "csrfmiddlewaretoken=${CSRF}" \
  "$LOGIN_URL" -o /dev/null

echo "Downloading allMeSH dataset..."
curl -sS -L -c "$COOKIE_JAR" -b "$COOKIE_JAR" \
  -e "${BASE_URL}/" \
  -H "Referer: ${BASE_URL}/" \
  "$DOWNLOAD_URL" \
  -o "$OUT"

SAMPLE="$(head -c 500 "$OUT")"
if [[ "$SAMPLE" == *"<!DOCTYPE"* ]] || [[ "$SAMPLE" == *"<html"* ]]; then
  echo "Download looks like HTML, not JSON (wrong credentials, CSRF, or URL)." >&2
  echo "First bytes: ${SAMPLE//$'\n'/ }" >&2
  exit 1
fi

echo "Wrote: $OUT"
