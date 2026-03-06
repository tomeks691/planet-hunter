#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn planet_hunter.main:app --host 127.0.0.1 --port 8420 --reload
