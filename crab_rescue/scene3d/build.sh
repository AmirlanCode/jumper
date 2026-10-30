#!/usr/bin/env bash
# Rebuild dist/app.js and dist/driver.js from src/ (needs node; `npm install` once).
set -euo pipefail
cd "$(dirname "$0")"
npx esbuild src/main.js --bundle --minify --format=esm --outfile=dist/app.js
npx esbuild src/driver.js --bundle --minify --format=esm --outfile=dist/driver.js
