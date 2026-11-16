#!/usr/bin/env bash
# Regenerates contracts/openapi.json from the backend code. Run after changing any endpoint or schema.
set -euo pipefail
cd "$(dirname "$0")/../backend"
python -c "import json; from app.main import app; json.dump(app.openapi(), open('../contracts/openapi.json', 'w'), indent=2)"
echo "Updated contracts/openapi.json"
