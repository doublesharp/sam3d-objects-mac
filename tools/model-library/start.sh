#!/bin/sh
set -eu
SAM3D_VIEWER_ROOT="$(CDPATH= cd "$(dirname "$0")/../.." && pwd)"
SAM3D_VIEWER_PORT="${SAM3D_VIEWER_PORT:-8766}"
echo "Open http://127.0.0.1:$SAM3D_VIEWER_PORT/tools/model-library/index.html"
exec python3 -m http.server "$SAM3D_VIEWER_PORT" --bind 127.0.0.1 --directory "$SAM3D_VIEWER_ROOT"
