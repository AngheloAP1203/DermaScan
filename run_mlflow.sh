#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-5001}"
HOST_ADDRESS="${HOST_ADDRESS:-127.0.0.1}"
BACKEND_STORE_URI="${BACKEND_STORE_URI:-sqlite:///mlflow.db}"
ARTIFACTS_DESTINATION="${ARTIFACTS_DESTINATION:-./mlartifacts}"

if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python"
fi

mkdir -p "$ARTIFACTS_DESTINATION"

echo "[DermaScan] MLflow Tracking Server"
echo "  Python:        $PYTHON"
echo "  URL:           http://$HOST_ADDRESS:$PORT"
echo "  Backend store: $BACKEND_STORE_URI"
echo "  Artifacts:     $ARTIFACTS_DESTINATION"
echo
echo "En otra terminal usa:"
echo "  export MLFLOW_TRACKING_URI=\"http://$HOST_ADDRESS:$PORT\""
echo

"$PYTHON" -m mlflow server \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --serve-artifacts \
  --artifacts-destination "$ARTIFACTS_DESTINATION" \
  --host "$HOST_ADDRESS" \
  --port "$PORT"
