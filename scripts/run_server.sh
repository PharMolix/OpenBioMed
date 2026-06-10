#!/bin/bash

echo "server start"

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate OpenBioMed

python -m uvicorn open_biomed.scripts.run_server:app \
    --host 0.0.0.0 \
    --port 8095 \
    --log-level info > ./tmp/server.log 2>&1 &

python -m uvicorn open_biomed.scripts.run_server_workflow:app \
    --host 0.0.0.0 \
    --port 8094 \
    --log-level info > ./tmp/workflow.log 2>&1 &

tail -f /dev/null