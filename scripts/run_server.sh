#!/bin/bash

echo "server start"

# Set external API endpoints (override default internal IPs)
export TFOLD_API_BASE_URL="http://43.142.171.112:11280/tFold"
export IGGM_API_BASE_URL="http://43.142.171.112:11280/IgGM"
# BOLTZ2_API_BASE_URL keeps default: http://172.16.20.44:17827/Boltz2

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