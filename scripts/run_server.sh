#!/bin/bash

echo "server start"

# Use empty GPUs (GPU 0,1 have zombie processes)
export CUDA_VISIBLE_DEVICES=2,3

# Activate the Conda environment
source /opt/conda/bin/activate OpenBioMed

python -m uvicorn open_biomed.scripts.run_server:app \
    --host 0.0.0.0 \
    --port 32520 \
    --log-level info > ./tmp/server_test.log 2>&1 &

python -m uvicorn open_biomed.scripts.run_server_workflow:app \
    --host 0.0.0.0 \
    --port 32521 \
    --log-level info > ./tmp/workflow_test.log 2>&1 &

tail -f /dev/null