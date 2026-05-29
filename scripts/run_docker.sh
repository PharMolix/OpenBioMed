#!/bin/bash

# docker build
# docker build -t openbiomed .

# docker run
docker stop openbiomed_container_junyu && docker rm openbiomed_container_junyu
docker run -it -d --gpus all \
    -p 8095:8095 -p 8094:8094 \
    -v /home/junyu/projects/OpenBioMed/OpenBioMed:/app \
    -v /share-vepfs/yk/projects/OpenBioMed/checkpoints:/app/checkpoints \
    --name openbiomed_container_junyu 95bb6d82611