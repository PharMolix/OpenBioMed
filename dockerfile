# Use the Ubuntu 20.04 image of NVIDIA CUDA 11.7.1 and cuDNN as the base image
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    libxrender1 \
    wget \
    g++ \
    cmake \
    zip \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# isntall Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /root/miniconda3 \
    && rm /tmp/miniconda.sh

# init Miniconda and create conda env
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda init bash \
    && . /root/.bashrc \
    && conda create -n OpenBioMed python=3.9 -y \
    && conda activate OpenBioMed \
    && pip install --upgrade pip setuptools

# Installing PyTorch and torchvision
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 \
    && pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
    && pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install additional packages from requirements.txt
RUN pip install -r requirements.txt

# Install visualization tools
RUN conda install -c conda-forge pymol-open-source -y \
    && pip install imageio

# Install AutoDockVina tools
RUN git config --global http.proxy http://100.68.173.241:3128 \
    && git config --global https.proxy http://100.68.173.241:3128 \
    && pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 \
    && pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# Install NLTK
RUN pip install spacy rouge_score nltk \
    && python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Set working directory
WORKDIR /app

# Activate the OpenBioMed environment by default
RUN echo "source activate OpenBioMed" >> ~/.bashrc

# Set default command
ENTRYPOINT ["./scripts/run_server.sh"]