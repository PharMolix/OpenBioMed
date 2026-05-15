# ------------------------------
# Base image: CUDA 11.7 + Ubuntu 20.04
# ------------------------------
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# ------------------------------
# Install system dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    libxrender1 \
    wget \
    g++ \
    cmake \
    zip \
    curl \
    ca-certificates \
    git \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Install Miniconda
# ------------------------------
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# ------------------------------
# Accept TOS and create conda environment
# ------------------------------
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda create -n OpenBioMed python=3.9 -y \
    && /opt/conda/bin/conda clean -afy

# ------------------------------
# Install PyTorch 1.13.1 + CUDA 11.7 in the environment
# ------------------------------
RUN /opt/conda/bin/conda install -n OpenBioMed \
    pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cudatoolkit=11.7 -c pytorch -c nvidia \
    && /opt/conda/bin/conda clean -afy

# ------------------------------
# Install PyG (torch_scatter, etc.) using environment pip
# ------------------------------
RUN /opt/conda/envs/OpenBioMed/bin/pip install --no-build-isolation \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# ------------------------------
# Install other Python packages
# ------------------------------
COPY requirements.txt . 
RUN /opt/conda/envs/OpenBioMed/bin/pip install \
    pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps -i https://mirrors.aliyun.com/pypi/simple \
    && /opt/conda/envs/OpenBioMed/bin/pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# ------------------------------
# Visualization & NLTK
# ------------------------------
RUN /opt/conda/bin/conda install -n OpenBioMed -c conda-forge pymol-open-source -y \
    && /opt/conda/envs/OpenBioMed/bin/pip install imageio rouge_score nltk alibabacloud_iqs20241111 alibabacloud_tea_openapi -i https://mirrors.aliyun.com/pypi/simple \
    && /opt/conda/envs/OpenBioMed/bin/python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# ------------------------------
# AutoDock Vina tools
# ------------------------------
RUN /opt/conda/envs/OpenBioMed/bin/pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2 \
    && /opt/conda/envs/OpenBioMed/bin/pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

# ------------------------------
# Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Activate conda environment by default
# ------------------------------
RUN echo "source /opt/conda/bin/activate OpenBioMed" >> ~/.bashrc

# ------------------------------
# Default entrypoint
# ------------------------------
ENTRYPOINT ["./scripts/run_server.sh"]