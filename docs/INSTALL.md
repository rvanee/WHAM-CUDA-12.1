# Installation

This CUDA 12.1 version of WHAM has been tested on Ubuntu 22.04 with python = 3.10. We suggest using an [anaconda](https://www.anaconda.com/) environment to run WHAM as below.

```bash
# Clone the repo
git clone https://github.com/rvanee/WHAM-CUDA-12.1.git --recursive
cd WHAM/

# Create Conda environment and install dependencies
conda env create -f environment.yml
conda activate wham

# Install DPVO
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

pip install .
```
