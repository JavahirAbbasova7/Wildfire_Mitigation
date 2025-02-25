# Installation

**Note:** *For the time being, a `requirements.txt` file will be used to install the project dependencies.*

1. **Clone the repository.**

```bash
git clone https://github.com/mitrefireline/simharness.git
```

2. **Create a conda environment.**

```bash
conda create --yes --name sh2 python=3.9.*
conda activate sh2
```

3. **Install required packages.**

<!-- cd simharness2/
sudo apt-get update && sudo apt-get install build-essential libgl1 -y -->

```bash
pip install -r requirements.txt
pip install ray[rllib]==2.6.2
```

## Troubleshooting
If you experience any `SSL` errors/warnings, try running the lines below, or append them to `$HOME/.bashrc` (preferred method). Then, rerun the installation commands above.

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
```

<!-- # Building Docker Image(s)

There are different flavors of docker images and only one of them ([simple](#simple)) currently works. The order of this section is by order of how close I think each dockerfile is to producing a working image once built.

## Simple

**File**: [`docker/simple.dockerfile`](docker/simple.dockerfile)

The simplest docker image just has the [`rayproject/ray:2.3.0-py39-gpu`](https://hub.docker.com/r/rayproject/ray) image as the base image and uses the [`requirements.txt`](requirements.txt) file to install the dependencies during build time. This doesn't install poetry or is using multi-stage builds because that was causing major headaches due to the way `ray` builds their docker images. The more complicated docker images below don't currently work but are described anyway.

The problem with this is that the [`requirements.txt`](requirements.txt) file has to be updated along with poetry to ensure the builds work. Not ideal.

**To build**:

```shell
docker build -f docker/simple.dockerfile .
```

## Ray

**File**: [`docker/ray.dockerfile`](docker/ray.dockerfile)

This was an attempt at using poetry to install the dependencies in the [`rayproject/ray:2.3.0-py39-gpu`](https://hub.docker.com/r/rayproject/ray) docker image, using it as a single stage build. This didn't work because I couldn't get the image to use the correct python when either installing the dependencies through poetry or running the test script after build when doing `docker run ...`.

**To build**:

```shell
docker build -f docker/ray.dockerfile .
```

## Multi

**File**: [`docker/multi.dockerfile`](docker/multi.dockerfile)

This is the most-tested multi-stage build dockerfile, but still does not work. It uses `continuumio/miniconda3:latest` as the build stage and `rayproject/ray:2.3.0-py39-gpu` as the deploy stage. I couldn't get the conda environment built in the first stage (using poetry) to correctly copy over to the second stage so that it could be used. This is because of how the `ray` image is built.


**To build**:

```shell
docker build -f docker/multi.dockerfile .
```

## Nvidia

**File**: [`docker/nvidia.dockerfile`](docker/nvidia.dockerfile)

This image is essentially the same as the [multi](docker/multi.dockerfile) image, but it uses `nvidia/cuda:11.2.0-runtime-ubuntu20.04` as the deploy image to get around the difficulties of `multi`. This didn't work either.

**To build**:

```shell
docker build -f docker/nvidia.dockerfile .
```

## Code Server

**File**: [`docker/code-server.dockerfile`](docker/code-server.dockerfile)

This was just the beginnings of putting the necessary packages into a docker image that also had code-server installed. Not tested all that much.

**To build**:

```shell
docker build -f docker/code-server.dockerfile .
``` -->
