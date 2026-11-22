FROM docker.io/python:3.12.1-slim-bookworm

LABEL org.opencontainers.image.authors="David Edelist <jedelist@bu.edu>, FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="pl-mnist_cnn" \
      org.opencontainers.image.description="A ChRIS plugin that trains, evaluates, or predicts with a small CNN on MNIST handwritten digits."

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG SRCDIR=/usr/local/src/app
WORKDIR ${SRCDIR}

# PyTorch's CPU wheels: the network is small enough to train on a CPU in minutes,
# and the default wheels would add several gigabytes of CUDA libraries to the image.
COPY requirements.txt .
RUN --mount=type=cache,sharing=private,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
ARG extras_require=none
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[${extras_require}]" \
    && cd / && rm -rf ${SRCDIR}

WORKDIR /
CMD ["mnist_cnn"]
