# GPU Development Dockerfile
# Provides NVIDIA JAX with GPU support, Java, SBT, and Python packages

FROM nvcr.io/nvidia/jax:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install Java and SBT
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ca-certificates \
    gnupg \
    apt-transport-https \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

# Install SBT
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list && \
    echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list && \
    curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add && \
    apt-get update && \
    apt-get install -y sbt && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY . /workspace/

# Create venv inheriting JAX from the base image, add extra packages
RUN uv venv .venv --system-site-packages && \
    uv pip install --python .venv/bin/python einops matplotlib pandas

# Skip uv sync — JAX is already available via system-site-packages
ENV DIMWIT_SKIP_SYNC=true

# Set Python path
ENV PYTHONPATH=/workspace/src/python

# Expose ports
EXPOSE 8888 5000

CMD ["/bin/bash"]
