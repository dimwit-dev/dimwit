# Docker Setup for DimWit

This directory contains Docker configuration for running the DimWit project with NVIDIA GPU support, Python, and JAX.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Docker Runtime** (nvidia-docker2):
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Docker Compose** (v1.28.0 or higher for GPU support)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and start the container:**
   ```bash
   docker-compose up -d
   ```

2. **Access the container:**
   ```bash
   docker-compose exec dimwit bash
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker CLI

1. **Build the GPU image:**
   ```bash
   docker build -t dimwit:latest .
   ```

2. **Run the container:**
   ```bash
   docker run --gpus all -it \
     -v $(pwd):/workspace \
     -v sbt-cache:/root/.sbt \
     -v ivy-cache:/root/.ivy2 \
     -p 8888:8888 \
     dimwit:latest
   ```

## Verify GPU Access

Once inside the container, verify GPU access:

```bash
# Check NVIDIA driver
nvidia-smi

# Test JAX GPU support
python3 -c "import jax; print(jax.devices())"
```

## Working with the Project

### Scala/SBT Commands

```bash
# Compile the project
sbt compile

# Run tests
sbt test

# Run examples
sbt "examples/run"

# Start SBT console
sbt console
```

### Python/JAX

```bash
# Run Python scripts
python3 src/python/jax_helper.py

# Start Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Development Workflow

The project directory is mounted as a volume, so changes made on your host machine are immediately reflected in the container and vice versa.

### Persistent Caches

The following directories are stored in Docker volumes for faster rebuilds:
- `/root/.sbt` - SBT cache
- `/root/.ivy2` - Ivy dependencies cache

## Customization

### Adjust GPU Settings

Edit `docker-compose.yml` to change which GPUs are visible:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1
```

### Add Python Packages

Edit `Dockerfile` and add packages to the pip install command:

```dockerfile
RUN pip install numpy scipy matplotlib pandas your-package-here
```

Then rebuild:
```bash
docker-compose build
```

## Troubleshooting

### GPU not detected

1. Verify nvidia-docker2 is installed:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
   ```

2. Check Docker daemon configuration (`/etc/docker/daemon.json`):
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     }
   }
   ```

### JAX not finding GPU

Ensure CUDA libraries are in the path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Ports

- `8888`: Jupyter notebook
- `5000`: Application port (modify as needed)

Add more ports in `docker-compose.yml` as required.

---

## CI/CD Images

dimwit provides two separate Dockerfiles:
- **Dockerfile** - GPU development image with NVIDIA JAX (~8 GB)
- **Dockerfile.ci** - CPU-only CI/CD image with JAX CPU (~2-3 GB)

The CPU variant is optimized for CI/CD pipelines, eliminating GPU dependencies while maintaining full functionality for build and test workflows.

### Why CPU-only for CI/CD?

- **Smaller size**: ~1 GB vs ~8 GB (GPU image)
- **Faster pulls**: 6-8x faster download times
- **No GPU required**: Works on standard GitHub Actions runners
- **Lower costs**: Reduced registry storage and bandwidth

### Image Location

The CI image is automatically built and published to GitHub Container Registry:

```
ghcr.io/marcelluethi/dimwit-ci:latest
```

Available tags:
- `latest` - Most recent build from main branch
- `sha-<commit>` - Specific commit SHA for reproducible builds

### Using the CI Image Locally

Pull and run the CI image for local testing:

```bash
# Pull the image
docker pull ghcr.io/marcelluethi/dimwit-ci:latest

# Run compilation
docker run --rm -v $(pwd):/workspace ghcr.io/marcelluethi/dimwit-ci:latest sbt compile

# Interactive shell
docker run --rm -it -v $(pwd):/workspace ghcr.io/marcelluethi/dimwit-ci:latest bash

# Or build locally
docker build -f Dockerfile.ci -t dimwit-ci .
```

### Automated Workflows

Two GitHub Actions workflows manage the CI infrastructure:

#### 1. Build CI Image (`build-ci-image.yml`)

Builds and pushes the CI image to GHCR when:
- `Dockerfile.ci` changes
- Build configuration changes (`build.sbt`, `project/`)
- Manually triggered via workflow dispatch

**Trigger manually:**
```bash
gh workflow run build-ci-image.yml
```

#### 2. CI Build (`ci.yml`)

Runs on every push and pull request:
- Compiles all three modules (core, nn, examples)
- Uses the CI Docker image as container
- Caches SBT dependencies between runs
- Checks code formatting (non-blocking)

### Image Contents

**CI image** (`Dockerfile.ci`):

**Base**: SBT Scala (`sbtscala/scala-sbt` with OpenJDK 11)

**System Packages**:
- OpenJDK 11
- SBT (Scala Build Tool)
- Git, curl, wget

**Python Packages**:
- JAX (CPU version)
- NumPy
- Einops
- Matplotlib, Pandas, Scikit-learn
- Jupyter

**Pre-cached**:
- SBT dependencies (from `build.sbt`)
- Coursier/Ivy cache

**Dev image** (`Dockerfile`):
- Base: `nvcr.io/nvidia/jax:24.04-py3`
- Size: ~8 GB
- JAX Backend: CUDA GPU
- NVIDIA Runtime: Required
- Use Case: Development with GPU

### Build Commands

| Variant | Command | Size | Use Case |
|---------|---------|------|----------|
| GPU Dev | `docker build -t dimwit .` | ~8 GB | Local development with GPU |
| CPU CI | `docker build -f Dockerfile.ci -t dimwit-ci .` | ~2-3 GB | CI/CD pipelines |

### Rebuilding the CI Image

When dependencies need updates (e.g., new JAX version, Python upgrade):

1. **Update `Dockerfile.ci`** with new versions
2. **Trigger the rebuild workflow**:
   ```bash
   gh workflow run build-ci-image.yml
   ```
3. **Wait for build completion** (~10-15 minutes)
4. **Verify the update**:
   ```bash
   docker pull ghcr.io/marcelluethi/dimwit-ci:latest
   docker run --rm ghcr.io/marcelluethi/dimwit-ci:latest pip list | grep jax
   ```

### Troubleshooting CI

**Image pull authentication failed**:
```bash
# Login to GHCR locally
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

**SBT compilation fails**:
- Check that `PYTHONPATH` includes `/workspace/src/python`
- Verify JAX CPU is installed: `docker run ... python -c "import jax; print(jax.devices())"`

**Out of disk space**:
- Clean old images: `docker system prune -a`
- Use specific SHA tags instead of always pulling `latest`
