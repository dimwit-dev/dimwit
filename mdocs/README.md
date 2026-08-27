# DimWit - Tensor programming with wit

> Programming is the art of telling another human being what one wants the computer to do.
> 
> Donald Knuth

<p align="center">
  <img src="assets/dimwit-logo.svg" alt="DimWit — Tensor programming with wit" width="360">
</p>


## Vision

We want to create a system for writing numerical and machine learning programs that puts human understanding first. While programming, the compiler should help us to keep concepts separate and sharpen our thinking about the problem.

## Why?

AI coding agents and modern numerical libraries, such as Jax, Pytorch or Tensorflow, make it ever easier to write numerical and machine learning programs. Yet understanding remains as difficult as ever. Untyped code, 
opaque tensor operations and a focus on performance instead of clarity often obscure the concepts underlying the code.

With *DimWit* we want to change this by
- allowing to express concepts clearly and on a high level
- leveraging the type system to enable the compiler to help us keep 
    concepts separate and check correctness.

## How?

DimWit uses the power of the Scala 3 type system to encode tensor dimensions as types. It combines this with a high-level API inspired by 
JAX and einops, and efficient implementations of tensor operations using JAX as a backend.

## Example

```scala mdoc:silent
import dimwit.*

// Labels are simply Scala types
trait Batch derives Label
trait Feature derives Label

// Create a 2D tensor with shape (3, 2), labeled with Batch and Feature
val t = Tensor(
    Shape(Axis[Batch] -> 3, Axis[Feature] -> 2),
).fromArray(
    Array(
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    )
)

// Function to normalize a single feature vector
def normalize(x: Tensor1[Feature, Float32]) : Tensor1[Feature, Float32] = 
    (x -! x.mean) /! x.std

// Apply the normalization function across the Batch dimension
val normalized: Tensor2[Batch, Feature, Float32] = 
    t.vmap(Axis[Batch])(normalize)
```

See our [quickstart guide](docs/quickstart.md) for a more detailed introduction to the core concepts and API and 
check out the [examples](examples/src/main/scala). 

## Using DimWit as a Library

### Requirements

- Scala **3.8** or newer
- sbt **1.11** or newer

### Installation

Add to your `build.sbt` (minimal complete example):

```scala
ThisBuild / scalaVersion := "3.8.1"

lazy val myProject = (project in file("."))
  .settings(
    name := "my-project",
    libraryDependencies ++= Seq(
      "ch.contrafactus" %% "dimwit-core" % "0.1.0"
    ),
    fork := true
  )
```

### Python Environment Setup

DimWit requires **Python 3.9+** and **JAX** since it uses JAX as the backend for tensor operations via ScalaPy. It also relies on **Einops** for tensor reshaping and manipulation.

The easiest way to set up the Python environment is to install the [uv package manager](https://github.com/astral-sh/uv) and add a `pyproject.toml` to your **project root** (the directory where you run `sbt`):

```toml
[project]
name = "my-project-python-env"
version = "0.1.0"
requires-python = "==3.13.*"
dependencies = [
    "einops>=0.8.1",
    # Use jax[cpu] for CPU-only environments (e.g. development/CI without a GPU)
    # Use jax[cuda12] for NVIDIA GPU support
    "jax[cpu]>=0.8.2",
]
```

DimWit provides the command `dimwit.initialize()` which you can call at the start of your application to automatically set up the Python environment. This will check for the required dependencies and set the necessary environment variables for ScalaPy.

```scala 
import dimwit.*

@main def runApp(): Unit = {
    // Initialize the Python environment for DimWit
    dimwit.initialize()

    println(Tensor0(42.0f)) 
}   
```

Alternatively, you can configure the Python environment manually using these environment variables:

- `DIMWIT_PYTHON_PATH` — path to the Python executable (find it with `which python3` or `uv run python -c "import sys; print(sys.executable)"`)
- `DIMWIT_PYTHON_LIBRARY` — path to the Python shared library (find it with `python3 -c "import ctypes.util; print(ctypes.util.find_library('python3'))"`)
- `DIMWIT_SKIP_SYNC` — set to `true` to skip uv environment sync on startup

## Status 

**Early but functional.** DimWit successfully runs complex models including GPT-2 (see [example](https://github.com/dimwit-dev/deepwit/tree/main/examples/src/main/scala/deepwit/examples/gpt)). The core concepts are stable, but the API is still evolving.

**Not production-ready** - expect breaking changes.

## Contributing

If your interests align with our vision, we would love to have you on board! Feel free to open issues or pull requests on GitHub.