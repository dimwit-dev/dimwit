# DimWit - Tensor programming with wit

> Programming is the art of telling another human being what one wants the computer to do.
> 
> Donald Knuth


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
def normalize(x: Tensor1[Feature, Float]) : Tensor1[Feature, Float] = 
    (x -! x.mean) /! x.std

// Apply the normalization function across the Batch dimension
val normalized: Tensor2[Batch, Feature, Float] = 
    t.vmap(Axis[Batch])(normalize)
```

See [examples](examples/src/main/scala/basic/) for more examples.


## Using DimWit as a Library

**Note**: DimWit is currently in early development (`0.1.0-SNAPSHOT`) and not yet published to Maven Central.

### Installation

To use dimwit as a library, clone the repository and publish locally:

```bash
git clone https://github.com/dimwit-dev/dimwit.git
cd dimwit
sbt publishLocal
```

Then add to your `build.sbt`:

```scala
libraryDependencies ++= Seq(
  "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT",  // Core tensor library
  "ch.contrafactus" %% "dimwit-nn" % "0.1.0-SNAPSHOT"     // Neural network components (optional)
)

resolvers += Resolver.mavenLocal
```

### Python Environment Setup


DimWit requires **Python 3.9+** and **JAX** since it uses JAX as the backend for tensor operations via ScalaPy. It also relies on **Einops** for tensor reshaping and manipulation.

The easiest way to set up the Python environment is to install the [uv package manager](https://github.com/astral-sh/uv) and add a file `pyproject.toml` with the following content to your project:

```toml
[project]
name = "dimwit-python-env"
version = "0.1.0"
requires-python = "==3.13.*"
dependencies = [
    "einops>=0.8.1",
    "jax[cuda12]>=0.8.2",
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

Alternatively, you can set up the Python environment manually by setting 
the environment variables `DIMWIT_PYTHON_PATH` and `DIMWIT_PYTHON_LIBRARY` to the path 
of your Python installation and tell DimWit to bypass uv by setting `DIMWIT_SKIP_SYNC` to `true`.

## Status 

**Early but functional.** DimWit successfully runs complex models including GPT-2 (see [example](examples/src/main/scala/complex/GPT2.scala)). The core concepts are stable, but the API is still evolving.

**Not production-ready** - expect breaking changes.

## Contributing

If your interests align with our vision, we would love to have you on board! Feel free to open issues or pull requests on GitHub.