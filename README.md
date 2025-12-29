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

```scala mdoc
import dimwit.*

// Labels are simply Scala types
trait Batch derives Label
trait Feature derives Label

// Create a 2D tensor with shape (3, 2), labeled with Batch and Feature
val t = Tensor(
        Shape(
            Axis[Batch] -> 3, 
            Axis[Feature] -> 2
        ), 
        VType[Float],
        Array(
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
    ))

// Function to normalize a single feature vector
def normalize(x: Tensor1[Feature, Float]) : Tensor1[Feature, Float] = 
    (x - x.mean) / x.std

// Apply the normalization function across the Batch dimension
val normalized: Tensor2[Batch, Feature, Float] = 
    vmap(Axis[Batch])(t)(normalize)
```

## Getting Started

To get started with DimWit, check out the [examples](examples/src/main/scala/examples/basic) directory which contains a variety of example programs demonstrating the core concepts and features of the library.

To run the examples, you will need to have SBT, Python and JAX installed. 
If you don't want to set up the environment manually, you can use the provided Docker image 

```bash
docker pull ghcr.io/marcelluethi/dimwit-ci:latest
docker run -it ghcr.io/marcelluethi/dimwit-ci:latest /bin/bash
```

## Status 

**Early but functional.** DimWit successfully runs complex models including GPT-2 (see [example](examples/src/main/scala/examples/basic/gpt2)). The core concepts are stable, but the API is still evolving.

**Not production-ready** - expect breaking changes.

## Contributing

If your interests align with our vision, we would love to have you on board! Feel free to open issues or pull requests on GitHub.