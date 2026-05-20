# Quickstart

Welcome to DimWit! This quickstart guide will give you an overview of the main features of DimWit and how to use them. 
It is not meant to be an exhaustive tutorial, but rather a quick introduction to the main concepts and operations in DimWit. 
For more detailed information, please refer to the API documentation, the examples and the tests.

### An introductory example

Before we start exploring the features of DimWit, let's look at a simple example that illustrates the main concepts and operations in DimWit. The example shows a linear regression model, implemented in machine learning style, using a model, loss function and a gradient-based training method.

```scala
// main imports for basic tensor operations and automatic differentiation
import dimwit.*
import dimwit.Autodiff.grad // TODO replace with cleaner import after PR is merged
import dimwit.optimizer.GradientDescent // TODO replace with cleaner import after refactoring 

// labels for tensor axes
trait Batch derives Label
trait Feature derives Label

// parameters are explicitly defined and usually bundled in a case class
case class Params(w: Tensor1[Feature, Float32], b: Tensor0[Float32]) derives TensorTree

// the model as a function of data and parametesrs
def model(x: Tensor2[Batch, Feature, Float32], y: Tensor1[Batch, Float32])(params: Params): Tensor1[Batch, Float32] =
  x.dot(Axis[Feature])(params.w) +! params.b

// the loss function as a function of data and parameters
def loss(x: Tensor2[Batch, Feature, Float32], y: Tensor1[Batch, Float32])(params: Params): Tensor0[Float32] =
  val pred = model(x, y)(params)
  (pred - y).pow(Tensor0(2.0f)).mean

// the training loop, which produces an iterator of parameters
def fit(x: Tensor2[Batch, Feature, Float32], y: Tensor1[Batch, Float32]): Iterator[Params] =

  // initialize parameters
  val p0 = Params(
    w = Tensor(Shape(Axis[Feature] -> 2)).fill(0f),
    b = Tensor0(0f)
  )

  // gradient function via automatic differentiation
  val gradFn = grad(loss(x, y))

  // gradient based optimization
  val gd = GradientDescent(learningRate = Tensor0(0.1f)) // this is wrong, should be 0.1f not Tensor0
  gd.iterate(p0)(gradFn)
```



We will learn the details as we go through the different sections of this quickstart guide, but let's briefly look at the main features that this example illustrates.

First, we see that we have labels for the axes of our tensors, which are `Batch` and `Feature`. These labels appear again in function signatures and make explicit what type of data a function 
expects and what type of data it returns. This is a key feature of DimWit, as it allows us to catch many errors at compile time, which would otherwise only be caught at runtime.

Second, we see that all the parameters of the model are explicitly defined and bundled in a case class `Params`. This will be the case even in much more complex models where we have many parameters. This explicit definition of parameters is a key feature of DimWit. Together with the named types of 
the tensors, this makes the code much more readable and maintainable, as it is always clear what the parameters are and how they are used in the model and the loss function.

Finally, we see that we can compute the gradients of the loss function with respect to the parameters using automatic differentiation, which is a key feature of DimWit. This allows us to easily implement gradient-based optimization algorithms, such as gradient descent, which is illustrated in the example.

### Getting started 

We assume that you have already added DimWit as a dependency to your project and that you have configured the Python environment as described in the README. 

To use DimWit in your Scala code, you need to import the main package as follows:

```scala
import dimwit.*
```

This import will give you everything that you need for working with Tensors. For more specialized operations, such as statistical functions or automatic differentiation, separate imports are required, which we will discuss later in this guide.

The first statement in every DimWit program should always be 
```
dimwit.initialize()
```. 
This initializes the Python environment and the JAX backend. 


### Labels, Axis, Extents and Shapes

The core concept in DimWit is that of a named axis, represented by a Scala type. 
Each axis has an associated label, which we define when we create the shape of a tensor and use to refer to that axis in operations.

A label is simply a Scala type that derives from the `Label` trait. For example:

```scala mdoc:invisible:reset
import dimwit.*
dimwit.initialize()
```

```scala
trait Batch derives Label
trait Feature derives Label
```

To create an axis, we use the `Axis` class, which takes a label as a type parameter:

```scala
val batchAxis = Axis[Batch]
val featureAxis = Axis[Feature]
```

An axis has an associated extent, which is the size of that axis. We can create an extent by creating an `AxisExtent` object as follows:

```scala
val batchExtent = AxisExtent(Axis[Batch], 3) 
``` 

or using the convenient `->` operator:

```scala
val featureExtent = Axis[Feature] -> 2
```

Finally, we can use these axes and extents to create a shape for a tensor. A shape is simply an ordered collection of axes and their corresponding extents. We can create a shape using the `Shape` class by passing the extents as arguments:

```scala
val shape : Shape[(Batch, Feature)]= Shape(batchExtent, featureExtent)
```
Note that we annotated here the type of the shape to illustrate that the resulting Shape type is parameterized by a tuple of the labels of the axes. Annotating the type
is usually not necessary in practice, as Scala can infer the types automatically. 

The labels that we specified are not only used for type-level safety, but represented at runtime as well. This means that we can print the shape and get a human-readable representation of the shape, showing the labels and their corresponding extents:

```scala
println(shape) 
// Shape(Batch -> 3, Feature -> 2)
```


### Creating Tensors

Now that we know how to create shapes, we can create tensors. A tensor is simply data that has a shape. To create a tensor in dimwit, 
we write `Tensor(shape)`, which creates a tensor factory for the specified shape. We can then use this factory to create tensors using several 
convenient methods. For example, we can create a tensor from an array of data using the `fromArray` method

```scala
val data = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
val tensor = Tensor(shape).fromArray(data)
```

This creates a tensor with the specified shape and data. The data is stored in a flat array
Let's inspect the tensor more closely: 

```scala
tensor
// res2: Tensor[Tuple2[Batch, Feature], Float32] = [[1. 2.]
//  [3. 4.]
//  [5. 6.]]
```
We see that the full type of the tensor is `Tensor[(Batch, Feature), Float32]`, which indicates that the tensor has two axes, `Batch` and `Feature`, and that the data type of the tensor is `Float`. As this type is rather bulky to write, we can also use the convenient type aliases `Tensor2`, `Tensor1` and `Tensor0` for tensors of rank 2, rank 1 and rank 0 respectively. In this case, the type of the tensor could also be written as `Tensor2[Batch, Feature, Float32]`.


In addition to the type aliases, we also have convenient factory methods for tensors of rank 0 to 2. These allow us to create tensors without having to explicitly create the shape. 

A `Tensor0` is a scalar and has no axis and therefore its shape is empty. To create a scalar tensor, it suffices to specify its value:
```scala
val scalar = Tensor0(42.0f)
```

A `Tensor1` is a vector and has one axis. When creating the tensor from an Array, it suffices to specify the axis and the data and dimwit will infer the shape from the length of the data array:
```scala
val vector = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f))
```

A `Tensor2` represents a matrix. The Tensor2 factory provides convenient methods to create special matrices, such as for example the identity 
matrix:
```scala
val eye = Tensor2.eye(Axis[Feature] -> 3)
``` 
Of course, we can also create a Tensor2 from an array of data, just like we did for the general Tensor factory:
```scala
val matrix = Tensor2(Axis[Feature], Axis[Batch]).fromArray(
    Array(
        Array(1f, 2f), 
        Array(3f, 4f), 
        Array(5f, 6f)
    )
)    
```

##### A note on type annotations for tensors

In DimWit, the type of a tensor is represented at the type level as `Tensor[ShapeTuple, VType]`, where `ShapeTuple` is a tuple of the labels of the axes and `DataType` is the type of the data. Hence a tensor with shape `Shape(Axis[Batch] -> 3, Axis[Feature] -> 2)` and data type `Float` has the type `Tensor[(Batch, Feature), Float32]`. To make type annotations more convenient, we have the type aliases `Tensor0`, `Tensor1` and `Tensor2`, etc. to 
refer to Tensors of a specific rank. For example, a `Tensor[(Batch, Feature), Float32]` can be referred to as `Tensor2[Batch, Feature, Float32]`, a Tensor `Tensor[Tuple1[Batch], Float32]` can be referred to as `Tensor1[Batch, Float32]` and a `Tensor[EmptyTuple, Float32]` can be referred to as `Tensor0[Float32]`.

### Arithmetic Operations on Tensors and broadcasting

DimWit provides the usual arithmetic operations on tensors, such as addition, multiplication, etc. These operations are defined in a way that respects the labels of the axes. For example, we can add two tensors with the same shape as follows:


```scala
trait A derives Label
trait B derives Label
val tensor1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
val tensor2 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(2.0f)
val sum = tensor1 + tensor2
```

However, if we try to add two tensors with incompatible shapes, we will get a compile-time error:

```scala
trait C derives Label
val tensor1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
val tensor3 = Tensor(Shape(Axis[A] -> 3, Axis[C] -> 2)).fill(2.0f)
tensor1 + tensor3 
// error:
// 
// A tuple of axis labels MdocApp1.this.A *: (MdocApp1.this.B *: EmptyTuple | MdocApp1.this.C *:
//   EmptyTuple) was given or inferred that does not have a valid Labels instance. 
// 
// Ensure that all of the types in the tuple have a 'derives Label' clause.
// .
// I found:
// 
//     dimwit.tensor.Labels.given_Labels_A_B[A², B²](
//       dimwit.tensor.Labels.lift[MdocApp1.this.A](this.A.derived$Label),
//       dimwit.tensor.Labels.lift[B²](/* missing */summon[dimwit.tensor.Label[B²]]))
// 
// But no implicit values were found that match type dimwit.tensor.Label[B²]
// 
// where:    A  is a trait in class MdocApp1
//           A² is a type variable
//           B  is a trait in class MdocApp1
//           B² is a type variable
// .
// tensor1 + tensor3 
//           ^^^^^^^
// error:
// Conflicting definitions:
// val tensor1:
//   dimwit.tensor.Tensor[(MdocApp1.this.A, MdocApp1.this.B),
//     dimwit.tensor.DType.Float32] in class MdocApp1 at line 63 and
// val tensor1:
//   dimwit.tensor.Tensor[(MdocApp1.this.A, MdocApp1.this.B),
//     dimwit.tensor.DType.Float32] in class MdocApp1 at line 67
// 
// val tensor1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
//     ^
```

Another key difference between DimWit and other tensor libraries is the broadcasting behavior. As silent broadcasting
is often a source of bugs, DimWit does not allow it. The following code will not compile, even though the shapes of the tensors are compatible for broadcasting:

```scala
trait C derives Label
val tensor3 = Tensor(Shape(Axis[A] -> 3)).fill(1.0f)
tensor1 + tensor3
// error:
// 
// A tuple of axis labels MdocApp1.this.A *: (MdocApp1.this.B *: EmptyTuple | EmptyTuple) was given or inferred that does not have a valid Labels instance. 
// 
// Ensure that all of the types in the tuple have a 'derives Label' clause.
// .
// I found:
// 
//     dimwit.tensor.Labels.concat[head, tail](this.A.derived$Label,
//       dimwit.tensor.Labels.lift[tail](/* missing */summon[dimwit.tensor.Label[tail]]
//         )
//     )
// 
// But no implicit values were found that match type dimwit.tensor.Label[tail].
// tensor1 + tensor3
//           ^^^^^^^
```

If we want to use broadcasting, we have to use the explicit broadcasting versions of the operations, which are suffixed with a `!`.
The following operation compiles successfully:
```scala
trait C derives Label
val tensor3 = Tensor(Shape(Axis[A] -> 3)).fill(1.0f)
tensor1 +! tensor3
```

Another common source of bugs is the use of the wrong axis in an operation. In DimWit we can specify the 
axis to sum over using the labels of the axes, which ensures that we are summing over the correct axis. 
The resulting tensor has the correct shape, which is inferred from the labels of the axes:

```scala
val sumOverB : Tensor1[A, Float32] = tensor1.sum(Axis[B])
// sumOverB: Tensor[Tuple1[A], Float32] = [2. 2. 2.]
val sumOverA : Tensor1[B, Float32] = tensor1.sum(Axis[A])
// sumOverA: Tensor[Tuple1[B], Float32] = [3. 3.]
```

### Transforming the shape of tensors


DimWit provides several operations to transform the shape of tensors, without changing the underlying data. 
We consider in the following always the following 3D tensor as an example:
```scala
trait A derives Label
trait B derives Label
trait C derives Label
val tensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
``` 

#### Flattening and unflattening axes
The first operation we will discuss is `flatten` which flattens part of the tensor into a single axis. Invoked without arguments, `flatten` will flatten all axes into a single axis, resulting in a Tensor1. 
```scala
val flattened : Tensor1[A |*| B |*| C, Float32] = tensor.flatten
```
Note that the resulting axis has a label that is a combination of the labels of the original axes. When flattening, we can also specify which axes to flatten, and the resulting axis will have a label that is a combination of the labels of the flattened axes. For example, we can flatten only the last two axes as follows:

```scala
  val partiallyFlattened: Tensor2[A, B |*| C, Float32] = tensor.flatten((Axis[B], Axis[C]))
```

The counterpart of flatten is `unflatten`, which takes an axis that was previously flattened and restores the original axes. Since in the process of flattening we lost the information about the original shape, 
we have to specify the shape of the original axes when unflattening. For example, we can unflatten the previously flattened tensor as follows:

```scala
flattened.unflatten(tensor.shape)
```

To unflatten a partially flattened tensor, we need to specify the axis that we want to unflatten (here the previously flattend axis with label `B |*| C`) and the shape of the original axes (here `Shape(Axis[B] -> 2, Axis[C] -> 4)`):

```scala
partiallyFlattened.unflatten(Axis[B |*| C], Shape(Axis[B] -> 2, Axis[C] -> 4))
```
#### Concatenating, splitting and slicing tensors

Flatten and unflatten takes a single tensor and transform its shape. In contrast, concatenate and split take multiple tensors and combine them into a single tensor or split a single tensor into multiple tensors.
Given two tensors with the same shape except for one axis, we can concatenate them along that axis as follows:

```scala
  val part1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
  val part2 = Tensor(Shape(Axis[A] -> 7, Axis[B] -> 2)).fill(1.0f)

  val concatenated: Tensor[(A, B), Float32] = concatenate(Seq(part1, part2), Axis[A])
```

The concatenated tensor can be split back into the original tensors using the split method
```scala
  val (split1, split2) = concatenated.split((Axis[A].at(0)))
```
If we want a split into several tensors, we can specify the split points as follows:
```scala
  val (splt1, splt2, splt3) = concatenated.split(Axis[A].at((1, 2)))
```

The method `slice` works in a similar way to split, but instead of splitting the tensor into several tensors, 
it returns a single tensor that is a slice of the original tensor. For example, we can extract the slice 
at index 1 along axis A as follows:

```scala
  val sliced1 : Tensor1[B, Float32]= concatenated.slice(Axis[A].at(1))
```

As for split, we can also provide multiple a tuple (or sequence) of indices, which will then select all slices in the tuple.

```scala
  val slicedMultiple : Tensor2[A, B, Float32] = concatenated.slice(Axis[A].at((0, 2)))
```

#### Squeezing, Expanding and transposing axes

Given a tensor, for which one axis has extent 1, we can remove that axis using the `squeeze`
method. 

```scala
val squeezableTensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 1, Axis[C] -> 4)).fill(1.0f)
val squeezedTensor : Tensor[(A, C), Float32] = squeezableTensor.squeeze(Axis[B])
```
Similarly, we can add a new axis with extent 1 using the method `appendAxis`:
```scala
val appendedTensor : Tensor[(A, C, B), Float32] = squeezedTensor.appendAxis(Axis[B])
```
Finally, we can permute the axes of a tensor using the `transpose` method. 
The order of the axes is the order of the labels in the tuple that we pass as an argument to the method. For example, we can reorder the above tensor to have the order of axes `A, B, C` as follows:

```scala
val restoredTensor : Tensor[(A, B, C), Float32] = appendedTensor.transpose((Axis[A], Axis[B], Axis[C]))
```

### Mapping over axes

We often want to apply functions to each slice of a tensor along a given axis. 
Let's take again the following tensor as an example:


```scala
trait A derives Label
trait B derives Label
trait C derives Label
val tensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
``` 

The simplest method is `vapply`. `vapply` applies a function from a `Tensor1` to a `Tensor1` to each slice of the tensor along the specified axis. For example, we can apply the function that multiplies each element by 2 to each slice along axis A as follows:    

```scala
val doubled : Tensor3[A, B, C, Float32] = tensor.vapply(Axis[A])((slice : Tensor1[A, Float32]) => slice *! Tensor0(2.0f))
```

Similar to `vapply`is `vreduce`. `vreduce` applies a function that reduces a `Tensor1` to a `Tensor0` to each slice of the tensor along the specified axis. It effectively reduces the specified axis to a scalar. 

```scala
val summedA : Tensor2[B, C, Float32] = tensor.vreduce(Axis[A])((slice : Tensor1[A, Float32]) => slice.sum)
```

A more general method is `vmap`, which applies a function to the slice of the tensor along the specified axis.  The function can return a tensor of any shape, not just a `Tensor1`. The resulting tensor will have the same shape as the original tensor, except that the specified axis will be replaced by the shape of the output of the function. The following example takes a Tensor2 as input and computes the 
mean of each slice along axis C


```scala
  val res : Tensor[(A, B), Float32] = tensor.vmap(Axis[A])((slice : Tensor2[B, C, Float32]) => slice.mean(Axis[C]))
```
`zipmap` is a variant of `vmap` that applies a function to the slices of multiple tensors along the specified axis. The function takes as input a tuple of slices, one from each tensor, and, as `vmap` returns a tensor of any shape. The resulting tensor will have the same shape as the original tensors, except that the specified axis will be replaced by the shape of the output of the function. For example, we can use `zipmap` to add two tensors along axis A as follows:

```scala
val t1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
val t2 = Tensor(Shape(Axis[A] -> 3, Axis[C] -> 3)).fill(2.0f)

val sumAlongA : Tensor1[A, Float32] = zipvmap(Axis[A])(t1, t2)((s1: Tensor2[B, C, Float32], s2: Tensor1[C, Float32]) => s1.sum + s2.sum)
``` 


### Automatic differentiation

A key feature of DimWit is the support for automatic differentiation. 
As long as a function expressed computations using the tensor operations provided by DimWit, we can compute the gradients of the function automatically. Functions for automatic differentiation are defined in the package `dimwit.autodiff`which we can import as follows:


```scala
import dimwit.autodiff.*
import Autodiff.grad 
```

Let's take a simple quadratic function as an example. 
```scala
def f(x: Tensor1[A, Float32]): Tensor0[Float32] = x.dot(Axis[A])(x)
```

To compute the gradient of this function with respect to its input, we can use the `grad` method as follows:

```scala
val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val gradient : Tensor1[A, Float32] => Grad[Tensor1[A, Float32]] = grad(f)
```

Note that the result of `grad` is a function that takes as input a tensor and returns the gradient of the function with respect to that tensor. The gradient is a normal tensor, but wrapped in a `Grad` object, to make sure that we don't accidentally use it in a computation without realizing that it is a gradient. To get the actual tensor from the `Grad` object, we can use the `value` method as follows:

```scala
val gradValue : Tensor1[A, Float32] = gradient(x).value
```

#### Tensor trees and gradients of multiple parameters

In practice, we often have functions that take as input multiple tensors. DimWit borrows 
the concept of tensor trees from Jax to handle this case. A tensor tree is simply a nested structure of tensors, such as a case class that contains tensors, or a tuple of tensors, etc. 
For larger models the most convenient representation of the parameters is usually a (nested) case class that contains all the parameters as fields. To mark a case class as a tensor tree, we need to make it derive the `TensorTree` type class:


```scala
case class Params(w: Tensor1[Feature, Float32], b: Tensor0[Float32]) derives TensorTree
```

To compute the gradient of a function that takes as input a tensor tree, we can use the same `grad` method, as long as the input type of the function is a tensor tree. The resulting gradient will then be a tensor tree of the same shape as the input tensor tree, as illustrated in the following example:

```scala
def f(params: Params): Tensor0[Float32] = params.w.dot(Axis[Feature])(params.w) + params.b.pow(Tensor0(2.0f))

val params = Params(
  w = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f)),
  b = Tensor0(3.0f)
)
val gradient : Params => Grad[Params] = grad(f)
val gradValue : Params = gradient(params).value
```

### Working with random numbers

DimWit is based on Jax.  Jax uses a functional approach to random number generation, which means that instead of having a global random state, we have to explicitly pass a random key to the functions that generate random numbers. DimWit follows the same approach, which means that we have to create a random key, whenever a method has a stochastic component. 

Let's say we want to create a random number drawn from a normal distribution. We first generate the corresponding distribution object:


```scala
import dimwit.stats.*
val normalDist = Normal(Tensor0(0.0f), Tensor0(1.0f))
```
To sample from this distribution, we need to create a random key and pass it to the `sample` method of the distribution object:

```scala
import dimwit.random.*
import Random.Key
val key = Key(42)
val srandomValue = normalDist.sample(key)
```

Whenever we want to generate a new random number, we have to split the key to get a new key. Dimwit provides several convenient methods to split keys. 
For example, we can split a key into two new keys as follows:

```scala
val (key1, key2) = key.split2()
```
Alternatively, we can split a key into a sequence of new keys as follows:
```scala
val keys = key.split(5)
```

Often we need to create a sequence of random numbers. In this case, we can use the `splitvmap` method, which splits a key into a tensor of new keys and applies a function to each of the new keys. For example, we can create a vector of random numbers drawn from the normal distribution as follows:
```scala
val sampleVec: Tensor1[A, Float32] = key.splitvmap(Axis[A] -> 3)((k: Key) => normalDist.sample(k))
```

A more flexible, but less performant way to create a tensor of keys and to use it together with the `vmap` or `zipvmap` method. For example, we can create a tensor of keys and use it to create a tensor of random numbers as follows:
```scala
val paramTensor = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val keyTensor : Tensor1[A, Key] = key.splitToTensor(Axis[A] -> 3)

val sampleVec2 = zipvmap(Axis[A])(paramTensor, keyTensor)((param, key) => Normal(param, Tensor0(1.0f)).sample(key.item))
```
Note that in the above example, we had to use `key.item` to get the actual key from the tensor of keys, as the function passed to `zipvmap` takes as input a slice of the tensor, which is a `Tensor0` and not a `Key`.

