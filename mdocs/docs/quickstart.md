# Quickstart

Welcome to DimWit! This quickstart guide will give you an overview of the main features of DimWit and how to use them. 
It is not meant to be an exhaustive tutorial, but rather a quick introduction to the main concepts and operations in DimWit. 
For more detailed information, please refer to the API documentation, the examples and the tests.

### Getting started

We assume that you have already added DimWit as a dependency to your project and that you have configured the Python environment as described in the README. 

To use DimWit in your Scala code, you need to import the main package as follows:

```scala mdoc:silent
import dimwit.*
```

This import will give you everything that you need for working with Tensors. For more specialized operations, such as statistical functions or automatic differentiation, separate imports are required, which we will discuss later in this guide.


### Labels, Axis, Extents and Shapes

The core concept in DimWit is that of a named axis, represented by a Scala type. 
Each axis has an associated label, which we define when we create the shape of a tensor and use to refer to that axis in operations.

A label is simply a Scala type that derives from the `Label` trait. For example:

```scala mdoc:silent
trait Batch derives Label
trait Feature derives Label
```

To create an axis, we use the `Axis` class, which takes a label as a type parameter:

```scala mdoc:silent
val batchAxis = Axis[Batch]
val featureAxis = Axis[Feature]
```

An axis has an associated extent, which is the size of that axis. We can create an extent by creating an `AxisExtent` object as follows:

```scala mdoc:silent
val batchExtent = AxisExtent(Axis[Batch], 3) 
``` 

or using the convenient `->` operator:

```scala mdoc:silent
val featureExtent = Axis[Feature] -> 2
```

Finally, we can use these axes and extents to create a shape for a tensor. A shape is simply an ordered collection of axes and their corresponding extents. We can create a shape using the `Shape` class by passing the extents as arguments:

```scala mdoc:silent
val shape : Shape[(Batch, Feature)]= Shape(batchExtent, featureExtent)
```
Note that we annotated here the type of the shape to illustrate that the resulting Shape type is parameterized by a tuple of the labels of the axes. Annotating the type
is usually not necessary in practice, as Scala can infer the types automatically. 

The labels that we specified are not only used for type-level safety, but represented at runtime as well. This means that we can print the shape and get a human-readable representation of the shape, showing the labels and their corresponding extents:

```scala mdoc
println(shape) 
```


### Creating Tensors

Now that we know how to create shapes, we can create tensors. A tensor is simply data that has a shape. To create a tensor in dimwit, 
we write `Tensor(shape)`, which creates a tensor factory for the specified shape. We can then use this factory to create tensors using several 
convenient methods. For example, we can create a tensor from an array of data using the `fromArray` method

```scala mdoc:silent
val data = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
val tensor = Tensor(shape).fromArray(data)
```

This creates a tensor with the specified shape and data. The data is stored in a flat array
Let's inspect the tensor more closely: 

```scala mdoc
tensor
```
We see that the full type of the tensor is `Tensor[(Batch, Feature), Float]`, which indicates that the tensor has two axes, `Batch` and `Feature`, and that the data type of the tensor is `Float`. As this type is rather bulky to write, we can also use the convenient type aliases `Tensor2`, `Tensor1` and `Tensor0` for tensors of rank 2, rank 1 and rank 0 respectively. In this case, the type of the tensor could also be written as `Tensor2[Batch, Feature, Float]`.


In addition to the type aliases, we also have convenient factory methods for tensors of rank 0 to 2. These allow us to create tensors without having to explicitly create the shape. 

A `Tensor0` is a scalar and has no axis and therefore its shape is empty. To create a scalar tensor, it suffices to specify its value:
```scala mdoc:silent
val scalar = Tensor0(42.0f)
```

A `Tensor1` is a vector and has one axis. When creating the tensor from an Array, it suffices to specify the axis and the data and dimwit will infer the shape from the length of the data array:
```scala mdoc:silent
val vector = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f))
```

A `Tensor2` represents a matrix. The Tensor2 factory provides convenient methods to create special matrices, such as for example the identity 
matrix:
```scala mdoc:silent
val eye = Tensor2.eye(Axis[Feature] -> 3)
``` 
Of course, we can also create a Tensor2 from an array of data, just like we did for the general Tensor factory:
```scala mdoc:silent
val matrix = Tensor2(Axis[Feature], Axis[Batch]).fromArray(
    Array(
        Array(1f, 2f), 
        Array(3f, 4f), 
        Array(5f, 6f)
    )
)    
```

##### A note on type annotations for tensors

In DimWit, the type of a tensor is represented at the type level as `Tensor[ShapeTuple, VType]`, where `ShapeTuple` is a tuple of the labels of the axes and `DataType` is the type of the data. Hence a tensor with shape `Shape(Axis[Batch] -> 3, Axis[Feature] -> 2)` and data type `Float` has the type `Tensor[(Batch, Feature), Float]`. To make type annotations more convenient, we have the type aliases `Tensor0`, `Tensor1` and `Tensor2`, etc. to 
refer to Tensors of a specific rank. For example, a `Tensor[(Batch, Feature), Float]` can be referred to as `Tensor2[Batch, Feature, Float]`, a Tensor `Tensor[Tuple1[Batch], Float]` can be referred to as `Tensor1[Batch, Float]` and a `Tensor[EmptyTuple, Float]` can be referred to as `Tensor0[Float]`.

### Arithmetic Operations on Tensors and broadcasting

DimWit provides the usual arithmetic operations on tensors, such as addition, multiplication, etc. These operations are defined in a way that respects the labels of the axes. For example, we can add two tensors with the same shape as follows:

```scala mdoc:invisible:reset
import dimwit.*
``` 

```scala mdoc:silent
trait A derives Label
trait B derives Label
val tensor1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
val tensor2 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(2.0f)
val sum = tensor1 + tensor2
```

However, if we try to add two tensors with incompatible shapes, we will get a compile-time error:

```scala mdoc:fail
trait C derives Label
val tensor1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
val tensor3 = Tensor(Shape(Axis[A] -> 3, Axis[C] -> 2)).fill(2.0f)
tensor1 + tensor3 
```

Another key difference between DimWit and other tensor libraries is the broadcasting behavior. As silent broadcasting
is often a source of bugs, DimWit does not allow it. The following code will not compile, even though the shapes of the tensors are compatible for broadcasting:

```scala mdoc:fail
trait C derives Label
val tensor3 = Tensor(Shape(Axis[A] -> 3)).fill(1.0f)
tensor1 + tensor3
```

If we want to use broadcasting, we have to use the explicit broadcasting versions of the operations, which are suffixed with a `!`.
The following operation compiles successfully:
```scala mdoc:silent
trait C derives Label
val tensor3 = Tensor(Shape(Axis[A] -> 3)).fill(1.0f)
tensor1 +! tensor3
```

Another common source of bugs is the use of the wrong axis in an operation. In DimWit we can specify the 
axis to sum over using the labels of the axes, which ensures that we are summing over the correct axis. 
The resulting tensor has the correct shape, which is inferred from the labels of the axes:

```scala mdoc
val sumOverB : Tensor1[A, Float] = tensor1.sum(Axis[B])
val sumOverA : Tensor1[B, Float] = tensor1.sum(Axis[A])
```

### Transforming the shape of tensors

```scala mdoc:invisible:reset
import dimwit.*
``` 

DimWit provides several operations to transform the shape of tensors, without changing the underlying data. 
We consider in the following always the following 3D tensor as an example:
```scala mdoc:silent
trait A derives Label
trait B derives Label
trait C derives Label
val tensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
``` 

#### Flattening and unflattening axes
The first operation we will discuss is `flatten` which flattens part of the tensor into a single axis. Invoked without arguments, `flatten` will flatten all axes into a single axis, resulting in a Tensor1. 
```scala mdoc:silent
val flattened : Tensor1[A |*| B |*| C, Float] = tensor.flatten
```
Note that the resulting axis has a label that is a combination of the labels of the original axes. When flattening, we can also specify which axes to flatten, and the resulting axis will have a label that is a combination of the labels of the flattened axes. For example, we can flatten only the last two axes as follows:

```scala mdoc:silent
  val partiallyFlattened: Tensor2[A, B |*| C, Float] = tensor.flatten((Axis[B], Axis[C]))
```

The counterpart of flatten is `unflatten`, which takes an axis that was previously flattened and restores the original axes. Since in the process of flattening we lost the information about the original shape, 
we have to specify the shape of the original axes when unflattening. For example, we can unflatten the previously flattened tensor as follows:

```scala mdoc:silent
flattened.unflatten(tensor.shape)
```

To unflatten a partially flattened tensor, we need to specify the axis that we want to unflatten (here the previously flattend axis with label `B |*| C`) and the shape of the original axes (here `Shape(Axis[B] -> 2, Axis[C] -> 4)`):

```scala mdoc:silent
partiallyFlattened.unflatten(Axis[B |*| C], Shape(Axis[B] -> 2, Axis[C] -> 4))
```
#### Concatenating, splitting and slicing tensors

Flatten and unflatten takes a single tensor and transform its shape. In contrast, concatenate and split take multiple tensors and combine them into a single tensor or split a single tensor into multiple tensors.
Given two tensors with the same shape except for one axis, we can concatenate them along that axis as follows:

```scala mdoc:silent
  val part1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2)).fill(1.0f)
  val part2 = Tensor(Shape(Axis[A] -> 7, Axis[B] -> 2)).fill(1.0f)

  val concatenated: Tensor[(A, B), Float] = concatenate(Seq(part1, part2), Axis[A])
```

The concatenated tensor can be split back into the original tensors using the split method
```scala mdoc:silent
  val (split1, split2) = concatenated.split((Axis[A].at(0)))
```
If we want a split into several tensors, we can specify the split points as follows:
```scala mdoc:silent
  val (splt1, splt2, splt3) = concatenated.split(Axis[A].at((1, 2)))
```

The method `slice` works in a similar way to split, but instead of splitting the tensor into several tensors, 
it returns a single tensor that is a slice of the original tensor. For example, we can extract the slice 
at index 1 along axis A as follows:

```scala mdoc:silent
  val sliced1 : Tensor1[B, Float]= concatenated.slice(Axis[A].at(1))
```

As for split, we can also provide multiple a tuple (or sequence) of indices, which will then select all slices in the tuple.

```scala mdoc:silent
  val slicedMultiple : Tensor2[A, B, Float] = concatenated.slice(Axis[A].at((0, 2)))
```

#### Squeezing, Expanding and transposing axes

Given a tensor, for which one axis has extent 1, we can remove that axis using the `squeeze`
method. 

```scala mdoc:silent
val squeezableTensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 1, Axis[C] -> 4)).fill(1.0f)
val squeezedTensor : Tensor[(A, C), Float] = squeezableTensor.squeeze(Axis[B])
```
Similarly, we can add a new axis with extent 1 using the method `appendAxis`:
```scala mdoc:silent
val appendedTensor : Tensor[(A, C, B), Float] = squeezedTensor.appendAxis(Axis[B])
```
Finally, we can permute the axes of a tensor using the `transpose` method. 
The order of the axes is the order of the labels in the tuple that we pass as an argument to the method. For example, we can reorder the above tensor to have the order of axes `A, B, C` as follows:

```scala mdoc:silent
val restoredTensor : Tensor[(A, B, C), Float] = appendedTensor.transpose((Axis[A], Axis[B], Axis[C]))
```

### Mapping over axes

We often want to apply functions to each slice of a tensor along a given axis. 
Let's take again the following tensor as an example:

```scala mdoc:invisible:reset
import dimwit.*
``` 

```scala mdoc:silent
trait A derives Label
trait B derives Label
trait C derives Label
val tensor = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
``` 

The simplest method is `vapply`. `vapply` applies a function from a `Tensor1` to a `Tensor1` to each slice of the tensor along the specified axis. For example, we can apply the function that multiplies each element by 2 to each slice along axis A as follows:    

```scala mdoc:silent
val doubled : Tensor3[A, B, C, Float] = tensor.vapply(Axis[A])((slice : Tensor1[A, Float]) => slice *! Tensor0(2.0f))
```

Similar to `vapply`is `vreduce`. `vreduce` applies a function that reduces a `Tensor1` to a `Tensor0` to each slice of the tensor along the specified axis. It effectively reduces the specified axis to a scalar. 

```scala mdoc:silent
val summedA : Tensor2[B, C, Float] = tensor.vreduce(Axis[A])((slice : Tensor1[A, Float]) => slice.sum)
```

A more general method is `vmap`, which applies a function to the slice of the tensor along the specified axis.  The function can return a tensor of any shape, not just a `Tensor1`. The resulting tensor will have the same shape as the original tensor, except that the specified axis will be replaced by the shape of the output of the function. The following example takes a Tensor2 as input and computes the 
mean of each slice along axis C


```scala mdoc:silent
  val res : Tensor[(A, B), Float] = tensor.vmap(Axis[A])((slice : Tensor2[B, C, Float]) => slice.mean(Axis[C]))
```
`zipmap` is a variant of `vmap` that applies a function to the slices of multiple tensors along the specified axis. The function takes as input a tuple of slices, one from each tensor, and, as `vmap` returns a tensor of any shape. The resulting tensor will have the same shape as the original tensors, except that the specified axis will be replaced by the shape of the output of the function. For example, we can use `zipmap` to add two tensors along axis A as follows:

```scala mdoc:silent
val t1 = Tensor(Shape(Axis[A] -> 3, Axis[B] -> 2, Axis[C] -> 4)).fill(1.0f)
val t2 = Tensor(Shape(Axis[A] -> 3, Axis[C] -> 3)).fill(2.0f)

val sumAlongA : Tensor1[A, Float] = zipvmap(Axis[A])(t1, t2)((s1: Tensor2[B, C, Float], s2: Tensor1[C, Float]) => s1.sum + s2.sum)
``` 


### Automatic differentiation

A key feature of DimWit is the support for automatic differentiation. 
As long as a function expressed computations using the tensor operations provided by DimWit, we can compute the gradients of the function automatically. Functions for automatic differentiation are defined in the package `dimwit.autodiff`which we can import as follows:

```scala mdoc:invisible:reset
import dimwit.*
trait A derives Label
trait B derives Label
trait C derives Label
```

```scala mdoc:silent
import dimwit.autodiff.*
import Autodiff.grad 
```

Let's take a simple quadratic function as an example. 
```scala mdoc:silent
def f(x: Tensor1[A, Float]): Tensor0[Float] = x.dot(Axis[A])(x)
```

To compute the gradient of this function with respect to its input, we can use the `grad` method as follows:

```scala mdoc:silent
val x = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val gradient : Tensor1[A, Float] => Grad[Tensor1[A, Float]] = grad(f)
```

Note that the result of `grad` is a function that takes as input a tensor and returns the gradient of the function with respect to that tensor. The gradient is a normal tensor, but wrapped in a `Grad` object, to make sure that we don't accidentally use it in a computation without realizing that it is a gradient. To get the actual tensor from the `Grad` object, we can use the `value` method as follows:

```scala mdoc:silent
val gradValue : Tensor1[A, Float] = gradient(x).value
```


### Working with random numbers

DimWit is based on Jax.  Jax uses a functional approach to random number generation, which means that instead of having a global random state, we have to explicitly pass a random key to the functions that generate random numbers. DimWit follows the same approach, which means that we have to create a random key, whenever a method has a stochastic component. 

Let's say we want to create a random number drawn from a normal distribution. We first generate the corresponding distribution object:

```scala mdoc:silent
import dimwit.stats.*
val normalDist = Normal(Tensor0(0.0f), Tensor0(1.0f))
```
To sample from this distribution, we need to create a random key and pass it to the `sample` method of the distribution object:

```scala mdoc:silent
import dimwit.random.*
import Random.Key
val key = Key(42)
val srandomValue = normalDist.sample(key)
```

Whenever we want to generate a new random number, we have to split the key to get a new key. Dimwit provides several convenient methods to split keys. 
For example, we can split a key into two new keys as follows:

```scala mdoc:silent
val (key1, key2) = key.split2()
```
Alternatively, we can split a key into a sequence of new keys as follows:
```scala mdoc:silent
val keys = key.split(5)
```

Often we need to create a sequence of random numbers. In this case, we can use the `splitvmap` method, which splits a key into a tensor of new keys and applies a function to each of the new keys. For example, we can create a vector of random numbers drawn from the normal distribution as follows:
```scala mdoc:silent
val sampleVec: Tensor1[A, Float] = key.splitvmap(Axis[A] -> 3)((k: Key) => normalDist.sample(k))
```

A more flexible, but less performant way to create a tensor of keys and to use it together with the `vmap` or `zipvmap` method. For example, we can create a tensor of keys and use it to create a tensor of random numbers as follows:
```scala mdoc:silent
val paramTensor = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val keyTensor : Tensor1[A, Key] = key.splitToTensor(Axis[A] -> 3)

val sampleVec2 = zipvmap(Axis[A])(paramTensor, keyTensor)((param, key) => Normal(param, Tensor0(1.0f)).sample(key.item))
```
Note that in the above example, we had to use `key.item` to get the actual key from the tensor of keys, as the function passed to `zipvmap` takes as input a slice of the tensor, which is a `Tensor0` and not a `Key`.

