# Quickstart

```scala mdoc:silent
import dimwit.*
```

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