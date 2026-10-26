# DimWit: Type-Safe Tensor Programming Library

> **⚠️ AI-GENERATED DOCUMENTATION FOR CODING AGENTS ⚠️**
>
> **INTENDED AUDIENCE**: Large Language Models, AI Coding Agents, Automated Code Generators
>
> **NOT FOR HUMAN CONSUMPTION**: This document contains exhaustive, verbose examples optimized for machine learning model consumption. Human developers should refer to standard documentation.
>
> **AUTO-GENERATED**: This file is generated from `docs/AGENTS.md` using [mdoc](https://scalameta.org/mdoc/). All code examples are verified at compile-time and execution-time. Last generated: 2026-01-19
>
> **PURPOSE**: Provide comprehensive, executable API examples with explicit error cases to train coding agents on proper DimWit usage patterns.

---

## Table of Contents

1. [Core Concepts: Labels and Shapes](#core-concepts-labels-and-shapes)
2. [Tensor Creation](#tensor-creation)
3. [Tensor Operations](#tensor-operations)
   - [Elementwise Operations](#elementwise-operations)
   - [Reduction Operations](#reduction-operations)
   - [Broadcast Operations](#broadcast-operations)
   - [Contraction Operations](#contraction-operations)
   - [Structural Operations](#structural-operations)
4. [Functional Patterns](#functional-patterns)
5. [Automatic Differentiation](#automatic-differentiation)
6. [Training Workflows](#training-workflows)
7. [Advanced Topics](#advanced-topics)
8. [Common Error Patterns](#common-error-patterns)

---

## Core Concepts: Labels and Shapes

### Dimension Labels

DimWit uses **type-level dimension labels** to track tensor shapes at compile time. This prevents dimension mismatch errors.

```scala mdoc:silent
import dimwit.*

// Define custom dimension labels using Scala 3 derives clause
trait Batch derives Label
trait Feature derives Label
trait Hidden derives Label
trait Output derives Label

// Pre-defined test labels (available in test scope)
trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
```

### Shape Construction

Shapes are constructed using `Axis[L]` and dimension sizes:

```scala mdoc:silent
// Scalar (0-dimensional)
val shape0 = Shape.empty

// Vector (1-dimensional)
val shape1 = Shape(Axis[Feature] -> 10)

// Matrix (2-dimensional)
val shape2 = Shape(Axis[Batch] -> 32, Axis[Feature] -> 10)

// 3D Tensor
val shape3 = Shape(Axis[Batch] -> 32, Axis[Feature] -> 10, Axis[Hidden] -> 128)

// Type aliases for convenience
val s1: Shape1[Feature] = Shape1(Axis[Feature] -> 10)
val s2: Shape2[Batch, Feature] = Shape2(Axis[Batch] -> 32, Axis[Feature] -> 10)
val s3: Shape3[Batch, Feature, Hidden] = Shape3(Axis[Batch] -> 32, Axis[Feature] -> 10, Axis[Hidden] -> 128)
```

### Type Safety Guarantees

**CRITICAL**: Labels are tracked at the type level. Mismatched labels cause **compile-time errors**.

```scala mdoc:fail
// ERROR: Cannot use undefined label
trait UndefinedLabel // Missing `derives Label`
val badShape = Shape(Axis[UndefinedLabel] -> 10)
```

```scala mdoc:fail
// ERROR: Wrong axis in operation
val t = Tensor(Shape2(Axis[A] -> 3, Axis[B] -> 4)).fill(1.0f)
// Trying to sum over non-existent axis C
val wrong = t.sum(Axis[C])
```

---

## Tensor Creation

### Basic Creation with `fill`

```scala mdoc:silent
// Integer tensor
val intTensor = Tensor(Shape2(Axis[A] -> 4, Axis[B] -> 5)).fill(42)
println(s"DType: ${intTensor.dtype}") // Int32

// Float tensor
val floatTensor = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(3.14f)
println(s"DType: ${floatTensor.dtype}") // Float32

// Boolean tensor
val boolTensor = Tensor(Shape1(Axis[A] -> 10)).fill(true)
println(s"DType: ${boolTensor.dtype}") // Bool
```

### Creation from Arrays with `fromArray`

```scala mdoc:silent
// 1D tensor from array
val t1d = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(1.0f, 2.0f, 3.0f))

// 2D tensor from flattened array (row-major order)
val t2d = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 3)).fromArray(
  Array(1.0f, 2.0f, 3.0f,  // First row
        4.0f, 5.0f, 6.0f)  // Second row
)

// Using nested arrays
val t2dNested = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f, 3.0f),
    Array(4.0f, 5.0f, 6.0f)
  )
)
```

### Identity Matrices with `eye`

`eye` is a creation method on the rank 2 factory: fix the shape first, then ask for the identity matrix.

```scala mdoc:silent
// Identity matrix, both axes labelled explicitly
val eye = Tensor2(Axis[A] -> 3, Axis[B] -> 3).eye

// From a single extent: the second axis is the primed copy of the first,
// i.e. the type is Tensor2[A, Prime[A], Float32]
val primedEye = Tensor2(Axis[A] -> 3).eye

// From a shape
val eyeFromShape = Tensor2(Shape2(Axis[A] -> 3, Axis[B] -> 3)).eye

// Non-square: the diagonal stops at the shorter axis
val wideEye = Tensor2(Axis[A] -> 2, Axis[B] -> 3).eye

// Unlike fill and fromArray, eye has no values to derive the value type from.
// It defaults to Float32 and takes the value type as an argument.
val intEye = Tensor2(Axis[A] -> 3, Axis[B] -> 3).eye(VType[Int32])
```

```scala mdoc:fail
// ERROR: eye only exists on the rank 2 factory
val notAMatrix = Tensor1(Axis[A] -> 3).eye
```

### Integer Ranges with `fromRange`

`fromRange` creates a vector with the elements of a Scala `Range`, like `jnp.arange`.
The extent of the axis is the length of the range, so it is a method on the `Tensor1(Axis[L])` factory.

```scala mdoc:silent
// 0, 1, 2, 3
val range = Tensor1(Axis[A]).fromRange(0 until 4)

// 2, 3, 4, 5
val inclusive = Tensor1(Axis[A]).fromRange(2 to 5)

// 0, 3, 6
val stepped = Tensor1(Axis[A]).fromRange(0 until 7 by 3)

// 3, 2, 1
val descending = Tensor1(Axis[A]).fromRange(3 until 0 by -1)

// A Range has no value type to derive from, so fromRange defaults to Int32
// and takes the (integer) value type as an argument ...
val byteRange = Tensor1(Axis[A]).fromRange(0 until 4, VType[Int8])

// ... or from the typed factory
val shortRange = Tensor1(Axis[A], VType[Int16]).fromRange(0 until 4)
```

```scala mdoc:fail
// ERROR: fromRange only exists on the rank 1 factory
val notAVector = Tensor2(Axis[A], Axis[B]).fromRange(0 until 4)
```

```scala mdoc:fail
// ERROR: a Range holds integers, so the value type must be an integer type
val floatRange = Tensor1(Axis[A]).fromRange(0 until 4, VType[Float32])
```

**Note**: A Scala `Range` only holds integers. There is deliberately no `arange` with a floating-point step,
because accumulating a step like `0.1f` is imprecise (see the `numpy.arange` docs). Use `linspace` instead.

### Evenly Spaced Values with `linspace`

`linspace` creates a vector of `num` evenly spaced values from `start` to `stop`, like `jnp.linspace`.
The extent of the axis is `num`, so it is a method on the `Tensor1(Axis[L])` factory.
As for other computation values (`clip`, `pow`, `learningRate`, ...), `start` and `stop` are `Tensor0`s
and determine the value type. They may be traced, e.g. `x.min` and `x.max`; only `num` must be static.

```scala mdoc:silent
// 0.0, 0.25, 0.5, 0.75, 1.0
val spaced = Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 5)

// endpoint = false excludes stop: 0.0, 0.25, 0.5, 0.75
val halfOpen = Tensor1(Axis[A]).linspace(Tensor0(0.0f), Tensor0(1.0f), 4, endpoint = false)

// start > stop counts down: 1.0, 0.5, 0.0
val descendingSpaced = Tensor1(Axis[A]).linspace(Tensor0(1.0f), Tensor0(0.0f), 3)

// Data-dependent bounds, e.g. histogram bin edges
val samples = Tensor1(Axis[B]).fromArray(Array(4.0f, 2.0f, 8.0f))
val binEdges = Tensor1(Axis[A]).linspace(samples.min, samples.max, 4)

// The typed factory fixes the value type; plain literals are converted to Tensor0 of that type
val halfSpaced = Tensor1(Axis[A], VType[Float16]).linspace(0.0f, 1.0f, 5)
```

```scala mdoc:fail
// ERROR: linspace produces floating point values, so start and stop must be floating
val intSpaced = Tensor1(Axis[A]).linspace(Tensor0(0), Tensor0(1), 5)
```

### Type Aliases for Common Shapes

```scala mdoc:silent
// Scalar (0D)
val scalar: Tensor0[Float32] = Tensor0(42.0f)

// Vector (1D)
val vector: Tensor1[Feature, Float32] = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f, 3.0f))

// Matrix (2D)
val matrix: Tensor2[Batch, Feature, Float32] = Tensor2(Axis[Batch], Axis[Feature]).fromArray(
  Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
)

// 3D Tensor
val tensor3d: Tensor3[Batch, Feature, Hidden, Float32] = 
  Tensor(Shape3(Axis[Batch] -> 2, Axis[Feature] -> 3, Axis[Hidden] -> 4)).fill(0.0f)
```

### Common Creation Errors

```scala mdoc:fail
// ERROR: Missing Label instance for axis type
trait NoLabel  // No derives Label!
val noLabel = Tensor(Shape1(Axis[NoLabel] -> 2)).fill(1.0f)
```

```scala mdoc:fail
// ERROR: Axis not found in tensor shape
val wrongAxis = Tensor(Shape1(Axis[A] -> 2)).fill(1.0f)
val summed = wrongAxis.sum(Axis[B])  // B not in shape!
```

---

## Tensor Operations

### Elementwise Operations

Operations applied element-by-element to tensor(s).

```scala mdoc:reset:silent
import dimwit.*

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))

// Arithmetic
val added = t + t  // [[2, 4], [6, 8]]
val subtracted = t - t  // [[0, 0], [0, 0]]
val multiplied = t * t  // [[1, 4], [9, 16]]
val divided = t / t  // [[1, 1], [1, 1]]

// Math functions
val absolute = t.abs
val exponential = t.exp
val logarithm = (t +! Tensor0(1.0f)).log  // Avoid log(0)
val power = t.pow(Tensor0(2.0f))
val sqrt = t.sqrt
val sign = t.sign

// Trigonometric
val sine = t.sin
val cosine = t.cos
val tanh = t.tanh

// Clipping
val clipped = t.clip(Tensor0(1.5f), Tensor0(3.5f))

// Logical operations on Bool tensors (exact shape match)
val bx = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(true, false), Array(false, true)))
val by = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(true, true), Array(false, false)))

val both = bx and by  // [[true, false], [false, false]]
val either = bx or by  // [[true, true], [false, true]]
val exclusive = bx xor by  // [[false, true], [false, true]]
val negated = !bx  // [[false, true], [true, false]]

// Broadcasting variants (! suffix, as for +! / *!)
val row = Tensor1(Axis[A]).fromArray(Array(true, false))
val bothB = bx and_! row  // [[true, false], [false, false]]
val eitherB = bx or_! row  // [[true, true], [false, true]]
val exclusiveB = bx xor_! row  // [[false, true], [false, true]]

// Typical use: combine comparison masks
val inRange = (t > Tensor.like(t).fill(1.0f)) and (t < Tensor.like(t).fill(4.0f))
```

**Note**: `and` / `or` / `xor` are elementwise and always evaluate both operands - unlike Scala's
short-circuiting `&&` / `||` on `Boolean`, which is why those symbols are deliberately not provided.

### Reduction Operations

Reduce tensor along axis or to scalar.

```scala mdoc:silent
val data = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f, 3.0f),
    Array(4.0f, 5.0f, 6.0f)
  )
)

// Reduce to scalar
val totalSum: Tensor0[Float32] = data.sum
val totalMean: Tensor0[Float32] = data.mean
val totalMax: Tensor0[Float32] = data.max
val totalMin: Tensor0[Float32] = data.min
val totalStd: Tensor0[Float32] = data.std

println(s"Sum: ${totalSum.item}")  // 21.0

// Reduce along axis A (across rows)
val sumA: Tensor1[B, Float32] = data.sum(Axis[A])
println(s"Sum along A: ${sumA}")  // [5.0, 7.0, 9.0]

// Reduce along axis B (across columns)
val sumB: Tensor1[A, Float32] = data.sum(Axis[B])
println(s"Sum along B: ${sumB}")  // [6.0, 15.0]

// Mean along axes
val meanA = data.mean(Axis[A])  // [2.5, 3.5, 4.5]
val meanB = data.mean(Axis[B])  // [2.0, 5.0]

// Argmax / Argmin (returns indices)
val argmaxB: Tensor1[A, Int32] = data.argmax(Axis[B])
val argminB: Tensor1[A, Int32] = data.argmin(Axis[B])
```

**Error: Reducing on non-existent axis**

```scala mdoc:fail
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Axis[C] not in tensor
val wrong = t.sum(Axis[C])
```

### Broadcast Operations

Operations with automatic broadcasting for scalars and compatible shapes.

```scala mdoc:silent
val tensor = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

// Scalar broadcast (note: ! suffix for broadcast operations)
val scalarAdd = tensor +! Tensor0(5.0f)  // [[15, 25], [35, 45]]
val scalarMul = tensor *! Tensor0(2.0f)  // [[20, 40], [60, 80]]
val scalarDiv = tensor /! Tensor0(10.0f) // [[1, 2], [3, 4]]
val scalarSub = tensor -! Tensor0(5.0f)  // [[5, 15], [25, 35]]

// Reverse broadcast
val reverseAdd = Tensor0(100.0f) +! tensor  // [[110, 120], [130, 140]]
val reverseSub = Tensor0(100.0f) -! tensor  // [[90, 80], [70, 60]]

// Tensor0 broadcast to tensor shape
val scalarBroadcast = Tensor0(5.0f).broadcastTo(tensor.shape)

// Comparison with broadcast (element-wise comparison)
val greater = tensor > Tensor0(25.0f).broadcastTo(tensor.shape)
```

**Important**: Standard operators `+`, `-`, `*`, `/` require **exact shape match**. Use `+!`, `-!`, `*!`, `/!` for broadcasting. The same holds for the logical operators on `Bool` tensors: `and`, `or`, `xor` require an exact shape match, `and_!`, `or_!`, `xor_!` broadcast.

```scala mdoc:fail
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Shape mismatch - cannot add scalar without broadcast operator
val wrong = t + 5.0f  // Use +! instead
```

### Contraction Operations

Dot products and matrix multiplication.

```scala mdoc:reset:silent
import dimwit.*

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label

// Dot product (vector · vector)
val v1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val v2 = Tensor1(Axis[A]).fromArray(Array(4.0f, 5.0f, 6.0f))
val dotProduct: Tensor0[Float32] = v1.dot(Axis[A])(v2)
println(s"Dot product: ${dotProduct.item}")  // 32.0

// Matrix-vector multiplication
val matrix = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f),
    Array(3.0f, 4.0f)
  )
)
val vec = Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f))
val result = matrix.dot(Axis[B])(vec)
println(s"Matrix-vec result: ${result}")  // [5.0, 11.0]

// Matrix-matrix multiplication
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
val m2 = Tensor2(Axis[B], Axis[C]).fromArray(Array(Array(5.0f, 6.0f), Array(7.0f, 8.0f)))
val matmul = m1.dot(Axis[B])(m2)
```

**Error: Dimension mismatch in contraction**

```scala mdoc:fail
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f, 4.0f)))
// ERROR: No shared axis for contraction
val wrong = m1.dot(Axis[B])(m2)
```

### Structural Operations

Reshape, transpose, and dimension manipulation.

```scala mdoc:silent
val original = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f))
)

// Transpose
val transposed: Tensor2[B, A, Float32] = original.transpose
println(s"Original shape: ${original.shape}")
println(s"Transposed shape: ${transposed.shape}")

// Reshape
// Reshaping back requires fromArray with proper data

// Unsqueeze (add size-1 dimension)
val vec1d = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val vec2d = vec1d.appendAxis(Axis[B])  // Add new axis B

// Select at an index (removes the axis)
val firstRow = original.select(Axis[A].at(0))
val secondRow = original.select(Axis[A].at(1))

// Select at a range (keeps the axis)
val data3d = Tensor(Shape3(Axis[A] -> 5, Axis[B] -> 3, Axis[C] -> 4)).fill(1.0f)
val middleSelection = data3d.select(Axis[A].at(1 until 4))  // Takes indices 1, 2, 3
println(s"Middle selection shape: ${middleSelection.shape}")  // Shape(A -> 3, B -> 3, C -> 4)

// Concatenate
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
val concatenated = concatenate(t1, t2, Axis[A])
```

### Selecting and Updating

Every selection along an axis is an `Axis[L].at(...)` selector. `select` reads the selection, and applying
the tensor to the selector, `t(selector)`, is the same as `t.select(selector)`, as `seq(i)` reads a Scala collection.
`updated` returns a copy with new values at the selection, as `updated` does for Scala's immutable collections.
A selector selects along exactly one axis; to select along several axes, chain the selectors,
for reading `t(s1)(s2)` and for writing `t.focus(s1).focus(s2).updated(value)`.
`t.focus(selector)` focuses on the selection: `.updated(value)` writes it, `.updatedWith(f)` applies `f` to it
and `.get` reads it; `t.updated(selector)(value)` is short for `t.focus(selector).updated(value)`.

```scala mdoc:silent
// Shape: A=4, B=3, the value at (a, b) is 10a + b
val grid = Tensor2(Axis[A], Axis[B]).fromArray(Array.tabulate(4, 3)((a, b) => 10.0f * a + b))

// Index: removes the axis
val gridRow: Tensor1[B, Float32] = grid.select(Axis[A].at(1)) // [10, 11, 12]

// Range, sequence or tuple of indices: keeps the axis, with the number of indices as extent
val gridRows = grid.select(Axis[A].at(1 until 3)) // rows 1, 2
val gridPicked = grid.select(Axis[A].at(Seq(3, 0))) // rows 3, 0
val gridTupled = grid.select(Axis[A].at((3, 0))) // the same, note the double parentheses

// Vector of indices, like jnp.take: the axis is replaced by the axis of the indices
val order = Tensor1(Axis[C]).fromArray(Array(2, 0, 2))
val gridGathered: Tensor2[A, C, Float32] = grid.select(Axis[B].at(order))

// Window of static size, like jax.lax.dynamic_slice: keeps the axis with the window size as extent
val gridWindow = grid.select(Axis[A].at(1, 2)) // rows 1, 2

// Several axes: chain the calls
val gridEntry: Tensor0[Float32] = grid.select(Axis[A].at(1)).select(Axis[B].at(2)) // 12

// Apply: the same as select
val gridEntryApplied: Tensor0[Float32] = grid(Axis[A].at(1))(Axis[B].at(2)) // 12

// updated takes a value with the shape of the selection
val gridZeroRow = grid.updated(Axis[A].at(0))(Tensor1(Axis[B]).fromArray(Array(0.0f, 0.0f, 0.0f)))
val gridZeroWindow = grid.updated(Axis[A].at(1, 2))(Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 3)).fill(0.0f))

// Several axes: chain focus and write the value, which has the shape of grid(s1)(s2)
val gridEntryWritten = grid.focus(Axis[A].at(1)).focus(Axis[B].at(2)).updated(Tensor0(-1.0f)) // entry (1, 2) is -1
val gridBlockWritten = grid.focus(Axis[A].at(1 until 3)).focus(Axis[B].at(order)).updated(Tensor(Shape2(Axis[A] -> 2, Axis[C] -> 3)).fill(0.0f))

// updatedWith applies a function to the selection, get reads it
val gridEntryRead: Tensor0[Float32] = grid.focus(Axis[A].at(1)).focus(Axis[B].at(2)).get // 12
val gridDoubledRow = grid.focus(Axis[A].at(1)).updatedWith(_ *! Tensor0(2.0f)) // row 1 is [20, 22, 24]
```

The index of `Axis[L].at(index)` and the start of a window may also be a `Tensor0[Int32]`, also a traced one,
e.g. a loop position under `jit`. Its value is then only known at runtime, but the window size stays static:

```scala mdoc:silent
// Write a row of ones into a buffer at a position known only at runtime
val writeRow = jit((buffer: Tensor2[A, B, Float32], position: Tensor0[Int32]) =>
  buffer.updated(Axis[A].at(position, 1))(Tensor(Shape2(Axis[A] -> 1, Axis[B] -> 3)).fill(1.0f))
)
val gridWritten = writeRow(grid, Tensor0(2)) // row 2 is now ones
```

```scala mdoc:fail
// ERROR: a selector selects along one axis, chain the calls instead
val gridTuple = grid.select((Axis[A].at(1), Axis[B].at(2)))
```

```scala mdoc:fail
// ERROR: an index removes the axis, so the value must not have it
val gridWrongValue = grid.updated(Axis[A].at(0))(grid)
```

```scala mdoc:crash
// ERROR at runtime: a window with an Int start must lie inside the axis
grid.select(Axis[A].at(3, 2))
```

---

## Functional Patterns

### vmap: Vectorized Mapping

`vmap` applies a function over a specified axis, parallelizing computation.

```scala mdoc:reset:silent
import dimwit.*

trait Batch derives Label
trait Feature derives Label
val data = Tensor2(Axis[Batch], Axis[Feature]).fromArray(
  Array(
    Array(1.0f, 2.0f, 3.0f),
    Array(4.0f, 5.0f, 6.0f),
    Array(7.0f, 8.0f, 9.0f)
  )
)

// Normalize each sample (row) independently
def normalize(x: Tensor1[Feature, Float32]): Tensor1[Feature, Float32] =
  val mean = x.mean
  val std = x.std + Tensor0(1e-6f)  // Avoid division by zero
  (x -! mean) /! std

val normalized: Tensor2[Batch, Feature, Float32] = data.vmap(Axis[Batch])(normalize)
println(s"Normalized data: ${normalized}")

// Sum each row
val rowSums: Tensor1[Batch, Float32] = data.vmap(Axis[Batch])(_.sum)
println(s"Row sums: ${rowSums}")  // [6.0, 15.0, 24.0]

// vmap over columns (note: axis moves to front)
val colSums: Tensor1[Feature, Float32] = data.vmap(Axis[Feature])(_.sum)
println(s"Column sums: ${colSums}")  // [12.0, 15.0, 18.0]

// Identity vmap doesn't change data, only axis order
val identity = data.vmap(Axis[Batch])(x => x)  // Same as data
val reordered = data.vmap(Axis[Feature])(x => x)  // Same as data.transpose
```

**Error: Wrong function signature for vmap**

```scala mdoc:fail
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Function expects Tensor2, but vmap provides Tensor1
def wrongFunc(x: Tensor2[A, B, Float32]): Tensor0[Float32] = x.sum
val wrong = t.vmap(Axis[A])(wrongFunc)
```

### zipvmap: Parallel Mapping Over Multiple Tensors

```scala mdoc:reset:silent
import dimwit.*

trait A derives Label
trait B derives Label

val t1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
val t2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

// Compute L2 distance between corresponding rows
def l2Distance(v1: Tensor1[B, Float32], v2: Tensor1[B, Float32]): Tensor0[Float32] =
  (v1 - v2).pow(Tensor0(2.0f)).sum.sqrt

val distances: Tensor1[A, Float32] = zipvmap(Axis[A])(t1, t2)(l2Distance)
println(s"L2 distances: ${distances}")

// zipvmap with 4 tensors
val t3 = t1 *! Tensor0(2.0f)
val t4 = t2 /! Tensor0(2.0f)
val result = zipvmap(Axis[A])(t1, t2, t3, t4) { (a, b, c, d) =>
  a.sum + b.sum - c.sum + d.sum
}
```

### vapply: Axis-wise Function Application

Unlike `vmap`, `vapply` allows applying **different** functions to each slice.

```scala mdoc:silent
val matrix = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
)

// Normalize each row by its L2 norm
val normalized = matrix.vapply(Axis[A]) { row =>
  val norm = row.pow(Tensor0(2.0f)).sum.sqrt + Tensor0(1e-8f)
  row /! norm
}

// Identity vapply
val same = matrix.vapply(Axis[A])(identity)
```

### vreduce: Functional Reduction

```scala mdoc:reset:silent
import dimwit.*

trait A derives Label
trait B derives Label

val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)))

// Equivalent to .sum(Axis[A])
val summedA = t.vreduce(Axis[A])(_.sum)
val summedB = t.vreduce(Axis[B])(_.sum)

println(s"Reduced A: ${summedA}")
println(s"Reduced B: ${summedB}")
```

### Loops and Branches: scan, foriLoop, whileLoop, cond

A Scala loop inside `jit` is unrolled while tracing: 100 iterations become 100 copies of the body,
and the compile time grows with them. `scan`, `foriLoop`, `whileLoop` and `cond` wrap the `jax.lax`
primitives of the same names, whose body is traced once. They work eagerly and under `jit`.

The state carried from one iteration to the next may be any tensor tree (a tensor, a tuple, a case class).
Since the body is traced once, it must not change the shape of the carry.

```scala mdoc:reset:silent
import dimwit.*

trait Time derives Label
trait Feature derives Label

val signal = Tensor1(Axis[Time]).fromArray(Array(1.0f, 3.0f, 2.0f, 6.0f))

// scan: loop over the axis Time of a tensor. The body gets the carry and the slice at each
// position along Time, and returns the next carry and an output; the outputs are stacked along Time.
val (lastAverage, averages) = scan(Axis[Time])(Tensor0(0.0f), signal): (average, x) =>
  val next = average * 0.5f + x * 0.5f
  (next, next)
// averages: Tensor1[Time, Float32] = [0.5, 1.75, 1.875, 3.9375]

// scan over several tensors along the same axis, like zipvmap; () means no outputs
val weights = Tensor1(Axis[Time]).fromArray(Array(1.0f, 1.0f, 2.0f, 2.0f))
val (weightedSum, _) = scan(Axis[Time])(Tensor0(0.0f), (signal, weights)):
  case (sum, (x, weight)) => (sum + x * weight, ())

// foriLoop: loop over the indices lower until upper. The index is a traced Tensor0[Int32],
// e.g. to write a window at it
val squares = foriLoop(0, 4)(Tensor(Shape(Axis[Time] -> 4)).fill(0)): (i, buffer) =>
  buffer.updated(Axis[Time].at(i, 1))((i * i).broadcastTo(Shape(Axis[Time] -> 1)))
// squares: [0, 1, 4, 9]

// whileLoop: loop while a condition holds, here Newton's method for the square root of 2
val (root, iterations) = whileLoop((Tensor0(1.0f), Tensor0(0)))((x, _) => (x * x - Tensor0(2.0f)).abs > Tensor0(1e-5f)):
  case (x, n) => ((x + Tensor0(2.0f) / x) * 0.5f, n + Tensor0(1))

// cond: choose a branch by a predicate that may be traced, unlike Scala's if;
// both branches must return the same shapes
val relu = jit((x: Tensor0[Float32]) => cond(x > Tensor0(0.0f))(x)(Tensor0(0.0f)))
```

`scan` and `foriLoop` are differentiable, so `Autodiff.grad` through them gives the same gradient as through the
unrolled loop. JAX cannot reverse-differentiate `whileLoop`, since the number of iterations is only known at runtime;
use forward mode (`Autodiff.jacFwd`) for it instead.

```scala mdoc:fail
// ERROR: the scanned axis must exist in the tensor
val notAnAxis = scan(Axis[Feature])(Tensor0(0.0f), signal)((sum, x) => (sum + x, ()))
```

```scala mdoc
// ERROR at runtime: the body must not change the shape of the carry.
// (The error is raised inside the traced body; here only its message is shown.)
val shapeError = scala.util.Try:
  scan(Axis[Time])(Tensor1(Axis[Feature]).fromArray(Array(0.0f)), signal): (state, x) =>
    (concatenate(state, state, Axis[Feature]), ())
.failed.get.getMessage.linesIterator.map(_.trim).find(_.contains("requirement failed")).get
```

---

## Automatic Differentiation

DimWit provides automatic differentiation via `Autodiff.grad`.

### First-Order Gradients

```scala mdoc:reset:silent
import dimwit.*
import dimwit.autodiff.Autodiff

trait A derives Label
import dimwit.autodiff.Autodiff

// Scalar function: f(x) = x²
def f(x: Tensor0[Float32]): Tensor0[Float32] = x * x

val df = Autodiff.grad(f)
val x = Tensor0(3.0f)
val gradient = df(x)
println(s"df/dx at x=3: ${gradient.value.item}")  // 6.0

// Vector function: f(x) = sum(x²)
def g(x: Tensor1[A, Float32]): Tensor0[Float32] = (x * x).sum

val dg = Autodiff.grad(g)
val xVec = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val vecGradient = dg(xVec)
println(s"dg/dx: ${vecGradient}")  // [2.0, 4.0, 6.0]
```

### Higher-Order Derivatives

```scala mdoc:silent
def f2(x: Tensor0[Float32]): Tensor0[Float32] = x * x

// First derivative
val df2 = Autodiff.grad(f2)

// Second derivative
val ddf2 = Autodiff.grad((x: Tensor0[Float32]) => df2(x).value)

// Third derivative
val dddf2 = Autodiff.grad((x: Tensor0[Float32]) => ddf2(x).value)

val x2 = Tensor0(3.0f)
println(s"f''(3) = ${ddf2(x2).value.item}")   // 2.0
println(s"f'''(3) = ${dddf2(x2).value.item}") // 0.0
```

### Gradients with Multiple Parameters

```scala mdoc:silent
// f(x, y) = (x + 2y)²
def twoParam(x: Tensor1[A, Float32], y: Tensor1[A, Float32]): Tensor0[Float32] =
  ((x + (y *! Tensor0(2.0f))).pow(Tensor0(2.0f))).sum

val dtwoParam = Autodiff.grad(twoParam)

val x3 = Tensor1(Axis[A]).fromArray(Array(1.0f))
val y3 = Tensor1(Axis[A]).fromArray(Array(1.0f))

val (xGrad, yGrad) = dtwoParam(x3, y3).value
println(s"∂f/∂x = ${xGrad}")  // [6.0]
println(s"∂f/∂y = ${yGrad}")  // [12.0]
```

### Gradients with vmap

```scala mdoc:reset:silent
import dimwit.*
import dimwit.autodiff.*

trait Batch derives Label
trait Feature derives Label

// Gradient works seamlessly with vmap
def batched(x: Tensor2[Batch, Feature, Float32]): Tensor0[Float32] =
  x.vmap(Axis[Batch])(_.sum).sum

val dBatched = Autodiff.grad(batched)

val xBatch = Tensor(Shape(Axis[Batch] -> 2, Axis[Feature] -> 2)).fill(1.0f)
val batchGrad = dBatched(xBatch)
println(s"Batch gradient: ${batchGrad}")
```

### TensorTree for Parameter Structures

Use **case classes** to group parameters. DimWit automatically derives `TensorTree` instances.

```scala mdoc:reset:silent
import dimwit.*
import dimwit.autodiff.Autodiff
import dimwit.tensortree.TensorTree

trait Feature derives Label
trait Hidden derives Label

case class LinearParams(
  weight: Tensor2[Feature, Hidden, Float32],
  bias: Tensor1[Hidden, Float32]
)

// Define a model
def linear(params: LinearParams)(input: Tensor1[Feature, Float32]): Tensor1[Hidden, Float32] =
  val weighted = params.weight.transpose.dot(Axis[Feature])(input)
  weighted + params.bias

// Define loss function
def loss(data: Tensor1[Feature, Float32], target: Tensor1[Hidden, Float32])(params: LinearParams): Tensor0[Float32] =
  val prediction = linear(params)(data)
  (prediction - target).pow(Tensor0(2.0f)).sum

// Compute gradient with respect to ALL parameters
val sampleData = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f, 3.0f))
val sampleTarget = Tensor1(Axis[Hidden]).fromArray(Array(0.0f, 1.0f))
val initParams = LinearParams(
  weight = Tensor(Shape(Axis[Feature] -> 3, Axis[Hidden] -> 2)).fill(0.1f),
  bias = Tensor(Shape(Axis[Hidden] -> 2)).fill(0.0f)
)

val dLoss = Autodiff.grad(loss(sampleData, sampleTarget))
val paramGradients: LinearParams = dLoss(initParams).value
println(s"Weight gradient shape: ${paramGradients.weight.shape}")
println(s"Bias gradient shape: ${paramGradients.bias.shape}")
```

**Error: Non-differentiable types**

```scala mdoc:fail
// ERROR: Cannot differentiate with respect to Int tensors
def intFunc(x: Tensor1[A, Int32]): Tensor0[Int32] = x.sum
val wrong = Autodiff.grad(intFunc)
```

### Jacobian Matrices

```scala mdoc:reset:silent
import dimwit.*
import dimwit.autodiff.*

trait A derives Label

// Jacobian of f: R² -> R², f(x) = 2x
def linearMap(x: Tensor1[A, Float32]): Tensor1[A, Float32] = x *! Tensor0(2.0f)

val jacobian = Autodiff.jacobian(linearMap)
val xJac = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
val jacResult = jacobian(xJac)
println(s"Jacobian: ${jacResult}")  // Should be 2 * identity matrix

// jacRev and jacFwd for larger Jacobians
val jacRev = Autodiff.jacRev(linearMap)
val jacFwd = Autodiff.jacFwd(linearMap)
```

### Hessian Matrices

```scala mdoc:reset:silent
import dimwit.*
import dimwit.autodiff.*

trait A derives Label
trait B derives Label

// Hessian of f: R -> R, f(x) = x²
val hScalar = Autodiff.hessian((x: Tensor0[Float32]) => x * x)
val xScalar = Tensor0(3.0f)
println(s"Hessian of x² at x=3: ${hScalar(xScalar)}")  // 2.0

// Hessian of f: R² -> R, f(x) = sum(x²)
def sumSquares(x: Tensor1[A, Float32]): Tensor0[Float32] = (x * x).sum
val hVec = Autodiff.hessian(sumSquares)
val xVec = Tensor1(Axis[A]).fromArray(Array(1.0f, 5.0f))
println(s"Hessian of sum(x²): ${hVec(xVec)}")  // 2 * identity matrix

// Block Hessian of f: R² x R² -> R, f(x1, x2) = sum(x1 * x2)
def mixed(x1: Tensor1[A, Float32], x2: Tensor1[A, Float32]): Tensor0[Float32] =
  (x1 * x2).sum
val hBlock = Autodiff.hessian(mixed.tupled)
val x1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val x2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
val ((h_x1x1, h_x1x2), (h_x2x1, h_x2x2)) = hBlock(x1, x2)
println(s"Block Hessian shapes: ${h_x1x1.shape}, ${h_x1x2.shape}, ${h_x2x1.shape}, ${h_x2x2.shape}")
```

**Note**: `Autodiff.hessian` is only defined for scalar-output functions (`f: In => Tensor0[V]`). For vector-output functions, use `Autodiff.jacobian` instead.

---

## Training Workflows

### Gradient Descent Optimizer

```scala mdoc:reset:silent
import dimwit.*
import dimwit.optimizer.{GradientDescent, GradientOptimizer}
import dimwit.random.Random

trait Feature derives Label
trait Batch derives Label

// Define model parameters
case class SimpleModelParams(w: Tensor1[Feature, Float32], b: Tensor0[Float32])

// Define loss function
def mse(data: Tensor2[Batch, Feature, Float32], labels: Tensor1[Batch, Float32])
       (params: SimpleModelParams): Tensor0[Float32] =
  val predictions = data.vmap(Axis[Batch]) { sample =>
    sample.dot(Axis[Feature])(params.w) + params.b
  }
  ((predictions - labels).pow(Tensor0(2.0f))).mean

// Initialize parameters
val key = Random.Key(42)
val numFeatures = 5
val initW = Tensor1(Axis[Feature]).fromArray(Array.fill(numFeatures)(0.1f))
val initB = Tensor0(0.0f)
val initModelParams = SimpleModelParams(initW, initB)

// Create dummy data
val trainData = Tensor(Shape(Axis[Batch] -> 10, Axis[Feature] -> numFeatures)).fill(1.0f)
val trainLabels = Tensor1(Axis[Batch]).fromArray(Array.fill(10)(1.0f))

// Compute gradient function
val lossFunc = mse(trainData, trainLabels)
val gradFunc = Autodiff.grad(lossFunc)

// Create optimizer
val optimizer = GradientDescent(learningRate = 0.01f)

// Training loop with iterator
val trained = optimizer.iterate(initModelParams)(gradFunc)
  .take(5)  // Run 5 iterations
  .foreach { params =>
    val currentLoss = lossFunc(params)
    println(f"Loss: ${currentLoss.item}%.4f")
  }
```

### Lion Optimizer

```scala mdoc:silent
import dimwit.optimizer.Lion

// Lion optimizer with momentum
val lionOptimizer = Lion(learningRate = 1e-3f, beta1 = 0.9f, beta2 = 0.99f, weightDecay = 0.0f)

// Training with Lion
val trainedLion = lionOptimizer.iterate(initModelParams)(gradFunc)
  .drop(100)  // Skip first 100 iterations
  .take(10)   // Train for 10 more
  .toList     // Collect results
```

### Complete Training Example: Linear Regression

```scala mdoc:silent
// Define problem dimensions
trait Sample derives Label
trait InputDim derives Label

// Generate synthetic data: y = 2x + 1 + noise
val numSamples = 100
val xData = Tensor2(Axis[Sample], Axis[InputDim]).fromArray(
  Array.tabulate(numSamples, 1)((i, _) => i.toFloat / numSamples)
)
val yData = Tensor1(Axis[Sample]).fromArray(
  Array.tabulate(numSamples)(i => 2.0f * i.toFloat / numSamples + 1.0f)
)

// Model parameters
case class RegressionParams(slope: Tensor1[InputDim, Float32], intercept: Tensor0[Float32])

// Loss function (MSE)
def regressionLoss(x: Tensor2[Sample, InputDim, Float32], y: Tensor1[Sample, Float32])
                  (params: RegressionParams): Tensor0[Float32] =
  val predictions = x.vmap(Axis[Sample]) { xi =>
    xi.dot(Axis[InputDim])(params.slope) + params.intercept
  }
  ((predictions - y).pow(Tensor0(2.0f))).mean

// Initialize
val initSlope = Tensor(Shape1(Axis[InputDim] -> 1)).fill(0.0f)
val initIntercept = Tensor0(0.0f)
val initRegressionParams = RegressionParams(initSlope, initIntercept)

// Train
val regressionGrad = Autodiff.grad(regressionLoss(xData, yData))
val gdOptimizer = GradientDescent(learningRate = 0.1f)

val finalParams = gdOptimizer.iterate(initRegressionParams)(regressionGrad)
  .take(100)
  .toList
  .last

println(s"Learned slope: ${finalParams.slope}")
println(s"Learned intercept: ${finalParams.intercept}")
println(s"Expected: slope ≈ 2.0, intercept ≈ 1.0")
```

---

## Advanced Topics

### JIT Compilation

JIT (Just-In-Time) compilation speeds up repeated function calls.

```scala mdoc:reset:silent
import dimwit.*
import dimwit.jax.Jit

trait A derives Label

// Define a complex function
def complexComputation(x: Tensor1[A, Float32]): Tensor1[A, Float32] =
  val y = x.exp
  val z = y.log
  val w = z.sin
  w.pow(Tensor0(2.0f))

// JIT compile the function
val jitComplex = jit(complexComputation)

// First call: compilation overhead
val input = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val result1 = jitComplex(input)

// Subsequent calls: fast execution
val result2 = jitComplex(input *! Tensor0(2.0f))
val result3 = jitComplex(input *! Tensor0(3.0f))

println(s"JIT result: ${result1}")
```

### JIT with Donation (Memory Efficiency)

```scala mdoc:silent
import dimwit.jitDonating
import dimwit.jitDonatingUnsafe

// jitDonating allows reusing input memory
def inPlaceOp(x: Tensor1[A, Float32]): Tensor1[A, Float32] = x *! Tensor0(2.0f)
val (jitDonate, jitF, jitReclaim) = jitDonating(inPlaceOp)
val inputDonate = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val donated = jitDonate(inputDonate)
val resultDonate = jitReclaim(jitF(donated))
// Note: inputDonate may be invalidated after jitDonate call

// Unsafe variant (more convenient but risky)
val jitUnsafe = jitDonatingUnsafe(inPlaceOp)
```

### Random Number Generation

DimWit uses **functional** random keys (inspired by JAX).

```scala mdoc:silent
import dimwit.random.Random
import dimwit.stats.{Normal, Uniform}

// Create root key
val rootKey = Random.Key(42)

// Split key for independent random streams
val (key1, key2) = rootKey.split2()
val keys = rootKey.split(5)  // Split into 5 keys

// Generate random numbers
val normalDist = Normal(loc = Tensor0(0.0f), scale = Tensor0(1.0f))
val randomSample = normalDist.sample(key1)
println(s"Normal samples: ${randomSample}")

val uniformDist = Uniform(low = Tensor0(0.0f), high = Tensor0(1.0f))
val uniformSample = uniformDist.sample(key2)
println(s"Uniform samples: ${uniformSample}")

// Permutations
val (permKey, _) = rootKey.split2()
val shuffled = Random.permutation(Axis[A] -> 5)(permKey)
println(s"Shuffled: ${shuffled}")
```

### Random Key Splitting with vmap

```scala mdoc:silent
trait B derives Label

// Split keys in parallel
val batchKey = Random.Key(123)
val batchKeys = batchKey.splitvmap(Axis[A] -> 8)(k => Normal.standardSample(k))
println(s"Batch keys shape: ${batchKeys.shape}")

```

---

## Common Error Patterns

This section demonstrates **compile-time** and **runtime** errors to help coding agents avoid common mistakes.

### Type Constraint Violations

```scala mdoc:reset
import dimwit.*

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
```

```scala mdoc:fail
// ERROR: Cannot perform floating-point operations on Int tensors
val intTensor = Tensor1(Axis[A]).fromArray(Array(1, 2, 3))
val wrong = intTensor.exp  // exp requires IsFloating constraint
```

```scala mdoc:fail
// ERROR: Cannot compute mean of Boolean tensor
val boolTensor = Tensor1(Axis[A]).fromArray(Array(true, false, true))
val wrong = boolTensor.mean
```

### Shape Mismatches

```scala mdoc:fail
// ERROR: Incompatible shapes for elementwise operation
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f, 5.0f))
val wrong = t1 + t2  // Different labels AND different sizes
```

```scala mdoc:fail
// ERROR: Matrix dimensions don't align for multiplication
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))  // Shape: (1, 2)
val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f), Array(4.0f)))  // Shape: (2, 1)
val wrong = m1.dot(Axis[B])(m2)  // Axis[B] not in m2
```

### Broadcast vs Non-Broadcast Confusion

```scala mdoc:fail
// ERROR: Forgot broadcast operator
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val wrong = t + 10.0f  // Should use +! for scalar broadcast
```

```scala mdoc:fail
// ERROR: Used broadcast when shapes already match
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
// This works but is semantically wrong (use + instead)
val wrong = t1 +! t2
```

### Axis Errors

```scala mdoc:fail
// ERROR: Reducing on non-existent axis
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val wrong = t.sum(Axis[C])  // Axis[C] not in tensor
```

```scala mdoc:fail
// ERROR: vmap with wrong axis
val t = Tensor2(Axis[A], Axis[B]).fill(1.0f)
val wrong = t.vmap(Axis[C])(_.sum)  // Axis[C] doesn't exist
```

### Gradient Errors

```scala mdoc:fail
// ERROR: Cannot differentiate Integer functions
def intFunc(x: Tensor0[Int32]): Tensor0[Int32] = x + x
val wrong = Autodiff.grad(intFunc)
```

```scala mdoc:fail
// ERROR: Function doesn't return scalar for grad
def nonScalar(x: Tensor1[A, Float32]): Tensor1[A, Float32] = x * x
val wrong = Autodiff.grad(nonScalar)  // Use jacobian instead
```

### Device Mismatches

```scala mdoc:fail
// ERROR: Operations between CPU and GPU tensors (when GPUs available)
// val cpuTensor = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
// val gpuTensor = cpuTensor.toDevice(Device.GPU(0))
// val wrong = cpuTensor + gpuTensor  // Device mismatch
```

---

## Best Practices for Coding Agents

### 1. Always Use Explicit Labels

```scala mdoc:reset:silent
import dimwit.*

// GOOD: Semantic labels
trait Batch derives Label
trait SeqLen derives Label
trait EmbedDim derives Label

val embeddings = Tensor(Shape3(Axis[Batch] -> 8, Axis[SeqLen] -> 128, Axis[EmbedDim] -> 64)).fill(0.0f)

// AVOID: Generic A, B, C in production code
```

### 2. Use Type Aliases for Clarity

```scala mdoc:silent
trait Feature derives Label

// GOOD: Clear type signatures
def process(input: Tensor2[Batch, Feature, Float32]): Tensor1[Batch, Float32] = 
  input.vmap(Axis[Batch])(_.sum)

// AVOID: Opaque Tensor types without explicit parameters
```

### 3. Prefer Case Classes for Parameters

```scala mdoc:silent
trait InputDim derives Label
trait Hidden derives Label
trait Output derives Label

// GOOD: Structured parameters with TensorTree
case class ModelParams(
  encoder: Tensor2[InputDim, Hidden, Float32],
  decoder: Tensor2[Hidden, Output, Float32],
  bias: Tensor1[Output, Float32]
)

// AVOID: Tuples or loose parameters
```

### 4. Use Broadcast Operators Explicitly

```scala mdoc:silent
trait Sample derives Label

val data = Tensor1(Axis[Sample]).fromArray(Array(1.0f, 2.0f, 3.0f))
val mean = data.mean
val std = data.std

// GOOD: Clear intent
val normalized = (data -! mean) /! std

// AVOID: Mixing broadcast and non-broadcast without clarity
```

### 5. Leverage Functional Random Keys

```scala mdoc:silent
import dimwit.random.Random
import dimwit.stats.{Normal, Uniform}

trait A derives Label

val rootKey = Random.Key(42)
val normalDist = Normal(loc = Tensor0(0.0f), scale = Tensor0(1.0f))
val uniformDist = Uniform(low = Tensor0(0.0f), high = Tensor0(1.0f))

// GOOD: Explicit key threading
val (key1, key2) = rootKey.split2()
val sample1 = normalDist.sample(key1)
val sample2 = uniformDist.sample(key2)

// AVOID: Stateful random number generators
```

### 6. JIT Compile Performance-Critical Functions

```scala mdoc:silent
import dimwit.jax.Jit.jit

trait Input derives Label

val simpleFunc = (x: Tensor1[Input, Float32]) => x *! Tensor0(2.0f)

// GOOD: JIT for repeated calls
val jitFunc = jit(simpleFunc)
val testData = Tensor1(Axis[Input]).fromArray(Array(1.0f, 2.0f, 3.0f))
val predictions = jitFunc(testData)

// AVOID: JIT for one-off computations (overhead exceeds benefit)
```

---

## Summary

This document provides comprehensive, executable examples of DimWit's API designed for AI/coding agent consumption:

1. **Type-safe dimension tracking** prevents shape errors at compile time
2. **Functional patterns** (vmap, vapply, zipvmap) enable elegant vectorized code
3. **Automatic differentiation** with TensorTree supports complex parameter structures
4. **Gradient-based optimization** with functional state management
5. **Advanced features** (JIT, functional random keys) for performance and reproducibility

**Key Differentiators from NumPy/PyTorch**:
- Compile-time dimension checking
- Labeled dimensions (not integer indices)
- Functional API (no mutation)
- JAX backend (XLA compilation, GPU support)

**For AI agents**: Use error showcases to understand type constraints. Prefer case classes over tuples. Thread random keys explicitly. Use `!` suffix for broadcast operations.

---

*End of AI-generated documentation. Generated: 2026-01-19*
