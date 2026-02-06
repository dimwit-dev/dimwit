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

```scala
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

```scala
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

```scala
// ERROR: Cannot use undefined label
trait UndefinedLabel // Missing `derives Label`
val badShape = Shape(Axis[UndefinedLabel] -> 10)
// error:
// 
// An axis label repl.MdocSession.MdocApp.UndefinedLabel was given or inferred, which does not have a Label instance.
// Ensure that all axis types repl.MdocSession.MdocApp.UndefinedLabel are defined with 'derives Label' (e.g. 'trait T derives Label')
// 
// val badShape = Shape(Axis[UndefinedLabel] -> 10)
//                                                ^
```

```scala
// ERROR: Wrong axis in operation
val t = Tensor(Shape2(Axis[A] -> 3, Axis[B] -> 4)).fill(1.0f)
// Trying to sum over non-existent axis C
val wrong = t.sum(Axis[C])
// error:
// Axis[repl.MdocSession.MdocApp.C] not found in Tensor[(repl.MdocSession.MdocApp.A, repl.MdocSession.MdocApp.B)].
// I found:
// 
//     dimwit.tensor.ShapeTypeHelpers.AxisRemover.bridge[
//       (repl.MdocSession.MdocApp.A, repl.MdocSession.MdocApp.B),
//       repl.MdocSession.MdocApp.C, Tuple](
//       dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[repl.MdocSession.MdocApp.A,
//         repl.MdocSession.MdocApp.B *: EmptyTuple.type, repl.MdocSession.MdocApp.C](
//         dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[repl.MdocSession.MdocApp.B,
//           EmptyTuple.type, repl.MdocSession.MdocApp.C](
//           dimwit.tensor.ShapeTypeHelpers.AxisIndex.concatRight[A², B², L])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.ShapeTypeHelpers.AxisIndex[EmptyTuple.type,
//   repl.MdocSession.MdocApp.C]
// 
// where:    A  is a trait in object MdocApp
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in object MdocApp
//           B² is a type variable with constraint <: Tuple
// .
// val wrong = t.sum(Axis[C])
//                          ^
```

---

## Tensor Creation

### Basic Creation with `fill`

```scala
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

```scala
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

### Type Aliases for Common Shapes

```scala
// Scalar (0D)
val scalar: Tensor0[Float] = Tensor0(42.0f)

// Vector (1D)
val vector: Tensor1[Feature, Float] = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f, 3.0f))

// Matrix (2D)
val matrix: Tensor2[Batch, Feature, Float] = Tensor2(Axis[Batch], Axis[Feature]).fromArray(
  Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
)

// 3D Tensor
val tensor3d: Tensor3[Batch, Feature, Hidden, Float] = 
  Tensor(Shape3(Axis[Batch] -> 2, Axis[Feature] -> 3, Axis[Hidden] -> 4)).fill(0.0f)
```

### Common Creation Errors

```scala
// ERROR: Missing Label instance for axis type
trait NoLabel  // No derives Label!
val noLabel = Tensor(Shape1(Axis[NoLabel] -> 2)).fill(1.0f)
// error:
// 
// An axis label repl.MdocSession.MdocApp.NoLabel was given or inferred, which does not have a Label instance.
// Ensure that all axis types repl.MdocSession.MdocApp.NoLabel are defined with 'derives Label' (e.g. 'trait T derives Label')
// 
// val noLabel = Tensor(Shape1(Axis[NoLabel] -> 2)).fill(1.0f)
//                                                ^
```

```scala
// ERROR: Axis not found in tensor shape
val wrongAxis = Tensor(Shape1(Axis[A] -> 2)).fill(1.0f)
val summed = wrongAxis.sum(Axis[B])  // B not in shape!
// error:
// Axis[repl.MdocSession.MdocApp.B] not found in Tensor[Tuple1[repl.MdocSession.MdocApp.A]].
// I found:
// 
//     dimwit.tensor.ShapeTypeHelpers.AxisRemover.bridge[
//       Tuple1[repl.MdocSession.MdocApp.A], repl.MdocSession.MdocApp.B, Tuple](
//       dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[repl.MdocSession.MdocApp.A,
//         EmptyTuple.type, repl.MdocSession.MdocApp.B](
//         dimwit.tensor.ShapeTypeHelpers.AxisIndex.concatRight[A², B², L]),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.ShapeTypeHelpers.AxisIndex[EmptyTuple.type,
//   repl.MdocSession.MdocApp.B]
// 
// where:    A  is a trait in object MdocApp
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in object MdocApp
//           B² is a type variable with constraint <: Tuple
// .
// val summed = wrongAxis.sum(Axis[B])  // B not in shape!
//                                   ^
```

---

## Tensor Operations

### Elementwise Operations

Operations applied element-by-element to tensor(s).

```scala
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
```

### Reduction Operations

Reduce tensor along axis or to scalar.

```scala
val data = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f, 3.0f),
    Array(4.0f, 5.0f, 6.0f)
  )
)

// Reduce to scalar
val totalSum: Tensor0[Float] = data.sum
val totalMean: Tensor0[Float] = data.mean
val totalMax: Tensor0[Float] = data.max
val totalMin: Tensor0[Float] = data.min
val totalStd: Tensor0[Float] = data.std

println(s"Sum: ${totalSum.item}")  // 21.0

// Reduce along axis A (across rows)
val sumA: Tensor1[B, Float] = data.sum(Axis[A])
println(s"Sum along A: ${sumA}")  // [5.0, 7.0, 9.0]

// Reduce along axis B (across columns)
val sumB: Tensor1[A, Float] = data.sum(Axis[B])
println(s"Sum along B: ${sumB}")  // [6.0, 15.0]

// Mean along axes
val meanA = data.mean(Axis[A])  // [2.5, 3.5, 4.5]
val meanB = data.mean(Axis[B])  // [2.0, 5.0]

// Argmax / Argmin (returns indices)
val argmaxB: Tensor1[A, Int] = data.argmax(Axis[B])
val argminB: Tensor1[A, Int] = data.argmin(Axis[B])
```

**Error: Reducing on non-existent axis**

```scala
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Axis[C] not in tensor
val wrong = t.sum(Axis[C])
// error:
// Axis[MdocApp0.this.C] not found in Tensor[(MdocApp0.this.A, MdocApp0.this.B)].
// I found:
// 
//     dimwit.tensor.ShapeTypeHelpers.AxisRemover.bridge[
//       (MdocApp0.this.A, MdocApp0.this.B), MdocApp0.this.C, Tuple](
//       dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp0.this.A,
//         MdocApp0.this.B *: EmptyTuple.type, MdocApp0.this.C](
//         dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp0.this.B,
//           EmptyTuple.type, MdocApp0.this.C](
//           dimwit.tensor.ShapeTypeHelpers.AxisIndex.concatRight[A², B², L])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.ShapeTypeHelpers.AxisIndex[EmptyTuple.type, MdocApp0.this.C]
// 
// where:    A  is a trait in class MdocApp0
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in class MdocApp0
//           B² is a type variable with constraint <: Tuple
// .
// val wrong = t.sum(Axis[C])
//                          ^
// error: 
// Conflicting definitions:
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 53 and
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 88
//
```

### Broadcast Operations

Operations with automatic broadcasting for scalars and compatible shapes.

```scala
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

**Important**: Standard operators `+`, `-`, `*`, `/` require **exact shape match**. Use `+!`, `-!`, `*!`, `/!` for broadcasting.

```scala
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Shape mismatch - cannot add scalar without broadcast operator
val wrong = t + 5.0f  // Use +! instead
// error: 
// Found:    (5.0f : Float)
// Required: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float]
// error: 
// Conflicting definitions:
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 53 and
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 97
//
```

### Contraction Operations

Dot products and matrix multiplication.

```scala
import dimwit.*

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label

// Dot product (vector · vector)
val v1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val v2 = Tensor1(Axis[A]).fromArray(Array(4.0f, 5.0f, 6.0f))
val dotProduct: Tensor0[Float] = v1.dot(Axis[A])(v2)
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

```scala
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f, 4.0f)))
// ERROR: No shared axis for contraction
val wrong = m1.dot(Axis[B])(m2)
// error: 
// Axis[MdocApp1.this.B] not found in Tensor[(MdocApp1.this.C, MdocApp1.this.D)].
// I found:
// 
//     dimwit.tensor.ShapeTypeHelpers.AxisRemover.bridge[
//       (MdocApp1.this.C, MdocApp1.this.D), MdocApp1.this.B, Tuple](
//       dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp1.this.C,
//         MdocApp1.this.D *: EmptyTuple.type, MdocApp1.this.B](
//         dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp1.this.D,
//           EmptyTuple.type, MdocApp1.this.B](
//           dimwit.tensor.ShapeTypeHelpers.AxisIndex.concatRight[A, B², L])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.ShapeTypeHelpers.AxisIndex[EmptyTuple.type, MdocApp1.this.B]
// 
// where:    B  is a trait in class MdocApp1
//           B² is a type variable with constraint <: Tuple
// .
// error:
// Axis[MdocApp1.this.B] not found in Tensor[(MdocApp1.this.C, MdocApp1.this.D)].
// I found:
// 
//     dimwit.tensor.ShapeTypeHelpers.AxisRemover.bridge[
//       (MdocApp1.this.C, MdocApp1.this.D), MdocApp1.this.B, Tuple](
//       dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp1.this.C,
//         MdocApp1.this.D *: EmptyTuple.type, MdocApp1.this.B](
//         dimwit.tensor.ShapeTypeHelpers.AxisIndex.tail[MdocApp1.this.D,
//           EmptyTuple.type, MdocApp1.this.B](
//           dimwit.tensor.ShapeTypeHelpers.AxisIndex.concatRight[A, B², L])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.ShapeTypeHelpers.AxisIndex[EmptyTuple.type, MdocApp1.this.B]
// 
// where:    B  is a trait in class MdocApp1
//           B² is a type variable with constraint <: Tuple
// .
// val wrong = m1.dot(Axis[B])(m2)
//                               ^
// error: 
// Conflicting definitions:
// val m1: dimwit.tensor.Tensor[(MdocApp1.this.A, MdocApp1.this.B), Float] in class MdocApp1 at line 119 and
// val m1: dimwit.tensor.Tensor[(MdocApp1.this.A, MdocApp1.this.B), Float] in class MdocApp1 at line 122
// 
// error: 
// Conflicting definitions:
// val m2: dimwit.tensor.Tensor[(MdocApp1.this.B, MdocApp1.this.C), Float] in class MdocApp1 at line 120 and
// val m2: dimwit.tensor.Tensor[(MdocApp1.this.C, MdocApp1.this.D), Float] in class MdocApp1 at line 123
//
```

### Structural Operations

Reshape, transpose, and dimension manipulation.

```scala
val original = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f))
)

// Transpose
val transposed: Tensor2[B, A, Float] = original.transpose
println(s"Original shape: ${original.shape}")
println(s"Transposed shape: ${transposed.shape}")

// Reshape
// Reshaping back requires fromArray with proper data

// Unsqueeze (add size-1 dimension)
val vec1d = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val vec2d = vec1d.appendAxis(Axis[B])  // Add new axis B

// Take (indexing with single index)
val firstRow = original.slice(Axis[A].at(0))
val secondRow = original.slice(Axis[A].at(1))

// Take (indexing with range)
val data3d = Tensor(Shape3(Axis[A] -> 5, Axis[B] -> 3, Axis[C] -> 4)).fill(1.0f)
val middleSlice = data3d.slice(Axis[A].at(1 until 4))  // Takes indices 1, 2, 3
println(s"Middle slice shape: ${middleSlice.shape}")  // Shape(A -> 3, B -> 3, C -> 4)

// Concatenate
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
val concatenated = concatenate(t1, t2, Axis[A])
```

---

## Functional Patterns

### vmap: Vectorized Mapping

`vmap` applies a function over a specified axis, parallelizing computation.

```scala
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
def normalize(x: Tensor1[Feature, Float]): Tensor1[Feature, Float] =
  val mean = x.mean
  val std = x.std + Tensor0(1e-6f)  // Avoid division by zero
  (x -! mean) /! std

val normalized: Tensor2[Batch, Feature, Float] = data.vmap(Axis[Batch])(normalize)
println(s"Normalized data: ${normalized}")

// Sum each row
val rowSums: Tensor1[Batch, Float] = data.vmap(Axis[Batch])(_.sum)
println(s"Row sums: ${rowSums}")  // [6.0, 15.0, 24.0]

// vmap over columns (note: axis moves to front)
val colSums: Tensor1[Feature, Float] = data.vmap(Axis[Feature])(_.sum)
println(s"Column sums: ${colSums}")  // [12.0, 15.0, 18.0]

// Identity vmap doesn't change data, only axis order
val identity = data.vmap(Axis[Batch])(x => x)  // Same as data
val reordered = data.vmap(Axis[Feature])(x => x)  // Same as data.transpose
```

**Error: Wrong function signature for vmap**

```scala
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Function expects Tensor2, but vmap provides Tensor1
def wrongFunc(x: Tensor2[A, B, Float]): Tensor0[Float] = x.sum
val wrong = t.vmap(Axis[A])(wrongFunc)
// error: 
// Not found: type A
// error: 
// Not found: type B
// error: 
// 
// An axis label Any was given or inferred, which does not have a Label instance.
// Ensure that all axis types Any are defined with 'derives Label' (e.g. 'trait T derives Label')
// 
// error:
// Not found: type A
// def wrongFunc(x: Tensor2[A, B, Float]): Tensor0[Float] = x.sum
//                          ^
// error:
// Not found: type B
// def wrongFunc(x: Tensor2[A, B, Float]): Tensor0[Float] = x.sum
//                             ^
```

### zipvmap: Parallel Mapping Over Multiple Tensors

```scala
import dimwit.*

trait A derives Label
trait B derives Label

val t1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
val t2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))

// Compute L2 distance between corresponding rows
def l2Distance(v1: Tensor1[B, Float], v2: Tensor1[B, Float]): Tensor0[Float] =
  (v1 - v2).pow(Tensor0(2.0f)).sum.sqrt

val distances: Tensor1[A, Float] = zipvmap(Axis[A])(t1, t2)(l2Distance)
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

```scala
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

```scala
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

---

## Automatic Differentiation

DimWit provides automatic differentiation via `Autodiff.grad`.

### First-Order Gradients

```scala
import dimwit.*
import dimwit.autodiff.Autodiff

trait A derives Label
import dimwit.autodiff.Autodiff

// Scalar function: f(x) = x²
def f(x: Tensor0[Float]): Tensor0[Float] = x * x

val df = Autodiff.grad(f)
val x = Tensor0(3.0f)
val gradient = df(x)
println(s"df/dx at x=3: ${gradient.value.item}")  // 6.0

// Vector function: f(x) = sum(x²)
def g(x: Tensor1[A, Float]): Tensor0[Float] = (x * x).sum

val dg = Autodiff.grad(g)
val xVec = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
val vecGradient = dg(xVec)
println(s"dg/dx: ${vecGradient}")  // [2.0, 4.0, 6.0]
```

### Higher-Order Derivatives

```scala
def f2(x: Tensor0[Float]): Tensor0[Float] = x * x

// First derivative
val df2 = Autodiff.grad(f2)

// Second derivative
val ddf2 = Autodiff.grad((x: Tensor0[Float]) => df2(x).value)

// Third derivative
val dddf2 = Autodiff.grad((x: Tensor0[Float]) => ddf2(x).value)

val x2 = Tensor0(3.0f)
println(s"f''(3) = ${ddf2(x2).value.item}")   // 2.0
println(s"f'''(3) = ${dddf2(x2).value.item}") // 0.0
```

### Gradients with Multiple Parameters

```scala
// f(x, y) = (x + 2y)²
def twoParam(x: Tensor1[A, Float], y: Tensor1[A, Float]): Tensor0[Float] =
  ((x + (y *! Tensor0(2.0f))).pow(Tensor0(2.0f))).sum

val dtwoParam = Autodiff.grad(twoParam)

val x3 = Tensor1(Axis[A]).fromArray(Array(1.0f))
val y3 = Tensor1(Axis[A]).fromArray(Array(1.0f))

val (xGrad, yGrad) = dtwoParam(x3, y3).value
println(s"∂f/∂x = ${xGrad}")  // [6.0]
println(s"∂f/∂y = ${yGrad}")  // [12.0]
```

### Gradients with vmap

```scala
import dimwit.*
import dimwit.autodiff.*

trait Batch derives Label
trait Feature derives Label

// Gradient works seamlessly with vmap
def batched(x: Tensor2[Batch, Feature, Float]): Tensor0[Float] =
  x.vmap(Axis[Batch])(_.sum).sum

val dBatched = Autodiff.grad(batched)

val xBatch = Tensor(Shape(Axis[Batch] -> 2, Axis[Feature] -> 2)).fill(1.0f)
val batchGrad = dBatched(xBatch)
println(s"Batch gradient: ${batchGrad}")
```

### TensorTree for Parameter Structures

Use **case classes** to group parameters. DimWit automatically derives `TensorTree` instances.

```scala
import dimwit.*
import dimwit.autodiff.{TensorTree, FloatTensorTree, Autodiff}

trait Feature derives Label
trait Hidden derives Label

case class LinearParams(
  weight: Tensor2[Feature, Hidden, Float],
  bias: Tensor1[Hidden, Float]
)

// Define a model
def linear(params: LinearParams)(input: Tensor1[Feature, Float]): Tensor1[Hidden, Float] =
  val weighted = params.weight.transpose.dot(Axis[Feature])(input)
  weighted + params.bias

// Define loss function
def loss(data: Tensor1[Feature, Float], target: Tensor1[Hidden, Float])(params: LinearParams): Tensor0[Float] =
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

```scala
// ERROR: Cannot differentiate with respect to Int tensors
def intFunc(x: Tensor1[A, Int]): Tensor0[Int] = x.sum
val wrong = Autodiff.grad(intFunc)
// error:
// Not found: type A
// def intFunc(x: Tensor1[A, Int]): Tensor0[Int] = x.sum
//                        ^
// error:
// No given instance of type dimwit.autodiff.ToPyTree[dimwit.tensor.Tensor1[<error Not found: type A>, Int]] was found for parameter inTree of method grad in object Autodiff
// val wrong = Autodiff.grad(intFunc)
//                                  ^
```

### Jacobian Matrices

```scala
import dimwit.*
import dimwit.autodiff.*

trait A derives Label

// Jacobian of f: R² -> R², f(x) = 2x
def linearMap(x: Tensor1[A, Float]): Tensor1[A, Float] = x *! Tensor0(2.0f)

val jacobian = Autodiff.jacobian(linearMap)
val xJac = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
val jacResult = jacobian(xJac)
println(s"Jacobian: ${jacResult}")  // Should be 2 * identity matrix

// jacRev and jacFwd for larger Jacobians
val jacRev = Autodiff.jacRev(linearMap)
val jacFwd = Autodiff.jacFwd(linearMap)
```

---

## Training Workflows

### Gradient Descent Optimizer

```scala
import dimwit.*
import nn.{GradientDescent, GradientOptimizer}
import dimwit.random.Random

trait Feature derives Label
trait Batch derives Label

// Define model parameters
case class SimpleModelParams(w: Tensor1[Feature, Float], b: Tensor0[Float])

// Define loss function
def mse(data: Tensor2[Batch, Feature, Float], labels: Tensor1[Batch, Float])
       (params: SimpleModelParams): Tensor0[Float] =
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
val optimizer = GradientDescent(learningRate = Tensor0(0.01f))

// Training loop with iterator
val trained = optimizer.iterate(initModelParams)(gradFunc)
  .take(5)  // Run 5 iterations
  .foreach { params =>
    val currentLoss = lossFunc(params)
    println(f"Loss: ${currentLoss.item}%.4f")
  }
```

### Lion Optimizer

```scala
import nn.Lion

// Lion optimizer with momentum
val lionOptimizer = Lion(learningRate = Tensor0(1e-3f), beta1 = Tensor0(0.9f), beta2 = Tensor0(0.99f), weightDecay = Tensor0(0.0f))

// Training with Lion
val trainedLion = lionOptimizer.iterate(initModelParams)(gradFunc)
  .drop(100)  // Skip first 100 iterations
  .take(10)   // Train for 10 more
  .toList     // Collect results
```

### Complete Training Example: Linear Regression

```scala
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
case class RegressionParams(slope: Tensor1[InputDim, Float], intercept: Tensor0[Float])

// Loss function (MSE)
def regressionLoss(x: Tensor2[Sample, InputDim, Float], y: Tensor1[Sample, Float])
                  (params: RegressionParams): Tensor0[Float] =
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
val gdOptimizer = GradientDescent(learningRate = Tensor0(0.1f))

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

```scala
import dimwit.*
import dimwit.jax.Jit

trait A derives Label

// Define a complex function
def complexComputation(x: Tensor1[A, Float]): Tensor1[A, Float] =
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

```scala
import dimwit.jitDonating
import dimwit.jitDonatingUnsafe

// jitDonating allows reusing input memory
def inPlaceOp(x: Tensor1[A, Float]): Tensor1[A, Float] = x *! Tensor0(2.0f)
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

```scala
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

```scala
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

```scala
import dimwit.*

trait A derives Label
trait B derives Label
trait C derives Label
trait D derives Label
// ERROR: Cannot perform floating-point operations on Int tensors
val intTensor = Tensor1(Axis[A]).fromArray(Array(1, 2, 3))
val wrong = intTensor.exp  // exp requires IsFloating constraint
// error: 
// value exp is not a member of dimwit.tensor.Tensor[Tuple1[MdocApp11.this.A], Int].
// An extension method was tried, but could not be fully constructed:
// 
//     dimwit.exp[Tuple1[MdocApp11.this.A], Int](this.intTensor)(
//       dimwit.tensor.Labels.concat[MdocApp11.this.A, EmptyTuple.type](
//         this.A.derived$Label, dimwit.tensor.Labels.namesOfEmpty),
//       /* missing */summon[dimwit.tensor.TensorOps.IsFloating[Int]])
// 
//     failed with:
// 
//         Operation only valid for Floating tensors.
```

```scala
// ERROR: Cannot compute mean of Boolean tensor
val boolTensor = Tensor1(Axis[A]).fromArray(Array(true, false, true))
val wrong = boolTensor.mean
// error: 
// Not found: Tensor1
// error: 
// Not found: type A
// error: 
// Not found: Axis
```

### Shape Mismatches

```scala
// ERROR: Incompatible shapes for elementwise operation
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[B]).fromArray(Array(3.0f, 4.0f, 5.0f))
val wrong = t1 + t2  // Different labels AND different sizes
// error: 
// Not found: Tensor1
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error:
// Not found: Tensor1
// val wrong = t1 + t2  // Different labels AND different sizes
//                     ^
// error: 
// Not found: type B
// error: 
// Not found: Axis
```

```scala
// ERROR: Matrix dimensions don't align for multiplication
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))  // Shape: (1, 2)
val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f), Array(4.0f)))  // Shape: (2, 1)
val wrong = m1.dot(Axis[B])(m2)  // Axis[B] not in m2
// error: 
// Not found: Tensor2
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error: 
// Not found: type B
// error: 
// Not found: Axis
// error: 
// Not found: Tensor2
// error: 
// Not found: type C
// error: 
// Not found: Axis
// error: 
// Not found: type D
// error: 
// Not found: Axis
```

### Broadcast vs Non-Broadcast Confusion

```scala
// ERROR: Forgot broadcast operator
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val wrong = t + 10.0f  // Should use +! for scalar broadcast
// error: 
// Not found: Tensor2
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error: 
// Not found: type B
// error: 
// Not found: Axis
```

```scala
// ERROR: Used broadcast when shapes already match
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
val t2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
// This works but is semantically wrong (use + instead)
// val result = t1 +! t2  // Compiles but misleading
// error: 
// Not found: Tensor1
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error: 
// Not found: Tensor1
// error: 
// Not found: type A
// error: 
// Not found: Axis
```

### Axis Errors

```scala
// ERROR: Reducing on non-existent axis
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val wrong = t.sum(Axis[C])  // Axis[C] not in tensor
// error: 
// Not found: Tensor2
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error: 
// Not found: type B
// error: 
// Not found: Axis
```

```scala
// ERROR: vmap with wrong axis
val t = Tensor2(Axis[A], Axis[B]).fill(1.0f)
val wrong = t.vmap(Axis[C])(_.sum)  // Axis[C] doesn't exist
// error: 
// Not found: Tensor2
// error: 
// Not found: type A
// error: 
// Not found: Axis
// error: 
// Not found: type B
// error: 
// Not found: Axis
```

### Gradient Errors

```scala
// ERROR: Cannot differentiate Integer functions
def intFunc(x: Tensor0[Int]): Tensor0[Int] = x + x
val wrong = Autodiff.grad(intFunc)
// error:
// Not found: type Tensor0
// def intFunc(x: Tensor0[Int]): Tensor0[Int] = x + x
//                ^^^^^^^
// error:
// Not found: type Tensor0
// def intFunc(x: Tensor0[Int]): Tensor0[Int] = x + x
//                               ^^^^^^^
// error:
// Not found: Autodiff
// val wrong = Autodiff.grad(intFunc)
//             ^^^^^^^^
```

```scala
// ERROR: Function doesn't return scalar for grad
def nonScalar(x: Tensor1[A, Float]): Tensor1[A, Float] = x * x
val wrong = Autodiff.grad(nonScalar)  // Use jacobian instead
// error: 
// Not found: type Tensor1
// error: 
// Not found: type A
// error:
// Not found: type Tensor1
// val wrong = Autodiff.grad(nonScalar)  // Use jacobian instead
//                                     ^
// error: 
// Not found: type A
// error: 
// Not found: Autodiff
```

### Device Mismatches

```scala
// ERROR: Operations between CPU and GPU tensors (when GPUs available)
// val cpuTensor = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
// val gpuTensor = cpuTensor.toDevice(Device.GPU(0))
// val wrong = cpuTensor + gpuTensor  // Device mismatch
```

---

## Best Practices for Coding Agents

### 1. Always Use Explicit Labels

```scala
import dimwit.*

// GOOD: Semantic labels
trait Batch derives Label
trait SeqLen derives Label
trait EmbedDim derives Label

val embeddings = Tensor(Shape3(Axis[Batch] -> 8, Axis[SeqLen] -> 128, Axis[EmbedDim] -> 64)).fill(0.0f)

// AVOID: Generic A, B, C in production code
```

### 2. Use Type Aliases for Clarity

```scala
trait Feature derives Label

// GOOD: Clear type signatures
def process(input: Tensor2[Batch, Feature, Float]): Tensor1[Batch, Float] = 
  input.vmap(Axis[Batch])(_.sum)

// AVOID: Opaque Tensor types without explicit parameters
```

### 3. Prefer Case Classes for Parameters

```scala
trait InputDim derives Label
trait Hidden derives Label
trait Output derives Label

// GOOD: Structured parameters with TensorTree
case class ModelParams(
  encoder: Tensor2[InputDim, Hidden, Float],
  decoder: Tensor2[Hidden, Output, Float],
  bias: Tensor1[Output, Float]
)

// AVOID: Tuples or loose parameters
```

### 4. Use Broadcast Operators Explicitly

```scala
trait Sample derives Label

val data = Tensor1(Axis[Sample]).fromArray(Array(1.0f, 2.0f, 3.0f))
val mean = data.mean
val std = data.std

// GOOD: Clear intent
val normalized = (data -! mean) /! std

// AVOID: Mixing broadcast and non-broadcast without clarity
```

### 5. Leverage Functional Random Keys

```scala
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

```scala
import dimwit.jax.Jit.jit

trait Input derives Label

val simpleFunc = (x: Tensor1[Input, Float]) => x *! Tensor0(2.0f)

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
