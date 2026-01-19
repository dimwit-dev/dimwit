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
// shape0: Shape[EmptyTuple] = Shape(dimensions = List())

// Vector (1-dimensional)
val shape1 = Shape(Axis[Feature] -> 10)
// shape1: Shape[*:[Feature, EmptyTuple]] = Shape(dimensions = List(10))

// Matrix (2-dimensional)
val shape2 = Shape(Axis[Batch] -> 32, Axis[Feature] -> 10)
// shape2: Shape[*:[Batch, *:[Feature, EmptyTuple]]] = Shape(
//   dimensions = List(32, 10)
// )

// 3D Tensor
val shape3 = Shape(Axis[Batch] -> 32, Axis[Feature] -> 10, Axis[Hidden] -> 128)
// shape3: Shape[*:[Batch, *:[Feature, *:[Hidden, EmptyTuple]]]] = Shape(
//   dimensions = List(32, 10, 128)
// )

// Type aliases for convenience
val s1: Shape1[Feature] = Shape1(Axis[Feature] -> 10)
// s1: Shape[*:[Feature, EmptyTuple]] = Shape(dimensions = List(10))
val s2: Shape2[Batch, Feature] = Shape2(Axis[Batch] -> 32, Axis[Feature] -> 10)
// s2: Shape[*:[Batch, *:[Feature, EmptyTuple]]] = Shape(
//   dimensions = List(32, 10)
// )
val s3: Shape3[Batch, Feature, Hidden] = Shape3(Axis[Batch] -> 32, Axis[Feature] -> 10, Axis[Hidden] -> 128)
// s3: Shape[*:[Batch, *:[Feature, *:[Hidden, EmptyTuple]]]] = Shape(
//   dimensions = List(32, 10, 128)
// )
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
// Ensure that:
// 1. All axis types repl.MdocSession.MdocApp.UndefinedLabel are defined with 'derives Label' (e.g. 'trait T derives Label'), or
// 2. You are not accidentally mixing incompatible label types (these would be unified by the compiler to a new type, which does not have a Label instance.
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
//       dimwit.tensor.AxisIndex.tail[repl.MdocSession.MdocApp.A,
//         repl.MdocSession.MdocApp.B *: EmptyTuple.type, L](
//         dimwit.tensor.AxisIndex.tail[repl.MdocSession.MdocApp.B, EmptyTuple.type, L²
//           ](dimwit.tensor.AxisIndex.concatRight[A², B², L³])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.AxisIndex[EmptyTuple.type, L²]
// 
// where:    A  is a trait in object MdocApp
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in object MdocApp
//           B² is a type variable with constraint <: Tuple
//           L  is a type variable
//           L² is a type variable
//           L³ is a type variable
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
// intTensor: Tensor[Tuple2[A, B], Int] = [[42 42 42 42 42]
//  [42 42 42 42 42]
//  [42 42 42 42 42]
//  [42 42 42 42 42]]
println(s"DType: ${intTensor.dtype}") // Int32
// DType: Int32

// Float tensor
val floatTensor = Tensor(Shape3(Axis[A] -> 2, Axis[B] -> 3, Axis[C] -> 4)).fill(3.14f)
// floatTensor: Tensor[Tuple3[A, B, C], Float] = [[[3.14 3.14 3.14 3.14]
//   [3.14 3.14 3.14 3.14]
//   [3.14 3.14 3.14 3.14]]
// 
//  [[3.14 3.14 3.14 3.14]
//   [3.14 3.14 3.14 3.14]
//   [3.14 3.14 3.14 3.14]]]
println(s"DType: ${floatTensor.dtype}") // Float32
// DType: Float32

// Boolean tensor
val boolTensor = Tensor(Shape1(Axis[A] -> 10)).fill(true)
// boolTensor: Tensor[Tuple1[A], Boolean] = [ True  True  True  True  True  True  True  True  True  True]
println(s"DType: ${boolTensor.dtype}") // Bool
// DType: Bool
```

### Creation from Arrays with `fromArray`

```scala
// 1D tensor from array
val t1d = Tensor(Shape1(Axis[A] -> 3)).fromArray(Array(1.0f, 2.0f, 3.0f))
// t1d: Tensor[Tuple1[A], Float] = [1. 2. 3.]

// 2D tensor from flattened array (row-major order)
val t2d = Tensor(Shape2(Axis[A] -> 2, Axis[B] -> 3)).fromArray(
  Array(1.0f, 2.0f, 3.0f,  // First row
        4.0f, 5.0f, 6.0f)  // Second row
)
// t2d: Tensor[Tuple2[A, B], Float] = [[1. 2. 3.]
//  [4. 5. 6.]]

// Using nested arrays
val t2dNested = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f, 3.0f),
    Array(4.0f, 5.0f, 6.0f)
  )
)
// t2dNested: Tensor[Tuple2[A, B], Float] = [[1. 2. 3.]
//  [4. 5. 6.]]
```

### Type Aliases for Common Shapes

```scala
// Scalar (0D)
val scalar: Tensor0[Float] = Tensor0(42.0f)
// scalar: Tensor[EmptyTuple, Float] = 42.0

// Vector (1D)
val vector: Tensor1[Feature, Float] = Tensor1(Axis[Feature]).fromArray(Array(1.0f, 2.0f, 3.0f))
// vector: Tensor[Tuple1[Feature], Float] = [1. 2. 3.]

// Matrix (2D)
val matrix: Tensor2[Batch, Feature, Float] = Tensor2(Axis[Batch], Axis[Feature]).fromArray(
  Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
)
// matrix: Tensor[Tuple2[Batch, Feature], Float] = [[1. 2.]
//  [3. 4.]]

// 3D Tensor
val tensor3d: Tensor3[Batch, Feature, Hidden, Float] = 
  Tensor(Shape3(Axis[Batch] -> 2, Axis[Feature] -> 3, Axis[Hidden] -> 4)).fill(0.0f)
// tensor3d: Tensor[Tuple3[Batch, Feature, Hidden], Float] = [[[0. 0. 0. 0.]
//   [0. 0. 0. 0.]
//   [0. 0. 0. 0.]]
// 
//  [[0. 0. 0. 0.]
//   [0. 0. 0. 0.]
//   [0. 0. 0. 0.]]]
```

### Common Creation Errors

```scala
// ERROR: Missing Label instance for axis type
trait NoLabel  // No derives Label!
val noLabel = Tensor(Shape1(Axis[NoLabel] -> 2)).fill(1.0f)
// error:
// 
// An axis label repl.MdocSession.MdocApp.NoLabel was given or inferred, which does not have a Label instance.
// Ensure that:
// 1. All axis types repl.MdocSession.MdocApp.NoLabel are defined with 'derives Label' (e.g. 'trait T derives Label'), or
// 2. You are not accidentally mixing incompatible label types (these would be unified by the compiler to a new type, which does not have a Label instance.
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
//       dimwit.tensor.AxisIndex.tail[repl.MdocSession.MdocApp.A, EmptyTuple.type, L](
//         dimwit.tensor.AxisIndex.concatRight[A², B², L²]),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.AxisIndex[EmptyTuple.type, L]
// 
// where:    A  is a trait in object MdocApp
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in object MdocApp
//           B² is a type variable with constraint <: Tuple
//           L  is a type variable
//           L² is a type variable
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
// t: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]

// Arithmetic
val added = t + t  // [[2, 4], [6, 8]]
// added: Tensor[Tuple2[A, B], Float] = [[2. 4.]
//  [6. 8.]]
val subtracted = t - t  // [[0, 0], [0, 0]]
// subtracted: Tensor[Tuple2[A, B], Float] = [[0. 0.]
//  [0. 0.]]
val multiplied = t * t  // [[1, 4], [9, 16]]
// multiplied: Tensor[Tuple2[A, B], Float] = [[ 1.  4.]
//  [ 9. 16.]]
val divided = t / t  // [[1, 1], [1, 1]]
// divided: Tensor[Tuple2[A, B], Float] = [[1. 1.]
//  [1. 1.]]

// Math functions
val absolute = t.abs
// absolute: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]
val exponential = t.exp
// exponential: Tensor[Tuple2[A, B], Float] = [[ 2.7182817  7.389056 ]
//  [20.085537  54.59815  ]]
val logarithm = (t +! Tensor0(1.0f)).log  // Avoid log(0)
// logarithm: Tensor[Tuple2[A, B], Float] = [[0.6931472 1.0986123]
//  [1.3862944 1.609438 ]]
val power = t.pow(Tensor0(2.0f))
// power: Tensor[Tuple2[A, B], Float] = [[ 1.  4.]
//  [ 9. 16.]]
val sqrt = t.sqrt
// sqrt: Tensor[Tuple2[A, B], Float] = [[1.        1.4142135]
//  [1.7320508 2.       ]]
val sign = t.sign
// sign: Tensor[Tuple2[A, B], Float] = [[1. 1.]
//  [1. 1.]]

// Trigonometric
val sine = t.sin
// sine: Tensor[Tuple2[A, B], Float] = [[ 0.841471    0.90929747]
//  [ 0.14112    -0.7568025 ]]
val cosine = t.cos
// cosine: Tensor[Tuple2[A, B], Float] = [[ 0.5403023  -0.4161468 ]
//  [-0.9899925  -0.65364367]]
val tanh = t.tanh
// tanh: Tensor[Tuple2[A, B], Float] = [[0.7615942 0.9640275]
//  [0.9950547 0.9993292]]

// Clipping
val clipped = t.clip(Tensor0(1.5f), Tensor0(3.5f))
// clipped: Tensor[Tuple2[A, B], Float] = [[1.5 2. ]
//  [3.  3.5]]
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
// data: Tensor[Tuple2[A, B], Float] = [[1. 2. 3.]
//  [4. 5. 6.]]

// Reduce to scalar
val totalSum: Tensor0[Float] = data.sum
// totalSum: Tensor[EmptyTuple, Float] = 21.0
val totalMean: Tensor0[Float] = data.mean
// totalMean: Tensor[EmptyTuple, Float] = 3.5
val totalMax: Tensor0[Float] = data.max
// totalMax: Tensor[EmptyTuple, Float] = 6.0
val totalMin: Tensor0[Float] = data.min
// totalMin: Tensor[EmptyTuple, Float] = 1.0
val totalStd: Tensor0[Float] = data.std
// totalStd: Tensor[EmptyTuple, Float] = 1.7078251

println(s"Sum: ${totalSum.item}")  // 21.0
// Sum: 21.0

// Reduce along axis A (across rows)
val sumA: Tensor1[B, Float] = data.sum(Axis[A])
// sumA: Tensor[Tuple1[B], Float] = [5. 7. 9.]
println(s"Sum along A: ${sumA}")  // [5.0, 7.0, 9.0]
// Sum along A: [5. 7. 9.]

// Reduce along axis B (across columns)
val sumB: Tensor1[A, Float] = data.sum(Axis[B])
// sumB: Tensor[Tuple1[A], Float] = [ 6. 15.]
println(s"Sum along B: ${sumB}")  // [6.0, 15.0]
// Sum along B: [ 6. 15.]

// Mean along axes
val meanA = data.mean(Axis[A])  // [2.5, 3.5, 4.5]
// meanA: Tensor[*:[B, EmptyTuple], Float] = [2.5 3.5 4.5]
val meanB = data.mean(Axis[B])  // [2.0, 5.0]
// meanB: Tensor[*:[A, EmptyTuple], Float] = [2. 5.]

// Argmax / Argmin (returns indices)
val argmaxB: Tensor1[A, Int] = data.argmax(Axis[B])
// argmaxB: Tensor[Tuple1[A], Int] = [2 2]
val argminB: Tensor1[A, Int] = data.argmin(Axis[B])
// argminB: Tensor[Tuple1[A], Int] = [0 0]
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
//       dimwit.tensor.AxisIndex.tail[MdocApp0.this.A,
//         MdocApp0.this.B *: EmptyTuple.type, L](
//         dimwit.tensor.AxisIndex.tail[MdocApp0.this.B, EmptyTuple.type, L²](
//           dimwit.tensor.AxisIndex.concatRight[A², B², L³])
//       ),
//     ???)
// 
// But given instance concatRight in object AxisIndex does not match type dimwit.tensor.AxisIndex[EmptyTuple.type, L²]
// 
// where:    A  is a trait in class MdocApp0
//           A² is a type variable with constraint <: Tuple
//           B  is a trait in class MdocApp0
//           B² is a type variable with constraint <: Tuple
//           L  is a type variable
//           L² is a type variable
//           L³ is a type variable
// .
// val wrong = t.sum(Axis[C])
//                          ^
// error: 
// Double definition:
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 53 and
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 88
//
```

### Broadcast Operations

Operations with automatic broadcasting for scalars and compatible shapes.

```scala
val tensor = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))
// tensor: Tensor[Tuple2[A, B], Float] = [[10. 20.]
//  [30. 40.]]

// Scalar broadcast (note: ! suffix for broadcast operations)
val scalarAdd = tensor +! Tensor0(5.0f)  // [[15, 25], [35, 45]]
// scalarAdd: Tensor[Tuple2[A, B], Float] = [[15. 25.]
//  [35. 45.]]
val scalarMul = tensor *! Tensor0(2.0f)  // [[20, 40], [60, 80]]
// scalarMul: Tensor[Tuple2[A, B], Float] = [[20. 40.]
//  [60. 80.]]
val scalarDiv = tensor /! Tensor0(10.0f) // [[1, 2], [3, 4]]
// scalarDiv: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]
val scalarSub = tensor -! Tensor0(5.0f)  // [[5, 15], [25, 35]]
// scalarSub: Tensor[Tuple2[A, B], Float] = [[ 5. 15.]
//  [25. 35.]]

// Reverse broadcast
val reverseAdd = Tensor0(100.0f) +! tensor  // [[110, 120], [130, 140]]
// reverseAdd: Tensor[Tuple2[A, B], Float] = [[110. 120.]
//  [130. 140.]]
val reverseSub = Tensor0(100.0f) -! tensor  // [[90, 80], [70, 60]]
// reverseSub: Tensor[Tuple2[A, B], Float] = [[90. 80.]
//  [70. 60.]]

// Tensor0 broadcast to tensor shape
val scalarBroadcast = Tensor0(5.0f).broadcastTo(tensor.shape)
// scalarBroadcast: Tensor[Tuple2[A, B], Float] = [[5. 5.]
//  [5. 5.]]

// Comparison with broadcast (element-wise comparison)
val greater = tensor > Tensor0(25.0f).broadcastTo(tensor.shape)
// greater: Tensor[Tuple2[A, B], Boolean] = [[False False]
//  [ True  True]]
```

**Important**: Standard operators `+`, `-`, `*`, `/` require **exact shape match**. Use `+!`, `-!`, `*!`, `/!` for broadcasting.

```scala
val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
// ERROR: Shape mismatch - cannot add scalar without broadcast operator
val wrong = t + 5.0f  // Use +! instead
// error:
// Found:    (5.0f : Float)
// Required: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float]
// val wrong = t + 5.0f  // Use +! instead
//                 ^^^^
// error: 
// Double definition:
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 53 and
// val t: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 97
//
```

### Contraction Operations

Dot products and matrix multiplication.

```scala
// Dot product (vector · vector)
val v1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
// v1: Tensor[Tuple1[A], Float] = [1. 2. 3.]
val v2 = Tensor1(Axis[A]).fromArray(Array(4.0f, 5.0f, 6.0f))
// v2: Tensor[Tuple1[A], Float] = [4. 5. 6.]
val dotProduct: Tensor0[Float] = v1.dot(Axis[A])(v2)
// dotProduct: Tensor[EmptyTuple, Float] = 32.0
println(s"Dot product: ${dotProduct.item}")  // 32.0
// Dot product: 32.0

// Matrix-vector multiplication
val matrix = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(
    Array(1.0f, 2.0f),
    Array(3.0f, 4.0f)
  )
)
// matrix: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]
val vec = Tensor1(Axis[B]).fromArray(Array(1.0f, 2.0f))
// vec: Tensor[Tuple1[B], Float] = [1. 2.]
val result: Tensor1[A, Float] = matrix.dot(Axis[B])(vec)
// result: Tensor[Tuple1[A], Float] = [ 5. 11.]
println(s"Matrix-vec result: ${result}")  // [5.0, 11.0]
// Matrix-vec result: [ 5. 11.]

// Matrix-matrix multiplication
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f)))
// m1: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]
val m2 = Tensor2(Axis[B], Axis[C]).fromArray(Array(Array(5.0f, 6.0f), Array(7.0f, 8.0f)))
// m2: Tensor[Tuple2[B, C], Float] = [[5. 6.]
//  [7. 8.]]
val matmul: Tensor2[A, C, Float] = m1.dot(Axis[B])(m2)
// matmul: Tensor[Tuple2[A, C], Float] = [[19. 22.]
//  [43. 50.]]
```

**Error: Dimension mismatch in contraction**

```scala
val m1 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f)))
val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f, 4.0f)))
// ERROR: No shared axis for contraction
val wrong = m1.dot(Axis[B])(m2)
// error: 
// 
// A tuple of axis labels (MdocApp0.this.A, MdocApp0.this.C) was given or inferred that does not have a valid Labels instance. 
// 
// Common causes:
// - One of the types in the tuple is missing a 'derives Label' clause.
// - You are mixing incompatible label types, which the compiler automatically unifies to a new type, which may not have a Labels instance.
// .
// I found:
// 
//     dimwit.tensor.Labels.given_Labels_A_B[MdocApp0.this.A,
//       MdocApp0.this.D & MdocApp0.this.C](
//       dimwit.tensor.Labels.lift[MdocApp0.this.A](this.A.derived$Label),
//       dimwit.tensor.Labels.lift[MdocApp0.this.D & MdocApp0.this.C](
//         /* missing */summon[dimwit.tensor.Label[MdocApp0.this.D & MdocApp0.this.C]])
//       )
// 
// But no implicit values were found that match type dimwit.tensor.Label[MdocApp0.this.D & MdocApp0.this.C].
// error:
// 
// A tuple of axis labels Tuple was given or inferred that does not have a valid Labels instance. 
// 
// Common causes:
// - One of the types in the tuple is missing a 'derives Label' clause.
// - You are mixing incompatible label types, which the compiler automatically unifies to a new type, which may not have a Labels instance.
// .
// I found:
// 
//     dimwit.tensor.Labels.given_Labels_A_B[MdocApp0.this.A,
//       MdocApp0.this.D & MdocApp0.this.C](
//       dimwit.tensor.Labels.lift[MdocApp0.this.A](this.A.derived$Label),
//       dimwit.tensor.Labels.lift[MdocApp0.this.D & MdocApp0.this.C](
//         /* missing */summon[dimwit.tensor.Label[MdocApp0.this.D & MdocApp0.this.C]])
//       )
// 
// But no implicit values were found that match type dimwit.tensor.Label[MdocApp0.this.D & MdocApp0.this.C].
// val wrong = m1.dot(Axis[B])(m2)
//                               ^
// error: 
// Double definition:
// val m1: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 110 and
// val m1: dimwit.tensor.Tensor[(MdocApp0.this.A, MdocApp0.this.B), Float] in class MdocApp0 at line 113
// 
// error: 
// Double definition:
// val m2: dimwit.tensor.Tensor[(MdocApp0.this.B, MdocApp0.this.C), Float] in class MdocApp0 at line 111 and
// val m2: dimwit.tensor.Tensor[(MdocApp0.this.C, MdocApp0.this.D), Float] in class MdocApp0 at line 114
//
```

### Structural Operations

Reshape, transpose, and dimension manipulation.

```scala
val original = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f))
)
// original: Tensor[Tuple2[A, B], Float] = [[1. 2. 3.]
//  [4. 5. 6.]]

// Transpose
val transposed: Tensor2[B, A, Float] = original.transpose
// transposed: Tensor[Tuple2[B, A], Float] = [[1. 4.]
//  [2. 5.]
//  [3. 6.]]
println(s"Original shape: ${original.shape}")
// Original shape: Shape(A -> 2, B -> 3)
println(s"Transposed shape: ${transposed.shape}")
// Transposed shape: Shape(B -> 3, A -> 2)

// Reshape
val reshaped = original.ravel  // Flatten to 1D
// reshaped: Tensor[Tuple1[(A |*| B)], Float] = [1. 2. 3. 4. 5. 6.]
// Reshaping back requires fromArray with proper data

// Unsqueeze (add size-1 dimension)
val vec1d = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
// vec1d: Tensor[Tuple1[A], Float] = [1. 2. 3.]
val vec2d = vec1d.appendAxis(Axis[B])  // Add new axis B
// vec2d: Tensor[*:[A, Tuple1[B]], Float] = [[1.]
//  [2.]
//  [3.]]

// Take (indexing)
val firstRow = original.slice(Axis[A] -> 0)
// firstRow: Tensor[*:[B, EmptyTuple], Float] = [1. 2. 3.]
val secondRow = original.slice(Axis[A] -> 1)
// secondRow: Tensor[*:[B, EmptyTuple], Float] = [4. 5. 6.]

// Concatenate
val t1 = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f))
// t1: Tensor[Tuple1[A], Float] = [1. 2.]
val t2 = Tensor1(Axis[A]).fromArray(Array(3.0f, 4.0f))
// t2: Tensor[Tuple1[A], Float] = [3. 4.]
val concatenated = concatenate(t1, t2, Axis[A])
// concatenated: Tensor[Tuple1[A], Float] = [1. 2. 3. 4.]
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
// data: Tensor[Tuple2[Batch, Feature], Float] = [[1. 2. 3.]
//  [4. 5. 6.]
//  [7. 8. 9.]]

// Normalize each sample (row) independently
def normalize(x: Tensor1[Feature, Float]): Tensor1[Feature, Float] =
  val mean = x.mean
  val std = x.std + Tensor0(1e-6f)  // Avoid division by zero
  (x -! mean) /! std

val normalized: Tensor2[Batch, Feature, Float] = data.vmap(Axis[Batch])(normalize)
// normalized: Tensor[Tuple2[Batch, Feature], Float] = [[-1.2247434  0.         1.2247434]
//  [-1.2247434  0.         1.2247434]
//  [-1.2247434  0.         1.2247434]]
println(s"Normalized data: ${normalized}")
// Normalized data: [[-1.2247434  0.         1.2247434]
//  [-1.2247434  0.         1.2247434]
//  [-1.2247434  0.         1.2247434]]

// Sum each row
val rowSums: Tensor1[Batch, Float] = data.vmap(Axis[Batch])(_.sum)
// rowSums: Tensor[Tuple1[Batch], Float] = [ 6. 15. 24.]
println(s"Row sums: ${rowSums}")  // [6.0, 15.0, 24.0]
// Row sums: [ 6. 15. 24.]

// vmap over columns (note: axis moves to front)
val colSums: Tensor1[Feature, Float] = data.vmap(Axis[Feature])(_.sum)
// colSums: Tensor[Tuple1[Feature], Float] = [12. 15. 18.]
println(s"Column sums: ${colSums}")  // [12.0, 15.0, 18.0]
// Column sums: [12. 15. 18.]

// Identity vmap doesn't change data, only axis order
val identity = data.vmap(Axis[Batch])(x => x)  // Same as data
// identity: Tensor[*:[Batch, *:[Feature, EmptyTuple]], Float] = [[1. 2. 3.]
//  [4. 5. 6.]
//  [7. 8. 9.]]
val reordered = data.vmap(Axis[Feature])(x => x)  // Same as data.transpose
// reordered: Tensor[*:[Feature, *:[Batch, EmptyTuple]], Float] = [[1. 4. 7.]
//  [2. 5. 8.]
//  [3. 6. 9.]]
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
// Ensure that:
// 1. All axis types Any are defined with 'derives Label' (e.g. 'trait T derives Label'), or
// 2. You are not accidentally mixing incompatible label types (these would be unified by the compiler to a new type, which does not have a Label instance.
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
// t1: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]
val t2 = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(10.0f, 20.0f), Array(30.0f, 40.0f)))
// t2: Tensor[Tuple2[A, B], Float] = [[10. 20.]
//  [30. 40.]]

// Compute L2 distance between corresponding rows
def l2Distance(v1: Tensor1[B, Float], v2: Tensor1[B, Float]): Tensor0[Float] =
  (v1 - v2).pow(Tensor0(2.0f)).sum.sqrt

val distances: Tensor1[A, Float] = zipvmap(Axis[A])(t1, t2)(l2Distance)
// distances: Tensor[Tuple1[A], Float] = [20.12461 45.     ]
println(s"L2 distances: ${distances}")
// L2 distances: [20.12461 45.     ]

// zipvmap with 4 tensors
val t3 = t1 *! Tensor0(2.0f)
// t3: Tensor[Tuple2[A, B], Float] = [[2. 4.]
//  [6. 8.]]
val t4 = t2 /! Tensor0(2.0f)
// t4: Tensor[Tuple2[A, B], Float] = [[ 5. 10.]
//  [15. 20.]]
val result = zipvmap(Axis[A])(t1, t2, t3, t4) { (a, b, c, d) =>
  a.sum + b.sum - c.sum + d.sum
}
// result: Tensor[*:[A, EmptyTuple], Float] = [42. 98.]
```

### vapply: Axis-wise Function Application

Unlike `vmap`, `vapply` allows applying **different** functions to each slice.

```scala
val matrix = Tensor2(Axis[A], Axis[B]).fromArray(
  Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
)
// matrix: Tensor[Tuple2[A, B], Float] = [[1. 2.]
//  [3. 4.]]

// Normalize each row by its L2 norm
val normalized = matrix.vapply(Axis[A]) { row =>
  val norm = row.pow(Tensor0(2.0f)).sum.sqrt + Tensor0(1e-8f)
  row /! norm
}
// normalized: Tensor[*:[A, *:[B, EmptyTuple]], Float] = [[0.31622776 0.44721365]
//  [0.94868326 0.8944273 ]]

// Identity vapply
val same = matrix.vapply(Axis[A])(identity)
// same: Tensor[*:[A, *:[B, EmptyTuple]], Float] = [[1. 2.]
//  [3. 4.]]
```

### vreduce: Functional Reduction

```scala
import dimwit.*

trait A derives Label
trait B derives Label

val t = Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)))
// t: Tensor[Tuple2[A, B], Float] = [[1. 2. 3.]
//  [4. 5. 6.]]

// Equivalent to .sum(Axis[A])
val summedA = t.vreduce(Axis[A])(_.sum)
// summedA: Tensor[*:[B, EmptyTuple], Float] = [5. 7. 9.]
val summedB = t.vreduce(Axis[B])(_.sum)
// summedB: Tensor[*:[A, EmptyTuple], Float] = [ 6. 15.]

println(s"Reduced A: ${summedA}")
// Reduced A: [5. 7. 9.]
println(s"Reduced B: ${summedB}")
// Reduced B: [ 6. 15.]
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
// df: Function1[Tensor0[Float], Grad[Tensor0[Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@730f7f91
val x = Tensor0(3.0f)
// x: Tensor[EmptyTuple, Float] = 3.0
val gradient = df(x)
// gradient: Grad[Tensor0[Float]] = 6.0
println(s"df/dx at x=3: ${gradient.value.item}")  // 6.0
// df/dx at x=3: 6.0

// Vector function: f(x) = sum(x²)
def g(x: Tensor1[A, Float]): Tensor0[Float] = (x * x).sum

val dg = Autodiff.grad(g)
// dg: Function1[Tensor1[A, Float], Grad[Tensor1[A, Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@27d9957e
val xVec = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
// xVec: Tensor[Tuple1[A], Float] = [1. 2. 3.]
val vecGradient = dg(xVec)
// vecGradient: Grad[Tensor1[A, Float]] = [2. 4. 6.]
println(s"dg/dx: ${vecGradient}")  // [2.0, 4.0, 6.0]
// dg/dx: [2. 4. 6.]
```

### Higher-Order Derivatives

```scala
def f2(x: Tensor0[Float]): Tensor0[Float] = x * x

// First derivative
val df2 = Autodiff.grad(f2)
// df2: Function1[Tensor0[Float], Grad[Tensor0[Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@f0e78b8

// Second derivative
val ddf2 = Autodiff.grad((x: Tensor0[Float]) => df2(x).value)
// ddf2: Function1[Tensor0[Float], Grad[Tensor0[Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@2267aa63

// Third derivative
val dddf2 = Autodiff.grad((x: Tensor0[Float]) => ddf2(x).value)
// dddf2: Function1[Tensor0[Float], Grad[Tensor0[Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@599bad8e

val x2 = Tensor0(3.0f)
// x2: Tensor[EmptyTuple, Float] = 3.0
println(s"f''(3) = ${ddf2(x2).value.item}")   // 2.0
// f''(3) = 2.0
println(s"f'''(3) = ${dddf2(x2).value.item}") // 0.0
// f'''(3) = 0.0
```

### Gradients with Multiple Parameters

```scala
// f(x, y) = (x + 2y)²
def twoParam(x: Tensor1[A, Float], y: Tensor1[A, Float]): Tensor0[Float] =
  ((x + (y *! Tensor0(2.0f))).pow(Tensor0(2.0f))).sum

val dtwoParam = Autodiff.grad(twoParam)
// dtwoParam: Function2[Tensor1[A, Float], Tensor1[A, Float], Grad[Tuple2[Tensor1[A, Float], Tensor1[A, Float]]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092b6b8@5e67e56b

val x3 = Tensor1(Axis[A]).fromArray(Array(1.0f))
// x3: Tensor[Tuple1[A], Float] = [1.]
val y3 = Tensor1(Axis[A]).fromArray(Array(1.0f))
// y3: Tensor[Tuple1[A], Float] = [1.]

val (xGrad, yGrad) = dtwoParam(x3, y3).value
// xGrad: Tensor[Tuple1[A], Float] = [6.]
// yGrad: Tensor[Tuple1[A], Float] = [12.]
println(s"∂f/∂x = ${xGrad}")  // [6.0]
// ∂f/∂x = [6.]
println(s"∂f/∂y = ${yGrad}")  // [12.0]
// ∂f/∂y = [12.]
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
// dBatched: Function1[Tensor2[Batch, Feature, Float], Grad[Tensor2[Batch, Feature, Float]]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@409b15b6

val xBatch = Tensor(Shape(Axis[Batch] -> 2, Axis[Feature] -> 2)).fill(1.0f)
// xBatch: Tensor[*:[Batch, *:[Feature, EmptyTuple]], Float] = [[1. 1.]
//  [1. 1.]]
val batchGrad = dBatched(xBatch)
// batchGrad: Grad[Tensor2[Batch, Feature, Float]] = [[1. 1.]
//  [1. 1.]]
println(s"Batch gradient: ${batchGrad}")
// Batch gradient: [[1. 1.]
//  [1. 1.]]
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
// sampleData: Tensor[Tuple1[Feature], Float] = [1. 2. 3.]
val sampleTarget = Tensor1(Axis[Hidden]).fromArray(Array(0.0f, 1.0f))
// sampleTarget: Tensor[Tuple1[Hidden], Float] = [0. 1.]
val initParams = LinearParams(
  weight = Tensor(Shape(Axis[Feature] -> 3, Axis[Hidden] -> 2)).fill(0.1f),
  bias = Tensor(Shape(Axis[Hidden] -> 2)).fill(0.0f)
)
// initParams: LinearParams = LinearParams(
//   weight = [[0.1 0.1]
//  [0.1 0.1]
//  [0.1 0.1]],
//   bias = [0. 0.]
// )

val dLoss = Autodiff.grad(loss(sampleData, sampleTarget))
// dLoss: Function1[LinearParams, Grad[LinearParams]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@30b16c30
val paramGradients: LinearParams = dLoss(initParams).value
// paramGradients: LinearParams = LinearParams(
//   weight = [[ 1.2        -0.79999995]
//  [ 2.4        -1.5999999 ]
//  [ 3.6000001  -2.3999999 ]],
//   bias = [ 1.2        -0.79999995]
// )
println(s"Weight gradient shape: ${paramGradients.weight.shape}")
// Weight gradient shape: Shape(Feature -> 3, Hidden -> 2)
println(s"Bias gradient shape: ${paramGradients.bias.shape}")
// Bias gradient shape: Shape(Hidden -> 2)
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
// jacobian: Function1[Tensor1[A, Float], Tensor[*:[A, Tuple1[A]], Float]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b0934940@2fb28b07
val xJac = Tensor1(Axis[A]).fromArray(Array(1.0f, 1.0f))
// xJac: Tensor[Tuple1[A], Float] = [1. 1.]
val jacResult = jacobian(xJac)
// jacResult: Tensor[*:[A, Tuple1[A]], Float] = [[2. 0.]
//  [0. 2.]]
println(s"Jacobian: ${jacResult}")  // Should be 2 * identity matrix
// Jacobian: [[2. 0.]
//  [0. 2.]]

// jacRev and jacFwd for larger Jacobians
val jacRev = Autodiff.jacRev(linearMap)
// jacRev: Function1[Tensor1[A, Float], Tensor[*:[A, Tuple1[A]], Float]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b0935398@442cd1ba
val jacFwd = Autodiff.jacFwd(linearMap)
// jacFwd: Function1[Tensor1[A, Float], Tensor[*:[A, Tuple1[A]], Float]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b0935df0@2a8f17d6
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
// key: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [ 0 42]
// )
val numFeatures = 5
// numFeatures: Int = 5
val initW = Tensor1(Axis[Feature]).fromArray(Array.fill(numFeatures)(0.1f))
// initW: Tensor[Tuple1[Feature], Float] = [0.1 0.1 0.1 0.1 0.1]
val initB = Tensor0(0.0f)
// initB: Tensor[EmptyTuple, Float] = 0.0
val initModelParams = SimpleModelParams(initW, initB)
// initModelParams: SimpleModelParams = SimpleModelParams(
//   w = [0.1 0.1 0.1 0.1 0.1],
//   b = 0.0
// )

// Create dummy data
val trainData = Tensor(Shape(Axis[Batch] -> 10, Axis[Feature] -> numFeatures)).fill(1.0f)
// trainData: Tensor[*:[Batch, *:[Feature, EmptyTuple]], Float] = [[1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]
//  [1. 1. 1. 1. 1.]]
val trainLabels = Tensor1(Axis[Batch]).fromArray(Array.fill(10)(1.0f))
// trainLabels: Tensor[Tuple1[Batch], Float] = [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

// Compute gradient function
val lossFunc = mse(trainData, trainLabels)
// lossFunc: Function1[SimpleModelParams, Tensor0[Float]] = repl.MdocSession$MdocApp43$$Lambda/0x00007714b0944b20@206a3949
val gradFunc = Autodiff.grad(lossFunc)
// gradFunc: Function1[SimpleModelParams, Grad[SimpleModelParams]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@736dd81a

// Create optimizer
val optimizer = GradientDescent(learningRate = Tensor0(0.01f))
// optimizer: GradientDescent = GradientDescent(learningRate = 0.01)

// Training loop with iterator
val trained = optimizer.iterate(initModelParams)(gradFunc)
  .take(5)  // Run 5 iterations
  .foreach { params =>
    val currentLoss = lossFunc(params)
    println(f"Loss: ${currentLoss.item}%.4f")
  }
// Loss: 0.2500
// Loss: 0.1936
// Loss: 0.1499
// Loss: 0.1161
// Loss: 0.0899
```

### Lion Optimizer

```scala
import nn.Lion

// Lion optimizer with momentum
val lionOptimizer = Lion(learningRate = Tensor0(1e-3f), beta1 = Tensor0(0.9f), beta2 = Tensor0(0.99f), weightDecay = Tensor0(0.0f))
// lionOptimizer: Lion = Lion(
//   learningRate = 0.001,
//   weightDecay = 0.0,
//   beta1 = 0.9,
//   beta2 = 0.99
// )

// Training with Lion
val trainedLion = lionOptimizer.iterate(initModelParams)(gradFunc)
  .drop(100)  // Skip first 100 iterations
  .take(10)   // Train for 10 more
  .toList     // Collect results
// trainedLion: List[SimpleModelParams] = List(
//   SimpleModelParams(
//     w = [0.2000002 0.2000002 0.2000002 0.2000002 0.2000002],
//     b = 0.100000024
//   ),
//   SimpleModelParams(
//     w = [0.2010002 0.2010002 0.2010002 0.2010002 0.2010002],
//     b = 0.101000026
//   ),
//   SimpleModelParams(
//     w = [0.2020002 0.2020002 0.2020002 0.2020002 0.2020002],
//     b = 0.10200003
//   ),
//   SimpleModelParams(
//     w = [0.2030002 0.2030002 0.2030002 0.2030002 0.2030002],
//     b = 0.10300003
//   ),
//   SimpleModelParams(
//     w = [0.2040002 0.2040002 0.2040002 0.2040002 0.2040002],
//     b = 0.10400003
//   ),
//   SimpleModelParams(
//     w = [0.2050002 0.2050002 0.2050002 0.2050002 0.2050002],
//     b = 0.105000034
//   ),
//   SimpleModelParams(
//     w = [0.20600021 0.20600021 0.20600021 0.20600021 0.20600021],
//     b = 0.106000036
//   ),
//   SimpleModelParams(
//     w = [0.20700021 0.20700021 0.20700021 0.20700021 0.20700021],
//     b = 0.10700004
//   ),
//   SimpleModelParams(
//     w = [0.20800021 0.20800021 0.20800021 0.20800021 0.20800021],
//     b = 0.10800004
//   ),
//   SimpleModelParams(
//     w = [0.20900021 0.20900021 0.20900021 0.20900021 0.20900021],
//     b = 0.10900004
//   )
// )
```

### Complete Training Example: Linear Regression

```scala
// Define problem dimensions
trait Sample derives Label
trait InputDim derives Label

// Generate synthetic data: y = 2x + 1 + noise
val numSamples = 100
// numSamples: Int = 100
val xData = Tensor2(Axis[Sample], Axis[InputDim]).fromArray(
  Array.tabulate(numSamples, 1)((i, _) => i.toFloat / numSamples)
)
// xData: Tensor[Tuple2[Sample, InputDim], Float] = [[0.  ]
//  [0.01]
//  [0.02]
//  [0.03]
//  [0.04]
//  [0.05]
//  [0.06]
//  [0.07]
//  [0.08]
//  [0.09]
//  [0.1 ]
//  [0.11]
//  [0.12]
//  [0.13]
//  [0.14]
//  [0.15]
//  [0.16]
//  [0.17]
//  [0.18]
//  [0.19]
//  [0.2 ]
//  [0.21]
//  [0.22]
//  [0.23]
//  [0.24]
//  [0.25]
//  [0.26]
//  [0.27]
//  [0.28]
//  [0.29]
//  [0.3 ]
//  [0.31]
//  [0.32]
//  [0.33]
//  [0.34]
//  [0.35]
//  [0.36]
//  [0.37]
//  [0.38]
//  [0.39]
//  [0.4 ]
//  [0.41]
//  [0.42]
//  [0.43]
//  [0.44]
//  [0.45]
//  [0.46]
//  [0.47]
//  [0.48]
// ...
val yData = Tensor1(Axis[Sample]).fromArray(
  Array.tabulate(numSamples)(i => 2.0f * i.toFloat / numSamples + 1.0f)
)
// yData: Tensor[Tuple1[Sample], Float] = [1.        1.02      1.04      1.06      1.08      1.1       1.12
//  1.14      1.16      1.1800001 1.2       1.22      1.24      1.26
//  1.28      1.3       1.3199999 1.34      1.36      1.38      1.4
//  1.42      1.44      1.46      1.48      1.5       1.52      1.54
//  1.56      1.5799999 1.6       1.62      1.64      1.6600001 1.6800001
//  1.7       1.72      1.74      1.76      1.78      1.8       1.8199999
//  1.8399999 1.86      1.88      1.9       1.9200001 1.94      1.96
//  1.98      2.        2.02      2.04      2.06      2.08      2.1
//  2.12      2.1399999 2.1599998 2.1799998 2.2       2.22      2.24
//  2.26      2.28      2.3       2.3200002 2.3400002 2.3600001 2.38
//  2.4       2.42      2.44      2.46      2.48      2.5       2.52
//  2.54      2.56      2.58      2.6       2.62      2.6399999 2.6599998
//  2.6799998 2.7       2.72      2.74      2.76      2.78      2.8
//  2.8200002 2.8400002 2.8600001 2.88      2.9       2.92      2.94
//  2.96      2.98     ]

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
// initSlope: Tensor[Tuple1[InputDim], Float] = [0.]
val initIntercept = Tensor0(0.0f)
// initIntercept: Tensor[EmptyTuple, Float] = 0.0
val initRegressionParams = RegressionParams(initSlope, initIntercept)
// initRegressionParams: RegressionParams = RegressionParams(
//   slope = [0.],
//   intercept = 0.0
// )

// Train
val regressionGrad = Autodiff.grad(regressionLoss(xData, yData))
// regressionGrad: Function1[RegressionParams, Grad[RegressionParams]] = dimwit.autodiff.Autodiff$$$Lambda/0x00007714b092ae28@4e439f22
val gdOptimizer = GradientDescent(learningRate = Tensor0(0.1f))
// gdOptimizer: GradientDescent = GradientDescent(learningRate = 0.1)

val finalParams = gdOptimizer.iterate(initRegressionParams)(regressionGrad)
  .take(100)
  .toList
  .last
// finalParams: RegressionParams = RegressionParams(
//   slope = [1.6920885],
//   intercept = 1.1631879
// )

println(s"Learned slope: ${finalParams.slope}")
// Learned slope: [1.6920885]
println(s"Learned intercept: ${finalParams.intercept}")
// Learned intercept: 1.1631879
println(s"Expected: slope ≈ 2.0, intercept ≈ 1.0")
// Expected: slope ≈ 2.0, intercept ≈ 1.0
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
// jitComplex: Function1[Tensor1[A, Float], Tensor1[A, Float]] = dimwit.jax.JitInternal$$$Lambda/0x00007714b0954e58@3efe6a7

// First call: compilation overhead
val input = Tensor1(Axis[A]).fromArray(Array(1.0f, 2.0f, 3.0f))
// input: Tensor[Tuple1[A], Float] = [1. 2. 3.]
val result1 = jitComplex(input)
// result1: Tensor[Tuple1[A], Float] = [0.7080735  0.82682186 0.01991485]

// Subsequent calls: fast execution
val result2 = jitComplex(input *! Tensor0(2.0f))
// result2: Tensor[Tuple1[A], Float] = [0.82682186 0.57275003 0.07807302]
val result3 = jitComplex(input *! Tensor0(3.0f))
// result3: Tensor[Tuple1[A], Float] = [0.01991485 0.07807302 0.16984165]

println(s"JIT result: ${result1}")
// JIT result: [0.7080735  0.82682186 0.01991485]
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
// rootKey: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [ 0 42]
// )

// Split key for independent random streams
val (key1, key2) = rootKey.split2()
// key1: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [2465931498 3679230171]
// )
// key2: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [255383827 267815257]
// )
val keys = rootKey.split(5)  // Split into 5 keys
// keys: Seq[Key] = Vector(
//   Key(
//     jaxKey = Array((), dtype=key<fry>) overlaying:
// [2765691542 1385194879]
//   ),
//   Key(
//     jaxKey = Array((), dtype=key<fry>) overlaying:
// [ 831049250 3807460095]
//   ),
//   Key(
//     jaxKey = Array((), dtype=key<fry>) overlaying:
// [3616728933  824333390]
//   ),
//   Key(
//     jaxKey = Array((), dtype=key<fry>) overlaying:
// [1482326074 1083977345]
//   ),
//   Key(
//     jaxKey = Array((), dtype=key<fry>) overlaying:
// [2713995981 2812206153]
//   )
// )

// Generate random numbers
val normalDist = Normal(loc = Tensor0(0.0f), scale = Tensor0(1.0f))
// normalDist: Normal[EmptyTuple, EmptyTuple, EmptyTuple] = dimwit.stats.Normal@3cd168f6
val randomSample = normalDist.sample(key1)
// randomSample: Tensor[EmptyTuple, Float] = 0.13790321
println(s"Normal samples: ${randomSample}")
// Normal samples: 0.13790321

val uniformDist = Uniform(low = Tensor0(0.0f), high = Tensor0(1.0f))
// uniformDist: Uniform[EmptyTuple] = dimwit.stats.Uniform@6368f18e
val uniformSample = uniformDist.sample(key2)
// uniformSample: Tensor[EmptyTuple, Float] = 0.91457367
println(s"Uniform samples: ${uniformSample}")
// Uniform samples: 0.91457367

// Permutations
val (permKey, _) = rootKey.split2()
// permKey: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [2465931498 3679230171]
// )
val shuffled = Random.permutation(Axis[A] -> 5)(permKey)
// shuffled: Tensor[Tuple1[A], Int] = [1 0 4 3 2]
println(s"Shuffled: ${shuffled}")
// Shuffled: [1 0 4 3 2]
```

### Random Key Splitting with vmap

```scala
trait B derives Label

// Split keys in parallel
val batchKey = Random.Key(123)
// batchKey: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [  0 123]
// )
val batchKeys = batchKey.splitvmap(Axis[A] -> 8)(k => Normal.standardSample(k))
// batchKeys: Tensor[*:[A, EmptyTuple], Float] = [ 0.7345024  -1.2327704  -0.2328729  -0.3326209   0.59855974 -0.5103833
//   0.83037174  0.596546  ]
println(s"Batch keys shape: ${batchKeys.shape}")
// Batch keys shape: Shape(A -> 8)
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
val wrong = intTensor.exp  // exp requires IsFloat constraint
// error:
// value exp is not a member of dimwit.tensor.Tensor[Tuple1[MdocApp10.this.A], Int].
// An extension method was tried, but could not be fully constructed:
// 
//     dimwit.exp[Tuple1[MdocApp10.this.A], Int](this.intTensor)(
//       dimwit.tensor.Labels.concat[MdocApp10.this.A, EmptyTuple.type](
//         this.A.derived$Label, dimwit.tensor.Labels.namesOfEmpty),
//       /* missing */summon[dimwit.tensor.TensorOps.IsFloat[Int]])
// 
//     failed with:
// 
//         Operation only valid for Float tensors.
// val wrong = intTensor.exp  // exp requires IsFloat constraint
//             ^^^^^^^^^^^^^
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
// val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f), Array(4.0f)))  // Shape: (2, 1)
//                       ^
// error: 
// Not found: Axis
// error:
// Not found: type D
// val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f), Array(4.0f)))  // Shape: (2, 1)
//                                ^
// error:
// Not found: Axis
// val m2 = Tensor2(Axis[C], Axis[D]).fromArray(Array(Array(3.0f), Array(4.0f)))  // Shape: (2, 1)
//                           ^^^^
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
// def nonScalar(x: Tensor1[A, Float]): Tensor1[A, Float] = x * x
//                  ^^^^^^^
// error:
// Not found: type A
// def nonScalar(x: Tensor1[A, Float]): Tensor1[A, Float] = x * x
//                          ^
// error:
// Not found: type Tensor1
// def nonScalar(x: Tensor1[A, Float]): Tensor1[A, Float] = x * x
//                                      ^^^^^^^
// error:
// Not found: type A
// def nonScalar(x: Tensor1[A, Float]): Tensor1[A, Float] = x * x
//                                              ^
// error:
// Not found: Autodiff
// val wrong = Autodiff.grad(nonScalar)  // Use jacobian instead
//             ^^^^^^^^
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
// embeddings: Tensor[Tuple3[Batch, SeqLen, EmbedDim], Float] = [[[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]
// 
//  [[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]
// 
//  [[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]
// 
//  ...
// 
//  [[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]
// 
//  [[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]
// 
//  [[0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   ...
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]
//   [0. 0. 0. ... 0. 0. 0.]]]

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
// data: Tensor[Tuple1[Sample], Float] = [1. 2. 3.]
val mean = data.mean
// mean: Tensor[EmptyTuple, Float] = 2.0
val std = data.std
// std: Tensor[EmptyTuple, Float] = 0.81649655

// GOOD: Clear intent
val normalized = (data -! mean) /! std
// normalized: Tensor[Tuple1[Sample], Float] = [-1.2247449  0.         1.2247449]

// AVOID: Mixing broadcast and non-broadcast without clarity
```

### 5. Leverage Functional Random Keys

```scala
import dimwit.random.Random
import dimwit.stats.{Normal, Uniform}

trait A derives Label

val rootKey = Random.Key(42)
// rootKey: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [ 0 42]
// )
val normalDist = Normal(loc = Tensor0(0.0f), scale = Tensor0(1.0f))
// normalDist: Normal[EmptyTuple, EmptyTuple, EmptyTuple] = dimwit.stats.Normal@4630aca7
val uniformDist = Uniform(low = Tensor0(0.0f), high = Tensor0(1.0f))
// uniformDist: Uniform[EmptyTuple] = dimwit.stats.Uniform@796ad176

// GOOD: Explicit key threading
val (key1, key2) = rootKey.split2()
// key1: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [2465931498 3679230171]
// )
// key2: Key = Key(
//   jaxKey = Array((), dtype=key<fry>) overlaying:
// [255383827 267815257]
// )
val sample1 = normalDist.sample(key1)
// sample1: Tensor[EmptyTuple, Float] = 0.13790321
val sample2 = uniformDist.sample(key2)
// sample2: Tensor[EmptyTuple, Float] = 0.91457367

// AVOID: Stateful random number generators
```

### 6. JIT Compile Performance-Critical Functions

```scala
import dimwit.jax.Jit.jit

trait Input derives Label

val simpleFunc = (x: Tensor1[Input, Float]) => x *! Tensor0(2.0f)
// simpleFunc: Function1[Tensor1[Input, Float], Tensor[Tuple1[Input], Float]] = repl.MdocSession$MdocApp65$$Lambda/0x00007714b0963258@7a77384b

// GOOD: JIT for repeated calls
val jitFunc = jit(simpleFunc)
// jitFunc: Function1[Tensor1[Input, Float], Tensor[Tuple1[Input], Float]] = dimwit.jax.JitInternal$$$Lambda/0x00007714b0954e58@4bb7451a
val testData = Tensor1(Axis[Input]).fromArray(Array(1.0f, 2.0f, 3.0f))
// testData: Tensor[Tuple1[Input], Float] = [1. 2. 3.]
val predictions = jitFunc(testData)
// predictions: Tensor[Tuple1[Input], Float] = [2. 4. 6.]

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
