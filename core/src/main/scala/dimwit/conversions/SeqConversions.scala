package dimwit.conversions

import dimwit.tensor.*
import dimwit.tensor.DType.*
import dimwit.tensor.ValueTypeClasses.IsBoolean
import dimwit.tensor.ValueTypeClasses.IsFloating
import dimwit.tensor.ValueTypeClasses.IsInteger

import scala.annotation.targetName
import scala.collection.immutable.ArraySeq

/* Conversions between Scala sequences and tensors. They are opt-in: `import dimwit.conversions.*` makes them
 * available, `import dimwit.*` does not.
 *
 * From Seqs to tensors, e.g. `Seq(1.0f, 2.0f).toTensor1(Axis[A])`: the value type is derived from the element type
 * as for `fromArray` (Float to Float32, Double to Float64, Byte to Int8, Short to Int16, Int to Int32, Long to Int64,
 * Boolean to Bool), or given explicitly, e.g. `Seq(1.0f, 2.0f).toTensor1(Axis[A], VType[Float16])`.
 * For `toTensor2` the outer Seq runs along the first axis and the inner Seqs, the rows, along the second.
 * All rows must have the same length, and there must be at least one row.
 *
 * From tensors to Seqs with `toSeq`, the reverse: `xs.toTensor1(Axis[A]).toSeq == xs`.
 */

extension (values: Seq[Boolean])
  @targetName("booleanSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Bool] = Tensor1(axis).fromArray(values.toArray)
  @targetName("booleanSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsBoolean](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Byte])
  @targetName("byteSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Int8] = Tensor1(axis).fromArray(values.toArray)
  @targetName("byteSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsInteger](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Short])
  @targetName("shortSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Int16] = Tensor1(axis).fromArray(values.toArray)
  @targetName("shortSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsInteger](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Int])
  @targetName("intSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Int32] = Tensor1(axis).fromArray(values.toArray)
  @targetName("intSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsInteger](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Long])
  @targetName("longSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Int64] = Tensor1(axis).fromArray(values.toArray)
  @targetName("longSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsInteger](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Float])
  @targetName("floatSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Float32] = Tensor1(axis).fromArray(values.toArray)
  @targetName("floatSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsFloating](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Double])
  @targetName("doubleSeqToTensor1")
  def toTensor1[L: Label](axis: Axis[L]): Tensor1[L, Float64] = Tensor1(axis).fromArray(values.toArray)
  @targetName("doubleSeqToTensor1Typed")
  def toTensor1[L: Label, V: IsFloating](axis: Axis[L], vtype: VType[V]): Tensor1[L, V] = Tensor1(axis, vtype).fromArray(values.toArray)

extension (values: Seq[Seq[Boolean]])
  @targetName("booleanSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Bool] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("booleanSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsBoolean](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Byte]])
  @targetName("byteSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Int8] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("byteSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsInteger](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Short]])
  @targetName("shortSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Int16] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("shortSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsInteger](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Int]])
  @targetName("intSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Int32] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("intSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsInteger](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Long]])
  @targetName("longSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Int64] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("longSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsInteger](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Float]])
  @targetName("floatSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Float32] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("floatSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsFloating](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

extension (values: Seq[Seq[Double]])
  @targetName("doubleSeqToTensor2")
  def toTensor2[L1: Label, L2: Label](axis1: Axis[L1], axis2: Axis[L2]): Tensor2[L1, L2, Float64] = Tensor2(axis1, axis2).fromArray(values.map(_.toArray).toArray)
  @targetName("doubleSeqToTensor2Typed")
  def toTensor2[L1: Label, L2: Label, V: IsFloating](axis1: Axis[L1], axis2: Axis[L2], vtype: VType[V]): Tensor2[L1, L2, V] =
    Tensor2(axis1, axis2, vtype).fromArray(values.map(_.toArray).toArray)

// --- From tensors to Seqs ---
// The elements are read into a fresh array, which is wrapped without copying. Nothing else references it,
// so the result is a proper immutable IndexedSeq. Other collections are one call away, e.g. `t.toSeq.toList`.

extension [L, V, X](t: Tensor1[L, V])(using ev: HasScalar[V, X])
  /** The elements of the vector, in order. Not available inside `jit`, as the values of a traced tensor are unknown. */
  @targetName("tensor1ToSeq")
  def toSeq: IndexedSeq[X] =
    require(!t.isTracer, "Cannot convert a JAX Tracer to a Seq.")
    ArraySeq.unsafeWrapArray(ev.readFlat(t.jaxValue))

extension [L1, L2, V, X](t: Tensor2[L1, L2, V])(using ev: HasScalar[V, X])
  /** The rows of the matrix, i.e. the outer Seq runs along the first axis and the inner Seqs along the second,
    * as for `toTensor2`. Not available inside `jit`, as the values of a traced tensor are unknown.
    */
  @targetName("tensor2ToSeq")
  def toSeq: IndexedSeq[IndexedSeq[X]] =
    require(!t.isTracer, "Cannot convert a JAX Tracer to a Seq.")
    val rows = t.shape.dimensions(0)
    val columns = t.shape.dimensions(1)
    val values = ev.readFlat(t.jaxValue)
    IndexedSeq.tabulate(rows)(row => ArraySeq.unsafeWrapArray(values.slice(row * columns, (row + 1) * columns)))
