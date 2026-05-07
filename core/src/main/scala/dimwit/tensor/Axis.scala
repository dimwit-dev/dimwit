package dimwit.tensor

import dimwit.|*|

import scala.compiletime.{constValue, erasedValue, summonInline}

/** Instances of this class represent an axis in a tensor with a specific label `L`.
  * Axis objects are used whenever an axis needs to be selected at the value level,
  * such as when indexing into a tensor or defining the shape of a tensor.
  */
final class Axis[L: Label]:
  def extent(size: Int): AxisExtent[L] = AxisExtent(this, size)
  def ->(size: Int): AxisExtent[L] = this.extent(size)
  def at(index: Int): AxisAtIndex[L] = AxisAtIndex(this, index)
  def at(range: Range): AxisAtRange[L] = AxisAtRange(this, range)
  def at(indices: Seq[Int]): AxisAtIndices[L] = AxisAtIndices(this, indices)
  def at(index: Tensor0[Int]): AxisAtTensorIndex[L] = AxisAtTensorIndex(this, index)
  def at[I <: NonEmptyTuple](indices: I): AxisAtTupleIndices[L, I] = AxisAtTupleIndices(this, indices)
  def as[U](newAxis: Axis[U]): (Axis[L], Axis[U]) = (this, newAxis)

/** Represents the extent of an axis, which is a combination of an Axis and its size. */
case class AxisExtent[L: Label](axis: Axis[L], size: Int):

  /** Combines this AxisExtent with another AxisExtent to create a new AxisExtent that represents the combined axes and their sizes.
    *
    * @param other The other AxisExtent to combine with this one.
    * @return A new AxisExtent representing the combined axes and their sizes.
    */
  def *[L2: Label](other: AxisExtent[L2]): AxisExtent[L |*| L2] =
    AxisExtent(Axis[L |*| L2], size * other.size)

/** Trait hierarchy to represent different ways to select an axis in a tensor, such as by index, range, or specific indices.
  */
sealed trait AxisSelector[L]:
  def axis: Axis[L]

/** Represent an axis selection by a single index.
  */
case class AxisAtIndex[L](axis: Axis[L], index: Int) extends AxisSelector[L]

/** Represent an axis selection by a range of indices. */
case class AxisAtRange[L](axis: Axis[L], range: Range) extends AxisSelector[L]

/** Represent an axis selection by a sequence of specific indices. */
case class AxisAtIndices[L](axis: Axis[L], indices: Seq[Int]) extends AxisSelector[L]

/* Represent an axis selection by a tensor containing indices. This allows for dynamic indexing based on the contents of the tensor. */
case class AxisAtTensorIndex[L](axis: Axis[L], index: Tensor0[Int]) extends AxisSelector[L]

/* Represent an axis selection by a tuple containing indices. */
case class AxisAtTupleIndices[L, I <: NonEmptyTuple](axis: Axis[L], indices: I) extends AxisSelector[L]
