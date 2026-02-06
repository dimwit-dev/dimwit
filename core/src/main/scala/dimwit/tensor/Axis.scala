package dimwit.tensor

case class AxisExtent[T](axis: Axis[T], size: Int)

// Axis selectors for indexing operations
sealed trait AxisSelector[L]:
  def axis: Axis[L]

case class AxisAtIndex[L](axis: Axis[L], index: Int) extends AxisSelector[L]
case class AxisAtRange[L](axis: Axis[L], range: Range) extends AxisSelector[L]
case class AxisAtIndices[L](axis: Axis[L], indices: Seq[Int]) extends AxisSelector[L]
case class AxisAtTensorIndex[L](axis: Axis[L], index: Tensor0[Int]) extends AxisSelector[L]

object Axis:

  def apply[A]: Axis[A] = new AxisImpl[A]()

/** Represents an axis with A. This maps the type-level to a runtime representation. */
sealed trait Axis[A]:
  def extent(size: Int): AxisExtent[A] = AxisExtent(this, size)
  def ->(size: Int): AxisExtent[A] = this.extent(size)
  def at(index: Int): AxisAtIndex[A] = AxisAtIndex(this, index)
  def at(range: Range): AxisAtRange[A] = AxisAtRange(this, range)
  def at(indices: Seq[Int]): AxisAtIndices[A] = AxisAtIndices(this, indices)
  def at(index: Tensor0[Int]): AxisAtTensorIndex[A] = AxisAtTensorIndex(this, index)
  def as[U](newAxis: Axis[U]): (Axis[A], Axis[U]) = (this, newAxis)

class AxisImpl[A] extends Axis[A]
