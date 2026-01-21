package dimwit.tensor

import scala.compiletime.{constValue, erasedValue, summonInline}

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

trait AxisIndex[Shape <: Tuple, +Axis]:
  def value: Int

object AxisIndex:

  def apply[T <: Tuple, L](using idx: AxisIndex[T, L]): Int = idx.value

  given head[L, Tail <: Tuple]: AxisIndex[L *: Tail, L] with
    val value = 0

  given tail[H, T <: Tuple, L](using
      next: AxisIndex[T, L]
  ): AxisIndex[H *: T, L] with
    val value = 1 + next.value

  given concatRight[A <: Tuple, B <: Tuple, L](using
      sizeA: ValueOf[Tuple.Size[A]],
      idxB: AxisIndex[B, L]
  ): AxisIndex[Tuple.Concat[A, B], L] with
    val value = sizeA.value + idxB.value

  given concatEnd[A <: Tuple, L]: AxisIndex[Tuple.Concat[A, Tuple1[L]], L] with
    val value = -1

sealed trait AxisIndices[T <: Tuple, Axiss <: Tuple]:
  def values: List[Int]

object AxisIndices:

  class AxisIndicesImpl[T <: Tuple, Axiss <: Tuple](val values: List[Int]) extends AxisIndices[T, Axiss]

  private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => Nil
      case _: (head *: tail) =>
        summonInline[AxisIndex[InTuple, head]].value :: indicesOfList[InTuple, tail]

  inline given [T <: Tuple, ToFind <: Tuple]: AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

end AxisIndices
