package dimwit.tensor

import scala.compiletime.{constValue, erasedValue}
import scala.compiletime.error

type Dim[T] = (Axis[T], Int)

object Axis:
  def apply[A]: Axis[A] =
    new AxisImpl[A]()

  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple      => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail       => h *: UnwrapAxes[tail]

/** Represents an axis with A. This maps the type-level to a runtime representation.
  */
sealed trait Axis[A]
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

import scala.compiletime.{constValue, erasedValue, summonInline}
object AxisIndices:

  class AxisIndicesImpl[T <: Tuple, Axiss <: Tuple](val values: List[Int]) extends AxisIndices[T, Axiss]

  private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
    inline erasedValue[ToFind] match
      case _: EmptyTuple     => Nil
      case _: (head *: tail) =>
        summonInline[AxisIndex[InTuple, head]].value :: indicesOfList[InTuple, tail]

  inline given [T <: Tuple, ToFind <: Tuple]: AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

end AxisIndices
