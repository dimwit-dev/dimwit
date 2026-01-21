package dimwit.tensor

import scala.collection.View.Empty
import scala.annotation.publicInBinary
import dimwit.tensor.{Labels, Label}
import scala.annotation.unchecked.uncheckedVariance

/** Represents the (typed) Shape of a tensor with runtime labels
  */
final case class Shape[+T <: Tuple: Labels] @publicInBinary private (
    val dimensions: List[Int]
):

  lazy val labels: List[String] = summon[Labels[T]].names

  def rank: Int = dimensions.size
  def size: Int = dimensions.foldLeft(1)((acc, d) => acc * d.asInstanceOf[Int])
  def extent[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): AxisExtent[L] = AxisExtent(axis, this(axis))
  def apply[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Int = this.dimensions(axisIndex.value)

  override def toString: String =
    labels
      .zip(dimensions)
      .map((label, dim) => s"$label -> $dim")
      .mkString("Shape(", ", ", ")")

  override def equals(other: Any): Boolean = other match
    case s: Shape[?] => dimensions == s.dimensions && labels == s.labels
    case _           => false

  override def hashCode(): Int = dimensions.hashCode() ^ labels.hashCode()

object Shape:

  private[tensor] type ExtractLabels[Args <: Tuple] <: Tuple = Args match
    case EmptyTuple            => EmptyTuple
    case AxisExtent[l] *: tail => l *: ExtractLabels[tail]

  def empty: Shape[EmptyTuple] = new Shape(Nil)

  def apply[L: Label](dim: AxisExtent[L]): Shape[L *: EmptyTuple] =
    Shape.fromTuple(Tuple1(dim))

  def apply[A <: Tuple](args: A)(using n: Labels[ExtractLabels[A]]): Shape[ExtractLabels[A]] =
    Shape.fromTuple(args)

  def fromTuple[A <: Tuple](args: A)(using n: Labels[ExtractLabels[A]]): Shape[ExtractLabels[A]] =
    val sizes = args.toList.collect:
      case ae: AxisExtent[?] => ae.size
    new Shape(sizes)

  private[tensor] def fromSeq[T <: Tuple: Labels](dims: Seq[Int]) = new Shape[T](dims.toList)

type Shape0 = Shape[EmptyTuple]
type Shape1[L] = Shape[L *: EmptyTuple]
type Shape2[L1, L2] = Shape[L1 *: L2 *: EmptyTuple]
type Shape3[L1, L2, L3] = Shape[L1 *: L2 *: L3 *: EmptyTuple]

val Shape0 = Shape.empty

object Shape1:
  def apply[L: Label](dim: AxisExtent[L]): Shape[Tuple1[L]] = Shape(dim)

object Shape2:
  def apply[L1: Label, L2: Label](
      dim1: AxisExtent[L1],
      dim2: AxisExtent[L2]
  ): Shape[(L1, L2)] = Shape.fromTuple(dim1, dim2)

object Shape3:
  def apply[L1: Label, L2: Label, L3: Label](
      dim1: AxisExtent[L1],
      dim2: AxisExtent[L2],
      dim3: AxisExtent[L3]
  ): Shape[(L1, L2, L3)] = Shape.fromTuple(dim1, dim2, dim3)
