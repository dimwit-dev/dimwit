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

  require(
    dimensions.size == labels.size,
    s"Dimensions and labels must have the same size but got ${dimensions.size} dims and ${labels.size} labels, dimensions: $dimensions, labels: ${labels.mkString(", ")}"
  )
  require(dimensions.forall(_ > 0), "All dimensions must be positive")
  // TODO maybe same Axis must means symetric along these axes? => same length
  // require(labels.distinct.size == labels.size, "Labels must be unique")

  def rank: Int = dimensions.size
  def size: Int = dimensions.foldLeft(1)((acc, d) => acc * d.asInstanceOf[Int])
  def dim[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Dim[L] = axis -> this(axis)
  def apply[L](axis: Axis[L])(using axisIndex: AxisIndex[T @uncheckedVariance, L]): Int = this.dimensions(axisIndex.value)

  def *:[U <: Tuple: Labels](other: Shape[U]): Shape[Tuple.Concat[U, T]] =
    import Labels.ForConcat.given
    new Shape(other.dimensions ++ dimensions)

  override def toString: String =
    labels
      .zip(dimensions)
      .map((label, dim) => s"$label -> $dim")
      .mkString("Shape(", ", ", ")")

  override def equals(other: Any): Boolean = other match
    case s: Shape[?] => dimensions == s.dimensions && labels == s.labels
    case _           => false

  override def hashCode(): Int = dimensions.hashCode() ^ labels.hashCode()

  def ++[U <: Tuple: Labels](other: Shape[U]): Shape[Tuple.Concat[U, T]] =
    import Labels.ForConcat.given
    new Shape(other.dimensions ++ dimensions)

  def +:[NewL: Label](dim: (Axis[NewL], Int)): Shape[NewL *: T] =
    new Shape(dim._2 :: dimensions)

object Shape:

  def empty: Shape[EmptyTuple] = new Shape(Nil)

  type ExtractLabels[Args <: Tuple] <: Tuple = Args match
    case EmptyTuple             => EmptyTuple
    case (Axis[l], Int) *: tail => l *: ExtractLabels[tail]

  def apply[L: Label](dim: (Axis[L], Int)): Shape[L *: EmptyTuple] =
    Shape.fromTuple(Tuple1(dim))

  def apply[A <: Tuple](args: A)(using
      n: Labels[ExtractLabels[A]]
  ): Shape[ExtractLabels[A]] = Shape.fromTuple(args)

  def fromTuple[A <: Tuple](args: A)(using
      n: Labels[ExtractLabels[A]]
  ): Shape[ExtractLabels[A]] =
    val sizes = args.toList.collect { case (_, s: Int) =>
      s
    }
    new Shape(sizes)

  private[tensor] def fromList[T <: Tuple: Labels](dims: List[Int]) = new Shape[T](dims)

type Shape0 = Shape[EmptyTuple]
type Shape1[L] = Shape[L *: EmptyTuple]
type Shape2[L1, L2] = Shape[L1 *: L2 *: EmptyTuple]
type Shape3[L1, L2, L3] = Shape[L1 *: L2 *: L3 *: EmptyTuple]

val Shape0 = Shape.empty

object Shape1:
  def apply[L: Label](dim: Dim[L]): Shape[Tuple1[L]] = Shape(dim)

object Shape2:
  def apply[L1: Label, L2: Label](
      dim1: Dim[L1],
      dim2: Dim[L2]
  ): Shape[(L1, L2)] = Shape.fromTuple(dim1, dim2)

object Test:
  val x = Shape2(Axis["A"] -> 3, Axis["B"] -> 4)

object Shape3:
  def apply[L1: Label, L2: Label, L3: Label](
      dim1: Dim[L1],
      dim2: Dim[L2],
      dim3: Dim[L3]
  ): Shape[(L1, L2, L3)] = Shape.fromTuple(dim1, dim2, dim3)
