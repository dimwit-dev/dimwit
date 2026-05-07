package dimwit.tensor

import scala.collection.View.Empty
import scala.annotation.publicInBinary
import ShapeTypeHelpers.AxisIndex
import dimwit.tensor.{Labels, Label}

/** Represents the Shape of a tensor. Conceptually, a shape is an order list of AxisExtents,
  * where each AxisExtent is a label associated with a size.
  */
final case class Shape[T <: Tuple: Labels] @publicInBinary private (
    val dimensions: List[Int]
):

  /** Returns the labels of the shape
    */
  lazy val labels: List[String] = summon[Labels[T]].names

  /** Returns the rank of the shape, which is the number of dimensions.
    */
  def rank: Int = dimensions.size

  /** Returns the total number of elements in the tensor represented by this shape.
    */
  def size: Int = dimensions.foldLeft(1)((acc, d) => acc * d.asInstanceOf[Int])

  /** Returns the extent of the specified axis.
    *
    * @param axis The axis for which to retrieve the extent.
    * @return The extent of the specified axis.
    */
  def extent[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): AxisExtent[L] = AxisExtent(axis, this(axis))

  /** Returns the size of the specified axis.
    *
    * @param axis The axis for which to retrieve the size.
    * @return The size of the specified axis.
    */
  def apply[L](axis: Axis[L])(using ev: AxisIndex[T, L]): Int = this.dimensions(ev.index)

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

  private[tensor] type ExtractLabels[Extents <: Tuple] <: Tuple = Extents match
    case EmptyTuple            => EmptyTuple
    case AxisExtent[l] *: tail => l *: ExtractLabels[tail]

  /** Creates an empty shape with no dimensions.
    * @return An empty shape with no dimensions.
    */
  def empty: Shape[EmptyTuple] = new Shape(Nil)

  /** Creates a shape with a single dimension.
    *
    * @param dim The extent of the single dimension.
    * @return A shape with one dimension.
    */
  def apply[L: Label](dim: AxisExtent[L]): Shape[L *: EmptyTuple] =
    Shape.fromTuple(Tuple1(dim))

  /** Create a shape from a tuple of AxisExtents.
    *
    * @param axisExtends A tuple of AxisExtents from which to create the shape.
    * @return A shape with dimensions corresponding to the sizes of the AxisExtents in the tuple.
    */
  def apply[Extents <: Tuple](axisExtends: Extents)(using n: Labels[ExtractLabels[Extents]]): Shape[ExtractLabels[Extents]] =
    fromTuple(axisExtends)

  /** Create a shape from a tuple of AxisExtents.
    * @param axisExtents A tuple of AxisExtents from which to create the shape.
    * @return A shape with dimensions corresponding to the sizes of the AxisExtents in the tuple.
    */
  def fromTuple[Extents <: Tuple](axisExtents: Extents)(using n: Labels[ExtractLabels[Extents]]): Shape[ExtractLabels[Extents]] =
    val sizes = axisExtents.toList.collect:
      case ae: AxisExtent[?] => ae.size
    new Shape(sizes)

  private[tensor] def fromSeq[T <: Tuple: Labels](dims: Seq[Int]) = new Shape[T](dims.toList)

/** Type alias for an empty shape (rank 0). */
type Shape0 = Shape[EmptyTuple]

/** Type alias for a shape with one dimension (rank 1). */
type Shape1[L] = Shape[L *: EmptyTuple]

/** Type alias for a shape with two dimensions (rank 2). */
type Shape2[L1, L2] = Shape[L1 *: L2 *: EmptyTuple]

/** Type alias for a shape with three dimensions (rank 3). */
type Shape3[L1, L2, L3] = Shape[L1 *: L2 *: L3 *: EmptyTuple]

/** Companion object for Shape0, providing a convenient way to create an empty shape. */
val Shape0 = Shape.empty

/** Companion object for Shape1, providing a convenient way to create a shape with one dimension. */
object Shape1:

  /** Creates a shape with a single dimension.
    *
    * @param dim The extent of the single dimension.
    * @return A shape with one dimension.
    */
  def apply[L: Label](dim: AxisExtent[L]): Shape[Tuple1[L]] = Shape(dim)

/** Companion object for Shape2, providing a convenient way to create a shape with two dimensions. */
object Shape2:

  /** Creates a shape with two dimensions.
    *
    * @param dim1 The extent of the first dimension.
    * @param dim2 The extent of the second dimension.
    * @return A shape with two dimensions.
    */
  def apply[L1: Label, L2: Label](
      dim1: AxisExtent[L1],
      dim2: AxisExtent[L2]
  ): Shape[(L1, L2)] = Shape.fromTuple((dim1, dim2))

object Shape3:

  /** Creates a shape with three dimensions.
    *
    * @param dim1 The extent of the first dimension.
    * @param dim2 The extent of the second dimension.
    * @param dim3 The extent of the third dimension.
    * @return A shape with three dimensions.
    */
  def apply[L1: Label, L2: Label, L3: Label](
      dim1: AxisExtent[L1],
      dim2: AxisExtent[L2],
      dim3: AxisExtent[L3]
  ): Shape[(L1, L2, L3)] = Shape.fromTuple((dim1, dim2, dim3))
