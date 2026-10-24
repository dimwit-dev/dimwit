package dimwit.tensor.tensorops

import dimwit.python.PyIndex.itemAt
import dimwit.jax.Einops
import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.AxisAtIndex
import dimwit.tensor.AxisAtIndices
import dimwit.tensor.AxisAtRange
import dimwit.tensor.AxisAtTensorIndex
import dimwit.tensor.AxisAtTensorIndices
import dimwit.tensor.AxisAtWindow
import dimwit.tensor.AxisAtTupleIndices
import dimwit.tensor.AxisExtent
import dimwit.tensor.DType.Bool
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Shape
import dimwit.tensor.ShapeTypeHelpers.AxesMerger
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.AxisReplacer
import dimwit.tensor.ShapeTypeHelpers.AxisReplacerAll
import dimwit.tensor.ShapeTypeHelpers.DimExtractor
import dimwit.tensor.ShapeTypeHelpers.MergeLabels
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.ShapeTypeHelpers.UnwrapDims
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor1
import dimwit.tensor.TupleHelpers
import dimwit.tensor.TupleHelpers.StrictSubset
import dimwit.tensor.TensorEvidence.CheckValid
import dimwit.tensor.TensorEvidence.ComputeMissing
import dimwit.tensor.TensorEvidence.IsPermutation
import dimwit.tensor.TensorEvidence.ValidationResult
import dimwit.tensor.Broadcast3
import dimwit.|+|
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Reader
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.implicitNotFound
import scala.util.NotGiven

/** Structural operations on tensors, e.g. `Tensor.stack` or `Tensor.concatenate`, and the type classes they need. */
private[dimwit] object StructuralOps:

  /** Inserts axis `New` directly after axis `Anchor`. */
  trait AxisInserter[T <: Tuple, Anchor, New]:
    type Out <: Tuple

  object AxisInserter extends AxisInserterLowPriority:
    type Aux[T <: Tuple, Anchor, New, O <: Tuple] = AxisInserter[T, Anchor, New] { type Out = O }

    private[tensorops] def instance[T <: Tuple, Anchor, New, O <: Tuple]: Aux[T, Anchor, New, O] =
      new AxisInserter[T, Anchor, New]:
        type Out = O

    given found[Anchor, New, T <: Tuple]: Aux[Anchor *: T, Anchor, New, Anchor *: New *: T] = instance

  trait AxisInserterLowPriority:
    given search[H, Anchor, New, T <: Tuple, O <: Tuple](using
        tail: AxisInserter.Aux[T, Anchor, New, O]
    ): AxisInserter.Aux[H *: T, Anchor, New, H *: O] = AxisInserter.instance

  /** Exchanges two axes in a shape, leaving every other axis in place.
    *
    * The three priority tiers resolve `swap(a, a)` to [[firstTuple]] rather than
    * leaving it ambiguous between [[firstTuple]] and [[AxisSwapperSecond.secondTuple]].
    */
  trait AxisSwapper[T <: Tuple, L1, L2]:
    type Out <: Tuple

  object AxisSwapper extends AxisSwapperSecond:
    type Aux[T <: Tuple, L1, L2, O <: Tuple] = AxisSwapper[T, L1, L2] { type Out = O }

    private[tensorops] def instance[T <: Tuple, L1, L2, O <: Tuple]: Aux[T, L1, L2, O] =
      new AxisSwapper[T, L1, L2]:
        type Out = O

    given emptyTuple[L1, L2]: Aux[EmptyTuple, L1, L2, EmptyTuple] = instance

    given firstTuple[L1, L2, T <: Tuple, O <: Tuple](using tail: Aux[T, L1, L2, O]): Aux[L1 *: T, L1, L2, L2 *: O] = instance

  trait AxisSwapperSecond extends AxisSwapperOther:
    given secondTuple[L1, L2, T <: Tuple, O <: Tuple](using
        tail: AxisSwapper.Aux[T, L1, L2, O]
    ): AxisSwapper.Aux[L2 *: T, L1, L2, L1 *: O] = AxisSwapper.instance

  trait AxisSwapperOther:
    given otherTuple[H, L1, L2, T <: Tuple, O <: Tuple](using
        tail: AxisSwapper.Aux[T, L1, L2, O]
    ): AxisSwapper.Aux[H *: T, L1, L2, H *: O] = AxisSwapper.instance

  private[dimwit] object Util:

    @implicitNotFound("The axis ${L} is already present in the tensor shape ${T}.")
    trait AxisAbsent[T, L]
    object AxisAbsent:
      given notContained[T <: Tuple, L](using NotGiven[Tuple.Contains[T, L] =:= true]): AxisAbsent[T, L] = new AxisAbsent[T, L] {}

  object TensorWhere:

    /** Returns a new tensor selecting elements from `ifTrue` where the condition is true
      * and from `ifFalse` where it is false.
      */
    def where[T <: Tuple: Labels, V](
        condition: Tensor[T, Bool],
        ifTrue: Tensor[T, V],
        ifFalse: Tensor[T, V]
    ): Tensor[T, V] =
      ElementWiseOps.requireSameShape(condition, ifTrue)
      ElementWiseOps.requireSameShape(ifTrue, ifFalse)
      Tensor(Jax.jnp.where(condition.jaxValue, ifTrue.jaxValue, ifFalse.jaxValue))

    /** Like [[where]], but broadcasts condition, `ifTrue` and `ifFalse` to their common shape,
      * which is the one of the three shapes containing all axes of the other two.
      */
    def where_![C <: Tuple, T <: Tuple, F <: Tuple, V](
        condition: Tensor[C, Bool],
        ifTrue: Tensor[T, V],
        ifFalse: Tensor[F, V]
    )(using
        @implicitNotFound(
          "Cannot broadcast condition ${C}, ifTrue ${T} and ifFalse ${F} to a common shape. One of them must contain all axes of the other two. If all have the same shape nothing has to be broadcast, use `where` instead of `where_!`."
        ) bc: Broadcast3[C, T, F, V]
    ): Tensor[bc.Out, V] =
      val (c, t, f) = bc.broadcast(condition, ifTrue, ifFalse)
      where(c, t, f)(using bc.labelsOut)

  export TensorWhere.where
  export TensorWhere.where_!

  /** Returns a new tensor with the upper triangular part of the input tensor,
    * setting elements below the kth diagonal to zero.
    *
    * @param tensor The input tensor from which to extract the upper triangular part.
    * @param kthDiagonal The diagonal above which to set elements to zero.
    *
    * @return A new tensor with the upper triangular part of the input tensor.
    */
  def triu[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
    Tensor(Jax.jnp.triu(tensor.jaxValue, k = kthDiagonal))

  /** Returns a new tensor with the lower triangular part of the input tensor,
    * setting elements above the kth diagonal to zero.
    *
    * @param tensor The input tensor from which to extract the lower triangular part.
    * @param kthDiagonal The diagonal below which to set elements to zero.
    *
    * @return A new tensor with the lower triangular part of the input tensor.
    */
  def tril[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
    Tensor(Jax.jnp.tril(tensor.jaxValue, k = kthDiagonal))

  /** Stacks a sequence of tensors along a new axis.
    * The new axis is inserted as the first axis of the resulting tensor.
    *
    * @param tensors A sequence of tensors to be stacked. All tensors must have the same shape and type.
    * @param newAxis The new axis to be inserted.
    * @return A new tensor with the stacked tensors.
    */
  def stack[L: Label, T <: Tuple: Labels, V](
      tensors: Seq[Tensor[T, V]],
      newAxis: Axis[L]
  ): Tensor[L *: T, V] =
    require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = 0)
    Tensor(stackedJaxValue)

  /** Stacks a sequence of tensors along a new axis, inserting the new axis
    * after the specified existing axis.
    *
    * @param tensors A sequence of tensors to be stacked. All tensors must have the same shape and type.
    * @param newAxis The new axis to be inserted.
    * @param afterAxis The existing axis after which the new axis will be inserted.
    * @return A new tensor with the stacked tensors.
    */
  def stack[NewL, L, T <: Tuple: Labels, V](
      tensors: Seq[Tensor[T, V]],
      newAxis: Axis[NewL],
      afterAxis: Axis[L]
  )(using
      newLabel: Label[NewL],
      axisIndex: AxisIndex[T, L],
      inserter: AxisInserter[T, L, NewL]
  ): Tensor[inserter.Out, V] =
    require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
    val axisIdx = axisIndex.index + 1 // we are inserting after the given axis, so shift by 1
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = axisIdx)
    val names = summon[Labels[T]].names
    val newNames = names.take(axisIdx) ++ Seq(newLabel.name) ++ names.drop(axisIdx)
    given Labels[inserter.Out] with
      val names = newNames.toSeq
    Tensor(stackedJaxValue)

  /** Concatenates a sequence of tensors along the specified axis, returning a new tensor with the concatenated values.
    *
    * @param tensors A sequence of tensors to be concatenated.
    *                All tensors must have the same shape and type,
    *                except for the dimension corresponding to the concatenation axis.
    * @param concatAxis The axis along which the tensors will be concatenated.
    * @return A new tensor with the concatenated values.
    */
  def concatenate[L: Label, T <: Tuple: Labels, V](
      tensors: Seq[Tensor[T, V]],
      concatAxis: Axis[L]
  )(using
      axisIndex: AxisIndex[T, L]
  ): Tensor[T, V] =
    require(tensors.nonEmpty, "Cannot concatenate an empty sequence of tensors")
    val axisIdx = axisIndex.index
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
    Tensor(concatenatedJaxValue)

  /** Concatenates two tensors along the specified axis,
    * returning a new tensor with the concatenated values.
    *
    * @param t1 The first tensor to be concatenated.
    * @param t2 The second tensor to be concatenated.
    * @param concatAxis The axis along which the tensors will be concatenated.
    * @return A new tensor with the concatenated values.
    */
  def concatenate[L: Label, T <: Tuple: Labels, V](
      t1: Tensor[T, V],
      t2: Tensor[T, V],
      concatAxis: Axis[L]
  )(using
      axisIndex: AxisIndex[T, L]
  ): Tensor[T, V] = concatenate(Seq(t1, t2), concatAxis)

  /** Concatenates two tensors along the common axis, returning a new tensor with the concatenated values.
    */
  def concatenate[T1 <: Tuple, T2 <: Tuple, V, R <: Tuple](
      t1: Tensor[T1, V],
      t2: Tensor[T2, V]
  )(using
      canConcat: ValidConcat.Aux[T1, T2, R],
      label: Labels[R]
  ): Tensor[R, V] =
    val jaxValues = List(t1.jaxValue, t2.jaxValue).toPythonProxy
    Tensor(Jax.jnp.concatenate(jaxValues, axis = canConcat.index))

  trait ValidConcat[T1 <: Tuple, T2 <: Tuple]:
    type Out <: Tuple
    def index: Int

  object ValidConcat:
    type Aux[T1 <: Tuple, T2 <: Tuple, O <: Tuple] = ValidConcat[T1, T2] { type Out = O }

    given sameAxisTuple[H, T1Tail <: Tuple, T2Tail <: Tuple, OutTail <: Tuple](using
        next: ValidConcat.Aux[T1Tail, T2Tail, OutTail]
    ): ValidConcat[H *: T1Tail, H *: T2Tail] with
      type Out = H *: OutTail
      def index: Int = next.index + 1

    given concatAxisTuple[H1, H2, Tail <: Tuple](using
        isDifferent: NotGiven[H1 =:= H2]
    ): ValidConcat[H1 *: Tail, H2 *: Tail] with
      type Out = (H1 |+| H2) *: Tail
      def index: Int = 0

  type SplitComponents[L, I <: Tuple] <: Tuple = I match
    case EmptyTuple => L *: EmptyTuple
    case _ *: tail  => L *: SplitComponents[L, tail]

  trait Deconcatenator[L]:
    type Components <: Tuple
    def labels: List[Label[?]]

  object Deconcatenator extends DeconcatenatorLowPriority:
    type Aux[L, C <: Tuple] = Deconcatenator[L] { type Components = C }

    given concatenated[A, B, CA <: Tuple, CB <: Tuple](using
        da: Aux[A, CA],
        db: Aux[B, CB]
    ): Aux[A |+| B, Tuple.Concat[CA, CB]] =
      new Deconcatenator[A |+| B]:
        type Components = Tuple.Concat[CA, CB]
        def labels = da.labels ++ db.labels

  trait DeconcatenatorLowPriority:
    given single[L](using l: Label[L]): Deconcatenator.Aux[L, L *: EmptyTuple] =
      new Deconcatenator[L]:
        type Components = L *: EmptyTuple
        def labels = List(l)

  trait TensorTupleMaker[Components <: Tuple, FullShape <: Tuple, SplitAxis, V]:
    type Out <: Tuple
    def apply(arrays: Seq[Jax.PyDynamic], compLabels: List[Label[?]], originalLabels: Seq[String], splitIndex: Int): Out

  object TensorTupleMaker:
    type Aux[C <: Tuple, F <: Tuple, S, V, O <: Tuple] =
      TensorTupleMaker[C, F, S, V] { type Out = O }

    given emptyTuple[F <: Tuple, S, V]: Aux[EmptyTuple, F, S, V, EmptyTuple] =
      new TensorTupleMaker[EmptyTuple, F, S, V]:
        type Out = EmptyTuple
        def apply(a: Seq[Jax.PyDynamic], c: List[Label[?]], o: Seq[String], i: Int) = EmptyTuple

    given consTuple[Head, Tail <: Tuple, F <: Tuple, S, V, NewShape <: Tuple](using
        replacer: TupleHelpers.Replacer[F, S, Head] { type Out = NewShape },
        tailMaker: TensorTupleMaker[Tail, F, S, V]
    ): Aux[Head *: Tail, F, S, V, Tensor[NewShape, V] *: tailMaker.Out] =

      new TensorTupleMaker[Head *: Tail, F, S, V]:
        type Out = Tensor[NewShape, V] *: tailMaker.Out

        def apply(arrays: Seq[Jax.PyDynamic], compLabels: List[Label[?]], originalLabels: Seq[String], splitIndex: Int): Out =
          val currentArr = arrays.head
          val currentLabel = compLabels.head
          val newNames = originalLabels.updated(splitIndex, currentLabel.name).toList
          val newLabelsWitness = new Labels[NewShape]:
            val names = newNames
          val headTensor = Tensor[NewShape, V](currentArr)(using newLabelsWitness)
          headTensor *: tailMaker(arrays.tail, compLabels.tail, originalLabels, splitIndex)

/** Extension methods for structural operations, e.g. `t.transpose`, `t.select(...)` or `t.rearrange(...)`. */
private[dimwit] object StructuralExtensions:

  import StructuralOps.*
  import StructuralOps.Util.*

  /** The Python list of `indices`. */
  private def pyIndices(indices: Seq[Int]): py.Any =
    indices.map(py.Any.from).toPythonCopy // TODO find out why Copy is needed here

  /** Requires that `value` has the shape `extents` of the selection it is written to, so that JAX does not broadcast it. */
  private def requireSelectionShape(value: Tensor[?, ?], extents: Seq[Int]): Unit =
    require(
      value.shape.dimensions == extents,
      s"Cannot set a value of shape ${value.shape} into a selection of shape ${value.axes.zip(extents).map((a, n) => s"$a -> $n").mkString("(", ", ", ")")}"
    )

  extension [T <: Tuple, V](tensor: Tensor[T, V])

    /** takes a concatenated tensor and splits it into a tuple of tensors along the specified axis,
      *  using the provided dimensions for each component.
      *
      * @param axis The axis along which to deconcatenate the tensor.
      * @param dims A tuple of AxisExtent specifying the sizes of each component along the specified axis.
      * @return A tuple of tensors corresponding to the deconcatenated components.
      *
      * Example usage:
      * {{{
      *   val t : Tensor2[Axis[A], Axis[B |+| C]) = ???
      *   val (partB, partC) = t.deconcatenate(Axis[B |+| C], (Axis[B] -> 2, Axis[C] -> 3)
      * }}}
      */
    def deconcatenate[L, Dims <: Tuple, Comps <: Tuple, Result](
        axis: Axis[L],
        dims: Dims
    )(using
        labels: Labels[T],
        axisIndex: AxisIndex[T, L],
        decon: Deconcatenator.Aux[L, Comps],
        extractor: DimExtractor[Dims],
        maker: TensorTupleMaker[Comps, T, L, V]
    ): maker.Out =
      val orderedSizes = dims.toList.asInstanceOf[List[Any]].map {
        case ae: AxisExtent[?] => ae.size
        case _                 => throw new IllegalArgumentException("Invalid dims format - expected AxisExtent")
      }

      require(orderedSizes.size == decon.labels.size, s"Provided ${orderedSizes.size} sizes but axis has ${decon.labels.size} components")

      val splitIndices = orderedSizes.scanLeft(0)(_ + _).tail.init
      val pyIndices = me.shadaj.scalapy.py.Dynamic.global.list(splitIndices.toPythonProxy)
      val splitArrays = Jax.jnp.split(tensor.jaxValue, pyIndices, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      val originalNames = summon[Labels[T]].names.toSeq

      maker.apply(splitArrays, decon.labels, originalNames, axisIndex.index)

    /** Flattens all axes of the tensor into a single axis.
      * The resulting tensor will have a single axis named by concatenating the original axis names with "*".
      *
      * @return a Tensor1 with the merged axis
      */
    def flatten(using labels: Labels[T]): Tensor1[MergeLabels[T], V] =
      given Labels[Tuple1[MergeLabels[T]]] with
        def names = List(summon[Labels[T]].names.mkString("*"))
      Tensor(Jax.jnp.ravel(tensor.jaxValue))

    /** Flattens the specified axes of the tensor into a single axis.
      * The resulting tensor will have the specified axes merged into a single axis named by concatenating the original axis names with "*"
      * The other axes remain unchanged.
      *
      * @param axes the axes to flatten, specified as a tuple of Axis (e.g. (Axis[Ax1], Axis[Ax2]))
      * @return a Tensor with the specified axes merged into a single axis
      */
    def flatten[AxesTuple <: Tuple, R <: Tuple](
        axes: AxesTuple
    )(using
        merger: AxesMerger.Aux[T, UnwrapAxes[AxesTuple], R],
        labels: Labels[R]
    ): Tensor[R, V] =
      val permuted = Jax.jnp.transpose(tensor.jaxValue, merger.permutation.toPythonProxy)

      val originalDims = tensor.shape.dimensions
      val mergedSize = merger.mergeIndices.map(originalDims).product

      val remainingDims = originalDims.zipWithIndex
        .filterNot((d, i) => merger.mergeIndices.contains(i))
        .map(_._1)

      val newDimensions = remainingDims.patch(merger.mergedIndex, Seq(mergedSize), 0)

      Tensor(Jax.jnp.reshape(permuted, newDimensions.toPythonProxy))

    /** Unflattens splitAxis into a new shape specified by newShape. The other axes remain unchanged.
      *
      * The user must ensure that the size of splitAxis matches the product of the dimensions in newShape, otherwise a runtime error will occur.
      *
      * @param splitAxis the axis to unflatten
      * @param newShape the new shape to unflatten into, specified as a Shape
      * @return a Tensor with the specified axis unflattened into the new shape
      */
    def unflatten[SplitL, NewT <: Tuple, R <: Tuple](
        splitAxis: Axis[SplitL],
        newShape: Shape[NewT]
    )(using
        ev: AxisReplacerAll.Aux[T, SplitL, NewT, R],
        labels: Labels[R]
    ): Tensor[R, V] =
      val before = tensor.shape.dimensions.take(ev.index)
      val after = tensor.shape.dimensions.drop(ev.index + 1)
      val fullNewShape = before ++ newShape.dimensions ++ after
      Tensor(
        Jax.jnp.reshape(
          tensor.jaxValue,
          py.Dynamic.global.tuple(
            fullNewShape.map(py.Any.from).toPythonProxy
          )
        )
      )

    /** Unflattens the tensor into a new shape specified by newShape.
      *
      * The user must ensure that the size of the tensor matches the product of the dimensions in newShape, otherwise a runtime error will occur.
      *
      * @param newShape the new shape to unflatten into, specified as a Shape
      * @return a Tensor with the new shape
      */
    def unflatten[NewT <: Tuple: Labels](
        newShape: Shape[NewT]
    )(using
        @implicitNotFound("unflatten without axis can only be used on Tensor1 types.")
        ev: T <:< Tuple1[Any] // <--- Ensures this only works on Tensor1
    ): Tensor[NewT, V] =
      val fullNewShape = newShape.dimensions
      Tensor(
        Jax.jnp.reshape(
          tensor.jaxValue,
          py.Dynamic.global.tuple(
            fullNewShape.map(py.Any.from).toPythonProxy
          )
        )
      )

    /** Transposes the tensor according to the specified new order of axes.
      *
      * @param NewOrder A tuple representing the new order of axes for the tensor.
      * @return A new tensor with the axes transposed according to the specified order.
      */
    def transpose[NewOrder <: Tuple, Status <: ValidationResult](newOrder: NewOrder)(using
        ev: AxisIndices[T, UnwrapAxes[NewOrder]],
        newLabels: Labels[UnwrapAxes[NewOrder]]
    )(using
        allAxesEv: IsPermutation[T, UnwrapAxes[NewOrder]]
    ): Tensor[UnwrapAxes[NewOrder], V] =
      val indices = ev.indices
      Tensor(Jax.jnp.transpose(tensor.jaxValue, indices.toPythonProxy))

    /** Splits the tensor along the specified axis at the given indices, returning a tuple of tensors corresponding to the splits.
      *
      * @param selector of the form Axis[L].at((idx1, idx2, ...)) specifying the axis to split and the indices to split at
      * @return the tuple of tensors resulting from the split
      */
    def split[L: Label, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using
        axisIndex: AxisIndex[T, L],
        maker: TensorTupleMaker[SplitComponents[L, I], T, L, V],
        labels: Labels[T]
    ): maker.Out =
      val splitList = selector.indices.toList.asInstanceOf[List[Int]]
      val pyIndices = me.shadaj.scalapy.py.Dynamic.global.list(splitList.toPythonProxy)
      val splitArrays = Jax.jnp.split(tensor.jaxValue, pyIndices, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      val axisLabelInstance = summon[Label[L]]
      val compLabels = List.fill(splitList.size + 1)(axisLabelInstance.asInstanceOf[Label[?]])
      maker.apply(splitArrays, compLabels, labels.names.toSeq, axisIndex.index)

    /** Splits the tensor along the specified axis at the given index,
      * returning a tuple of two tensors corresponding to the splits.
      *
      * @param selector of the form Axis[L].at(idx) specifying the axis to split and the index to split at
      * @return a tuple of two tensors resulting from the split
      */
    def split[L: Label](selector: AxisAtIndex[L])(using
        axisIndex: AxisIndex[T, L],
        maker: TensorTupleMaker[L *: L *: EmptyTuple, T, L, V],
        labels: Labels[T]
    ): maker.Out =
      split(AxisAtTupleIndices(selector.axis, Tuple1(selector.index)))

    /** Unstacks the tensor along the specified axis at the given indices, returning a sequence of tensors corresponding to the splits.
      *
      * @param unstackAxis the axis to split, specified as an Axis (e.g. Axis[Ax1])
      * @return a sequence of tensors resulting from the split, each with the specified axis removed
      */
    def unstack[L: Label](unstackAxis: Axis[L])(using
        labels: Labels[T],
        ev: AxisRemover[T, L],
        labelR: Labels[ev.RemainingAxes]
    ): Seq[Tensor[ev.RemainingAxes, V]] =
      (0 until tensor.shape.dimensions(ev.index)).map: i =>
        val slicedJax = Jax.jnp.take(tensor.jaxValue, Jax.jnp.array(i), axis = ev.index)
        Tensor[ev.RemainingAxes, V](slicedJax)

    /** splits the tensor into chunks of the specified size along the given axis
      * returning a sequence of tensors corresponding to the chunks.
      */
    def chunk[splitL: Label](splitAxis: Axis[splitL], chunkSize: Int)(using
        labels: Labels[T],
        axisIndex: AxisIndex[T, splitL]
    ): Seq[Tensor[T, V]] =
      val res = Jax.jnp.split(tensor.jaxValue, chunkSize, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      res.map(x => Tensor[T, V](x))

    /** Select the tensor at an index along an axis, removing the axis.
      *
      * @param selector of the form `Axis[L].at(index)`
      */
    def select[L](selector: AxisAtIndex[L])(using
        ev: AxisRemover[T, L],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      Tensor(tensor.jaxValue.itemAt(tensor.indexAlong(ev.index, py.Any.from(selector.index))))

    /** Select the tensor at an index given by a tensor along an axis, removing the axis.
      * The index may be traced, e.g. a loop index under `jit`.
      *
      * @param selector of the form `Axis[L].at(index: Tensor0[Int32])`
      */
    def select[L](selector: AxisAtTensorIndex[L])(using
        ev: AxisRemover[T, L],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      Tensor(tensor.jaxValue.itemAt(tensor.indexAlong(ev.index, selector.index.jaxValue)))

    /** Select the tensor at a range along an axis, keeping the axis with the length of the range as extent.
      *
      * @param selector of the form `Axis[L].at(range)`
      */
    def select[L](selector: AxisAtRange[L])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): Tensor[T, V] =
      Tensor(tensor.jaxValue.itemAt(tensor.indexAlong(axisIndex.index, tensor.pyRange(selector.range, axisIndex.index))))

    /** Select the tensor at a sequence of indices along an axis, keeping the axis with the number of indices as extent.
      *
      * @param selector of the form `Axis[L].at(Seq(i, j, ...))`
      */
    def select[L](selector: AxisAtIndices[L])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): Tensor[T, V] =
      Tensor(tensor.jaxValue.itemAt(tensor.indexAlong(axisIndex.index, pyIndices(selector.indices))))

    /** Select the tensor at a tuple of indices along an axis, keeping the axis with the number of indices as extent.
      *
      * @param selector of the form `Axis[L].at((i, j, ...))`
      */
    def select[L, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): Tensor[T, V] =
      Tensor(tensor.jaxValue.itemAt(tensor.indexAlong(axisIndex.index, pyIndices(selector.indices.toList.asInstanceOf[List[Int]]))))

    /** Select the tensor at the indices of a vector along an axis, like `jnp.take`.
      * The axis is replaced by the axis of the indices. Out-of-bounds indices yield NaN for floating tensors.
      *
      * @param selector of the form `Axis[L].at(indices: Tensor1[L2, Int32])`
      */
    def select[L, L2](selector: AxisAtTensorIndices[L, L2])(using
        ev: AxisReplacer[T, L, L2],
        labels: Labels[ev.NewShape]
    ): Tensor[ev.NewShape, V] =
      Tensor(Jax.jnp.take(tensor.jaxValue, selector.indices.jaxValue, axis = ev.index))

    /** Select a window of static size along an axis, like `jax.lax.dynamic_slice`.
      * Its start may be a tensor, also a traced one, e.g. a loop index under `jit`.
      *
      * @param selector of the form `Axis[L].at(start, windowSize)`
      * @return The window, which keeps the axis with extent `windowSize`.
      */
    def select[L](selector: AxisAtWindow[L])(using
        axisIndex: AxisIndex[T, L],
        labels: Labels[T]
    ): Tensor[T, V] =
      val window = tensor.windowExtents(selector, axisIndex.index)
      Tensor(Jax.lax.dynamic_slice(tensor.jaxValue, tensor.windowStarts(selector, axisIndex.index), window.toPythonCopy))

    // `tensor(selector)` is `tensor.select(selector)`, as `seq(i)` reads a Scala collection.

    def apply[L](selector: AxisAtIndex[L])(using ev: AxisRemover[T, L], labels: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] =
      tensor.select(selector)(using ev, labels)

    def apply[L](selector: AxisAtTensorIndex[L])(using ev: AxisRemover[T, L], labels: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] =
      tensor.select(selector)(using ev, labels)

    def apply[L](selector: AxisAtRange[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): Tensor[T, V] =
      tensor.select(selector)

    def apply[L](selector: AxisAtIndices[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): Tensor[T, V] =
      tensor.select(selector)

    def apply[L, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): Tensor[T, V] =
      tensor.select(selector)

    def apply[L, L2](selector: AxisAtTensorIndices[L, L2])(using ev: AxisReplacer[T, L, L2], labels: Labels[ev.NewShape]): Tensor[ev.NewShape, V] =
      tensor.select(selector)(using ev, labels)

    def apply[L](selector: AxisAtWindow[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): Tensor[T, V] =
      tensor.select(selector)

    // `tensor.updated(selector)(value)` returns a copy of the tensor with `value` at the selection, as `updated` does for
    // Scala's immutable collections. `tensor.updated(selector)` is a [[TensorUpdate]]: its `updated` with a further selector
    // selects along another axis of the selection, so `tensor.updated(s1).updated(s2)(value)` writes what `tensor(s1)(s2)` reads.

    /** Update the tensor at an index along an axis.
      *
      * @param selector of the form `Axis[L].at(index)`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L](selector: AxisAtIndex[L])(using ev: AxisRemover[T, L], labels: Labels[T]): TensorUpdate[T, ev.RemainingAxes, V] =
      TensorUpdate(
        selected => tensor.select(selector)(using ev, selected),
        value =>
          requireSelectionShape(value, tensor.shape.dimensions.patch(ev.index, Nil, 1))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(ev.index, py.Any.from(selector.index))).set(value.jaxValue))
      )

    /** Update the tensor at an index given by a tensor along an axis. The index may be traced.
      * An out-of-bounds index leaves the tensor unchanged.
      *
      * @param selector of the form `Axis[L].at(index: Tensor0[Int32])`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L](selector: AxisAtTensorIndex[L])(using ev: AxisRemover[T, L], labels: Labels[T]): TensorUpdate[T, ev.RemainingAxes, V] =
      TensorUpdate(
        selected => tensor.select(selector)(using ev, selected),
        value =>
          requireSelectionShape(value, tensor.shape.dimensions.patch(ev.index, Nil, 1))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(ev.index, selector.index.jaxValue)).set(value.jaxValue))
      )

    /** Update the tensor at a range along an axis.
      *
      * @param selector of the form `Axis[L].at(range)`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L](selector: AxisAtRange[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): TensorUpdate[T, T, V] =
      TensorUpdate(
        _ => tensor.select(selector),
        value =>
          requireSelectionShape(value, tensor.shape.dimensions.updated(axisIndex.index, selector.range.length))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(axisIndex.index, tensor.pyRange(selector.range, axisIndex.index))).set(value.jaxValue))
      )

    /** Update the tensor at a sequence of indices along an axis.
      *
      * @param selector of the form `Axis[L].at(Seq(i, j, ...))`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L](selector: AxisAtIndices[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): TensorUpdate[T, T, V] =
      TensorUpdate(
        _ => tensor.select(selector),
        value =>
          requireSelectionShape(value, tensor.shape.dimensions.updated(axisIndex.index, selector.indices.length))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(axisIndex.index, pyIndices(selector.indices))).set(value.jaxValue))
      )

    /** Update the tensor at a tuple of indices along an axis.
      *
      * @param selector of the form `Axis[L].at((i, j, ...))`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): TensorUpdate[T, T, V] =
      TensorUpdate(
        _ => tensor.select(selector),
        value =>
          val indices = selector.indices.toList.asInstanceOf[List[Int]]
          requireSelectionShape(value, tensor.shape.dimensions.updated(axisIndex.index, indices.length))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(axisIndex.index, pyIndices(indices))).set(value.jaxValue))
      )

    /** Update the tensor at the indices of a vector along an axis. Out-of-bounds indices are skipped;
      * for repeated indices, it is unspecified which of their values is written.
      *
      * @param selector of the form `Axis[L].at(indices: Tensor1[L2, Int32])`
      * @return the update, to apply to the new values, with the shape of `tensor.select(selector)`
      */
    def updated[L, L2](selector: AxisAtTensorIndices[L, L2])(using ev: AxisReplacer[T, L, L2], labels: Labels[T]): TensorUpdate[T, ev.NewShape, V] =
      TensorUpdate(
        selected => tensor.select(selector)(using ev, selected),
        value =>
          requireSelectionShape(value, tensor.shape.dimensions.updated(ev.index, selector.indices.shape.dimensions.head))
          Tensor(tensor.jaxValue.at.itemAt(tensor.indexAlong(ev.index, selector.indices.jaxValue)).set(value.jaxValue))
      )

    /** Update the tensor in a window of static size along an axis, like `jax.lax.dynamic_update_slice`.
      * Its start may be a tensor, also a traced one, e.g. a loop index under `jit`.
      *
      * @param selector of the form `Axis[L].at(start, windowSize)`
      * @return the update, to apply to the new window, with the shape of `tensor.select(selector)`
      */
    def updated[L](selector: AxisAtWindow[L])(using axisIndex: AxisIndex[T, L], labels: Labels[T]): TensorUpdate[T, T, V] =
      TensorUpdate(
        _ => tensor.select(selector),
        value =>
          requireSelectionShape(value, tensor.windowExtents(selector, axisIndex.index))
          Tensor(Jax.lax.dynamic_update_slice(tensor.jaxValue, value.jaxValue, tensor.windowStarts(selector, axisIndex.index)))
      )

    /** The Python index `tensor[:, ..., index, ..., :]`, with `index` at the axis `dimIndex`. */
    private def indexAlong(dimIndex: Int, index: py.Any): Jax.PyDynamic =
      val all = py.Dynamic.global.slice(py.None)
      Jax.Dynamic.global.tuple(Seq.tabulate[py.Any](tensor.shape.rank)(i => if i == dimIndex then index else all).toPythonProxy)

    /** The Python slice for `range` on the axis `dimIndex`, which the range must not leave. */
    private def pyRange(range: Range, dimIndex: Int): py.Any =
      val dimSize = tensor.shape.dimensions(dimIndex)
      require(range.isEmpty || (range.min >= 0 && range.max < dimSize), s"$range is out of bounds for axis of size $dimSize")
      // map Scala Range to Python Range which is exclusive
      val stop = range match
        case r: Range.Inclusive => r.end + r.step.sign
        case r: Range.Exclusive => r.end
      // the range is in bounds, so a negative stop can only mean "before the first element", which Python spells as None
      py.Dynamic.global.slice(range.start, if stop < 0 then py.None else py.Any.from(stop), range.step)

    /** The extents of the window `selector` on the axis at `dimIndex`: `windowSize` there, the whole axis elsewhere. */
    private def windowExtents(selector: AxisAtWindow[?], dimIndex: Int): Seq[Int] =
      val dimSize = tensor.shape.dimensions(dimIndex)
      val size = selector.windowSize
      require(size >= 0 && size <= dimSize, s"Window of size $size does not fit axis ${tensor.axes(dimIndex)} of size $dimSize")
      selector.start match
        case start: Int =>
          require(start >= 0 && start + size <= dimSize, s"Window $start until ${start + size} is out of bounds for axis ${tensor.axes(dimIndex)} of size $dimSize")
        case _ => () // a tensor start may be traced, so JAX clamps it
      tensor.shape.dimensions.updated(dimIndex, size)

    /** The start of the window `selector` on every axis: its start on the axis at `dimIndex`, 0 elsewhere. */
    private def windowStarts(selector: AxisAtWindow[?], dimIndex: Int): Jax.PyDynamic =
      val start: py.Any = selector.start match
        case start: Int          => py.Any.from(start)
        case start: Tensor[?, ?] => start.jaxValue
      val starts = Seq.tabulate[py.Any](tensor.shape.rank)(i => if i == dimIndex then start else py.Any.from(0))
      Jax.Dynamic.global.tuple(starts.toPythonProxy)

    def rearrange[Axes <: Tuple, Status <: ValidationResult](newOrder: Axes)(using
        Labels[UnwrapAxes[Axes]]
    )(using
        computer: ComputeMissing[UnwrapAxes[Axes], T, EmptyTuple, Status],
        guard: CheckValid[Status]
    ): Tensor[UnwrapAxes[Axes], V] =
      rearrange[Axes, EmptyTuple, Status](newOrder, EmptyTuple)

    // Convenience overload for 1 dims (to support error messages with single axis)
    inline def rearrange[Axes <: Tuple, L1, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Tuple1[AxisExtent[L1]]], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[Tuple1[AxisExtent[L1]]]): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, Tuple1(d1))

    // Convenience overload for 2 dims
    inline def rearrange[Axes <: Tuple, L1, L2, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2])]): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2))

    // Convenience overload for 3 dims
    inline def rearrange[Axes <: Tuple, L1, L2, L3, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])]): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2, d3))

    // Convenience overload for 4 dims
    inline def rearrange[Axes <: Tuple, L1, L2, L3, L4, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3], d4: AxisExtent[L4])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])]): Tensor[UnwrapAxes[Axes], V] =
      rearrange(newOrder, (d1, d2, d3, d4))

    def rearrange[Axes <: Tuple, Dims <: Tuple, Status <: ValidationResult](
        newOrder: Axes,
        dims: Dims
    )(using
        computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Dims], Status],
        guard: CheckValid[Status]
    )(using
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[Dims]
    ): Tensor[UnwrapAxes[Axes], V] =
      def cleanPatternPrime(pattern: String): String =
        // Support dimwit.Prime by replacing ' with "Prime"
        pattern.replaceAll(
          "'",
          "Prime"
        )
      def createEinopsPattern(fromPattern: String, toPattern: String): String =
        def cleanPatternStar(pattern: String): String =
          // to replace all a*b*c in pattern with (a b c), example:
          // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
          val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
          regex.replaceAllIn(
            pattern,
            _.group(1).split("\\*").mkString("(", " ", ")")
          )
        def cleanPatternPlus(pattern: String): String =
          // Support dimwit.|+| by replacing + with underlines
          val regex = raw"([a-zA-Z0-9_]+(\+[a-zA-Z0-9_]+)+)".r
          regex.replaceAllIn(
            pattern,
            _.group(1).replace("+", "_")
          )
        def cleanPattern(pattern: String): String =
          cleanPatternPlus(cleanPatternStar(cleanPatternPrime(pattern)))
        s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
      val fromPattern = tensor.shape.labels.mkString(" ")
      val toPattern = newLabels.names.mkString(" ")
      val pattern = createEinopsPattern(fromPattern, toPattern)
      val dimSizesMap = extractor.extract(dims)
      val cleanDimSizesMap = dimSizesMap.map { case (k, v) =>
        val newKey = cleanPatternPrime(k)
        (newKey, v)
      }
      Tensor(
        Einops.rearrange(
          tensor.jaxValue,
          pattern,
          kwargsMap = cleanDimSizesMap
        )
      )

    def broadcastTo[O <: Tuple: Labels](newShape: Shape[O])(using
        labels: Labels[T],
        ev: StrictSubset[T, O]
    ): Tensor[O, V] =
      /* Disallow implicit broadcasting where an *existing* axis changes size (implicitly).
       * dimwit broadcasting only adds missing axes, never changes existing ones.
       * 
       * This is a required check to prevent implicit broadcasting across dimwit.
       * If this check is not explicitly present, Jax.jnp.broadcast_to would implicit broadcast.*/
      def disallowImplicitShapeBroadcasting(): Unit =
        val tAxesDims = tensor.axes.zip(tensor.shape.dimensions).toMap
        val newShapeAxesDims = newShape.labels.zip(newShape.dimensions).toMap
        tensor.axes.foreach(axisName =>
          require(
            tAxesDims(axisName) == newShapeAxesDims(axisName),
            s"Broadcasting only adds missing axes. Present axes must have the same size. Axis ${axisName} has size ${tAxesDims(axisName)} in the current tensor but size ${newShapeAxesDims(axisName)} in the target shape."
          )
        )

      disallowImplicitShapeBroadcasting() // Make dimwit coders, good coders :)

      val t = tensor

      val currentNames = summon[Labels[T]].names
      val targetNames = summon[Labels[O]].names

      val targetOrder = targetNames.filter(currentNames.contains)
      val permutation = targetOrder.map(n => currentNames.indexOf(n))

      val alignedJax =
        if permutation != currentNames.indices.toList then Jax.jnp.transpose(t.jaxValue, permutation.toPythonProxy)
        else t.jaxValue

      val currentShapeMap = currentNames.zip(t.shape.dimensions).toMap

      val intermediateShape = targetNames.map { name =>
        currentShapeMap.getOrElse(name, 1)
      }

      val reshapedJax = Jax.jnp.reshape(alignedJax, intermediateShape.toPythonProxy)
      Tensor(Jax.jnp.broadcast_to(reshapedJax, newShape.dimensions.toPythonProxy))

    def relabel[OldLabel: Label, NewLabel: Label](
        rename: (Axis[OldLabel], Axis[NewLabel])
    )(using
        ev: AxisReplacer[T, OldLabel, NewLabel],
        newLabels: Labels[ev.NewShape]
    ): Tensor[ev.NewShape, V] = Tensor(tensor.jaxValue)

    def relabelAll[newT <: Tuple](
        newAxes: newT
    )(using
        newLabels: Labels[UnwrapAxes[newT]],
        @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
        evSameSize: Tuple.Size[newT] =:= Tuple.Size[T]
    ): Tensor[UnwrapAxes[newT], V] = Tensor[UnwrapAxes[newT], V](tensor.jaxValue)

    def swap[L1: Label, L2: Label](
        axis1: Axis[L1],
        axis2: Axis[L2]
    )(using
        labels: Labels[T],
        axisIndex1: AxisIndex[T, L1],
        axisIndex2: AxisIndex[T, L2],
        swapper: AxisSwapper[T, L1, L2]
    ): Tensor[swapper.Out, V] =
      given Labels[swapper.Out] with
        def names =
          val originalNames = summon[Labels[T]].names
          val ax1Name = summon[Label[L1]].name
          val ax2Name = summon[Label[L2]].name
          originalNames.map {
            case n if n == ax1Name => ax2Name
            case n if n == ax2Name => ax1Name
            case n                 => n
          }
      Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.index, axisIndex2.index))

    def appendAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[T, Tuple1[L]], V] =
      val newShape = tensor.shape.dimensions :+ 1
      Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

    def prependAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[Tuple1[L], T], V] =
      val newShape = 1 +: tensor.shape.dimensions
      Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

    def squeeze[L: Label](axis: Axis[L])(using
        ev: AxisRemover[T, L],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      require(
        tensor.shape.dimensions(ev.index) == 1,
        s"Cannot squeeze axis ${summon[Label[L]].name} of size ${tensor.shape.dimensions(ev.index)}"
      )
      Tensor(Jax.jnp.squeeze(tensor.jaxValue, axis = ev.index))
