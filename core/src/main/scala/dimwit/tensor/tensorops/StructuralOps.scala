package dimwit.tensor.tensorops

import dimwit.jax.Einops
import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.AxisAtIndex
import dimwit.tensor.AxisAtIndices
import dimwit.tensor.AxisAtRange
import dimwit.tensor.AxisAtTensorIndex
import dimwit.tensor.AxisAtTupleIndices
import dimwit.tensor.AxisExtent
import dimwit.tensor.DType.Bool
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Shape
import dimwit.tensor.ShapeTypeHelpers.AxesConditionalRemover
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
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import dimwit.tensor.TupleHelpers
import dimwit.tensor.TupleHelpers.StrictSubset
import dimwit.tensor.TupleHelpers.TensorEvidence.CheckValid
import dimwit.tensor.TupleHelpers.TensorEvidence.ComputeMissing
import dimwit.tensor.TupleHelpers.TensorEvidence.IsPermutation
import dimwit.tensor.TupleHelpers.TensorEvidence.ValidationResult
import dimwit.|+|
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Reader
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.implicitNotFound
import scala.util.NotGiven

object StructuralOps:

  private object Util:

    type InsertBefore[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => B *: EmptyTuple
      case A *: tail  => B *: A *: tail
      case h *: tail  => h *: InsertBefore[tail, A, B]

    type InsertAfter[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => B *: EmptyTuple
      case A *: tail  => A *: B *: tail
      case h *: tail  => h *: InsertAfter[tail, A, B]

    type SliceIndex = Int | List[Int] | Range | Tensor0[Int32]
    type ExtractLabel[X] = X match
      case AxisAtIndex[l]           => l
      case AxisAtRange[l]           => l
      case AxisAtIndices[l]         => l
      case AxisAtTupleIndices[l, ?] => l
      case AxisAtTensorIndex[l]     => l
    type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

    trait SliceLabelExtractor[Inputs <: Tuple, Out <: Tuple]

    object SliceLabelExtractor:

      given empty: SliceLabelExtractor[EmptyTuple, EmptyTuple] =
        new SliceLabelExtractor[EmptyTuple, EmptyTuple] {}

      // New givens for AxisSelector types
      given consAxisAtIndex[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[AxisAtIndex[L] *: Tail, L *: TailOut] =
        new SliceLabelExtractor[AxisAtIndex[L] *: Tail, L *: TailOut] {}

      given consAxisAtRange[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[AxisAtRange[L] *: Tail, TailOut] =
        new SliceLabelExtractor[AxisAtRange[L] *: Tail, TailOut] {}

      given consAxisAtIndices[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[AxisAtIndices[L] *: Tail, TailOut] =
        new SliceLabelExtractor[AxisAtIndices[L] *: Tail, TailOut] {}

      given consAxisAtTupleIndices[L, I <: NonEmptyTuple, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[AxisAtTupleIndices[L, I] *: Tail, TailOut] =
        new SliceLabelExtractor[AxisAtTupleIndices[L, I] *: Tail, TailOut] {}

      given consAxisAtTensorIndex[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[AxisAtTensorIndex[L] *: Tail, L *: TailOut] =
        new SliceLabelExtractor[AxisAtTensorIndex[L] *: Tail, L *: TailOut] {}

      // Keep backward compatibility with tuple syntax
      given consInt[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] =
        new SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] {}

      given consTensor0Int[L, Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[(Axis[L], Tensor0[Int32]) *: Tail, L *: TailOut] =
        new SliceLabelExtractor[(Axis[L], Tensor0[Int32]) *: Tail, L *: TailOut] {}

      given consSeq[L, SeqT <: Seq[Int], Tail <: Tuple, TailOut <: Tuple](using
          tailExt: SliceLabelExtractor[Tail, TailOut]
      ): SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] =
        new SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] {}

    type Swap[T <: Tuple, A, B] <: Tuple = T match
      case EmptyTuple => EmptyTuple
      case A *: tail  => B *: Swap[tail, A, B]
      case B *: tail  => A *: Swap[tail, A, B]
      case h *: tail  => h *: Swap[tail, A, B]

    @implicitNotFound("The axis ${L} is already present in the tensor shape ${T}.")
    trait AxisAbsent[T, L]
    object AxisAbsent:
      given [T <: Tuple, L](using NotGiven[Tuple.Contains[T, L] =:= true]): AxisAbsent[T, L] = new AxisAbsent[T, L] {}

  import Util.*

  object TensorWhere:
    /** Returns a new tensor where elements are selected from `x` or `y`
      * depending on the boolean condition.
      *
      * @param condition A tensor of boolean values that determines which elements to select.
      * @param x A tensor from which to select elements when the condition is true.
      * @param y A tensor from which to select elements when the condition is false.
      *
      * @return A new tensor with elements from `x` where the condition is true, and elements from `y` where the condition is false.
      */
    def where[T <: Tuple: Labels, V](
        condition: Tensor[T, Bool],
        x: Tensor[T, V],
        y: Tensor[T, V]
    ): Tensor[T, V] =
      Tensor(Jax.jnp.where(condition.jaxValue, x.jaxValue, y.jaxValue))

  export TensorWhere.where

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
      axisIndex: AxisIndex[T, L]
  ): Tensor[InsertAfter[T, L, NewL], V] =
    require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
    val axisIdx = axisIndex.index + 1 // we are inserting after the given axis, so shift by 1
    val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
    val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = axisIdx)
    val names = summon[Labels[T]].names
    val newNames = names.take(axisIdx) ++ Seq(newLabel.name) ++ names.drop(axisIdx)
    given Labels[InsertAfter[T, L, NewL]] with
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

    given recursive[H, T1Tail <: Tuple, T2Tail <: Tuple, OutTail <: Tuple](using
        next: ValidConcat.Aux[T1Tail, T2Tail, OutTail]
    ): ValidConcat[H *: T1Tail, H *: T2Tail] with
      type Out = H *: OutTail
      def index: Int = next.index + 1

    given concatAxis[H1, H2, Tail <: Tuple](using
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

    given recursive[A, B, CA <: Tuple, CB <: Tuple](using
        da: Aux[A, CA],
        db: Aux[B, CB]
    ): Aux[A |+| B, Tuple.Concat[CA, CB]] =
      new Deconcatenator[A |+| B]:
        type Components = Tuple.Concat[CA, CB]
        def labels = da.labels ++ db.labels

  trait DeconcatenatorLowPriority:
    given base[L](using l: Label[L]): Deconcatenator.Aux[L, L *: EmptyTuple] =
      new Deconcatenator[L]:
        type Components = L *: EmptyTuple
        def labels = List(l)

  trait TensorTupleMaker[Components <: Tuple, FullShape <: Tuple, SplitAxis, V]:
    type Out <: Tuple
    def apply(arrays: Seq[Jax.PyDynamic], compLabels: List[Label[?]], originalLabels: Seq[String], splitIndex: Int): Out

  object TensorTupleMaker:
    type Aux[C <: Tuple, F <: Tuple, S, V, O <: Tuple] =
      TensorTupleMaker[C, F, S, V] { type Out = O }

    given empty[F <: Tuple, S, V]: Aux[EmptyTuple, F, S, V, EmptyTuple] =
      new TensorTupleMaker[EmptyTuple, F, S, V]:
        type Out = EmptyTuple
        def apply(a: Seq[Jax.PyDynamic], c: List[Label[?]], o: Seq[String], i: Int) = EmptyTuple

    given cons[Head, Tail <: Tuple, F <: Tuple, S, V, NewShape <: Tuple](using
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

    private def calcPyIndices[Inputs <: Tuple](
        inputs: Inputs,
        targetDims: List[Int]
    ) =

      val PySlice = py.Dynamic.global.slice
      val Colon = PySlice(py.None)
      val rank = tensor.shape.rank
      val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

      val inputList = inputs.toList.asInstanceOf[List[Any]]

      targetDims.zip(inputList).foreach { case (dimIndex, input) =>
        val dimSize = tensor.shape.dimensions(dimIndex)
        input match
          // New AxisSelector types
          case AxisAtIndex(_, idx) =>
            indicesBuffer(dimIndex) = py.Any.from(idx)
          case AxisAtRange(_, range) =>
            indicesBuffer(dimIndex) = PySlice(range.head, range.last + 1, range.step)
          case AxisAtIndices(_, indices) =>
            indicesBuffer(dimIndex) = indices.map(py.Any.from).toPythonCopy // TODO find out why Copy is needed here
          case AxisAtTupleIndices(_, indices) =>
            indicesBuffer(dimIndex) = indices.toList.asInstanceOf[List[Int]].map(py.Any.from).toPythonCopy
          case AxisAtTensorIndex(_, tensorIdx) =>
            indicesBuffer(dimIndex) = tensorIdx.jaxValue
          // Backward compatibility with tuples
          case (_, sliceIndex) =>
            sliceIndex match
              case sliceSeq: List[Int] @unchecked =>
                indicesBuffer(dimIndex) = sliceSeq.map(py.Any.from).toPythonProxy
              case range: Range @unchecked =>
                indicesBuffer(dimIndex) = PySlice(range.head, range.last + 1, range.step)
              case idx: Int =>
                indicesBuffer(dimIndex) = py.Any.from(idx)
              case tensorId: Tensor0[Int32] @unchecked =>
                indicesBuffer(dimIndex) = tensorId.jaxValue
      }

      Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)

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
      val axisIdx = ev.index
      val unstacked = Jax.jnp.split(tensor.jaxValue, tensor.shape.dimensions(axisIdx), axis = axisIdx).as[Seq[Jax.PyDynamic]]
      unstacked.map(x => Tensor[ev.RemainingAxes, V](x))

    /** splits the tensor into chunks of the specified size along the given axis
      * returning a sequence of tensors corresponding to the chunks.
      */
    def chunk[splitL: Label](splitAxis: Axis[splitL], chunkSize: Int)(using
        labels: Labels[T],
        axisIndex: AxisIndex[T, splitL]
    ): Seq[Tensor[T, V]] =
      val res = Jax.jnp.split(tensor.jaxValue, chunkSize, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
      res.map(x => Tensor[T, V](x))

    /** Slices the tensor according to the specified inputs,
      * removing the specified labels from the resulting tensor.
      *
      * @param inputs A tuple of inputs specifying how to slice the tensor.
      * @return The sliced tensor with the specified labels removed from its shape.
      */
    def slice[Inputs <: Tuple, LabelsToRemove <: Tuple](
        inputs: Inputs
    )(using
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Inputs]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
      Tensor(tensor.jaxValue.bracketAccess(pyIndices))

    /** Slice the given tensor, specifying the axis and index to slice at.
      *
      * @param selector An AxisAtIndex specifying the axis and index to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L, LabelsToRemove <: Tuple](
        selector: AxisAtIndex[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndex[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndex[L]]]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a given range to slice at.
      *
      * @param selector An AxisAtRange specifying the axis and range to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L, LabelsToRemove <: Tuple](
        selector: AxisAtRange[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtRange[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtRange[L]]]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a list of indices to slice at.
      *
      * @param selector An AxisAtIndices specifying the axis and indices to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L, LabelsToRemove <: Tuple](
        selector: AxisAtIndices[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndices[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndices[L]]]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a tensor of indices to slice at.
      *
      * @param selector An AxisAtTensorIndex specifying the axis and tensor of indices to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L, LabelsToRemove <: Tuple](
        selector: AxisAtTensorIndex[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtTensorIndex[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtTensorIndex[L]]]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = slice(Tuple1(selector))

    /** Slice the given tensor, specifying the axis and a tuple of indices to slice at.
      *
      * @param selector An AxisAtTupleIndices specifying the axis and tuple of indices to slice at.
      * @return A sliced tensor with the specified axis removed from its shape.
      */
    def slice[L, U <: NonEmptyTuple, LabelsToRemove <: Tuple](
        selector: AxisAtTupleIndices[L, U]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtTupleIndices[L, U]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtTupleIndices[L, U]]]],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] = slice(Tuple1(selector))

    def take[L1, L2: Label](
        axis: Axis[L1]
    )(
        indices: Tensor1[L2, Int32]
    )(using
        ev: AxisRemover[T, L1],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[Tuple.Concat[Tuple1[L2], ev.RemainingAxes], V] =
      val result = Jax.jnp.take(tensor.jaxValue, indices.jaxValue, axis = ev.index)
      Tensor(result)

    def set[Inputs <: Tuple, LabelsToRemove <: Tuple](
        inputs: Inputs
    )(using
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Inputs]],
        labels: Labels[T]
    )(value: Tensor[ev.RemainingAxes, V]): Tensor[T, V] =
      val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
      val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value.jaxValue)
      Tensor(result)

    // Convenience overload for Float
    def set[Inputs <: Tuple, LabelsToRemove <: Tuple](
        inputs: Inputs
    )(using
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        ev: AxesConditionalRemover.Aux[T, LabelsToRemove, ExtractLabels[Inputs], EmptyTuple],
        labels: Labels[T]
    )(value: Float): Tensor[T, V] =
      val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
      val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value)
      Tensor(result)

    // Convenience overload for AxisAtIndex
    def set[L, LabelsToRemove <: Tuple](
        selector: AxisAtIndex[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndex[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndex[L]]]],
        labels: Labels[T]
    )(value: Tensor[ev.RemainingAxes, V]): Tensor[T, V] = set(Tuple1(selector))(value)

    // Convenience overload for AxisAtRange
    def set[L, LabelsToRemove <: Tuple](
        selector: AxisAtRange[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtRange[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtRange[L]]]],
        labels: Labels[T]
    )(value: Tensor[ev.RemainingAxes, V]): Tensor[T, V] = set(Tuple1(selector))(value)

    // Convenience overload for AxisAtIndices
    def set[L, LabelsToRemove <: Tuple](
        selector: AxisAtIndices[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndices[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndices[L]]]],
        labels: Labels[T]
    )(value: Tensor[ev.RemainingAxes, V]): Tensor[T, V] = set(Tuple1(selector))(value)

    // Convenience overload for AxisAtTensorIndex
    def set[L, LabelsToRemove <: Tuple](
        selector: AxisAtTensorIndex[L]
    )(using
        sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtTensorIndex[L]], LabelsToRemove],
        ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtTensorIndex[L]]]],
        labels: Labels[T]
    )(value: Tensor[ev.RemainingAxes, V]): Tensor[T, V] = set(Tuple1(selector))(value)

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

    def retag[newT <: Tuple](using newLabels: Labels[newT]): Tensor[newT, V] =
      Tensor(tensor.jaxValue)(using newLabels)

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
        axisIndex2: AxisIndex[T, L2]
    ): Tensor[Swap[T, L1, L2], V] =
      given Labels[Swap[T, L1, L2]] with
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

  extension [L: Label, V](tensor: Tensor1[L, V])
    def roll(shift: Int): Tensor1[L, V] =
      Tensor(Jax.jnp.roll(tensor.jaxValue, shift = shift, axis = 0))
