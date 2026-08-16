package dimwit.tensor

import scala.annotation.implicitNotFound
import scala.compiletime.erasedValue
import scala.compiletime.summonInline
import scala.util.NotGiven

/* Helpers for tracking Tensor Shape types across various operations */
object ShapeTypeHelpers:

  import TupleHelpers.*

  /** Wraps each element of a tuple in an Axis */
  type WrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple => EmptyTuple
    case a *: tail  => Axis[a] *: WrapAxes[tail]

  /** Unwraps each Axis in a tuple to get the label types */
  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple      => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail       => h *: UnwrapAxes[tail]

  /** Unwrap each AxisExtent in a tuple to get the label types */
  type UnwrapDims[T <: Tuple] <: Tuple = T match
    case EmptyTuple            => EmptyTuple
    case AxisExtent[a] *: tail => a *: UnwrapDims[tail]

  /** Base trait for tracking an axis in a tensor shape */
  @implicitNotFound("Axis[${Axis}] not found in Tensor[${TensorShape}]")
  trait AxisInTensor[TensorShape <: Tuple, Axis]:
    def index: Int

  /** Finds the index of an axis in a tensor shape */
  trait AxisIndex[Shape <: Tuple, Axis] extends AxisInTensor[Shape, Axis]

  object AxisIndex:

    def apply[T <: Tuple, L](using idx: AxisIndex[T, L]): Int = idx.index

    given found[L, Tail <: Tuple]: AxisIndex[L *: Tail, L] with
      val index = 0

    given search[H, T <: Tuple, L](using
        next: AxisIndex[T, L]
    ): AxisIndex[H *: T, L] with
      val index = 1 + next.index

    given concatRight[A <: Tuple, B <: Tuple, L](using
        sizeA: ValueOf[Tuple.Size[A]],
        idxB: AxisIndex[B, L]
    ): AxisIndex[Tuple.Concat[A, B], L] with
      val index = sizeA.value + idxB.index

    given concatEnd[A <: Tuple, L]: AxisIndex[Tuple.Concat[A, Tuple1[L]], L] with
      val index = -1

  /** Removing an axis from a tensor shape.
    *
    * RemainingAxes is the resulting shape after removing the axis.
    */
  trait AxisRemover[TensorShape <: Tuple, Axis] extends AxisInTensor[TensorShape, Axis]:
    type RemainingAxes <: Tuple

  object AxisRemover:
    type Aux[S <: Tuple, A, R <: Tuple] = AxisRemover[S, A] { type RemainingAxes = R }

    given bridge[S <: Tuple, A, R <: Tuple](using
        axisIndex: AxisIndex[S, A],
        ev: RemoverAll.Aux[S, A *: EmptyTuple, R]
    ): AxisRemover.Aux[S, A, R] = new AxisRemover[S, A]:
      type RemainingAxes = R
      def index: Int = axisIndex.index

  /** Replaces an axis in a tensor shape with another axis.
    *
    * NewShape is the resulting shape after replacement.
    */
  trait AxisReplacer[TensorShape <: Tuple, Axis, AxisReplacement] extends AxisInTensor[TensorShape, Axis]:
    type NewShape <: Tuple

  object AxisReplacer:
    type Aux[S <: Tuple, A, AR, O <: Tuple] = AxisReplacer[S, A, AR] { type NewShape = O }

    given bridge[S <: Tuple, A, AR, O <: Tuple](using
        idx: AxisIndex[S, A],
        replacer: Replacer.Aux[S, A, AR, O]
    ): AxisReplacer.Aux[S, A, AR, O] = new AxisReplacer[S, A, AR]:
      def index: Int = idx.index
      type NewShape = O

  /** Replace Axis in given Tuple with the Axes in the AxisReplacements tuple */
  trait AxisReplacerAll[TensorShape <: Tuple, Axis, AxisReplacements <: Tuple] extends AxisInTensor[TensorShape, Axis]:
    type NewShape <: Tuple

  object AxisReplacerAll:
    type Aux[S <: Tuple, A, AR <: Tuple, O <: Tuple] = AxisReplacerAll[S, A, AR] { type NewShape = O }

    trait Splice[Source <: Tuple, Axis, Replacement <: Tuple]:
      type Out <: Tuple
      def index: Int

    object Splice:
      type Aux[S <: Tuple, A, R <: Tuple, O <: Tuple] = Splice[S, A, R] { type Out = O }

      given found[A, T <: Tuple, R <: Tuple]: Splice.Aux[A *: T, A, R, Tuple.Concat[R, T]] =
        new Splice[A *: T, A, R]:
          type Out = Tuple.Concat[R, T]
          def index = 0

      given search[H, T <: Tuple, A, R <: Tuple, TailOut <: Tuple](using
          ne: NotGiven[H =:= A],
          tailSplice: Splice.Aux[T, A, R, TailOut]
      ): Splice.Aux[H *: T, A, R, H *: TailOut] =
        new Splice[H *: T, A, R]:
          type Out = H *: TailOut
          def index = 1 + tailSplice.index

    given bridge[S <: Tuple, A, AR <: Tuple, O <: Tuple](using
        s: Splice.Aux[S, A, AR, O]
    ): AxisReplacerAll.Aux[S, A, AR, O] = new AxisReplacerAll[S, A, AR]:
      type NewShape = O
      def index: Int = s.index

  /** Base trait for tracking multiple axes in a tensor shape */
  @implicitNotFound("Axes [${Axes}] not all found in Tensor shape [${TensorShape}]")
  trait AxesInTensor[TensorShape <: Tuple, Axes <: Tuple]:
    def indices: List[Int]

  /** Finds the indices of multiple axes in a tensor shape */
  sealed trait AxisIndices[T <: Tuple, Axes <: Tuple] extends AxesInTensor[T, Axes]

  object AxisIndices:

    class AxisIndicesImpl[T <: Tuple, Axes <: Tuple](val indices: List[Int]) extends AxisIndices[T, Axes]

    private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
      inline erasedValue[ToFind] match
        case _: EmptyTuple     => Nil
        case _: (head *: tail) =>
          summonInline[AxisIndex[InTuple, head]].index :: indicesOfList[InTuple, tail]

    inline given indices[T <: Tuple, ToFind <: Tuple]: AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

  end AxisIndices

  /** Removes multiple axes from a tensor shape. */
  trait AxesRemover[TensorShape <: Tuple, Axes <: Tuple] extends AxesInTensor[TensorShape, Axes]:
    type RemainingAxes <: Tuple

  object AxesRemover:
    type Aux[T <: Tuple, Axes <: Tuple, R <: Tuple] = AxesRemover[T, Axes] { type RemainingAxes = R }

    given bridge[T <: Tuple, Axes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, Axes],
        ev: RemoverAll.Aux[T, Axes, R]
    ): AxesRemover.Aux[T, Axes, R] = new AxesRemover[T, Axes]:
      type RemainingAxes = R
      def indices: List[Int] = idx.indices

  /** Removes [[RemovedAxis]] from a tensor shape while computing runtime indices
    *  for [[IndexAxes]].
    */
  trait AxesConditionalRemover[TensorShape <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple] extends AxesInTensor[TensorShape, IndexAxes]:
    type RemainingAxes <: Tuple

  object AxesConditionalRemover:
    type Aux[T <: Tuple, RA <: Tuple, IA <: Tuple, R <: Tuple] = AxesConditionalRemover[T, RA, IA] { type RemainingAxes = R }

    given bridge[T <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, IndexAxes],
        ev: RemoverAll.Aux[T, RemovedAxis, R]
    ): AxesConditionalRemover.Aux[T, RemovedAxis, IndexAxes, R] = new AxesConditionalRemover[T, RemovedAxis, IndexAxes]:
      type RemainingAxes = R
      def indices = idx.indices

  /** Removes a shared axis from multiple tensor shapes while computing runtime indices
    * for the remaining axes.
    */
  @implicitNotFound("Axis[${Axis}] not found in ${Shapes}}")
  trait SharedAxisRemover[Shapes <: Tuple, Axis]:
    type RemainingAxes <: Tuple
    def indices: List[Int]
    def shapesLabels: List[List[String]]

  object SharedAxisRemover:
    type Aux[S <: Tuple, A, O <: Tuple] = SharedAxisRemover[S, A] { type RemainingAxes = O }

    given emptyTuple[Axis]: SharedAxisRemover.Aux[EmptyTuple, Axis, EmptyTuple] = new SharedAxisRemover[EmptyTuple, Axis]:
      type RemainingAxes = EmptyTuple
      def indices = Nil
      def shapesLabels = Nil

    given consTuple[H <: Tuple, T <: Tuple, Axis, R <: Tuple, TailOut <: Tuple](using
        evH: AxisRemover.Aux[H, Axis, R],
        evT: SharedAxisRemover.Aux[T, Axis, TailOut],
        rLabels: Labels[R]
    ): SharedAxisRemover.Aux[H *: T, Axis, R *: TailOut] = new SharedAxisRemover[H *: T, Axis]:
      type RemainingAxes = R *: TailOut
      def indices = evH.index :: evT.indices
      def shapesLabels = List(rLabels.names) ++ evT.shapesLabels

  /** Extracts the dimensions of a tensor shape into a Map of label names to sizes.
    */
  trait DimExtractor[T]:
    def extract(t: T): Map[String, Int]

  object DimExtractor:
    given emptyTuple: DimExtractor[EmptyTuple] with
      def extract(t: EmptyTuple) = Map.empty

    given consTuple[L, Tail <: Tuple](using
        label: Label[L],
        tailExtractor: DimExtractor[Tail]
    ): DimExtractor[AxisExtent[L] *: Tail] with
      def extract(t: AxisExtent[L] *: Tail) =
        val size = t.head.size
        Map(label.name -> size) ++ tailExtractor.extract(t.tail)

  /** Merges multiple axes in a tensor shape into a single axis.
    *
    * NewShape is the resulting shape after merging the axes.
    */
  @implicitNotFound("Cannot merge axes ${ToMerge} in shape ${S}. Ensure all axes exist.")
  trait AxesMerger[S <: Tuple, ToMerge <: Tuple]:
    type NewShape <: Tuple
    def permutation: List[Int] // To make axes contiguous
    def mergedIndex: Int // Where the new axis sits in NewShape
    def mergeIndices: List[Int] // Original indices of axes to be merged

  import dimwit.|*|
  type MergeLabels[T <: Tuple] = T match
    case head *: tail => MergeLabelsRec[tail, head]

  type MergeLabelsRec[T <: Tuple, Acc] = T match
    case EmptyTuple   => Acc
    case head *: tail => MergeLabelsRec[tail, Acc |*| head]

  object MergeLabels:
    given mergedLabel[T <: Tuple: Labels]: Label[MergeLabels[T]] with
      def name = summon[Labels[T]].names.mkString("*")

  object AxesMerger:
    type Aux[S <: Tuple, TM <: Tuple, Out <: Tuple] = AxesMerger[S, TM] { type NewShape = Out }

    given bridge[S <: Tuple, TM <: Tuple, R <: Tuple](using
        indices: AxisIndices[S, TM],
        remover: RemoverAll[S, Tuple.Tail[TM]],
        replacer: AxisReplacer.Aux[remover.Out, Tuple.Head[TM], MergeLabels[TM], R],
        valueOf: ValueOf[Tuple.Size[S]]
    ): AxesMerger[S, TM] with
      type NewShape = replacer.NewShape

      def mergeIndices = indices.indices

      def permutation: List[Int] =
        val toMerge = indices.indices
        val others = (0 until valueOf.value).filterNot(toMerge.contains).toList
        // Move all 'toMerge' indices to the position of the first one (the pivot)
        val pivotIdxInS = toMerge.head
        val (pref, suff) = others.partition(_ < pivotIdxInS)
        pref ++ toMerge ++ suff

      def mergedIndex: Int = replacer.index
