package dimwit.tensor

import scala.annotation.implicitNotFound
import scala.util.NotGiven
import scala.compiletime.{constValue, erasedValue, summonInline}

/* Helpers for tracking Tensor Shape types across various operations */
object ShapeTypeHelpers:

  import TupleHelpers.*

  type WrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple => EmptyTuple
    case a *: tail  => Axis[a] *: WrapAxes[tail]

  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple      => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail       => h *: UnwrapAxes[tail]

  type UnwrapDims[T <: Tuple] <: Tuple = T match
    case EmptyTuple            => EmptyTuple
    case AxisExtent[a] *: tail => a *: UnwrapDims[tail]

  @implicitNotFound("Axis[${Axis}] not found in Tensor[${TensorShape}]")
  trait AxisInTensor[TensorShape <: Tuple, Axis]:
    def index: Int

  trait AxisIndex[Shape <: Tuple, Axis] extends AxisInTensor[Shape, Axis]

  object AxisIndex:

    def apply[T <: Tuple, L](using idx: AxisIndex[T, L]): Int = idx.index

    given head[L, Tail <: Tuple]: AxisIndex[L *: Tail, L] with
      val index = 0

    given tail[H, T <: Tuple, L](using
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

  trait AxisRemover[TensorShape <: Tuple, Axis, RemainingShape <: Tuple] extends AxisInTensor[TensorShape, Axis]

  object AxisRemover:
    given bridge[S <: Tuple, A, R <: Tuple](using
        axisIndex: AxisIndex[S, A],
        ev: RemoverAll.Aux[S, A *: EmptyTuple, R]
    ): AxisRemover[S, A, R] with
      def index: Int = axisIndex.index

  // Replace single axis with single axis
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

  // Replace single axis with multiple axes
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

      given recurse[H, T <: Tuple, A, R <: Tuple, TailOut <: Tuple](using
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

  @implicitNotFound("Axes [${Axes}] not all found in Tensor shape [${TensorShape}]")
  trait AxesInTensor[TensorShape <: Tuple, Axes <: Tuple]:
    def indices: List[Int]

  sealed trait AxisIndices[T <: Tuple, Axes <: Tuple] extends AxesInTensor[T, Axes]

  object AxisIndices:

    class AxisIndicesImpl[T <: Tuple, Axes <: Tuple](val indices: List[Int]) extends AxisIndices[T, Axes]

    private inline def indicesOfList[InTuple <: Tuple, ToFind <: Tuple]: List[Int] =
      inline erasedValue[ToFind] match
        case _: EmptyTuple     => Nil
        case _: (head *: tail) =>
          summonInline[AxisIndex[InTuple, head]].index :: indicesOfList[InTuple, tail]

    inline given [T <: Tuple, ToFind <: Tuple]: AxisIndices[T, ToFind] = AxisIndicesImpl[T, ToFind](indicesOfList[T, ToFind])

  end AxisIndices
  trait AxesRemover[TensorShape <: Tuple, Axes <: Tuple, RemainingShape <: Tuple] extends AxesInTensor[TensorShape, Axes]

  object AxesRemover:
    given bridge[T <: Tuple, Axes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, Axes],
        ev: RemoverAll.Aux[T, Axes, R]
    ): AxesRemover[T, Axes, R] with
      def indices: List[Int] = idx.indices

  trait AxesConditionalRemover[TensorShape <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple, RemainingShape <: Tuple] extends AxesInTensor[TensorShape, IndexAxes]

  object AxesConditionalRemover:
    given bridge[T <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, IndexAxes],
        ev: RemoverAll.Aux[T, RemovedAxis, R]
    ): AxesConditionalRemover[T, RemovedAxis, IndexAxes, R] with
      def indices = idx.indices

  @implicitNotFound("Axis[${Axis}] not found in ${Shapes}}")
  trait SharedAxisRemover[Shapes <: Tuple, Axis, Sliced <: Tuple]:
    def indices: List[Int]

  object SharedAxisRemover:

    given empty[Axis]: SharedAxisRemover[EmptyTuple, Axis, EmptyTuple] with
      def indices = Nil
      type Sliced = EmptyTuple

    given cons[H <: Tuple, T <: Tuple, Axis, R <: Tuple, TailOut <: Tuple](using
        evH: AxisRemover[H, Axis, R],
        evT: SharedAxisRemover[T, Axis, TailOut]
    ): SharedAxisRemover[H *: T, Axis, R *: TailOut] with
      def indices = evH.index :: evT.indices

  trait DimExtractor[T]:
    def extract(t: T): Map[String, Int]

  object DimExtractor:
    given DimExtractor[EmptyTuple] with
      def extract(t: EmptyTuple) = Map.empty

    given [L, Tail <: Tuple](using
        label: Label[L],
        tailExtractor: DimExtractor[Tail]
    ): DimExtractor[AxisExtent[L] *: Tail] with
      def extract(t: AxisExtent[L] *: Tail) =
        val size = t.head.size
        Map(label.name -> size) ++ tailExtractor.extract(t.tail)

    given single[L](using label: Label[L]): DimExtractor[AxisExtent[L]] with
      def extract(t: AxisExtent[L]) =
        Map(label.name -> t.size)

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
