package dimwit.tensor

import scala.annotation.implicitNotFound

/* Helpers for tracking Tensor Shape types across various operations */
object ShapeTypeHelpers:

  import TupleHelpers.*

  type UnwrapAxes[T <: Tuple] <: Tuple = T match
    case EmptyTuple      => EmptyTuple
    case Axis[a] *: tail => a *: UnwrapAxes[tail]
    case h *: tail       => h *: UnwrapAxes[tail]

  @implicitNotFound("Axis[${Axis}] not found in Tensor[${TensorShape}]")
  trait AxisInTensor[TensorShape <: Tuple, Axis]:
    def index: Int

  trait AxisRemover[TensorShape <: Tuple, Axis, RemainingShape <: Tuple] extends AxisInTensor[TensorShape, Axis]

  object AxisRemover:
    given bridge[S <: Tuple, A, R <: Tuple](using
        axisIndex: AxisIndex[S, A],
        ev: RemoverAll.Aux[S, A *: EmptyTuple, R]
    ): AxisRemover[S, A, R] with
      def index: Int = axisIndex.value

  trait AxisReplacer[TensorShape <: Tuple, Axis, AxisReplacement] extends AxisInTensor[TensorShape, Axis]:
    type NewShape <: Tuple

  object AxisReplacer:
    type Aux[S <: Tuple, A, AR, O <: Tuple] = AxisReplacer[S, A, AR] { type NewShape = O }

    given bridge[S <: Tuple, A, AR, O <: Tuple](using
        idx: AxisIndex[S, A],
        replacer: Replacer.Aux[S, A, AR, O]
    ): AxisReplacer.Aux[S, A, AR, O] = new AxisReplacer[S, A, AR]:
      def index: Int = idx.value
      type NewShape = O

  @implicitNotFound("Axes [${Axes}] not all found in Tensor shape [${TensorShape}]")
  trait AxesInTensor[TensorShape <: Tuple, Axes <: Tuple]:
    def indices: List[Int]

  trait AxesRemover[TensorShape <: Tuple, Axes <: Tuple, RemainingShape <: Tuple] extends AxesInTensor[TensorShape, Axes]

  object AxesRemover:
    given bridge[T <: Tuple, Axes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, Axes],
        ev: RemoverAll.Aux[T, Axes, R]
    ): AxesRemover[T, Axes, R] with
      def indices: List[Int] = idx.values

  trait AxesConditionalRemover[TensorShape <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple, RemainingShape <: Tuple] extends AxesInTensor[TensorShape, IndexAxes]

  object AxesConditionalRemover:
    given bridge[T <: Tuple, RemovedAxis <: Tuple, IndexAxes <: Tuple, R <: Tuple](using
        idx: AxisIndices[T, IndexAxes],
        ev: RemoverAll.Aux[T, RemovedAxis, R]
    ): AxesConditionalRemover[T, RemovedAxis, IndexAxes, R] with
      def indices = idx.values

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
