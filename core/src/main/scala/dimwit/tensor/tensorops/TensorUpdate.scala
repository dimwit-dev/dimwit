package dimwit.tensor.tensorops

import dimwit.tensor.AxisAtIndex
import dimwit.tensor.AxisAtIndices
import dimwit.tensor.AxisAtRange
import dimwit.tensor.AxisAtTensorIndex
import dimwit.tensor.AxisAtTensorIndices
import dimwit.tensor.AxisAtTupleIndices
import dimwit.tensor.AxisAtWindow
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.AxisReplacer
import dimwit.tensor.Tensor

import StructuralExtensions.*

/** The update of a selection of a tensor, returned by `tensor.updated(selector)`.
  *
  * Applied to a value with the shape of the selection, it returns the updated copy of the tensor.
  * `updated` with a further selector selects along an axis of the selection, as `select` does, so that
  * `tensor.updated(s1).updated(s2)(value)` writes `value` where `tensor(s1)(s2)` reads, e.g. a single entry.
  *
  * @tparam T the shape of the tensor
  * @tparam S the shape of the selection
  */
final class TensorUpdate[T <: Tuple, S <: Tuple, V] private[tensorops] (
    private[tensorops] val selection: Labels[S] => Tensor[S, V],
    write: Tensor[S, V] => Tensor[T, V]
):

  /** The copy of the tensor with `value` at the selection. */
  def apply(value: Tensor[S, V]): Tensor[T, V] = write(value)

  // Selecting along a further axis writes back what it changes in the selection.

  def updated[L](selector: AxisAtIndex[L])(using ev: AxisRemover[S, L], labels: Labels[S]): TensorUpdate[T, ev.RemainingAxes, V] =
    andThen(selection(labels).updated(selector)(using ev, labels))

  def updated[L](selector: AxisAtTensorIndex[L])(using ev: AxisRemover[S, L], labels: Labels[S]): TensorUpdate[T, ev.RemainingAxes, V] =
    andThen(selection(labels).updated(selector)(using ev, labels))

  def updated[L](selector: AxisAtRange[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorUpdate[T, S, V] =
    andThen(selection(labels).updated(selector))

  def updated[L](selector: AxisAtIndices[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorUpdate[T, S, V] =
    andThen(selection(labels).updated(selector))

  def updated[L, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorUpdate[T, S, V] =
    andThen(selection(labels).updated(selector))

  def updated[L, L2](selector: AxisAtTensorIndices[L, L2])(using ev: AxisReplacer[S, L, L2], labels: Labels[S]): TensorUpdate[T, ev.NewShape, V] =
    andThen(selection(labels).updated(selector)(using ev, labels))

  def updated[L](selector: AxisAtWindow[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorUpdate[T, S, V] =
    andThen(selection(labels).updated(selector))

  /** The update of a selection of the selection: its value is written into the selection, and that into the tensor. */
  private def andThen[S2 <: Tuple](inner: TensorUpdate[S, S2, V]): TensorUpdate[T, S2, V] =
    TensorUpdate(inner.selection, value => write(inner(value)))
