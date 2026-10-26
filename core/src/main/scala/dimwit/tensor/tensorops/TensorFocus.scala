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

/** A focus on a selection of a tensor, returned by `tensor.focus(selector)`, to read or write it.
  *
  * `.get` reads the selection, `.updated(value)` returns a copy of the tensor with `value` at the selection,
  * and `.updatedWith(f)` returns a copy with `f` applied to the selection. `.focus` with a further selector selects
  * along an axis of the selection, as `select` does, so that `tensor.focus(s1).focus(s2).updated(value)` writes
  * `value` where `tensor(s1)(s2)` reads, e.g. a single entry. Each step reads its selection and writes it back.
  *
  * It behaves like a lens as long as the selected positions are distinct and in bounds. For repeated indices,
  * it is unspecified which of their values is written; a traced index out of bounds is read clamped and not written.
  *
  * @tparam T the shape of the tensor
  * @tparam S the shape of the selection
  */
final class TensorFocus[T <: Tuple, S <: Tuple, V] private[tensorops] (
    private[tensorops] val selection: Labels[S] => Tensor[S, V],
    write: Tensor[S, V] => Tensor[T, V]
):

  /** The selection, the same as the chained `select` calls. */
  def get(using labels: Labels[S]): Tensor[S, V] = selection(labels)

  /** The copy of the tensor with `value`, which has the shape of the selection, at the selection. */
  def updated(value: Tensor[S, V]): Tensor[T, V] = write(value)

  /** The copy of the tensor with `f` applied to the selection. `f` must keep the shape of the selection. */
  def updatedWith(f: Tensor[S, V] => Tensor[S, V])(using labels: Labels[S]): Tensor[T, V] = write(f(selection(labels)))

  // Focusing along a further axis writes back what it changes in the selection.

  def focus[L](selector: AxisAtIndex[L])(using ev: AxisRemover[S, L], labels: Labels[S]): TensorFocus[T, ev.RemainingAxes, V] =
    andThen(get.focus(selector)(using ev, labels))

  def focus[L](selector: AxisAtTensorIndex[L])(using ev: AxisRemover[S, L], labels: Labels[S]): TensorFocus[T, ev.RemainingAxes, V] =
    andThen(get.focus(selector)(using ev, labels))

  def focus[L](selector: AxisAtRange[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorFocus[T, S, V] =
    andThen(get.focus(selector))

  def focus[L](selector: AxisAtIndices[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorFocus[T, S, V] =
    andThen(get.focus(selector))

  def focus[L, I <: NonEmptyTuple](selector: AxisAtTupleIndices[L, I])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorFocus[T, S, V] =
    andThen(get.focus(selector))

  def focus[L, L2](selector: AxisAtTensorIndices[L, L2])(using ev: AxisReplacer[S, L, L2], labels: Labels[S]): TensorFocus[T, ev.NewShape, V] =
    andThen(get.focus(selector)(using ev, labels))

  def focus[L](selector: AxisAtWindow[L])(using axisIndex: AxisIndex[S, L], labels: Labels[S]): TensorFocus[T, S, V] =
    andThen(get.focus(selector))

  /** The focus on a selection of the selection: its value is written into the selection, and that into the tensor. */
  private def andThen[S2 <: Tuple](inner: TensorFocus[S, S2, V]): TensorFocus[T, S2, V] =
    TensorFocus(inner.selection, value => write(inner.updated(value)))
