package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.DType.Int32
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

object ReductionOps:

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    /** sums the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def sum[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.indices.toPythonProxy))
    def sum[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.index))
    def sum: Tensor0[V] = Tensor0(Jax.jnp.sum(t.jaxValue))

    /** takes the maximum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def max[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.indices.toPythonProxy))
    def max[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.index))
    def max: Tensor0[V] = Tensor0(Jax.jnp.max(t.jaxValue))

    /** takes the minimum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def min[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.indices.toPythonProxy))
    def min[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.index))
    def min: Tensor0[V] = Tensor0(Jax.jnp.min(t.jaxValue))

    /** argument of the maximum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def argmax[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.indices.toPythonProxy))
    def argmax[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.index))
    def argmax: Tensor0[Int32] = Tensor0(Jax.jnp.argmax(t.jaxValue))

    /** argument of the minimum of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def argmin[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.indices.toPythonProxy))
    def argmin[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.index))
    def argmin: Tensor0[Int32] = Tensor0(Jax.jnp.argmin(t.jaxValue))

    /** Returns a tensor of indices that would sort `t` along the specified axes */
    def argsort[Inputs <: Tuple](axes: Inputs)(using ev: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue, axis = ev.indices.toPythonProxy))
    def argsort[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue, axis = ev.index))
    def argsort: Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue))

    /** sorts the tensor `t` along the specified axis */
    def sort[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, V] = Tensor(Jax.jnp.sort(t.jaxValue, axis = ev.index))
    def sort: Tensor[T, V] = Tensor(Jax.jnp.sort(t.jaxValue))

    /** computes the cumulative sum of the tensor `t` along the specified axis. */
    def cumsum[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, V] = Tensor(Jax.jnp.cumsum(t.jaxValue, axis = ev.index))
    def cumsum: Tensor[T, V] = Tensor(Jax.jnp.cumsum(t.jaxValue, axis = -1))

    /** computes the cumulative product of the tensor `t` along the specified axis. */
    def cumprod[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, V] = Tensor(Jax.jnp.cumprod(t.jaxValue, axis = ev.index))
    def cumprod: Tensor[T, V] = Tensor(Jax.jnp.cumprod(t.jaxValue, axis = -1))

    /** computes the discrete difference of the tensor `t` along the specified axis, reducing that axis' size by one. */
    def diff[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, V] = Tensor(Jax.jnp.diff(t.jaxValue, axis = ev.index))
    def diff: Tensor[T, V] = Tensor(Jax.jnp.diff(t.jaxValue))

  // ---------------------------------------------------------
  // IsFloat operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    /** computes the mean of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def mean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.indices.toPythonProxy))
    def mean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.index))
    def mean: Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))

    /** computes the mean of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def std[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.indices.toPythonProxy))
    def std[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.index))
    def std: Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))

    /** computes the qth quantile of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def quantile[Inputs <: Tuple](q: Float, axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.indices.toPythonProxy))
    def quantile[L: Label](q: Float, axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.index))
    def quantile(q: Float): Tensor0[V] = Tensor0(Jax.jnp.quantile(t.jaxValue, q))

    /** computes the median of the tensor `t` along the specified axes, returning a new tensor with those axes removed. */
    def median: Tensor0[V] = Tensor0(Jax.jnp.median(t.jaxValue))
    def median[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.index))
    def median[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.indices.toPythonProxy))

    /** computes the mean of the tensor `t` along the specified axes, ignoring na values and returning a new tensor with those axes removed. */
    def nanmean: Tensor0[V] = Tensor0(Jax.jnp.nanmean(t.jaxValue))
    def nanmean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.index))
    def nanmean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.indices.toPythonProxy))

    /** computes the median of the tensor `t` along the specified axes, ignoring na values and returning a new tensor with those axes removed. */
    def nanmedian[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.indices.toPythonProxy))
    def nanmedian[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.index))
    def nanmedian: Tensor0[V] = Tensor0(Jax.jnp.nanmedian(t.jaxValue))
