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

  // ---------------------------------------------------------
  // IsNumber operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

    // --- Sum ---
    def sum: Tensor0[V] = Tensor0(Jax.jnp.sum(t.jaxValue))
    def sum[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.index))
    def sum[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Max ---
    def max: Tensor0[V] = Tensor0(Jax.jnp.max(t.jaxValue))
    def max[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.index))
    def max[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Min ---
    def min: Tensor0[V] = Tensor0(Jax.jnp.min(t.jaxValue))
    def min[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.index))
    def min[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Argmax ---
    def argmax: Tensor0[Int32] = Tensor0(Jax.jnp.argmax(t.jaxValue))
    def argmax[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.index))
    def argmax[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Argmin ---
    def argmin: Tensor0[Int32] = Tensor0(Jax.jnp.argmin(t.jaxValue))
    def argmin[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.index))
    def argmin[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, Int32] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Argsort ---
    def argsort: Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue))
    def argsort[L: Label](axis: Axis[L])(using ev: AxisIndex[T, L]): Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue, axis = ev.index))
    def argsort[Inputs <: Tuple](axes: Inputs)(using ev: AxisIndices[T, UnwrapAxes[Inputs]]): Tensor[T, Int32] = Tensor(Jax.jnp.argsort(t.jaxValue, axis = ev.indices.toPythonProxy))

  // ---------------------------------------------------------
  // IsFloat operations (IsFloat or IsInt)
  // ---------------------------------------------------------

  extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

    // --- Mean ---
    def mean: Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))
    def mean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.index))
    def mean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Std ---
    def std: Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))
    def std[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.index))
    def std[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.indices.toPythonProxy))

    // --- Quantile ---
    def quantile(q: Float): Tensor0[V] = Tensor0(Jax.jnp.quantile(t.jaxValue, q))
    def quantile[L: Label](q: Float, axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.index))
    def quantile[Inputs <: Tuple](q: Float, axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.indices.toPythonProxy))

    // --- Median ---
    def median: Tensor0[V] = Tensor0(Jax.jnp.median(t.jaxValue))
    def median[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.index))
    def median[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.indices.toPythonProxy))

    def nanmean: Tensor0[V] = Tensor0(Jax.jnp.nanmean(t.jaxValue))
    def nanmean[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.index))
    def nanmean[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmean(t.jaxValue, axis = ev.indices.toPythonProxy))

    def nanmedian: Tensor0[V] = Tensor0(Jax.jnp.nanmedian(t.jaxValue))
    def nanmedian[L: Label](axis: Axis[L])(using ev: AxisRemover[T, L], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.index))
    def nanmedian[Inputs <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs]], l: Labels[ev.RemainingAxes]): Tensor[ev.RemainingAxes, V] = Tensor(Jax.jnp.nanmedian(t.jaxValue, axis = ev.indices.toPythonProxy))
