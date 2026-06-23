package dimwit.tensor.tensorops

import dimwit.tensor.Tensor
import dimwit.tensor.Labels
import dimwit.jax.Jax
import dimwit.tensor.DType.Bool
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsBoolean
import dimwit.tensor.VType
import dimwit.tensor.DType.Int32
import dimwit.tensor.DType.Float32
import dimwit.tensor.TensorOps.IsInteger
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.Label
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.AxesRemover
import dimwit.tensor.Axis
import dimwit.tensor.ShapeTypeHelpers.UnwrapAxes
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}
import dimwit.tensor.TupleHelpers.PrimeConcat
import scala.annotation.targetName

object ContractionOps:

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])

    /** Computes the outer product of this tensor with another tensor.
      * Automatically primes the labels of the resulting tensor to avoid label collisions.
      */
    def outerProduct[OtherShape <: Tuple: Labels](other: Tensor[OtherShape, V])(using
        primeConcat: PrimeConcat[T, OtherShape],
        labels: Labels[primeConcat.Out]
    ): Tensor[primeConcat.Out, V] = Tensor(
      // Jax outer product flattens, reshape required
      Jax.jnp.reshape(
        Jax.jnp.outer(tensor.jaxValue, other.jaxValue),
        (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
      )
    )

    /** Computes the dot product of this tensor with another tensor along the specified axis.
      * The axis must be present in both tensors and will be contracted (removed) from the resulting tensor.
      *
      * @param axis The axis along which to contract. Must be present in both tensors.
      * @param other The other tensor to contract with.
      */
    def dot[
        ContractAxis,
        OtherShape <: Tuple
    ](axis: Axis[ContractAxis])(other: Tensor[OtherShape, V])(using
        ev: AxisRemover[T, ContractAxis],
        evOther: AxisRemover[OtherShape, ContractAxis]
    )(using
        primeConcat: PrimeConcat[ev.RemainingAxes, evOther.RemainingAxes],
        labelsOut: Labels[primeConcat.Out]
    ): Tensor[primeConcat.Out, V] =
      val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
      val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
      val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

      Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))

    /** Computes the dot product of this tensor with another tensor along the specified pair of axes.
      * The axes must be present in their respective tensors and will be contracted (removed) from the resulting tensor.
      *
      * @param axis The pair of axes along which to contract. Each axis must be present in its respective tensor.
      * @param other The other tensor to contract with.
      *
      * Example usage:
      * {{{
      * val t1: Tensor[("A", "B", "C"), Float] = ???
      * val t2: Tensor[("D", "E, "F), Float] = ???
      * val result = t1.dot(Axis["B" ~ "F])(t2)
      * }}}
      */
    @targetName("dotOn")
    def dot[
        ContractAxisA,
        ContractAxisB,
        OtherShape <: Tuple
    ](axisPair: (Axis[ContractAxisA], Axis[ContractAxisB]))(other: Tensor[OtherShape, V])(using
        ev: AxisRemover[T, ContractAxisA],
        evOther: AxisRemover[OtherShape, ContractAxisB]
    )(using
        primeConcat: PrimeConcat[ev.RemainingAxes, evOther.RemainingAxes],
        outLabels: Labels[primeConcat.Out]
    ): Tensor[primeConcat.Out, V] =
      val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
      val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
      val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

      Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))
