package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.prime.PrimeConcat
import dimwit.tensor.Axis
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.Tensor
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

import scala.annotation.targetName

/** Provides extension methods for tensor contraction operations,
  * including outer products and dot products.
  */
object ContractionOps:

  extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])

    /** Computes the outer product of this tensor with another tensor.
      * Automatically primes the labels of the resulting tensor to avoid label collisions.
      */
    def outerProduct[OtherShape <: Tuple: Labels](other: Tensor[OtherShape, V])(using
        primeConcat: PrimeConcat[T, OtherShape],
        labels: Labels[primeConcat.Out]
    ): Tensor[primeConcat.Out, V] = Tensor(
      Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = 0) // generalized outer product
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
