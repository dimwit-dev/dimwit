package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.DType._
import dimwit.tensor.HasScalar
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.Tensor1
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

object Tensor1Ops:

  extension [L, V](t: Tensor1[L, V])

    def relabelTo[NewL: Label](newAxis: Axis[NewL]): Tensor1[NewL, V] = Tensor[Tuple1[NewL], V](t.jaxValue)

    // TODO generalize to TensorN (like slice)
    def dynamicSlice(
        dynamicStart: Tensor0[Int32],
        staticSize: Int
    )(using
        label: Label[L]
    ): Tensor1[L, V] =
      // TODO understand why toPythonCopy is needed and toPythonProxy fails!
      Tensor(Jax.lax.dynamic_slice(t.jaxValue, Seq(dynamicStart.jaxValue).toPythonCopy, Seq(staticSize).toPythonCopy))

  extension [L, V, X](t: Tensor1[L, V])(using ev: HasScalar[V, X])
    /** Converts a Tensor1 to a Scala Array.
      * The user must ensure that the tensor is not a JAX Tracer
      * (i.e., it is not part of a JAX computation graph) before calling this method,
      * otherwise a runtime error will occur.
      */
    def toArray: Array[X] =
      require(!t.isTracer, "Cannot convert a JAX Tracer to an array.")
      ev.readFlat(t.jaxValue)
