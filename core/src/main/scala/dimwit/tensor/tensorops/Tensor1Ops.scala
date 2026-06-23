package dimwit.tensor.tensorops

import dimwit.tensor.Tensor0
import dimwit.tensor.DType.*
import dimwit.tensor.TensorOps
import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.Labels
import dimwit.tensor.tensorops.ElementWiseOps.add
import dimwit.tensor.Tensor
import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast
import dimwit.tensor.tensorops.ElementWiseOps.subtract
import dimwit.tensor.tensorops.ElementWiseOps.multiply
import dimwit.tensor.tensorops.ElementWiseOps.divide
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Tensor1
import dimwit.tensor.HasScalar
import dimwit.jax.Jax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.{Reader, Writer}

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
