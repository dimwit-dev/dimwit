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
import dimwit.tensor.Tensor2
import dimwit.tensor.Tensor3

object Tensor3Ops:

  extension [L1, L2, L3, V, X](t: Tensor3[L1, L2, L3, V])(using ev: HasScalar[V, X])
    /** Converts a Tensor3 to a nested Scala Array (Array of Arrays of Arrays).
      * The user must ensure that the tensor is not a JAX Tracer
      * (i.e., it is not part of a JAX computation graph) before calling this method,
      * otherwise a runtime error will occur.
      */
    def toArray: Array[Array[Array[X]]] =
      require(!t.isTracer, "Cannot convert a JAX Tracer to an array.")
      given scala.reflect.ClassTag[X] = ev.classTag
      val d1 = t.shape.dimensions(1); val d2 = t.shape.dimensions(2)
      ev.readFlat(t.jaxValue).grouped(d1 * d2).map(_.grouped(d2).toArray).toArray
