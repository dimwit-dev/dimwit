package dimwit.tensor.tensorops

import dimwit.tensor.Axis
import dimwit.tensor.HasScalar
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.Tensor2

object Tensor2Ops:

  extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

    // Support .transpose without arguments for 2D tensors while keeping (not shadowing) the general .transpose with arguments
    def transpose: Tensor2[L2, L1, V] = t.transpose(Axis[L2], Axis[L1])
    def transpose(axis2: Axis[L2], axis1: Axis[L1]): Tensor2[L2, L1, V] = StructuralOps.transpose(t)(axis2, axis1)

  extension [L1, L2, V, X](t: Tensor2[L1, L2, V])(using ev: HasScalar[V, X])
    /** Converts a Tensor2 to a nested Scala Array (Array of Arrays).
      * The user must ensure that the tensor is not a JAX Tracer
      * (i.e., it is not part of a JAX computation graph) before calling this method,
      * otherwise a runtime error will occur.
      */
    def toArray: Array[Array[X]] =
      require(!t.isTracer, "Cannot convert a JAX Tracer to an array.")
      given scala.reflect.ClassTag[X] = ev.classTag
      ev.readFlat(t.jaxValue).grouped(t.shape.dimensions(1)).toArray
