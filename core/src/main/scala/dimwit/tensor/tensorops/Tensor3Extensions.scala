package dimwit.tensor.tensorops

import dimwit.tensor.HasScalar
import dimwit.tensor.Tensor3

private[dimwit] object Tensor3Extensions:

  extension [L1, L2, L3, V, X](t: Tensor3[L1, L2, L3, V])(using ev: HasScalar[V, X])
    /** Converts a Tensor3 to a nested Scala Array (Array of Arrays of Arrays).
      * The user must ensure that the tensor is not a JAX Tracer
      * (i.e., it is not part of a JAX computation graph) before calling this method,
      * otherwise a runtime error will occur.
      */
    def toArray: Array[Array[Array[X]]] =
      require(!t.isTracer, "Cannot convert a JAX Tracer to an array.")
      given scala.reflect.ClassTag[X] = ev.classTag
      val d0 = t.shape.dimensions(0); val d1 = t.shape.dimensions(1); val d2 = t.shape.dimensions(2)
      val values = ev.readFlat(t.jaxValue)
      Array.tabulate(d0, d1)((i, j) => values.slice((i * d1 + j) * d2, (i * d1 + j + 1) * d2))
