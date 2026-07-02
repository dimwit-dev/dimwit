package dimwit.tensor.tensorops

import dimwit.tensor.DType.*
import dimwit.tensor.Tensor0

object Tensor0Ops:

  private inline def checkTracer[V, R](scalar: Tensor0[V]): Unit =
    require(
      !scalar.isTracer,
      """
        | Cannot convert a JAX Tracer to a scalar value. Tensor0 is part of a JAX computation graph (e.g., inside vmap or a jitted function).
        | Common mistakes leading to this error:
        |   - calling .slice(t0.item) rather than .slice(t0); breaking the computation graph unintentionally.
        |""".stripMargin
    )

  extension (scalar: Tensor0[Bool])
    /** return the underlying boolean value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Boolean =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Boolean]

  extension (scalar: Tensor0[Int8])
    /** return the underlying Byte value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Byte =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Byte]

  extension (scalar: Tensor0[Int16])
    /** return the underlying Short value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Short =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Int].toShort

  extension (scalar: Tensor0[Int32])
    /** return the underlying Int value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Int =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Int]

  extension (scalar: Tensor0[Int64])
    /** return the underlying Long value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Long =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Long]

  extension (scalar: Tensor0[Float32])
    /** return the underlying Float value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Float =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Float]

  extension (scalar: Tensor0[Float64])
    /** return the underlying Double value of the scalar tensor.
      * Attention! Breaks the computational graph.
      */
    def item: Double =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Double]
