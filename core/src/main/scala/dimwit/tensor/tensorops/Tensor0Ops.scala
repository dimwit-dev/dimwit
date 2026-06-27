package dimwit.tensor.tensorops

import dimwit.tensor.DType._
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
    def item: Boolean =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Boolean]

  extension (scalar: Tensor0[Int8])
    def item: Byte =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Byte]

  extension (scalar: Tensor0[Int16])
    def item: Short =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Int].toShort

  extension (scalar: Tensor0[Int32])
    def item: Int =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Int]

  extension (scalar: Tensor0[Int64])
    def item: Long =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Long]

  extension (scalar: Tensor0[Float32])
    def item: Float =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Float]

  extension (scalar: Tensor0[Float64])
    def item: Double =
      checkTracer(scalar)
      scalar.jaxValue.item().as[Double]
