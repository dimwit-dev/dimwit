package dimwit.random

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import dimwit.jax.{Jax, JaxDType}
import me.shadaj.scalapy.py.SeqConverters

/** JAX-based random number generation with proper key management.
  *
  * JAX uses a functional approach to randomness where:
  *   - Random keys must be explicitly managed
  *   - Keys are split to generate independent random streams
  *   - This ensures reproducibility and parallelizability
  *
  * This object provides low-level sampling primitives using JAX. For statistical modeling, prefer using distribution classes in dimwit.distributions.
  */
object Random:

  /** A random key for generating random numbers */
  case class Key(jaxKey: Jax.PyDynamic):
    /** Split this key into multiple independent keys */
    def split(num: Int): Seq[Key] =
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      (0 until num).map(i => Key(splitKeys.__getitem__(i)))

    /** Split into exactly 2 keys (common case) */
    def split2(): (Key, Key) =
      val keys = split(2)
      (keys(0), keys(1))

    /** Generate a new key by splitting */
    def next(): Key = split2()._2

  object Key:
    /** Create a random key from an integer seed */
    def apply(seed: Int): Key = Key(Jax.jrandom.key(seed))

    /** Create a random key from current time */
    def fromTime(): Key = Key(System.currentTimeMillis().toInt)

    /** Create a random key from Scala's random */
    def random(): Key = Key(scala.util.Random.nextInt())

  object Normal:

    class NormalFactory[T <: Tuple: Labels](shape: Shape[T]):
      def apply(
          key: Key
      )(using
          executionType: ExecutionType[Float]
      ): Tensor[T, Float] = this(key, Tensor0.zero(VType[Float]), Tensor0.one(VType[Float]))

      def apply(
          key: Key,
          mean: Tensor0[Float],
          std: Tensor0[Float]
      )(using
          executionType: ExecutionType[Float]
      ): Tensor[T, Float] =
        val jaxValues = Jax.jrandom.normal(
          key.jaxKey,
          shape.dimensions.toPythonProxy,
          dtype = executionType.dtype.jaxType
        )
        val standardNormal = Tensor[T, Float](jaxValues)
        standardNormal :* std :+ mean

    def apply[T <: Tuple: Labels](shape: Shape[T]) = new NormalFactory[T](shape)

  class Uniform:

    class UniformFactory[T <: Tuple: Labels](shape: Shape[T]):

      def apply(
          key: Key
      )(using
          executionType: ExecutionType[Float]
      ): Tensor[T, Float] = this(key, Tensor0.zero(VType[Float]), Tensor0.one(VType[Float]))

      /** Uniform distribution in [minval, maxval) */
      def apply(
          key: Key,
          minval: Tensor0[Float],
          maxval: Tensor0[Float]
      )(using
          executionType: ExecutionType[Float]
      ): Tensor[T, Float] =
        val jaxValues = Jax.jrandom.uniform(
          key.jaxKey,
          shape.dimensions.toPythonProxy,
          minval = minval.jaxValue,
          maxval = maxval.jaxValue,
          dtype = executionType.dtype.jaxType
        )
        Tensor(jaxValues)

    def apply[T <: Tuple: Labels](shape: Shape[T]) = new UniformFactory[T](shape)
