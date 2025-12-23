package shapeful.random

import shapeful.tensor.*
import shapeful.tensor.TensorOps.*
import shapeful.jax.{Jax, JaxDType}
import me.shadaj.scalapy.py.SeqConverters

/** JAX-based random number generation with proper key management.
  *
  * JAX uses a functional approach to randomness where:
  *   - Random keys must be explicitly managed
  *   - Keys are split to generate independent random streams
  *   - This ensures reproducibility and parallelizability
  *
  * This object provides low-level sampling primitives using JAX. For statistical modeling, prefer using distribution
  * classes in shapeful.distributions.
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

  /** Generate random samples from various distributions.
    *
    * These are low-level JAX primitives. For statistical modeling with logpdf support, use the distribution classes in
    * shapeful.distributions instead.
    */

  /** Normal distribution with specified mean and standard deviation */
  def normal[T <: Tuple : Labels, V : Value](
      key: Key,
      shape: Shape[T],
      mean: Tensor0[V] = Tensor0.of[DType.Float32.type].apply(0f),
      std: Tensor0[V] = Tensor0.of[DType.Float32.type].apply(1f)
  )(using ev: V =:= DType.Float32.type): Tensor[T, V] =
    val dtype = summon[Value[V]].dtype
    val jaxValues = Jax.jrandom.normal(
      key.jaxKey,
      shape.dimensions.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    val standardNormal = Tensor.fromPy[T, V](jaxValues)
    standardNormal :* std :+ mean

  /** Uniform distribution in [0, 1) */
  def uniform[T <: Tuple : Labels, V : Value](
      key: Key,
      shape: Shape[T]
  )(using ev: V =:= DType.Float32.type): Tensor[T, V] = 
    uniform(key, shape, Tensor0.of[V].apply(0f), Tensor0.of[V].apply(1f))

  /** Uniform distribution in [minval, maxval) */
  def uniform[T <: Tuple : Labels, V : Value](
      key: Key,
      shape: Shape[T],
      minval: Tensor0[V],
      maxval: Tensor0[V]
  ): Tensor[T, V] =
    val dtype = summon[Value[V]].dtype
    val jaxValues = Jax.jrandom.uniform(
      key.jaxKey,
      shape.dimensions.toPythonProxy,
      minval = minval.jaxValue,
      maxval = maxval.jaxValue,
      dtype = JaxDType.jaxDtype(dtype)
    )
    Tensor.fromPy[T, V](jaxValues)

