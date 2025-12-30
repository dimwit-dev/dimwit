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

    /** Split this key into multiple independent keys stored in a tensor */
    def splitToTensor[L: Label](axis: Axis[L], num: Int): Tensor1[L, Key] =
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      Tensor[Tuple1[L], Key](splitKeys)

    /** Split into exactly 2 keys (common case) */
    def split2(): (Key, Key) =
      val keys = split(2)
      (keys(0), keys(1))

    /** Generate a tensor of samples by splitting the key along the given axis and applying f to each sub-key ^ */
    def splitvmap[L: Label, T <: Tuple: Labels, V](axis: Axis[L], n: Int)(f: Key => Tensor[T, V]): Tensor[L *: T, V] =
      this.splitToTensor(axis, n).vmap(axis)(k => f(Key(k.jaxValue)))

    /** Generate a new key by splitting */
    def next(): Key = split2()._2

  object Key:
    /** Create a random key from an integer seed */
    def apply(seed: Int): Key = Key(Jax.jrandom.key(seed))

    /** Create a random key from current time */
    def fromTime(): Key = Key(System.currentTimeMillis().toInt)

    /** Create a random key from Scala's random */
    def random(): Key = Key(scala.util.Random.nextInt())

    /** Reader instance to enable .item on Tensor0[Key] */
    given me.shadaj.scalapy.readwrite.Reader[Key] with
      def read(v: me.shadaj.scalapy.py.Any): Key =
        Key(v.as[Jax.PyDynamic])
