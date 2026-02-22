package dimwit.random

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import dimwit.jax.{Jax, JaxDType}
import me.shadaj.scalapy.py.SeqConverters
import dimwit.python.PyBridge.liftPyTensor
import scala.compiletime.{requireConst, constValue, ops}
import Tuple.Size
import dimwit.tensor.TupleHelpers.TupleNOf

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

    /** Split this key into N independent keys as a Tuple */
    inline def splitToTuple[N <: Int & Singleton](inline n: N): TupleNOf[N, Key] =
      requireConst(n)
      val num = n.asInstanceOf[Int]
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      val keysArray = Array.tabulate(num)(i => Key(splitKeys.__getitem__(i)))
      Tuple.fromArray(keysArray).asInstanceOf[TupleNOf[N, Key]]

    /** Split this key into multiple independent keys stored in a tensor */
    def splitToTensor[L: Label](dim: AxisExtent[L]): Tensor1[L, Key] =
      val n = dim.size
      val splitKeys = Jax.jrandom.split(jaxKey, n)
      Tensor[Tuple1[L], Key](splitKeys)

    /** Split into exactly 2 keys (common case) */
    def split2(): (Key, Key) =
      val keys = split(2)
      (keys(0), keys(1))

    /** Generate a tensor of samples by splitting the key along the given axis and applying f to each sub-key ^ */
    def splitvmap[L: Label, T <: Tuple: Labels, V](dim: AxisExtent[L])(f: Key => Tensor[T, V]): Tensor[L *: T, V] =
      val axis = dim.axis
      this.splitToTensor(dim).vmap(axis)(k => f(k.item))

    /** Generate a new key by splitting */
    def next: Key = split2()._2

    override def equals(other: Any): Boolean =
      other match
        case that: Key => Jax.jnp.array_equal(this.jaxKey, that.jaxKey).item().as[Boolean]
        case _         => false

    override def hashCode(): Int = jaxKey.tobytes().hashCode()

  /** Generate a random permutation of indices from 0 to n-1.
    *
    * Common use case: shuffle a tensor along an axis by combining with `.take()`:
    * {{{}
    * val key = Random.Key(42)
    * val data = Tensor2.fromArray(Axis[Row], Axis[Col], VType[Int])(
    *    Array(Array(1, 2),
    *    Array(3, 4),
    *    Array(5, 6)) )
    *    val perm = Random.permutation(Axis[Row] -> 3)(key)
    *    val shuffled = data.take(Axis[Row])(perm)
    * }}}
    *
    * @param dim
    *   Specifies the axis label for the resulting permutation tensor and its size n
    * @param key
    *   The random key
    * @return
    *   A 1D tensor containing a random permutation of [0, 1, ..., n-1]
    */
  def permutation[L: Label](dim: AxisExtent[L])(key: Key): Tensor1[L, Int] =
    liftPyTensor(Jax.jrandom.permutation(key.jaxKey, dim.size))

  object Key:
    /** Create a random key from an integer seed */
    def apply(seed: Int): Key = Key(Jax.jrandom.key(seed))

    /** Create a random key from current time */
    def fromTime(): Key = Key(System.currentTimeMillis().toInt)

    /** Create a random key from Scala's random */
    def random(): Key = Key(scala.util.Random.nextInt())

  // Enable .item on Tensor0[Key] to extract the Key
  // Note that implementing a Reader instance and using
  // the standard jax.item does not work, as Key is
  // not a primitive type in  JAX.
  extension (tensorKey: Tensor0[Key])
    def item: Key = Key(tensorKey.jaxValue)
