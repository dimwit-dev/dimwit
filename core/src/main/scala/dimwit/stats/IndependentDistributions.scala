package dimwit.stats

import dimwit.*
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random

class Normal[T <: Tuple: Labels](
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:

  require(loc.shape == scale.shape, "loc and scale must have the same shape")

  override def logProb(x: Tensor[T, Float]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    val standardNormal = Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.normal(key.jaxKey, loc.shape.dimensions.toPythonProxy))
    standardNormal *! scale +! loc

object Normal:
  def standardNormal[T <: Tuple: Labels](shape: Shape[T]) = new Normal(
    loc = Tensor.zeros(shape, VType[Float]),
    scale = Tensor.ones(shape, VType[Float])
  )

class Uniform[T <: Tuple: Labels](
    val low: Tensor[T, Float],
    val high: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(low.shape == high.shape, "Low and high must have the same shape")

  override def logProb(x: Tensor[T, Float]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.uniform(loc = low.jaxValue, scale = (high - low).jaxValue).logpdf(x.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.uniform(key.jaxKey, shape = low.shape.dimensions.toPythonProxy, minval = low.jaxValue, maxval = high.jaxValue)
    )

class Bernoulli[T <: Tuple: Labels](
    val probs: Tensor[T, Float]
) extends IndependentDistribution[T, Int]:

  override def logProb(x: Tensor[T, Int]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.bernoulli(p = probs.jaxValue).logpmf(x.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.bernoulli(key.jaxKey, p = probs.jaxValue))

class Multinomial[L: Label](
    val n: Int,
    val probs: Tensor1[L, Float]
) extends IndependentDistribution[Tuple1[L], Int]:

  private lazy val logProbs: Tensor1[L, Float] = probs.log

  override def logProb(x: Tensor1[L, Int]): Tensor1[L, Float] =
    Tensor.fromPy(VType[Float])(jstats.multinomial(n = n, p = probs.jaxValue).logpmf(x.jaxValue))

  override def sample(key: Random.Key): Tensor1[L, Int] =
    Tensor.fromPy(VType[Int])(
      Jax.jrandom.multinomial(
        key.jaxKey,
        n = n,
        pvals = probs.jaxValue
      )
    )

class Categorical[L: Label](val probs: Tensor1[L, Float]) extends IndependentDistribution[EmptyTuple, Int]:

  private val numCategories = probs.shape.dimensions(0)
  private val logProbs = probs.log

  override def logProb(x: Tensor0[Int]): Tensor0[Float] =
    Tensor.fromPy(VType[Float]) {
      val indices = py.Dynamic.global.`range`(numCategories)
      jstats.rv_discrete(values = indices, probs.jaxValue).logpmf(x.jaxValue)
    }
  override def sample(key: Random.Key): Tensor0[Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.categorical(key.jaxKey, logProbs.jaxValue))

class Cauchy[T <: Tuple: Labels](
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(loc.shape == scale.shape, "Location and scale must have the same shape")

  override def logProb(x: Tensor[T, Float]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.cauchy(loc = loc.jaxValue, scale = scale.jaxValue).logpdf(x.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy))

class HalfNormal[T <: Tuple: Labels](
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:

  require(loc.shape == scale.shape, "Mean and scale must have the same shape")

  override def logProb(x: Tensor[T, Float]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.halfnorm(loc = loc.jaxValue, scale = scale.jaxValue).logpdf(x.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    (Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.halfnorm(k.jaxKey)) *! scale +! loc).abs

class StudentT[T <: Tuple: Labels](
    val df: Int,
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(loc.shape == scale.shape, "loc, and scale must have the same shape")

  override def logProb(x: Tensor[T, Float]): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(jstats.t(df = df, loc = loc.jaxValue, scale = scale.jaxValue).logpdf(x.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.t(k.jaxKey, df = df, shape = loc.shape.dimensions.toPythonProxy)
    ) *! scale +! loc
