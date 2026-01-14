package dimwit.stats

import dimwit.*
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random

class Normal[LocT <: T, ScaleT <: T, T <: Tuple: Labels](
    val loc: Tensor[LocT, Float],
    val scale: Tensor[ScaleT, Float]
) extends IndependentDistribution[T, Float]:

  require(loc.shape.dimensions == scale.shape.dimensions, "loc and scale must have the same dimensions")

  override def logProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    val standardNormal = Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.normal(key.jaxKey, loc.shape.dimensions.toPythonProxy))
    standardNormal * scale + loc

object Normal:
  def standardNormal[T <: Tuple: Labels](shape: Shape[T]) = new Normal(
    loc = Tensor.zeros(shape, VType[Float]),
    scale = Tensor.ones(shape, VType[Float])
  )

class Uniform[T <: Tuple: Labels](
    val low: Tensor[T, Float],
    val high: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(low.shape.dimensions == high.shape.dimensions, "Low and high must have the same dimensions")

  override def logProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.uniform.logpdf(x.jaxValue, loc = low.jaxValue, scale = (high - low).jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.uniform(key.jaxKey, shape = low.shape.dimensions.toPythonProxy, minval = low.jaxValue, maxval = high.jaxValue)
    )

class Bernoulli[T <: Tuple: Labels](
    val probs: Tensor[T, Float]
) extends IndependentDistribution[T, Int]:

  override def logProb(x: Tensor[T, Int]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.bernoulli.logpmf(x.jaxValue, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.bernoulli(key.jaxKey, p = probs.jaxValue))

class Cauchy[T <: Tuple: Labels](
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(loc.shape.dimensions == scale.shape.dimensions, "Location and scale must have the same dimensions")

  override def logProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy)) * scale + loc

class HalfNormal[T <: Tuple: Labels](
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:

  require(loc.shape.dimensions == scale.shape.dimensions, "Mean and scale must have the same dimensions")

  override def logProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    // Half-normal is a folded normal: logpdf = log(2) + norm.logpdf
    Tensor.fromPy(VType[LogProb])(
      Jax.jnp.log(2.0) + jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )

  override def sample(k: Random.Key): Tensor[T, Float] =
    // Half-normal: sample from normal and take absolute value
    val normal = Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.normal(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy))
    (normal * scale + loc).abs

class StudentT[T <: Tuple: Labels](
    val df: Int,
    val loc: Tensor[T, Float],
    val scale: Tensor[T, Float]
) extends IndependentDistribution[T, Float]:
  require(loc.shape.dimensions == scale.shape.dimensions, "loc, and scale must have the same dimensions")

  override def logProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.t.logpdf(x.jaxValue, df = df, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.t(k.jaxKey, df = df, shape = loc.shape.dimensions.toPythonProxy)
    ) * scale + loc
