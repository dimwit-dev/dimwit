package dimwit.stats

import dimwit.*
import dimwit.DType.Float32
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random
import dimwit.python.PyBridge.liftPyTensor

/** Normal (Gaussian) distribution */
class Normal[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    val standardNormal = liftPyTensor(loc.shape, loc.vtype)(Jax.jrandom.normal(key.jaxKey, loc.shape.dimensions.toPythonProxy))
    standardNormal * scale + loc

object Normal:

  /** Create a Normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): Normal[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc and scale must have the same dimensions")
    new Normal(loc, scale)

  def isotropic[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor0[V]): Normal[T, V] = new Normal(loc = loc, scale = scale.broadcastTo(loc.shape))
  def standardIsotropic[T <: Tuple: Labels, V: IsFloating](shape: Shape[T], scale: Tensor0[V]): Normal[T, V] = isotropic(loc = Tensor(shape, VType[V]).fill(0f), scale = scale)

  /** Sample from standard normal distribution N(0, 1) */
  def standardSample(key: Random.Key): Tensor0[Float32] = new Normal(Tensor0(0f), Tensor0(1f)).sample(key)
  def standardNormal[T <: Tuple: Labels](shape: Shape[T]): Normal[T, Float32] = Normal.standardIsotropic(shape, scale = Tensor0(VType[Float32])(1f))

/** Uniform distribution */
class Uniform[T <: Tuple: Labels, V: IsFloating](val low: Tensor[T, V], val high: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.uniform.logpdf(x.jaxValue, loc = low.jaxValue, scale = (high - low).jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.uniform(key.jaxKey, shape = low.shape.dimensions.toPythonProxy, minval = low.jaxValue, maxval = high.jaxValue)
    )

/** Uniform distribution */
class DiscreteUniform[T <: Tuple: Labels](val min: Tensor[T, Int], val max: Tensor[T, Int]) extends IndependentDistribution[T, Int]:

  override def elementWiseLogProb(x: Tensor[T, Int]): Tensor[T, LogProb] =
    liftPyTensor(jstats.randint.logpmf(x.jaxValue, low = min.jaxValue, high = max.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Int] =
    liftPyTensor(
      Jax.jrandom.randint(key.jaxKey, shape = min.shape.dimensions.toPythonProxy, minval = min.jaxValue, maxval = max.jaxValue)
    )

object Uniform:

  /** Create a Uniform distribution from low and high tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](low: Tensor[T, V], high: Tensor[T, V]): Uniform[T, V] =
    require(low.shape.dimensions == high.shape.dimensions, "Low and high must have the same dimensions")
    new Uniform(low, high)

  /** Create a discrete Uniform distribution from low and high int tensors */
  def apply[T <: Tuple: Labels](min: Tensor[T, Int], max: Tensor[T, Int]): DiscreteUniform[T] =
    require(min.shape.dimensions == max.shape.dimensions, "min and max must have the same dimensions")
    new DiscreteUniform(min, max)

/** Bernoulli distribution */
class Bernoulli[T <: Tuple: Labels](val probs: Tensor[T, Prob]) extends IndependentDistribution[T, Bool]:

  override def elementWiseLogProb(x: Tensor[T, Bool]): Tensor[T, LogProb] =
    liftPyTensor(jstats.bernoulli.logpmf(x.jaxValue, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Bool] =
    liftPyTensor(Jax.jrandom.bernoulli(key.jaxKey, p = probs.jaxValue))

object Bernoulli:

  /** Create a Bernoulli distribution from probability tensor */
  def apply[T <: Tuple: Labels](probs: Tensor[T, Prob]): Bernoulli[T] =
    new Bernoulli(probs)

/** Binomial distribution - number of successes in n independent Bernoulli trials */
class Binomial[T <: Tuple: Labels, V: IsInteger](val n: Tensor0[V], val probs: Tensor[T, Prob]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.binom.logpmf(x.jaxValue, n = n.jaxValue, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    liftPyTensor(probs.shape, VType[V])(
      Jax.jrandom.binomial(key.jaxKey, n = n.jaxValue, p = probs.jaxValue)
    )

object Binomial:

  /** Create a Binomial distribution from number of trials and probability tensor */
  def apply[T <: Tuple: Labels, V: IsInteger](n: Tensor0[V], probs: Tensor[T, Prob]): Binomial[T, V] =
    new Binomial(n, probs)

/** Cauchy distribution */
class Cauchy[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy)) * scale + loc

object Cauchy:

  /** Create a Cauchy distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): Cauchy[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Location and scale must have the same dimensions")
    new Cauchy(loc, scale)

/** Half-normal distribution */
class HalfNormal[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    // Half-normal logpdf = log(2) + norm.logpdf for x >= loc, -inf otherwise
    val rawLogProb = liftPyTensor(x.shape, VType[LogProb])(
      Jax.jnp.log(2.0) + jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )
    val valid = x >= loc
    val negInf = LogProb(Tensor.like(x).fill(Float.NegativeInfinity).asFloat32)
    where(valid, rawLogProb, negInf)

  override def sample(k: Random.Key): Tensor[T, V] =
    // Half-normal: |N(0,1)| * scale + loc
    val normal = liftPyTensor(loc.shape, VType[V])(Jax.jrandom.normal(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy))
    normal.abs * scale + loc

object HalfNormal:

  /** Create a half-normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): HalfNormal[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Mean and scale must have the same dimensions")
    new HalfNormal(loc, scale)

/** Student's t-distribution */
class StudentT[T <: Tuple: Labels, V: IsFloating](val df: Tensor0[V], val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.t.logpdf(x.jaxValue, df = df.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.t(k.jaxKey, df = df.jaxValue, shape = loc.shape.dimensions.toPythonProxy)
    ) * scale + loc

object StudentT:

  /** Create a Student's t-distribution from parameters */
  def apply[T <: Tuple: Labels, V: IsFloating](df: Tensor0[V], loc: Tensor[T, V], scale: Tensor[T, V]): StudentT[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc, and scale must have the same dimensions")
    new StudentT(df, loc, scale)

/** Beta distribution */
class Beta[T <: Tuple: Labels, V: IsFloating](val alpha: Tensor[T, V], val beta: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    liftPyTensor(jstats.beta.logpdf(x.jaxValue, a = alpha.jaxValue, b = beta.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.beta(k.jaxKey, a = alpha.jaxValue, b = beta.jaxValue, shape = alpha.shape.dimensions.toPythonProxy)
    )

object Beta:

  /** Create a Beta distribution from alpha and beta tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](alpha: Tensor[T, V], beta: Tensor[T, V]): Beta[T, V] =
    require(alpha.shape.dimensions == beta.shape.dimensions, "alpha and beta must have the same dimensions")
    new Beta(alpha, beta)
