package dimwit.stats

import dimwit.DType.Float32
import dimwit.*
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.python.PyBridge.liftPyTensor
import dimwit.random.Random
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Normal (Gaussian) distribution */
class Normal[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    val standardNormal = liftPyTensor(loc.shape, loc.vtype)(Jax.jrandom.normal(key.jaxKey, loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType))
    standardNormal * scale + loc

object Normal:

  /** Create a Normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): Normal[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc and scale must have the same dimensions")
    new Normal(loc, scale)

  def isotropic[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor0[V]): Normal[T, V] = new Normal(loc = loc, scale = scale.broadcastTo(loc.shape))
  def standardIsotropic[T <: Tuple: Labels, V: IsFloating](shape: Shape[T], scale: Tensor0[V]): Normal[T, V] = isotropic(loc = Tensor(shape, VType[V]).fill(0f), scale = scale)
  def standardIsotropic[T <: Tuple: Labels](shape: Shape[T], scale: Float): Normal[T, Float32] = isotropic(loc = Tensor(shape).fill(0f), scale = Tensor0(scale))

  /** Sample from standard normal distribution N(0, 1) */
  def standardSample(key: Random.Key): Tensor0[Float32] = new Normal(Tensor0(0f), Tensor0(1f)).sample(key)
  def standardNormal[T <: Tuple: Labels](shape: Shape[T]): Normal[T, Float32] = Normal.standardIsotropic(shape, scale = Tensor0(VType[Float32])(1f))

/** Truncated Normal (Gaussian) distribution */
class TruncatedNormal[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V], val low: Tensor[T, V], val high: Tensor[T, V]) extends IndependentDistribution[T, V]:

  private lazy val a = (low - loc) / scale
  private lazy val b = (high - loc) / scale

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.truncnorm.logpdf(x.jaxValue, a = a.jaxValue, b = b.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    val stdTruncated = liftPyTensor(loc.shape, VType[V])(
      Jax.jrandom.truncated_normal(key.jaxKey, lower = a.jaxValue, upper = b.jaxValue, shape = loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)
    )
    stdTruncated * scale + loc

object TruncatedNormal:

  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V], low: Tensor[T, V], high: Tensor[T, V]): TruncatedNormal[T, V] =
    new TruncatedNormal(loc, scale, low, high)

  def symmetric[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V], numStd: Float = 2.0f): TruncatedNormal[T, V] =
    val boundOffset = scale *! Tensor0(VType[V])(numStd)
    val low = loc - boundOffset
    val high = loc + boundOffset
    new TruncatedNormal(loc, scale, low, high)

/** Uniform distribution */
class Uniform[T <: Tuple: Labels, V: IsFloating](val low: Tensor[T, V], val high: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.uniform.logpdf(x.jaxValue, loc = low.jaxValue, scale = (high - low).jaxValue))

  override def sample(key: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.uniform(key.jaxKey, shape = low.shape.dimensions.toPythonProxy, minval = low.jaxValue, maxval = high.jaxValue, dtype = VType[V].dtype.jaxType)
    )

/** Discrete uniform distribution over the integers in the half-open interval `[min, max)`,
  *  i.e. min is inclusive, max exclusive.
  */
class DiscreteUniform[T <: Tuple: Labels](val min: Tensor[T, Int32], val max: Tensor[T, Int32]) extends IndependentDistribution[T, Int32]:

  override def elementWiseLogProb(x: Tensor[T, Int32]): Tensor[T, LogProb] =
    val logProb = LogProb(-(max - min).asFloat32.log)
    val inSupport = (x >= min) and (x < max)
    val negInf = LogProb(Tensor(x.shape).fill(Float.NegativeInfinity))
    where(inSupport, logProb, negInf)

  override def sample(key: Random.Key): Tensor[T, Int32] =
    liftPyTensor(
      Jax.jrandom.randint(key.jaxKey, shape = min.shape.dimensions.toPythonProxy, minval = min.jaxValue, maxval = max.jaxValue)
    )

object Uniform:

  /** Create a Uniform distribution from low and high tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](low: Tensor[T, V], high: Tensor[T, V]): Uniform[T, V] =
    require(low.shape.dimensions == high.shape.dimensions, "Low and high must have the same dimensions")
    new Uniform(low, high)

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
      // jax.random.binomial returns floating point counts
      Jax.jnp.astype(Jax.jrandom.binomial(key.jaxKey, n = n.jaxValue, p = probs.jaxValue), VType[V].dtype.jaxType)
    )

object Binomial:

  /** Create a Binomial distribution from number of trials and probability tensor */
  def apply[T <: Tuple: Labels, V: IsInteger](n: Tensor0[V], probs: Tensor[T, Prob]): Binomial[T, V] =
    new Binomial(n, probs)

/** Cauchy distribution */
class Cauchy[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)) * scale + loc

object Cauchy:

  /** Create a Cauchy distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): Cauchy[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Location and scale must have the same dimensions")
    new Cauchy(loc, scale)

/** Half-Cauchy distribution - absolute value of a Cauchy-distributed random variable */
class HalfCauchy[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    // Half-Cauchy logpdf = log(2) + cauchy.logpdf for x >= loc, -inf otherwise
    val rawLogProb = LogProb.fromPy[T](
      Jax.jnp.log(2.0) + jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )
    val valid = x >= loc
    val negInf = LogProb(Tensor.like(x).fill(Float.NegativeInfinity).asFloat32)
    where(valid, rawLogProb, negInf)

  override def sample(k: Random.Key): Tensor[T, V] =
    // Half-Cauchy: |Cauchy(0, 1)| * scale + loc
    val cauchy = liftPyTensor(loc.shape, VType[V])(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType))
    cauchy.abs * scale + loc

object HalfCauchy:

  /** Create a HalfCauchy distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): HalfCauchy[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Location and scale must have the same dimensions")
    new HalfCauchy(loc, scale)

/** Half-normal distribution */
class HalfNormal[T <: Tuple: Labels, V: IsFloating](val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    // Half-normal logpdf = log(2) + norm.logpdf for x >= loc, -inf otherwise
    val rawLogProb = LogProb.fromPy[T](
      Jax.jnp.log(2.0) + jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )
    val valid = x >= loc
    val negInf = LogProb(Tensor.like(x).fill(Float.NegativeInfinity).asFloat32)
    where(valid, rawLogProb, negInf)

  override def sample(k: Random.Key): Tensor[T, V] =
    // Half-normal: |N(0,1)| * scale + loc
    val normal = liftPyTensor(loc.shape, VType[V])(Jax.jrandom.normal(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType))
    normal.abs * scale + loc

object HalfNormal:

  /** Create a half-normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](loc: Tensor[T, V], scale: Tensor[T, V]): HalfNormal[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Mean and scale must have the same dimensions")
    new HalfNormal(loc, scale)

/** Student's t-distribution */
class StudentT[T <: Tuple: Labels, V: IsFloating](val df: Tensor0[V], val loc: Tensor[T, V], val scale: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.t.logpdf(x.jaxValue, df = df.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.t(k.jaxKey, df = df.jaxValue, shape = loc.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)
    ) * scale + loc

object StudentT:

  /** Create a Student's t-distribution from parameters */
  def apply[T <: Tuple: Labels, V: IsFloating](df: Tensor0[V], loc: Tensor[T, V], scale: Tensor[T, V]): StudentT[T, V] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc, and scale must have the same dimensions")
    new StudentT(df, loc, scale)

/** Beta distribution */
class Beta[T <: Tuple: Labels, V: IsFloating](val alpha: Tensor[T, V], val beta: Tensor[T, V]) extends IndependentDistribution[T, V]:

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.beta.logpdf(x.jaxValue, a = alpha.jaxValue, b = beta.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    liftPyTensor(
      Jax.jrandom.beta(k.jaxKey, a = alpha.jaxValue, b = beta.jaxValue, shape = alpha.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)
    )

object Beta:

  /** Create a Beta distribution from alpha and beta tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](alpha: Tensor[T, V], beta: Tensor[T, V]): Beta[T, V] =
    require(alpha.shape.dimensions == beta.shape.dimensions, "alpha and beta must have the same dimensions")
    new Beta(alpha, beta)

/** Exponential distribution */
class Exponential[T <: Tuple: Labels, V: IsFloating](val rate: Tensor[T, V]) extends IndependentDistribution[T, V]:

  lazy val scale = (Tensor0(1f).asType(VType[V]) /! rate).jaxValue

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.expon.logpdf(x.jaxValue, scale = scale))

  override def sample(k: Random.Key): Tensor[T, V] =
    // samples from jrandom.exponential are always with scale=1,
    // so we need to divide by the rate to get the correct distribution
    val stndExp = liftPyTensor[T, V](
      Jax.jrandom.exponential(k.jaxKey, shape = rate.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)
    )
    stndExp / rate

/** Gamma distribution with `concentration` (the shape parameter, often written alpha or k) and `rate` (often written beta).
  *
  * The parameter is called concentration rather than shape, as in Dirichlet, to avoid a clash with a tensor's `shape`.
  * Mean is concentration / rate, variance concentration / rate^2. Gamma(1, rate) is Exponential(rate).
  */
class Gamma[T <: Tuple: Labels, V: IsFloating](val concentration: Tensor[T, V], val rate: Tensor[T, V]) extends IndependentDistribution[T, V]:

  private lazy val scale = Tensor.like(rate).fill(1f) / rate

  override def elementWiseLogProb(x: Tensor[T, V]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.gamma.logpdf(x.jaxValue, a = concentration.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, V] =
    // jax.random.gamma samples with rate 1, so divide by the rate
    val standardGamma = liftPyTensor(concentration.shape, VType[V])(
      Jax.jrandom.gamma(k.jaxKey, a = concentration.jaxValue, shape = concentration.shape.dimensions.toPythonProxy, dtype = VType[V].dtype.jaxType)
    )
    standardGamma / rate

object Gamma:

  /** Create a Gamma distribution from concentration and rate tensors */
  def apply[T <: Tuple: Labels, V: IsFloating](concentration: Tensor[T, V], rate: Tensor[T, V]): Gamma[T, V] =
    require(concentration.shape.dimensions == rate.shape.dimensions, "concentration and rate must have the same dimensions")
    new Gamma(concentration, rate)

  /** Create a Gamma distribution from concentration and scale tensors, where scale = 1 / rate */
  def fromScale[T <: Tuple: Labels, V: IsFloating](concentration: Tensor[T, V], scale: Tensor[T, V]): Gamma[T, V] =
    require(concentration.shape.dimensions == scale.shape.dimensions, "concentration and scale must have the same dimensions")
    new Gamma(concentration, Tensor.like(scale).fill(1f) / scale)

class Poisson[T <: Tuple: Labels, VSample: IsInteger, VRate: IsFloating](val rate: Tensor[T, VRate]) extends IndependentDistribution[T, VSample]:

  override def elementWiseLogProb(x: Tensor[T, VSample]): Tensor[T, LogProb] =
    LogProb.fromPy(jstats.poisson.logpmf(x.jaxValue, mu = rate.jaxValue))

  override def sample(k: Random.Key): Tensor[T, VSample] =
    liftPyTensor(rate.shape, VType[VSample])(
      Jax.jrandom.poisson(k.jaxKey, lam = rate.jaxValue, shape = rate.shape.dimensions.toPythonProxy, dtype = VType[VSample].dtype.jaxType)
    )

object Poisson:

  /** Create a Poisson distribution from a rate tensor, sampling as Int32 counts */
  def apply[T <: Tuple: Labels, VRate: IsFloating](rate: Tensor[T, VRate]): Poisson[T, Int32, VRate] =
    new Poisson[T, Int32, VRate](rate)

  /** Create a Poisson distribution from a rate tensor, sampling counts of the given integer type */
  def apply[T <: Tuple: Labels, VSample: IsInteger, VRate: IsFloating](rate: Tensor[T, VRate], vtype: VType[VSample]): Poisson[T, VSample, VRate] =
    new Poisson[T, VSample, VRate](rate)
