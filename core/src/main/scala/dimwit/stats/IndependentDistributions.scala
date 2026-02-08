package dimwit.stats

import dimwit.*
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import dimwit.random.Random

/** Normal (Gaussian) distribution */
class Normal[T <: Tuple: Labels](val loc: Tensor[T, Float], val scale: Tensor[T, Float]) extends IndependentDistribution[T, Float]:

  override def elementWiseLogProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    val standardNormal = Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.normal(key.jaxKey, loc.shape.dimensions.toPythonProxy))
    standardNormal * scale + loc

object Normal:

  /** Create a Normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels](loc: Tensor[T, Float], scale: Tensor[T, Float]): Normal[T] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc and scale must have the same dimensions")
    new Normal(loc, scale)

  /** Sample from standard normal distribution N(0, 1) */
  def standardSample(key: Random.Key): Tensor0[Float] = new Normal(Tensor0(0f), Tensor0(1f)).sample(key)

  def standardNormal[T <: Tuple: Labels](shape: Shape[T])(using executionType: ExecutionType[Float]): Normal[T] =
    Normal(
      loc = Tensor(shape).fill(0f),
      scale = Tensor(shape).fill(1f)
    )

/** Uniform distribution */
class Uniform[T <: Tuple: Labels](val low: Tensor[T, Float], val high: Tensor[T, Float]) extends IndependentDistribution[T, Float]:

  override def elementWiseLogProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.uniform.logpdf(x.jaxValue, loc = low.jaxValue, scale = (high - low).jaxValue))

  override def sample(key: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.uniform(key.jaxKey, shape = low.shape.dimensions.toPythonProxy, minval = low.jaxValue, maxval = high.jaxValue)
    )

object Uniform:

  /** Create a Uniform distribution from low and high tensors */
  def apply[T <: Tuple: Labels](low: Tensor[T, Float], high: Tensor[T, Float]): Uniform[T] =
    require(low.shape.dimensions == high.shape.dimensions, "Low and high must have the same dimensions")
    new Uniform(low, high)

/** Bernoulli distribution */
class Bernoulli[T <: Tuple: Labels](val probs: Tensor[T, Prob]) extends IndependentDistribution[T, Int]:

  override def elementWiseLogProb(x: Tensor[T, Int]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.bernoulli.logpmf(x.jaxValue, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.bernoulli(key.jaxKey, p = probs.jaxValue))

object Bernoulli:

  /** Create a Bernoulli distribution from probability tensor */
  def apply[T <: Tuple: Labels](probs: Tensor[T, Prob]): Bernoulli[T] =
    new Bernoulli(probs)

/** Binomial distribution - number of successes in n independent Bernoulli trials */
class Binomial[T <: Tuple: Labels](val n: Int, val probs: Tensor[T, Prob]) extends IndependentDistribution[T, Int]:

  override def elementWiseLogProb(x: Tensor[T, Int]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.binom.logpmf(x.jaxValue, n = n, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor[T, Int] =
    // Sum n independent Bernoulli(p) trials
    trait Trials derives Label
    val trialSamples = key.splitvmap(Axis[Trials] -> n) { k =>
      Tensor.fromPy(VType[Int])(Jax.jrandom.bernoulli(k.jaxKey, p = probs.jaxValue))
    }
    trialSamples.sum(Axis[Trials])

object Binomial:

  /** Create a Binomial distribution from number of trials and probability tensor */
  def apply[T <: Tuple: Labels](n: Int, probs: Tensor[T, Prob]): Binomial[T] =
    require(n > 0, "Number of trials must be positive")
    new Binomial(n, probs)

/** Cauchy distribution */
class Cauchy[T <: Tuple: Labels](val loc: Tensor[T, Float], val scale: Tensor[T, Float]) extends IndependentDistribution[T, Float]:

  override def elementWiseLogProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy)) * scale + loc

object Cauchy:

  /** Create a Cauchy distribution from location and scale tensors */
  def apply[T <: Tuple: Labels](loc: Tensor[T, Float], scale: Tensor[T, Float]): Cauchy[T] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Location and scale must have the same dimensions")
    new Cauchy(loc, scale)

/** Half-normal distribution */
class HalfNormal[T <: Tuple: Labels](val loc: Tensor[T, Float], val scale: Tensor[T, Float]) extends IndependentDistribution[T, Float]:

  override def elementWiseLogProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    // Half-normal logpdf = log(2) + norm.logpdf for x >= loc, -inf otherwise
    val rawLogProb = Tensor.fromPy[T, LogProb](VType[LogProb])(
      Jax.jnp.log(2.0) + jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
    )
    val valid = x >= loc
    val negInf = LogProb(Tensor.like(x).fill(Float.NegativeInfinity))
    where(valid, rawLogProb, negInf)

  override def sample(k: Random.Key): Tensor[T, Float] =
    // Half-normal: |N(0,1)| * scale + loc
    val normal = Tensor.fromPy[T, Float](VType[Float])(Jax.jrandom.normal(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy))
    normal.abs * scale + loc

object HalfNormal:

  /** Create a half-normal distribution from location and scale tensors */
  def apply[T <: Tuple: Labels](loc: Tensor[T, Float], scale: Tensor[T, Float]): HalfNormal[T] =
    require(loc.shape.dimensions == scale.shape.dimensions, "Mean and scale must have the same dimensions")
    new HalfNormal(loc, scale)

/** Student's t-distribution */
class StudentT[T <: Tuple: Labels](val df: Float, val loc: Tensor[T, Float], val scale: Tensor[T, Float]) extends IndependentDistribution[T, Float]:

  override def elementWiseLogProb(x: Tensor[T, Float]): Tensor[T, LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.t.logpdf(x.jaxValue, df = df, loc = loc.jaxValue, scale = scale.jaxValue))

  override def sample(k: Random.Key): Tensor[T, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.t(k.jaxKey, df = df, shape = loc.shape.dimensions.toPythonProxy)
    ) * scale + loc

object StudentT:

  /** Create a Student's t-distribution from parameters */
  def apply[T <: Tuple: Labels](df: Float, loc: Tensor[T, Float], scale: Tensor[T, Float]): StudentT[T] =
    require(loc.shape.dimensions == scale.shape.dimensions, "loc, and scale must have the same dimensions")
    new StudentT(df, loc, scale)
