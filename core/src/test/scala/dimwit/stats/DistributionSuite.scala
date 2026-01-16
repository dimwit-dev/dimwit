package dimwit.stats

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.random.Random
import dimwit.jax.Jax.scipy_stats as jstats

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class DistributionSuite extends AnyFunSpec with Matchers:

  trait A derives Label
  trait Samples derives Label

  describe("Normal Distribution"):
    it("logProbs matches JAX"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = Normal(loc, scale)
      val scalaLogProbs = dist.logProb(x)
      val jaxLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates means"):
      val normal = Normal(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 1.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 0.5f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => normal.sample(k))
      val sampleMeans = samples.mean(Axis[Samples])
      val expectedMeans = normal.loc
      sampleMeans should approxEqual(expectedMeans, 0.2f)

    it("more specific types"):
      trait A derives Label
      trait LocA extends A derives Label
      trait ScaleA extends A derives Label
      val loc = Tensor(Shape(Axis[LocA] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[ScaleA] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))
      val dist = Normal(loc, scale)
      val scalaLogProbs = dist.logProb(x)
      scalaLogProbs shouldBe a[Tensor1[A, LogProb]]

  describe("Uniform Distribution"):
    it("logProbs matches JAX"):
      val low = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, -1.0f, 2.0f))
      val high = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 1.0f, 5.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 0.0f, 3.0f))

      val dist = Uniform(low, high)
      val scalaLogProbs = dist.logProb(x)
      val jaxLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.uniform.logpdf(x.jaxValue, loc = low.jaxValue, scale = (high - low).jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates means"):
      val uniform = Uniform(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(-1.0f, 0.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 2.0f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => uniform.sample(k))
      val sampleMeans = samples.mean(Axis[Samples])
      val expectedMeans = (uniform.low + uniform.high) *! 0.5f
      sampleMeans should approxEqual(expectedMeans, 0.2f)

  describe("Bernoulli"):
    it("logProbs matches JAX"):
      val probs = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.3f, 0.5f, 0.8f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0, 1, 1))

      val dist = Bernoulli(probs)
      val scalaLogProbs = dist.logProb(x)
      val jaxLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.bernoulli.logpmf(x.jaxValue, p = probs.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates probabilities"):
      val bernoulli = Bernoulli(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.3f, 0.7f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 1000)(k => bernoulli.sample(k))
      val sampleMeans = samples.asFloat.mean(Axis[Samples])
      val expectedMeans = bernoulli.probs
      sampleMeans should approxEqual(expectedMeans, 0.1f)

  describe("Cauchy"):
    it("logProbs matches JAX"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = Cauchy(loc, scale)
      val scalaLogProbs = dist.logProb(x)
      val jaxLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.cauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample medians approximates location"):
      val cauchy = Cauchy(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 2.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 0.5f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 50000)(k => cauchy.sample(k))
      val sampleMedian = samples.median(Axis[Samples])
      val expectedMedian = cauchy.loc
      sampleMedian should approxEqual(expectedMedian, 0.5f)

  describe("HalfNormal"):
    it("logProbs computed correctly"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 0.0f, 0.0f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.0f, 0.8f))

      val dist = HalfNormal(loc, scale)
      val scalaLogProbs = dist.logProb(x)
      // Compute expected manually: log(2) + norm.logpdf for x >= loc
      val expectedLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.norm.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
      ) +! math.log(2.0).toFloat
      scalaLogProbs.asFloat should approxEqual(expectedLogProbs)

    it("sample means approximates expected means"):
      val halfNormal = HalfNormal(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 0.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 2.0f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => halfNormal.sample(k))
      val sampleMeans = samples.mean(Axis[Samples])
      // Mean of half-normal is scale * sqrt(2/pi) + loc
      val sqrtTwoOverPi = math.sqrt(2.0 / math.Pi).toFloat
      val expectedMeans = halfNormal.scale *! sqrtTwoOverPi + halfNormal.loc
      sampleMeans should approxEqual(expectedMeans, 0.2f)

  describe("StudentT"):
    it("logProbs matches JAX"):
      val df = 5
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = StudentT(df, loc, scale)
      val scalaLogProbs = dist.logProb(x)
      val jaxLogProbs = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        jstats.t.logpdf(x.jaxValue, df = df, loc = loc.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates location"):
      val studentT = StudentT(
        df = 5,
        loc = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 2.0f)),
        scale = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 0.5f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => studentT.sample(k))
      val sampleMean = samples.mean(Axis[Samples])
      val expectedMean = studentT.loc
      sampleMean should approxEqual(expectedMean, 0.2f)

  describe("MVNormal"):
    it("logProb matches JAX"):
      val mean = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, 2.0f))
      val cov = Tensor(Shape(Axis[A] -> 3, Axis[Prime[A]] -> 3)).fromArray(
        Array(
          1.0f, 0.5f, 0.2f,
          0.5f, 2.0f, 0.3f,
          0.2f, 0.3f, 1.5f
        )
      )
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, 2.2f))

      val dist = MVNormal(mean, cov)
      val scalaLogProb = dist.logProb(x)
      val jaxLogProb = Tensor.fromPy[EmptyTuple, Float](VType[Float])(
        jstats.multivariate_normal.logpdf(x.jaxValue, mean = mean.jaxValue, cov = cov.jaxValue)
      )
      scalaLogProb.asFloat should approxEqual(jaxLogProb)

    it("sample mean approximates mean"):
      val mean = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 2.0f))
      val cov = Tensor(Shape(Axis[A] -> 2, Axis[Prime[A]] -> 2)).fromArray(
        Array(1.0f, 0.3f, 0.3f, 1.0f)
      )
      val mvNormal = MVNormal(mean, cov)
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => mvNormal.sample(k))
      val sampleMean = samples.mean(Axis[Samples])
      val expectedMean = mvNormal.mean
      sampleMean should approxEqual(expectedMean, 0.2f)

  describe("Dirichlet"):
    it("logProb matches JAX"):
      val concentration = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(2.0f, 3.0f, 5.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.2f, 0.3f, 0.5f))

      val dist = Dirichlet(concentration)
      val scalaLogProb = dist.logProb(x)
      val jaxLogProb = Tensor.fromPy[EmptyTuple, Float](VType[Float])(
        jstats.dirichlet.logpdf(x.jaxValue, alpha = concentration.jaxValue)
      )
      scalaLogProb.asFloat should approxEqual(jaxLogProb)

    it("sample mean approximates expected mean"):
      val concentration = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(2.0f, 5.0f, 3.0f))
      val dirichlet = Dirichlet(concentration)
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => dirichlet.sample(k))
      val sampleMean = samples.mean(Axis[Samples])
      // Expected mean for Dirichlet is concentration / sum(concentration)
      // For [2.0, 5.0, 3.0], sum=10.0, so expected is [0.2, 0.5, 0.3]
      val expectedMean = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.2f, 0.5f, 0.3f))
      sampleMean should approxEqual(expectedMean, 0.2f)

  describe("Multinomial"):
    it("logProb matches JAX"):
      val probsFloat = Tensor(Shape(Axis[A] -> 4)).fromArray(Array(0.1f, 0.2f, 0.3f, 0.4f))
      val probs = Prob(probsFloat)
      val x = Tensor(Shape(Axis[A] -> 4)).fromArray(Array(2, 1, 3, 4))
      val n = 10

      val dist = Multinomial[A](n, probs)
      val scalaLogProb = dist.logProb(x)
      val jaxLogProb = Tensor.fromPy[EmptyTuple, Float](VType[Float])(
        jstats.multinomial.logpmf(x.jaxValue, n = n, p = probs.jaxValue)
      )
      scalaLogProb.asFloat should approxEqual(jaxLogProb)

    it("sample mean approximates expected counts"):
      val probsFloat = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.2f, 0.5f, 0.3f))
      val probs = Prob(probsFloat)
      val n = 100
      val multinomial = Multinomial[A](n, probs)
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => multinomial.sample(k))
      val sampleMean = samples.asFloat.mean(Axis[Samples])
      // Expected mean counts are n * probs
      val expectedMean = multinomial.probs.asFloat *! n.toFloat
      sampleMean should approxEqual(expectedMean, 2.0f)

  describe("Categorical"):
    it("logProb matches expected value"):
      val probs = Tensor(Shape(Axis[A] -> 4)).fromArray(Array(0.1f, 0.2f, 0.3f, 0.4f))
      val x = Tensor0(2)

      val dist = Categorical(probs)
      val scalaLogProb = dist.logProb(x)
      val expectedLogProb = Tensor0(math.log(0.3f).toFloat)
      scalaLogProb.asFloat should approxEqual(expectedLogProb)

    it("sample distribution matches probabilities"):
      val probs = Tensor(Shape(Axis[A] -> 4)).fromArray(Array(0.1f, 0.2f, 0.3f, 0.4f))
      val categorical = Categorical(probs)
      val key = Random.Key(42)
      val numSamples = 10000
      val samples = key.splitvmap(Axis[Samples] -> numSamples)(k => categorical.sample(k))
      val counts = Tensor.fromPy[Tuple1[A], Float](VType[Float])(
        Jax.jnp.bincount(samples.jaxValue, minlength = 4).astype(Jax.jnp.float32)
      )
      val frequencies = counts *! (1.0f / numSamples.toFloat)
      frequencies should approxEqual(probs, 0.02f)
