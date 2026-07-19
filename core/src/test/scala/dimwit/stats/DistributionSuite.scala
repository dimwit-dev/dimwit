package dimwit.stats

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.random.Random
import dimwit.jax.Jax.scipy_stats as jstats

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import dimwit.python.PyBridge.{liftPyTensor0, liftPyTensor1}

class DistributionSuite extends DimwitTest:

  trait A derives Label
  trait Samples derives Label

  describe("Normal Distribution"):
    it("logProbs matches JAX"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = Normal(loc, scale)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
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

  describe("Uniform Distribution"):
    it("logProbs matches JAX"):
      val low = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, -1.0f, 2.0f))
      val high = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 1.0f, 5.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 0.0f, 3.0f))

      val dist = Uniform(low, high)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
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
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(false, true, true))

      val dist = Bernoulli(Prob(probs))
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.bernoulli.logpmf(x.jaxValue, p = probs.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates probabilities"):
      val bernoulli = Bernoulli(
        Prob(Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.3f, 0.7f)))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 1000)(k => bernoulli.sample(k))
      val sampleMeans = samples.asFloat32.mean(Axis[Samples])
      val expectedMeans = bernoulli.probs.asFloat
      sampleMeans should approxEqual(expectedMeans, 0.1f)

  describe("Binomial"):
    it("logProbs matches JAX"):
      val n = Tensor0(10)
      val probs = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.3f, 0.5f, 0.8f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(3, 5, 8))

      val dist = Binomial(n, Prob(probs))
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.binom.logpmf(x.jaxValue, n = n.jaxValue, p = probs.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates n*p"):
      val n = Tensor0(20)
      val binomial = Binomial(
        n,
        Prob(Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.3f, 0.7f)))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => binomial.sample(k))
      val sampleMeans = samples.asFloat32.mean(Axis[Samples])
      val expectedMeans = binomial.probs.asFloat *! n.item.toFloat
      sampleMeans should approxEqual(expectedMeans, 0.5f)

    it("reduces to Bernoulli when n=1"):
      val n = Tensor0(1)
      val probs = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.2f, 0.5f, 0.9f))

      val binomial = Binomial(n, Prob(probs))
      val key = Random.Key(123)
      val samples = key.splitvmap(Axis[Samples] -> 5000)(k => binomial.sample(k))
      val sampleMeans = samples.asFloat32.mean(Axis[Samples])
      sampleMeans should approxEqual(probs, 0.1f)

    it("handles edge cases p=0 and p=1"):
      val n = Tensor0(5)
      val probsEdge = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 1.0f))

      val binomial = Binomial(n, Prob(probsEdge))
      val key = Random.Key(456)
      val samples = key.splitvmap(Axis[Samples] -> 100)(k => binomial.sample(k))
      val sampleMeans = samples.asFloat32.mean(Axis[Samples])
      val expectedMeans = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, n.item.toFloat))
      sampleMeans should approxEqual(expectedMeans, 0.1f)

  describe("Cauchy"):
    it("logProbs matches JAX"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = Cauchy(loc, scale)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
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

  describe("HalfCauchy"):
    it("logProbs matches scipy"):
      val scipy_stats = py.module("scipy.stats")
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 0.0f, 0.0f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.0f, 0.8f))

      val dist = HalfCauchy(loc, scale)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val scipyLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        scipy_stats.halfcauchy.logpdf(x.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(scipyLogProbs)

    it("logProb is -inf for x < loc"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 1.0f, 1.0f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 1.0f, 1.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 0.5f, 0.9f))

      val dist = HalfCauchy(loc, scale)
      val logProbs = dist.elementWiseLogProb(x)
      logProbs.asFloat.toArray.foreach(v => v should be(Float.NegativeInfinity))

    it("sample medians approximates location"):
      val halfCauchy = HalfCauchy(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.0f, 0.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1.0f, 2.0f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 50000)(k => halfCauchy.sample(k))
      // All samples should be >= loc
      val minSample = samples.min(Axis[Samples])
      minSample.toArray.zip(halfCauchy.loc.toArray).foreach { (s, l) =>
        s should be >= l
      }
      // Median of HalfCauchy(0, scale) is scale
      val sampleMedian = samples.median(Axis[Samples])
      sampleMedian should approxEqual(halfCauchy.scale, 0.2f)

  describe("HalfNormal"):
    it("logProbs computed correctly"):
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 0.0f, 0.0f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.0f, 0.8f))

      val dist = HalfNormal(loc, scale)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      // Compute expected manually: log(2) + norm.logpdf for x >= loc
      val expectedLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
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
      val df = Tensor0(5.0f)
      val loc = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.0f, 1.0f, -0.5f))
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1.0f, 0.5f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.5f, -1.0f))

      val dist = StudentT(df, loc, scale)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.t.logpdf(x.jaxValue, df = df.jaxValue, loc = loc.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates location"):
      val studentT = StudentT(
        df = Tensor0(5.0f),
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
      val jaxLogProb = liftPyTensor0(VType[Float32])(
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
      val jaxLogProb = liftPyTensor0(VType[Float32])(
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
      val n = Tensor0(10)

      val dist = Multinomial[A](n, probs)
      val scalaLogProb = dist.logProb(x)
      val jaxLogProb = liftPyTensor0(VType[Float32])(
        jstats.multinomial.logpmf(x.jaxValue, n = n.jaxValue, p = probs.jaxValue)
      )
      scalaLogProb.asFloat should approxEqual(jaxLogProb)

    it("sample mean approximates expected counts"):
      val probsFloat = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.2f, 0.5f, 0.3f))
      val probs = Prob(probsFloat)
      val n = Tensor0(100)
      val multinomial = Multinomial[A](n, probs)
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => multinomial.sample(k))
      val sampleMean = samples.asFloat32.mean(Axis[Samples])
      // Expected mean counts are n * probs
      val expectedMean = multinomial.probs.asFloat *! n.item.toFloat
      sampleMean should approxEqual(expectedMean, 2.0f)

  describe("Categorical"):
    it("logProb matches expected value"):
      val probs = Prob(Tensor(Shape(Axis[A] -> 4)).fromArray(Array(0.1f, 0.2f, 0.3f, 0.4f)))
      val x = Tensor0(2)

      val dist = Categorical(probs)
      val scalaLogProb = dist.logProb(x)
      val expectedLogProb = Tensor0(math.log(0.3f).toFloat)
      scalaLogProb.asFloat should approxEqual(expectedLogProb)

    it("sample distribution matches probabilities"):
      val probs = Prob(Tensor(Shape(Axis[A] -> 4)).fromArray(Array(0.1f, 0.2f, 0.3f, 0.4f)))
      val categorical = Categorical(probs)
      val key = Random.Key(42)
      val numSamples = 10000
      val samples = key.splitvmap(Axis[Samples] -> numSamples)(k => categorical.sample(k))
      val counts = liftPyTensor1(Axis[A], VType[Float32])(
        Jax.jnp.bincount(samples.jaxValue, minlength = 4).astype(Jax.jnp.float32)
      )
      val frequencies = counts *! (1.0f / numSamples.toFloat)
      frequencies should approxEqual(probs.asFloat32, 0.02f)

  describe("Beta"):
    it("logProbs matches JAX"):
      val alpha = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 2.0f, 5.0f))
      val beta = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 2.0f, 1.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.3f, 0.5f, 0.8f))

      val dist = Beta(alpha, beta)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.beta.logpdf(x.jaxValue, a = alpha.jaxValue, b = beta.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates expected means"):
      val betaDist = Beta(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(2.0f, 5.0f)),
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(5.0f, 2.0f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => betaDist.sample(k))
      val sampleMeans = samples.mean(Axis[Samples])
      // Mean of Beta distribution is alpha / (alpha + beta)
      val expectedMeans = betaDist.alpha / (betaDist.alpha + betaDist.beta)
      sampleMeans should approxEqual(expectedMeans, 0.02f)

  describe("Exponential"):
    it("logProbs matches JAX"):
      val rate = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.0f, 2.0f))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(0.5f, 1.0f, 0.3f))

      val dist = Exponential(rate)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val scale = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(2.0f, 1.0f, 0.5f))
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.expon.logpdf(x.jaxValue, scale = scale.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates 1/rate"):
      val exponential = Exponential(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(0.5f, 2.0f))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => exponential.sample(k))
      val sampleMeans = samples.mean(Axis[Samples])
      // Mean of Exponential(rate) is 1/rate
      val expectedMeans = Tensor(Shape(Axis[A] -> 2)).fromArray(Array(2.0f, 0.5f))
      sampleMeans should approxEqual(expectedMeans, 0.1f)

  describe("Poisson"):
    it("logProbs matches JAX"):
      val rate = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1, 3, 10))
      val x = Tensor(Shape(Axis[A] -> 3)).fromArray(Array(1, 2, 8))

      val dist = Poisson(rate)
      val scalaLogProbs = dist.elementWiseLogProb(x)
      val jaxLogProbs = liftPyTensor1(Axis[A], VType[Float32])(
        jstats.poisson.logpmf(x.jaxValue, mu = rate.jaxValue)
      )
      scalaLogProbs.asFloat should approxEqual(jaxLogProbs)

    it("sample means approximates rate"):
      val poisson = Poisson(
        Tensor(Shape(Axis[A] -> 2)).fromArray(Array(1, 5))
      )
      val key = Random.Key(42)
      val samples = key.splitvmap(Axis[Samples] -> 10000)(k => poisson.sample(k))
      val sampleMeans = samples.asFloat32.mean(Axis[Samples])
      // Mean of Poisson(lambda) is lambda
      val expectedMeans = poisson.rate.asFloat32
      sampleMeans should approxEqual(expectedMeans, 0.2f)
