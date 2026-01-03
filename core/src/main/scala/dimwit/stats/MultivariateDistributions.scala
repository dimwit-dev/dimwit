package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class MVNormal[L: Label](
    val mean: Tensor1[L, Float],
    val covariance: Tensor2[L, Prime[L], Float]
) extends MultivariateDistribution[Tuple1[L], Float]:

  override def logProb(x: Tensor1[L, Float]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.multivariate_normal.logpdf(x.jaxValue, mean = mean.jaxValue, cov = covariance.jaxValue))

  override def sample(k: Random.Key): Tensor[Tuple1[L], Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.multivariate_normal(
        k.jaxKey,
        mean = mean.jaxValue,
        cov = covariance.jaxValue
      )
    )

class Dirichlet[L: Label](
    val concentration: Tensor1[L, Float]
) extends MultivariateDistribution[Tuple1[L], Float]:

  override def logProb(x: Tensor1[L, Float]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.dirichlet.logpdf(x.jaxValue, alpha = concentration.jaxValue))

  override def sample(k: Random.Key): Tensor1[L, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.dirichlet(
        k.jaxKey,
        alpha = concentration.jaxValue
      )
    )

class Multinomial[L: Label](
    val n: Int,
    val probs: Tensor1[L, Prob]
) extends MultivariateDistribution[Tuple1[L], Int]:

  private lazy val logProbs: Tensor1[L, LogProb] = probs.log

  override def logProb(x: Tensor1[L, Int]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.multinomial.logpmf(x.jaxValue, n = n, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor1[L, Int] =
    // Sample from categorical n times using vmap, then bincount
    val splitKeys = Jax.jrandom.split(key.jaxKey, n)
    // Use vmap to apply categorical to all keys
    val vmapCategorical = Jax.jax.vmap(
      py.Dynamic.global.eval("lambda k, lp: __import__('jax').random.categorical(k, lp)"),
      in_axes = (0, py.None)
    )
    val samples = vmapCategorical(splitKeys, logProbs.jaxValue)
    Tensor.fromPy(VType[Int])(
      Jax.jnp.bincount(samples, length = probs.shape.dimensions(0))
    )

class Categorical[L: Label](val probs: Tensor1[L, Float]) extends MultivariateDistribution[EmptyTuple, Int]:

  private val numCategories = probs.shape.dimensions(0)
  private val logProbs = probs.log

  override def logProb(x: Tensor0[Int]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(logProbs.jaxValue.__getitem__(x.jaxValue))

  override def sample(key: Random.Key): Tensor0[Int] =
    Tensor.fromPy(VType[Int])(Jax.jrandom.categorical(key.jaxKey, logProbs.jaxValue))
