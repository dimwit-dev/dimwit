package dimwit.stats

import dimwit.*
import dimwit.random.Random
import dimwit.jax.Jax
import dimwit.jax.Jax.scipy_stats as jstats
import dimwit.jax.Jax.PyDynamic

/** Distribution over a vector of random variables.
  */
trait MultivariateDistribution[L: Label, V] extends Distribution[Tuple1[L], V]

class MVNormal[L: Label](
    val mean: Tensor1[L, Float],
    val covariance: Tensor2[L, Prime[L], Float]
) extends MultivariateDistribution[L, Float]:

  override def logProb(x: Tensor1[L, Float]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.multivariate_normal.logpdf(x.jaxValue, mean = mean.jaxValue, cov = covariance.jaxValue))

  override def sample(k: Random.Key): Tensor1[L, Float] =
    Tensor.fromPy(VType[Float])(
      Jax.jrandom.multivariate_normal(
        k.jaxKey,
        mean = mean.jaxValue,
        cov = covariance.jaxValue
      )
    )

class Dirichlet[L: Label](
    val concentration: Tensor1[L, Float]
) extends MultivariateDistribution[L, Float]:

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
) extends MultivariateDistribution[L, Int]:

  private val categorical: Categorical[L] = Categorical(probs)

  override def logProb(x: Tensor1[L, Int]): Tensor0[LogProb] =
    Tensor.fromPy(VType[LogProb])(jstats.multinomial.logpmf(x.jaxValue, n = n, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor1[L, Int] =
    // Sample from categorical n times using splitvmap, then bincount
    trait Draws derives Label
    val draws = key.splitvmap(Axis[Draws] -> n)(k => categorical.sample(k))
    Tensor.fromPy(VType[Int])(
      Jax.jnp.bincount(draws.jaxValue, length = probs.shape.dimensions(0))
    )
