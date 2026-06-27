package dimwit.stats

import dimwit._
import dimwit.jax.Jax
import dimwit.jax.Jax.{scipy_stats => jstats}
import dimwit.python.PyBridge.liftPyTensor
import dimwit.random.Random

/** Distribution over a vector of random variables.
  */
trait MultivariateDistribution[L: Label, V] extends Distribution[Tuple1[L], V]

class MVNormal[L: Label, V: IsFloating](
    val mean: Tensor1[L, V],
    val covariance: Tensor2[L, Prime[L], V]
) extends MultivariateDistribution[L, V]:

  override def logProb(x: Tensor1[L, V]): Tensor0[LogProb] =
    liftPyTensor(jstats.multivariate_normal.logpdf(x.jaxValue, mean = mean.jaxValue, cov = covariance.jaxValue))

  override def sample(k: Random.Key): Tensor1[L, V] =
    liftPyTensor(
      Jax.jrandom.multivariate_normal(
        k.jaxKey,
        mean = mean.jaxValue,
        cov = covariance.jaxValue
      )
    )

class Dirichlet[L: Label, V: IsFloating](
    val concentration: Tensor1[L, V]
) extends MultivariateDistribution[L, V]:

  override def logProb(x: Tensor1[L, V]): Tensor0[LogProb] =
    liftPyTensor(jstats.dirichlet.logpdf(x.jaxValue, alpha = concentration.jaxValue))

  override def sample(k: Random.Key): Tensor1[L, V] =
    liftPyTensor(
      Jax.jrandom.dirichlet(
        k.jaxKey,
        alpha = concentration.jaxValue
      )
    )

class Multinomial[L: Label](
    val n: Tensor0[Int32],
    val probs: Tensor1[L, Prob]
) extends MultivariateDistribution[L, Int32]:

  private val categorical: Categorical[L] = Categorical[L](probs)

  override def logProb(x: Tensor1[L, Int32]): Tensor0[LogProb] =
    liftPyTensor(jstats.multinomial.logpmf(x.jaxValue, n = n.jaxValue, p = probs.jaxValue))

  override def sample(key: Random.Key): Tensor1[L, Int32] =
    // Sample from categorical n times using splitvmap, then bincount
    trait Draws derives Label
    val draws = key.splitvmap(Axis[Draws] -> n.item)(k => categorical.sample(k))
    liftPyTensor(
      Jax.jnp.bincount(draws.jaxValue, length = probs.shape.dimensions(0))
    )
