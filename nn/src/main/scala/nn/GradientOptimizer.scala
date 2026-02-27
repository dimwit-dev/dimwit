package nn

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.{Grad, TensorTree}
import dimwit.jax.Jax
import dimwit.jax.Jit

/** Gradient optimizer interface with functional state management.
  *
  * This API provides the following two styles of usage:
  *
  * 1. **Simple iterator API**
  *    {{{
  *    val optimizer = GradientDescent(lr = 0.1)
  *    optimizer.iterate(initParams)(gradientFunction).take(1000).foreach(...)
  *    }}}
  *
  * 2. **Functional state threading with foldLeft** for minibatch training
  *    {{{
  *    val optimizer = GradientDescent(lr = 0.1)
  *    val (finalParams, finalState) = batches.foldLeft((initParams, optimizer.init(initParams))):
  *      case ((params, state), batch) =>
  *        val grads = Autodiff.grad(loss(batch))(params)
  *        optimizer.update(grads, params, state)
  *    }}}
  */
trait GradientOptimizer:
  type State[_]

  // Core API
  def init[Params: ToFloatTensorTree](params: Params): State[Params]
  def update[Params: ToFloatTensorTree](gradients: Grad[Params], params: Params, state: State[Params]): (Params, State[Params])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[Params: ToFloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[(Params, State[Params])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[Params: ToFloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[Params] =
    iterateWithState(init)(df).map(_._1)

case class GradientDescent(learningRate: Tensor0[Float]) extends GradientOptimizer:
  import dimwit.Conversions.given

  type State[P] = Unit // Stateless optimizer

  def init[Params: ToFloatTensorTree](params: Params): Unit = ()

  def update[Params: ToFloatTensorTree](gradients: Grad[Params], params: Params, state: Unit): (Params, Unit) =
    val p = TensorTree.from(params)
    val g = TensorTree.from(gradients.value)
    val newParams = (p -- g.scale(learningRate)).toScala
    (newParams, ())

case class Lion(learningRate: Tensor0[Float], weightDecay: Tensor0[Float] = Tensor0(0.0f), beta1: Tensor0[Float] = Tensor0(0.9f), beta2: Tensor0[Float] = Tensor0(0.99f)) extends GradientOptimizer:

  type State[P] = P // momentum state has same structure as params

  def init[Params: ToFloatTensorTree](params: Params): Params =
    TensorTree.from(params).fillZeros.toScala

  def update[Params: ToFloatTensorTree](gradients: Grad[Params], params: Params, momentums: Params): (Params, Params) =
    val p = TensorTree.from(params)
    val g = TensorTree.from(gradients.value)
    val m = TensorTree.from(momentums)

    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = (m **! beta1 ++ g **! (1f - beta1)).sign

    val updatedParams = (p -- updateDirection.scale(learningRate) -- p.scale(weightDecay)).toScala
    val newMomentums = (m **! beta2 ++ g **! (1f - beta2)).toScala

    (updatedParams, newMomentums)

case class AdamState[P](
    momentums: TensorTree[P], // momentums
    velocities: TensorTree[P], // velocities
    b1: Tensor0[Float], // decay rate for momentums mᵗ
    b2: Tensor0[Float] // decay rate for velocities vᵗ
)

/** Implements the Adam optimization algorithm.
  *
  * @see [[https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization]]
  */
case class Adam(
    learningRate: Tensor0[Float], // step size (learning rate)
    b1: Tensor0[Float] = Tensor0(0.9f), // decay rate for momentums mᵗ
    b2: Tensor0[Float] = Tensor0(0.999f), // decay rate for velocities vᵗ
    epsilon: Tensor0[Float] = Tensor0(1e-8f) // small constant to prevent division by zero
) extends GradientOptimizer:

  private val β1 = b1
  private val β2 = b2

  type State[P] = AdamState[P]

  def init[Params: ToFloatTensorTree](params: Params): State[Params] =
    val zeros = TensorTree.from(params).fillZeros
    AdamState(zeros, zeros, b1 = Tensor0(1f), b2 = Tensor0(1f))

  def update[Params: ToFloatTensorTree](
      gradients: Grad[Params],
      params: Params,
      state: State[Params]
  ): (Params, State[Params]) =
    // rename state variables to last time step for clarity
    val `mₜ₋₁` = state.momentums
    val `vₜ₋₁` = state.velocities
    val `β1ₜ₋₁` = state.b1
    val `β2ₜ₋₁` = state.b2

    // rename parameters for internal clarity
    val α = learningRate
    val ε = epsilon
    val `θₜ₋₁` = TensorTree.from(params)

    // update moments for bias correction
    val β1ₜ = `β1ₜ₋₁` * β1
    val β2ₜ = `β2ₜ₋₁` * β2

    // Adam implementation
    val gₜ = TensorTree.from(gradients.value)
    val mᵗ = `mₜ₋₁` **! `β1` ++ gₜ **! (1f - `β1`)
    val vᵗ = `vₜ₋₁` **! `β2` ++ (gₜ ** gₜ) **! (1f - `β2`)
    val m̂ = mᵗ `//!` (1f - `β1ₜ`)
    val v̂ = vᵗ `//!` (1f - `β2ₜ`)
    val θₜ = (`θₜ₋₁` -- (m̂ **! α) `//` (v̂.sqrt ++! ε)).toScala

    (θₜ, AdamState(mᵗ, vᵗ, β1ₜ, β2ₜ))

/** Implements the AdamW algorithm (Adam with decoupled weight decay).
  *
  * This implementation follows the logic described in "Decoupled Weight Decay Regularization"
  * where weight decay is performed directly on parameters rather than added to gradients.
  *
  * @see [[https://arxiv.org/abs/1711.05101 Decoupled Weight Decay Regularization]]
  *
  * @param learningRate The step size.
  * @param weightDecayFactor The coefficient for weight decay (lambda).
  */
case class AdamW(
    val adam: Adam,
    val weightDecayFactor: Tensor0[Float]
) extends GradientOptimizer:

  type State[P] = adam.State[P]

  def init[Params: ToFloatTensorTree](params: Params): State[Params] = adam.init(params)

  def update[Params: ToFloatTensorTree](
      gradients: Grad[Params],
      params: Params,
      state: State[Params]
  ): (Params, State[Params]) =
    val α = adam.learningRate
    val `θₜ₋₁` = TensorTree.from(params)
    val `λ'` = weightDecayFactor
    val λ = `λ'` * α // Tie weight decay to learning rate
    val decayedParams = (`θₜ₋₁` -- `θₜ₋₁` **! λ).toScala
    val (θₜ, adamState) = adam.update(gradients, decayedParams, state)
    (θₜ, adamState)
