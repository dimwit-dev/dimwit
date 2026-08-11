package dimwit.optimizer

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.*
import dimwit.autodiff.Grad
import dimwit.tensortree.TreeOf
import dimwit.tensortree.TreeOf.ops.*

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
trait GradientOptimizer[State0[_]]:

  type State[P] = State0[P]

  // Core API
  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): State[P]

  def update[P: TensorTree, V](gradients: Grad[P], params: P, state: State[P])(using TreeOf[P, V])(using IsFloating[V]): (P, State[P])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[P: TensorTree, V](init: P)(df: P => Grad[P])(using TreeOf[P, V])(using IsFloating[V]): Iterator[(P, State[P])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[P: TensorTree, V](init: P)(df: P => Grad[P])(using TreeOf[P, V])(using IsFloating[V]): Iterator[P] =
    iterateWithState(init)(df).map(_._1)

type GradientDescentState[P] = Unit // empty state

class GradientDescent(val learningRate: Tensor0[Float32]) extends GradientOptimizer[GradientDescentState]:

  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): Unit = ()

  def update[P: TensorTree, V](gradients: Grad[P], params: P, state: Unit)(using ft: TreeOf[P, V])(using IsFloating[V]): (P, Unit) =
    val α = learningRate.asFloat(VType[V])
    val newParams = params -- gradients.value.scale(α)
    (newParams, ())

case class LionState[P](
    momentums: P,
    step: Tensor0[Int32]
)

class Lion(val learningRate: Tensor0[Float32], val weightDecay: Tensor0[Float32] = Tensor0(0.0f), val beta1: Tensor0[Float32] = Tensor0(0.9f), val beta2: Tensor0[Float32] = Tensor0(0.99f)) extends GradientOptimizer[LionState]:

  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): LionState[P] =
    LionState(params.fillCopy(0f), step = 1)

  def update[P: TensorTree, V](gradients: Grad[P], params: P, state: LionState[P])(using TreeOf[P, V])(using IsFloating[V]): (P, LionState[P]) =
    val α = learningRate.asFloat(VType[V])
    val β1 = beta1.asFloat(VType[V])
    val β2 = beta2.asFloat(VType[V])
    val λ = weightDecay.asFloat(VType[V])

    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = (state.momentums **! β1 ++ gradients.value **! (1f - β1)).sign

    val updatedParams = params -- updateDirection.scale(α) -- params.scale(λ)
    val newMomentums = state.momentums **! β2 ++ gradients.value **! (1f - β2)

    (updatedParams, LionState(newMomentums, state.step + 1))

case class AdamState[P](
    momentums: P,
    velocities: P,
    beta1t: Tensor0[Float32], // decay rate for momentums mᵗ, hard-coded precision to make State independent of V, making persisting and restoring easier
    beta2t: Tensor0[Float32] // decay rate for velocities vᵗ, hard-coded precision to make State independent of V, making persisting and restoring easier
)

/** Implements the Adam optimization algorithm.
  *
  * @see [[https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization]]
  */
class Adam(
    val learningRate: Tensor0[Float32],
    val beta1: Tensor0[Float32] = Tensor0(0.9f), // decay rate for momentums mᵗ
    val beta2: Tensor0[Float32] = Tensor0(0.999f), // decay rate for velocities vᵗ
    val epsilon: Tensor0[Float32] = Tensor0(1e-8f) // small constant to prevent division by zero
) extends GradientOptimizer[AdamState]:

  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): AdamState[P] =
    def zeros = params.fillCopy(0f)
    AdamState(zeros, zeros, beta1t = Tensor0(1f), beta2t = Tensor0(1f))

  def update[P: TensorTree, V](
      gradients: Grad[P],
      params: P,
      state: AdamState[P]
  )(using TreeOf[P, V])(using IsFloating[V]): (P, AdamState[P]) =
    // rename parameters for internal clarity
    val α = learningRate.asFloat(VType[V])
    val β1 = beta1.asFloat(VType[V])
    val β2 = beta2.asFloat(VType[V])
    val ε = epsilon.asFloat(VType[V])

    // rename state variables to last time step for clarity
    val `mₜ₋₁` = state.momentums
    val `vₜ₋₁` = state.velocities
    val `β1ₜ₋₁` = state.beta1t.asFloat(VType[V])
    val `β2ₜ₋₁` = state.beta2t.asFloat(VType[V])

    val `θₜ₋₁` = params

    // Adam implementation
    val gₜ = gradients.value
    val mᵗ = `β1` **! `mₜ₋₁` ++ (1f - `β1`) **! gₜ
    val vᵗ = `β2` **! `vₜ₋₁` ++ (1f - `β2`) **! gₜ.pow(2)
    val β1ₜ = `β1ₜ₋₁` * β1
    val β2ₜ = `β2ₜ₋₁` * β2
    val m̂ = mᵗ `//!` (1f - `β1ₜ`)
    val v̂ = vᵗ `//!` (1f - `β2ₜ`)
    val θₜ = `θₜ₋₁` -- (α **! m̂) `//` (v̂.sqrt ++! ε)

    (θₜ, AdamState(mᵗ, vᵗ, β1ₜ.asFloat32, β2ₜ.asFloat32))

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
class AdamW(
    val adam: Adam,
    val weightDecayFactor: Tensor0[Float32]
) extends GradientOptimizer[AdamState]:

  def init[P: TensorTree, V](params: P)(using TreeOf[P, V])(using IsFloating[V]): AdamState[P] = adam.init(params)

  def update[P: TensorTree, V](
      gradients: Grad[P],
      params: P,
      state: AdamState[P]
  )(using TreeOf[P, V])(using IsFloating[V]): (P, AdamState[P]) =
    val α = adam.learningRate.asFloat(VType[V])
    val `λ'` = weightDecayFactor.asFloat(VType[V])

    val `θₜ₋₁` = params
    val λ = `λ'` * α // Tie weight decay to learning rate
    val decayedParams = `θₜ₋₁` -- λ **! `θₜ₋₁`
    val (θₜ, adamState) = adam.update(gradients, decayedParams, state)
    (θₜ, adamState)
