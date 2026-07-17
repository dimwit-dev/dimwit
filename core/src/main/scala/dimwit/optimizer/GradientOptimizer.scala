package dimwit.optimizer

import dimwit.*
import dimwit.autodiff.FloatTree.*
import dimwit.autodiff.FloatTree.ops.*
import dimwit.autodiff.*

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
  def init[Params: TensorTree: FloatTreeFor[Float32]](params: Params): State[Params]
  def update[Params: TensorTree: FloatTreeFor[Float32]](gradients: Grad[Params], params: Params, state: State[Params]): (Params, State[Params])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[Params: TensorTree: FloatTreeFor[Float32]](init: Params)(df: Params => Grad[Params]): Iterator[(Params, State[Params])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[Params: TensorTree: FloatTreeFor[Float32]](init: Params)(df: Params => Grad[Params]): Iterator[Params] =
    iterateWithState(init)(df).map(_._1)

case class GradientDescent(learningRate: Double) extends GradientOptimizer:

  private val lr = Tensor0(learningRate.toFloat)

  type State[P] = Unit // Stateless optimizer

  def init[Params: TensorTree: FloatTreeFor[Float32]](params: Params): Unit = ()

  def update[Params: TensorTree: FloatTreeFor[Float32]](gradients: Grad[Params], params: Params, state: Unit): (Params, Unit) =
    val newParams = params -- gradients.value.scale(lr)
    (newParams, ())

case class Lion(learningRate: Double, weightDecay: Double = 0.0, beta1: Double = 0.9, beta2: Double = 0.99) extends GradientOptimizer:

  val beta1f = Tensor0(beta1.toFloat)
  val beta2f = Tensor0(beta2.toFloat)
  val lr = Tensor0(learningRate.toFloat)

  type State[P] = P // momentum state has same structure as params

  def init[Params: TensorTree: FloatTreeFor[Float32]](params: Params): Params =
    params.map([T <: Tuple] =>
      (n: Labels[T]) ?=>
        (t: Tensor[T, Float32]) =>
          Tensor(t.shape).fill(0f)
    )

  def update[Params: TensorTree: FloatTreeFor[Float32]](gradients: Grad[Params], params: Params, momentums: Params): (Params, Params) =

    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = (momentums **! beta1f ++ gradients.value **! (1f - beta1f)).sign

    val updatedParams = params -- updateDirection.scale(lr) -- params.scale(Tensor0(weightDecay.toFloat))
    val newMomentums = momentums **! beta2f ++ gradients.value **! (1f - beta2f)

    (updatedParams, newMomentums)

case class AdamState[P](
    momentums: P, // momentums
    velocities: P, // velocities
    b1: Tensor0[Float32], // decay rate for momentums mᵗ
    b2: Tensor0[Float32] // decay rate for velocities vᵗ
)

/** Implements the Adam optimization algorithm.
  *
  * @see [[https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization]]
  */
case class Adam(
    learningRate: Double, // step size (learning rate)
    b1: Double = 0.9, // decay rate for momentums mᵗ
    b2: Double = 0.999, // decay rate for velocities vᵗ
    epsilon: Double = 1e-8 // small constant to prevent division by zero
) extends GradientOptimizer:

  private val β1 = Tensor0(b1.toFloat)
  private val β2 = Tensor0(b2.toFloat)
  private val ε = Tensor0(epsilon.toFloat)

  type State[P] = AdamState[P]

  def init[Params: TensorTree: FloatTreeFor[Float32]](params: Params): State[Params] =
    def zeros = params.fillCopy(0f)
    AdamState(zeros, zeros, b1 = Tensor0(1.0f), b2 = Tensor0(1.0f))

  def update[Params: TensorTree: FloatTreeFor[Float32]](
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
    val α = Tensor0(learningRate.toFloat)
    val ε = Tensor0(epsilon.toFloat)
    val `θₜ₋₁` = params

    // update moments for bias correction
    val β1ₜ = `β1ₜ₋₁` * β1
    val β2ₜ = `β2ₜ₋₁` * β2

    // Adam implementation
    val gₜ = gradients.value
    val mᵗ = `β1` **! `mₜ₋₁` ++ (1f - `β1`) **! gₜ
    val vᵗ = `β2` **! `vₜ₋₁` ++ (1f - `β2`) **! gₜ.pow(2)
    val m̂ = mᵗ `//!` (1f - `β1ₜ`)
    val v̂ = vᵗ `//!` (1f - `β2ₜ`)
    val θₜ = `θₜ₋₁` -- (α **! m̂) `//` (v̂.sqrt ++! ε)

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
    val weightDecayFactor: Double
) extends GradientOptimizer:

  type State[P] = adam.State[P]

  def init[Params: TensorTree: FloatTreeFor[Float32]](params: Params): State[Params] = adam.init(params)

  def update[Params: TensorTree: FloatTreeFor[Float32]](
      gradients: Grad[Params],
      params: Params,
      state: State[Params]
  ): (Params, State[Params]) =
    val α = Tensor0(adam.learningRate.toFloat)
    val `θₜ₋₁` = params
    val `λ'` = Tensor0(weightDecayFactor.toFloat)
    val λ = `λ'` * α // Tie weight decay to learning rate
    val decayedParams = `θₜ₋₁` -- λ **! `θₜ₋₁`
    val (θₜ, adamState) = adam.update(gradients, decayedParams, state)
    (θₜ, adamState)
