package dimwit.optimizer

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.*
import dimwit.autodiff.Grad
import dimwit.tensortree.*
import dimwit.tensortree.FloatTree.ops.*

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
trait GradientOptimizer[V: IsFloating, State0[_]]:

  type State[P] = State0[P]

  // Core API
  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): State[P]
  def update[P](gradients: Grad[P], params: P, state: State[P])(using TensorTree[P], FloatTree[P, V]): (P, State[P])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[P](init: P)(df: P => Grad[P])(using TensorTree[P], FloatTree[P, V]): Iterator[(P, State[P])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[P](init: P)(df: P => Grad[P])(using TensorTree[P], FloatTree[P, V]): Iterator[P] =
    iterateWithState(init)(df).map(_._1)

object GradientDescent:

  def of[V: IsFloating](vtype: VType[V])(learningRate: Tensor0[V]): GradientDescent[V] = new GradientDescent(learningRate)

type GradientDescentState[P] = Unit // empty state

class GradientDescent[V: IsFloating](val learningRate: Tensor0[V]) extends GradientOptimizer[V, GradientDescentState]:

  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): Unit = ()

  def update[P](gradients: Grad[P], params: P, state: Unit)(using TensorTree[P], FloatTree[P, V]): (P, Unit) =
    val newParams = params -- gradients.value.scale(learningRate)
    (newParams, ())

case class LionState[P](
    momentums: P,
    step: Tensor0[Int32]
)

object Lion:

  def of[V](vtype: VType[V])(using IsFloating[V])(learningRate: Tensor0[V], weightDecay: Tensor0[V] = Tensor0(vtype)(0.0), beta1: Tensor0[V] = Tensor0(vtype)(0.9), beta2: Tensor0[V] = Tensor0(vtype)(0.99)): Lion[V] = new Lion(learningRate, weightDecay, beta1, beta2)

class Lion[V: IsFloating](val learningRate: Tensor0[V], val weightDecay: Tensor0[V] = Tensor0(0.0), val beta1: Tensor0[V] = Tensor0(0.9), val beta2: Tensor0[V] = Tensor0(0.99)) extends GradientOptimizer[V, LionState]:

  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): LionState[P] =
    LionState(params.fillCopy(0f), step = 1)

  def update[P](gradients: Grad[P], params: P, state: LionState[P])(using TensorTree[P], FloatTree[P, V]): (P, LionState[P]) =
    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = (state.momentums **! beta1 ++ gradients.value **! (1f - beta1)).sign

    val updatedParams = params -- updateDirection.scale(learningRate) -- params.scale(weightDecay)
    val newMomentums = state.momentums **! beta2 ++ gradients.value **! (1f - beta2)

    (updatedParams, LionState(newMomentums, state.step + 1))

case class AdamState[P](
    momentums: P,
    velocities: P,
    b1: Tensor0[Float32], // decay rate for momentums mᵗ, hard-coded precision to make State independent of V, making persisting and restoring easier
    b2: Tensor0[Float32] // decay rate for velocities vᵗ, hard-coded precision to make State independent of V, making persisting and restoring easier
)

object Adam:

  def of[V](vtype: VType[V])(using IsFloating[V])(learningRate: Tensor0[V], b1: Tensor0[V] = 0.9, b2: Tensor0[V] = 0.999, epsilon: Tensor0[V] = 1e-8): Adam[V] = new Adam(learningRate, b1, b2, epsilon)
  def apply[V: IsFloating](learningRate: Tensor0[V], b1: Double = 0.9, b2: Double = 0.999, epsilon: Double = 1e-8): Adam[V] = new Adam(learningRate, b1, b2, epsilon)

/** Implements the Adam optimization algorithm.
  *
  * @see [[https://arxiv.org/abs/1412.6980 Adam: A Method for Stochastic Optimization]]
  */
class Adam[V: IsFloating](
    val learningRate: Tensor0[V],
    b1: Tensor0[V], // decay rate for momentums mᵗ
    b2: Tensor0[V], // decay rate for velocities vᵗ
    epsilon: Tensor0[V] // small constant to prevent division by zero
) extends GradientOptimizer[V, AdamState]:

  private val vtype = VType[V]

  private val β1 = b1
  private val β2 = b2
  private val ε = epsilon

  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): AdamState[P] =
    def zeros = params.fillCopy(0f)
    AdamState(zeros, zeros, b1 = Tensor0(1f), b2 = Tensor0(1f))

  def update[P](
      gradients: Grad[P],
      params: P,
      state: AdamState[P]
  )(using TensorTree[P], FloatTree[P, V]): (P, AdamState[P]) =
    // rename state variables to last time step for clarity
    val `mₜ₋₁` = state.momentums
    val `vₜ₋₁` = state.velocities
    val `β1ₜ₋₁` = state.b1.asFloat(vtype)
    val `β2ₜ₋₁` = state.b2.asFloat(vtype)

    // rename parameters for internal clarity
    val α = learningRate

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
class AdamW[V: IsFloating](
    val adam: Adam[V],
    val weightDecayFactor: Tensor0[V]
) extends GradientOptimizer[V, [P] =>> AdamState[P]]:

  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): AdamState[P] = adam.init(params)

  def update[P](
      gradients: Grad[P],
      params: P,
      state: AdamState[P]
  )(using TensorTree[P], FloatTree[P, V]): (P, AdamState[P]) =
    val α = adam.learningRate
    val `θₜ₋₁` = params
    val `λ'` = weightDecayFactor
    val λ = `λ'` * α // Tie weight decay to learning rate
    val decayedParams = `θₜ₋₁` -- λ **! `θₜ₋₁`
    val (θₜ, adamState) = adam.update(gradients, decayedParams, state)
    (θₜ, adamState)

case class LearningRateSchedulerState[P, State[_]](
    step: Tensor0[Int32],
    optState: State[P]
)
type LearningRateSchedulerStateFor[State[_]] = [P] =>> LearningRateSchedulerState[P, State]

object LearningRateScheduler:

  def of[V: IsFloating, State[_]](
      vtype: VType[V]
  )(
      optF: Tensor0[V] => GradientOptimizer[V, State],
      schedule: Tensor0[Int32] => Tensor0[V]
  ): LearningRateScheduler[V, State] =
    new LearningRateScheduler(optF, schedule)

class LearningRateScheduler[V: IsFloating, State[_]](val optF: Tensor0[V] => GradientOptimizer[V, State], schedule: Tensor0[Int32] => Tensor0[V]) extends GradientOptimizer[V, LearningRateSchedulerStateFor[State]]:

  def init[P](params: P)(using TensorTree[P], FloatTree[P, V]): LearningRateSchedulerState[P, State] =
    val step = Tensor0(1)
    val opt = optF(schedule(step))
    LearningRateSchedulerState(step, opt.init(params))

  def update[P](gradients: Grad[P], params: P, state: LearningRateSchedulerState[P, State])(using TensorTree[P], FloatTree[P, V]): (P, LearningRateSchedulerState[P, State]) =
    val step = state.step
    val optState = state.optState
    val opt = optF(schedule(step))
    val (newParams, newOptState) = opt.update(gradients, params, optState)
    (newParams, LearningRateSchedulerState(step + 1, newOptState))
