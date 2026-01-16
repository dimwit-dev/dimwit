package nn

import dimwit.*
import dimwit.autodiff.Grad
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
  def init[Params: ToPyTree: FloatTensorTree](params: Params): State[Params]
  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, state: State[Params]): (Params, State[Params])

  // Convenience: iterator with fixed gradient function
  def iterateWithState[Params: ToPyTree: FloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[(Params, State[Params])] =
    Iterator.iterate((init, this.init(init))): (params, state) =>
      val grads = df(params)
      update(grads, params, state)

  def iterate[Params: ToPyTree: FloatTensorTree](init: Params)(df: Params => Grad[Params]): Iterator[Params] =
    iterateWithState(init)(df).map(_._1)

case class GradientDescent(learningRate: Tensor0[Float]) extends GradientOptimizer:
  import dimwit.Conversions.given

  type State[P] = Unit // Stateless optimizer

  def init[Params: ToPyTree: FloatTensorTree](params: Params): Unit = ()

  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, state: Unit): (Params, Unit) =
    val paramTree = summon[FloatTensorTree[Params]]
    val newParams = paramTree.zipMap(
      gradients.value,
      params,
      [T <: Tuple] => (n: Labels[T]) ?=> (g: Tensor[T, Float], p: Tensor[T, Float]) => p - g.scale(learningRate)
    )
    (newParams, ())

case class Lion(learningRate: Tensor0[Float], weightDecay: Tensor0[Float] = Tensor0(0.0f), beta1: Tensor0[Float] = Tensor0(0.9f), beta2: Tensor0[Float] = Tensor0(0.99f)) extends GradientOptimizer:
  import dimwit.Conversions.given

  type State[P] = P // momentum state has same structure as params

  def init[Params: ToPyTree: FloatTensorTree](params: Params): Params =
    val paramTree = summon[FloatTensorTree[Params]]
    paramTree.map(
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (t: Tensor[T, Float]) =>
            Tensor.zeros(t.shape, VType[Float])
    )

  def update[Params: ToPyTree: FloatTensorTree](gradients: Grad[Params], params: Params, momentums: Params): (Params, Params) =
    val paramTree = summon[FloatTensorTree[Params]]
    // the direction (1 or -1)
    // is determined by the sign of the momentum + gradient
    val updateDirection = paramTree.zipMap(
      gradients.value,
      momentums,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (grad: Tensor[T, Float], momentum: Tensor[T, Float]) =>
            (momentum *! beta1 + grad *! (1f - beta1)).sign
    )

    val updatedParams = paramTree.zipMap(
      updateDirection,
      params,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (updateDir: Tensor[T, Float], param: Tensor[T, Float]) =>
            param - updateDir *! learningRate - param *! weightDecay
    )

    val newMomentums = paramTree.zipMap(
      gradients.value,
      momentums,
      [T <: Tuple] =>
        (n: Labels[T]) ?=>
          (g: Tensor[T, Float], m: Tensor[T, Float]) =>
            m *! beta2 + g *! (1f - beta2)
    )

    (updatedParams, newMomentums)
