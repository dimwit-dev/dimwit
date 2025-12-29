package nn

import dimwit.*

case class GradientDescent[Params](df: Params => Params, lr: Tensor0[Float]):
  def step(params: Params)(using paramTree: FloatTensorTree[Params]) =
    val gradients = df(params)
    paramTree.zipMap(gradients, params, [T <: Tuple] => (n: Labels[T]) ?=> (g: Tensor[T, Float], p: Tensor[T, Float]) => p - g.scale(lr))
