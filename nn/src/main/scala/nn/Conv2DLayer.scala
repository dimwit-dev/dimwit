package nn

import dimwit.*
import dimwit.random.Random.Key
import dimwit.stats.Normal

object Conv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel, V](
      kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, V]
  )

  object Params:
    given [S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V]: TensorTree[Params[S1, S2, InChannel, OutChannel, V]] = TensorTree.derived

    def apply[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](paramKey: Key)(kernelShape: Shape[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple]): Params[S1, S2, InChannel, OutChannel, V] =
      Params(kernel = Normal.standardNormal(kernelShape).sample(paramKey).asFloat(VType[V]))

case class Conv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label, V: IsFloating](
    params: Conv2DLayer.Params[S1, S2, InChannel, OutChannel, V],
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
):

  def apply(x: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, V]): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, V] =
    x.conv2d(params.kernel, stride, padding)
