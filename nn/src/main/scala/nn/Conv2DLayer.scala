package nn

import dimwit.*
import dimwit.random.Random.Key
import dimwit.stats.Normal

object Conv2DLayer:

  case class Params[S1, S2, InChannel, OutChannel](
      kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, Float]
  )

  object Params:
    given [S1: Label, S2: Label, InChannel: Label, OutChannel: Label]: FloatTensorTree[Params[S1, S2, InChannel, OutChannel]] = FloatTensorTree.derived
    given [S1: Label, S2: Label, InChannel: Label, OutChannel: Label]: ToPyTree[Params[S1, S2, InChannel, OutChannel]] = ToPyTree.derived

    def apply[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](paramKey: Key)(kernelShape: Shape[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple])(using executionType: ExecutionType[Float]): Params[S1, S2, InChannel, OutChannel] =
      Params(kernel = Normal.standardNormal(kernelShape).sample(paramKey))

case class Conv2DLayer[S1: Label, S2: Label, InChannel: Label, OutChannel: Label](
    params: Conv2DLayer.Params[S1, S2, InChannel, OutChannel],
    stride: Stride2[S1, S2] | Int = 1,
    padding: Padding = Padding.SAME
):

  def apply(x: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, Float]): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, Float] =
    x.conv2d(params.kernel, stride, padding)
