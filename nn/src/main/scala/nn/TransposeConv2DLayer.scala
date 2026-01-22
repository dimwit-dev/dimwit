package nn

import dimwit.*
import dimwit.random.Random.Key
import dimwit.stats.Normal

object TransposeConvLayer:

  case class Params[S1, S2, InChannels, OutChannels](
      kernel: Tensor[S1 *: S2 *: InChannels *: OutChannels *: EmptyTuple, Float]
  )

  object Params:
    given [S1: Label, S2: Label, IC: Label, OC: Label]: FloatTensorTree[Params[S1, S2, IC, OC]] = FloatTensorTree.derived
    given [S1: Label, S2: Label, IC: Label, OC: Label]: ToPyTree[Params[S1, S2, IC, OC]] = ToPyTree.derived

    /** Initialize transpose convolutional layer parameters
      *
      * @param paramKey
      *   Random key for parameter initialization
      * @param kernelShape
      *   Shape of the convolutional kernel, e.g., (KernelH, KernelW, InChannels, OutChannels) for 2D transpose conv
      */
    def apply[S1: Label, S2: Label, InChannels: Label, OutChannels: Label](paramKey: Key)(
        kernelShape: Shape[S1 *: S2 *: InChannels *: OutChannels *: EmptyTuple]
    )(using
        executionType: ExecutionType[Float]
    ): Params[S1, S2, InChannels, OutChannels] =
      Params(
        kernel = Normal.standardNormal(kernelShape).sample(paramKey)
      )

case class TransposeConvLayer[S1: Label, S2: Label, InChannels: Label, OutChannels: Label](
    params: TransposeConvLayer.Params[S1, S2, InChannels, OutChannels],
    stride: Int = 1,
    padding: Padding = Padding.SAME
):
  /** Apply transpose convolution to input tensor
    *
    * Note: For transpose convolution, the input has OutChannels (matching forward conv output) and the output has InChannels (matching forward conv input). This is the adjoint operation to forward convolution.
    *
    * Input: (Spatial..., OutChannels) Output: (Spatial..., InChannels)
    */
  def apply(x: Tensor[S1 *: S2 *: OutChannels *: EmptyTuple, Float]): Tensor[S1 *: S2 *: InChannels *: EmptyTuple, Float] =
    x.transposeConv2d(params.kernel, stride, padding)
