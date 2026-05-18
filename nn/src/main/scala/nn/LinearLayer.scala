package nn

import dimwit.*
import dimwit.random.Random
import dimwit.random.Random.Key
import dimwit.tensor.VType
import dimwit.stats.Normal

object LinearLayer:

  case class Params[In, Out](weight: Tensor2[In, Out, Float32], bias: Tensor1[Out, Float32])

  object Params:
    given [I: Label, O: Label]: TensorTree[Params[I, O]] = TensorTree.derived

    def apply[In: Label, Out: Label](paramKey: Key)(
        inputDim: AxisExtent[In],
        outputDim: AxisExtent[Out]
    ): Params[In, Out] =
      Params(
        weight = Normal.standardNormal(Shape(inputDim, outputDim)).sample(paramKey),
        bias = Tensor(Shape(outputDim)).fill(0.0f)
      )

case class LinearLayer[In: Label, Out: Label](params: LinearLayer.Params[In, Out]) extends Function[Tensor1[In, Float32], Tensor1[Out, Float32]]:
  override def apply(x: Tensor1[In, Float32]): Tensor1[Out, Float32] =
    import params.{weight, bias}
    x.dot(Axis[In])(weight) + bias

object LinearMap:

  case class Params[In](weight: Tensor1[In, Float32], bias: Tensor0[Float32])

  object Params:
    given [In: Label]: TensorTree[Params[In]] = TensorTree.derived

    def apply[In: Label](paramKey: Key)(inputDim: AxisExtent[In]): Params[In] =
      Params(
        weight = Normal.standardNormal(Shape(inputDim)).sample(paramKey),
        bias = Tensor0(0.0f)
      )

case class LinearMap[In: Label](params: LinearMap.Params[In]) extends Function[Tensor1[In, Float32], Tensor0[Float32]]:
  override def apply(x: Tensor1[In, Float32]): Tensor0[Float32] =
    import params.{weight, bias}
    x.dot(Axis[In])(weight) + bias
