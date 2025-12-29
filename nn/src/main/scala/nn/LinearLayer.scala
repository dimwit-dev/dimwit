package nn

import dimwit.*
import dimwit.random.Random
import dimwit.random.Random.Key
import dimwit.tensor.VType
import dimwit.tensor.ExecutionType

object LinearLayer:

  case class Params[In, Out](weight: Tensor2[In, Out, Float], bias: Tensor1[Out, Float])

  object Params:
    given [I: Label, O: Label]: TensorTree[Params[I, O]] = TensorTree.derived
    given [I: Label, O: Label]: ToPyTree[Params[I, O]] = ToPyTree.derived

    def apply[In: Label, Out: Label](paramKey: Key)(
        inputDim: Dim[In],
        outputDim: Dim[Out]
    )(using
        executionType: ExecutionType[Float]
    ): Params[In, Out] =
      Params(
        weight = Random.Normal(Shape(inputDim, outputDim))(paramKey),
        bias = Tensor.zeros(Shape(outputDim), VType[Float])
      )

case class LinearLayer[In: Label, Out: Label](params: LinearLayer.Params[In, Out]) extends Function[Tensor1[In, Float], Tensor1[Out, Float]]:
  override def apply(x: Tensor1[In, Float]): Tensor1[Out, Float] =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias

object LinearMap:

  case class Params[In](weight: Tensor1[In, Float], bias: Tensor0[Float])

  object Params:
    given [In: Label]: TensorTree[Params[In]] = TensorTree.derived
    given [In: Label]: ToPyTree[Params[In]] = ToPyTree.derived

    def apply[In: Label](paramKey: Key)(inputDim: Dim[In])(using
        executionType: ExecutionType[Float]
    ): Params[In] =
      Params(
        weight = Random.Normal(Shape(inputDim))(paramKey),
        bias = Tensor0(0.0f)
      )

case class LinearMap[In: Label](params: LinearMap.Params[In]) extends Function[Tensor1[In, Float], Tensor0[Float]]:
  override def apply(x: Tensor1[In, Float]): Tensor0[Float] =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias
