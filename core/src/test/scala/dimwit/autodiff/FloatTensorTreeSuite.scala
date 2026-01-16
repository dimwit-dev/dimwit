package dimwit.autodiff

import dimwit.*
import dimwit.Conversions.given
import dimwit.*
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class FloatTensorTreeSuite extends AnyFunSpec with Matchers:

  describe("map"):
    it("1-level case class"):
      case class Params(
          val w1: Tensor1[A, Float],
          val b1: Tensor0[Float],
          val w2: Tensor2[A, B, Float],
          val b2: Tensor0[Float]
      )
      val params = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0.5f),
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f))),
        Tensor0(0.25f)
      )
      val ftTree = summon[FloatTensorTree[Params]]
      def add5[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, Float] = t +! 0.5f
      val res = ftTree.map(params, [T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, Float]) => add5[T](x))
      res.w1 should approxEqual(params.w1 +! 0.5f)
      res.b1 should approxEqual(params.b1 + 0.5f)
      res.w2 should approxEqual(params.w2 +! 0.5f)
      res.b2 should approxEqual(params.b2 + 0.5f)

    it("2-level case class"):
      case class LayerParams(
          val w: Tensor2[A, B, Float],
          val b: Tensor0[Float]
      )
      case class ModelParams(
          val layer1: LayerParams,
          val layer2: LayerParams
      )
      val layer1Params = LayerParams(
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f))),
        Tensor0(0.25f)
      )
      val layer2Params = LayerParams(
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.7f, 0.8f), Array(0.9f, 1.0f), Array(1.1f, 1.2f))),
        Tensor0(0.75f)
      )
      val params = ModelParams(layer1Params, layer2Params)
      val ftTree = summon[FloatTensorTree[ModelParams]]
      def add5[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, Float] = t +! 0.5f
      val res = ftTree.map(params, [T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, Float]) => add5[T](x))

      res.layer1.w should approxEqual(params.layer1.w +! 0.5f)
      res.layer1.b should approxEqual(params.layer1.b + 0.5f)
      res.layer2.w should approxEqual(params.layer2.w +! 0.5f)
      res.layer2.b should approxEqual(params.layer2.b + 0.5f)

    it("case class with tuples"):
      case class LayerParams(
          val weightBias: (Tensor2[A, B, Float], Tensor0[Float])
      )
      val layerParams = LayerParams(
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f))),
        Tensor0(0.25f)
      )
      val ftTree = summon[FloatTensorTree[LayerParams]]
      def add5[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, Float] = t +! 0.5f
      val res = ftTree.map(layerParams, [T <: Tuple] => (labels: Labels[T]) ?=> (x: Tensor[T, Float]) => add5[T](x))

      res.weightBias._1 should approxEqual(layerParams.weightBias._1 +! 0.5f)
      res.weightBias._2 should approxEqual(layerParams.weightBias._2 + 0.5f)

  describe("zipmap"):
    it("1-level case class"):
      case class Params(
          val w1: Tensor1[A, Float],
          val b1: Tensor0[Float]
      )
      val params1 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0.5f)
      )
      val params2 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.4f, 0.5f, 0.6f)),
        Tensor0(1.5f)
      )
      val ftTree = summon[FloatTensorTree[Params]]
      def addTensors[T <: Tuple: Labels](t1: Tensor[T, Float], t2: Tensor[T, Float]): Tensor[T, Float] = t1 + t2
      val res = ftTree.zipMap(params1, params2, [T <: Tuple] => (labels: Labels[T]) ?=> (x1: Tensor[T, Float], x2: Tensor[T, Float]) => addTensors[T](x1, x2))
      res.w1 should approxEqual(params1.w1 + params2.w1)
      res.b1 should approxEqual(params1.b1 + params2.b1)
