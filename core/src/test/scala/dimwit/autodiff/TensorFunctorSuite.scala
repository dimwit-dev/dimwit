package dimwit.autodiff

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class TensorFunctorSuite extends AnyFunSpec with Matchers:

  describe("map"):
    it("1-level case class (mixed dtypes)"):
      case class Data(
          val numbers: Tensor1[A, Float],
          val counts: Tensor1[A, Int],
          val flags: Tensor1[A, Boolean]
      )
      val params = Data(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor1(Axis[A]).fromArray(Array(1, 2, 3)),
        Tensor1(Axis[A]).fromArray(Array(true, false, true))
      )
      val tree = summon[TensorFunctor[Data]]
      val tree2 = tree.map(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)
      tree2.numbers should approxEqual(params.numbers)
      tree2.counts should equal(params.counts)
      tree2.flags should equal(params.flags)

    it("1-level case class (all float)"):
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
      val ftTree = summon[TensorFunctor[Params]]
      def add5[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] = t // identity for generic V
      val res = ftTree.map(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)
      res.w1 should approxEqual(params.w1)
      res.b1 should approxEqual(params.b1)
      res.w2 should approxEqual(params.w2)
      res.b2 should approxEqual(params.b2)

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
      val ftTree = summon[TensorFunctor[ModelParams]]
      val res = ftTree.map(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)

      res.layer1.w should approxEqual(params.layer1.w)
      res.layer1.b should approxEqual(params.layer1.b)
      res.layer2.w should approxEqual(params.layer2.w)
      res.layer2.b should approxEqual(params.layer2.b)

    it("case class with tuple"):
      case class LayerParams(
          val weightBias: (Tensor2[A, B, Float], Tensor0[Float])
      )
      val layerParams = LayerParams(
        Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f))),
        Tensor0(0.25f)
      )
      val ftTree = summon[TensorFunctor[LayerParams]]
      val res = ftTree.map(layerParams, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)

      res.weightBias._1 should approxEqual(layerParams.weightBias._1)
      res.weightBias._2 should approxEqual(layerParams.weightBias._2)

    it("case class with list"):
      case class Params(
          val layerWeights: List[Tensor2[A, B, Float]]
      )
      val layerParams = Params(
        List(
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f))),
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.1f, 1.2f), Array(1.3f, 1.4f), Array(1.5f, 1.6f)))
        )
      )
      val ftTree = summon[TensorFunctor[Params]]
      val res = ftTree.map(layerParams, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)

      res.layerWeights(0) should approxEqual(layerParams.layerWeights(0))
      res.layerWeights(1) should approxEqual(layerParams.layerWeights(1))

    it("case class with map"):
      case class Params(
          val layerWeights: Map[String, Tensor2[A, B, Float]]
      )
      val layerParams = Params(
        Map(
          ("layer1", Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f), Array(0.5f, 0.6f)))),
          ("layer2", Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.1f, 1.2f), Array(1.3f, 1.4f), Array(1.5f, 1.6f))))
        )
      )
      val ftTree = summon[TensorFunctor[Params]]
      val res = ftTree.map(layerParams, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)

      res.layerWeights("layer1") should approxEqual(layerParams.layerWeights("layer1"))
      res.layerWeights("layer2") should approxEqual(layerParams.layerWeights("layer2"))

  describe("zipmap"):
    it("1-level case class (mixed dtypes)"):
      case class Params(
          val w1: Tensor1[A, Float],
          val b1: Tensor0[Int]
      )
      val params1 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0)
      )
      val params2 = Params(
        Tensor1(Axis[A]).fromArray(Array(0.4f, 0.5f, 0.6f)),
        Tensor0(1)
      )
      val ftTree = summon[TensorFunctor[Params]]
      val res = ftTree.zipMap(params1, params2, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x1: Tensor[T, V], x2: Tensor[T, V]) => maximum(x1, x2))
      res.w1 should approxEqual(maximum(params1.w1, params2.w1))
      res.b1 should equal(maximum(params1.b1, params2.b1))

    it("1-level case class (all float)"):
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
      val ftTree = summon[TensorFunctor[Params]]
      val res = ftTree.zipMap(params1, params2, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x1: Tensor[T, V], x2: Tensor[T, V]) => maximum(x1, x2))
      res.w1 should approxEqual(maximum(params1.w1, params2.w1))
      res.b1 should approxEqual(maximum(params1.b1, params2.b1))
