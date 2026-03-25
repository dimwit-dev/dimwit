package dimwit.autodiff

import dimwit.*
import dimwit.jax.Jax
import dimwit.Conversions.given
import me.shadaj.scalapy.py
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class ToPyTreeSuite extends AnyFunSpec with Matchers:

  describe("TensorTree Identity (fromPyTree(toPyTree(x)) == x)"):

    it("1-level case class"):
      case class Params(
          val w: Tensor1[A, Float],
          val b: Tensor0[Float]
      )
      val params = Params(
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0.5f)
      )

      val tc = summon[TensorTree[Params]]
      val reconstructed = tc.fromPyTree(tc.toPyTree(params))

      reconstructed.w should approxEqual(params.w)
      reconstructed.b should approxEqual(params.b)

    it("2-level case class"):
      case class LayerParams(
          val w: Tensor2[A, B, Float],
          val b: Tensor0[Float]
      )
      case class ModelParams(
          val layer1: LayerParams,
          val layer2: LayerParams
      )

      val params = ModelParams(
        LayerParams(
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f))),
          Tensor0(0.25f)
        ),
        LayerParams(
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.5f, 0.6f), Array(0.7f, 0.8f))),
          Tensor0(0.75f)
        )
      )

      val tc = summon[TensorTree[ModelParams]]
      val reconstructed = tc.fromPyTree(tc.toPyTree(params))

      reconstructed.layer1.w should approxEqual(params.layer1.w)
      reconstructed.layer1.b should approxEqual(params.layer1.b)
      reconstructed.layer2.w should approxEqual(params.layer2.w)
      reconstructed.layer2.b should approxEqual(params.layer2.b)

    it("tuple"):
      val myTuple = (
        Tensor1(Axis[A]).fromArray(Array(0.1f, 0.2f, 0.3f)),
        Tensor0(0.5f)
      )

      val tc = summon[TensorTree[(Tensor1[A, Float], Tensor0[Float])]]
      val reconstructed = tc.fromPyTree(tc.toPyTree(myTuple))

      reconstructed._1 should approxEqual(myTuple._1)
      reconstructed._2 should approxEqual(myTuple._2)

    it("case class with list"):
      case class Params(
          val layerWeights: List[Tensor2[A, B, Float]]
      )
      val params = Params(
        List(
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(0.1f, 0.2f), Array(0.3f, 0.4f))),
          Tensor2(Axis[A], Axis[B]).fromArray(Array(Array(1.1f, 1.2f), Array(1.3f, 1.4f)))
        )
      )

      val tc = summon[TensorTree[Params]]
      val reconstructed = tc.fromPyTree(tc.toPyTree(params))

      reconstructed.layerWeights.size shouldBe params.layerWeights.size
      reconstructed.layerWeights(0) should approxEqual(params.layerWeights(0))
      reconstructed.layerWeights(1) should approxEqual(params.layerWeights(1))
