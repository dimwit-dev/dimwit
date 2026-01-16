package dimwit.autodiff

import dimwit.*
import dimwit.Conversions.given
import dimwit.*
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

class TensorTreeSuite extends AnyFunSpec with Matchers:

  describe("map"):
    it("1-level case class"):
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
      val tree = summon[TensorTree[Data]]
      val tree2 = tree.map(params, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x: Tensor[T, V]) => x)
      tree2.numbers should approxEqual(params.numbers)
      tree2.counts should equal(params.counts)
      tree2.flags should equal(params.flags)

  describe("zipmap"):
    it("1-level case class"):
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
      val ftTree = summon[TensorTree[Params]]
      val res = ftTree.zipMap(params1, params2, [T <: Tuple, V] => (labels: Labels[T]) ?=> (x1: Tensor[T, V], x2: Tensor[T, V]) => maximum(x1, x2))
      res.w1 should approxEqual(maximum(params1.w1, params2.w1))
      res.b1 should equal(maximum(params1.b1, params2.b1))
