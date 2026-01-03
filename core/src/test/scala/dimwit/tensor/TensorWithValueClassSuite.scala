package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import scala.compiletime.testing.typeCheckErrors

class TensorWithValueClassSuite extends AnyFunSpec with ScalaCheckPropertyChecks with Matchers:

  it("Nice error message when axis not found in tensor for sum"):
    object ValueClassScope:
      opaque type V1 = Float
      opaque type V2 = Float

      object V1:
        def apply[T <: Tuple](t: Tensor[T, Float]): Tensor[T, V1] = t // lift
        given IsFloat[V1] with {} // make all IsFloat ops available
      object V2:
        def apply[T <: Tuple](t: Tensor[T, Float]): Tensor[T, V2] = t // lift
        given IsFloat[V2] with {} // make all IsFloat ops available

    import ValueClassScope.*
    val t = Tensor.zeros(Shape(Axis[A] -> 1, Axis[B] -> 2), VType[Float])
    val v1 = V1(t)
    val v2 = V2(t)
    "v1 + v1" should compile
    "v2 + v2" should compile
    "v1 + v2" shouldNot compile
