package dimwit.tensor

import dimwit.*
import scala.compiletime.testing.typeCheckErrors

class TensorWithValueClassSuite extends DimwitTest:

  it("Value class support for more specific types in tensors"):
    object ValueClassScope:
      opaque type V1 = Float32
      opaque type V2 = Float32

      object V1:
        def apply[T <: Tuple](t: Tensor[T, Float32]): Tensor[T, V1] = t // lift
        given IsFloating[V1] = summon[IsFloating[Float32]] // make all IsFloating ops available
      object V2:
        def apply[T <: Tuple](t: Tensor[T, Float32]): Tensor[T, V2] = t // lift
        given IsFloating[V2] = summon[IsFloating[Float32]] // make all IsFloating ops available

    import ValueClassScope.*
    val t = Tensor(Shape(Axis[A] -> 1, Axis[B] -> 2)).fill(0f)
    val v1 = V1(t)
    val v2 = V2(t)
    "v1 + v1" should compile
    "v2 + v2" should compile
    "v1 + v2" shouldNot compile
