package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class ShapeSuite extends AnyFunSpec with Matchers:

  it("Basic shape functions"):
    val shape = Shape(Axis[A] -> 2, Axis[B] -> 3)
    shape.dimensions shouldBe List(2, 3)
    shape.rank shouldBe 2
    shape.size shouldBe 6
    shape(Axis[A]) shouldBe 2
    shape(Axis[B]) shouldBe 3
    shape.extent(Axis[A]) shouldBe a[AxisExtent[A]]
    shape.extent(Axis[A]).size shouldBe shape(Axis[A])
    shape.extent(Axis[B]) shouldBe a[AxisExtent[B]]
    shape.extent(Axis[B]).size shouldBe shape(Axis[B])
    shape.labels shouldBe List("A", "B")
