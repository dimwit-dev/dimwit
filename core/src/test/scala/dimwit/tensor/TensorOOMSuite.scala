package dimwit.tensor

import dimwit.*
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors

class TensorMemorySuite extends AnyFunSpec with Matchers:

  def createLargeTensor(): Tensor[?, Float] =
    // 10k x 10k Float tensor ~ 0.4 GB
    Tensor.zeros(Shape(Axis[A] -> 10_000, Axis[B] -> 10_000), VType[Float])

  it("OOM when allocating and referencing infinite tensors"):
    def oomBlock(): Unit =
      var l = List.empty[Tensor[?, Float]]
      try while true do l = createLargeTensor() :: l
      finally info(s"Allocated ${l.size} tensors before OOM")
    an[Exception] should be thrownBy oomBlock()

  it("No OOM when allocating N tensors"):
    def oomAtN: Int =
      var l = List.empty[Tensor[?, Float]]
      try while true do l = createLargeTensor() :: l
      catch case _: Exception => () // ignore OOM exception
      l.size + 1
    def block(n: Int): Unit =
      // Each iteration: Create tensor ready for garbage collection
      for _ <- 0 until n do
        PythonMemoryGuard.withRetry:
          createLargeTensor()
    val N = 2 * oomAtN // guaranteed to be above available memory
    noException should be thrownBy (block(N))
