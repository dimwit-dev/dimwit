package dimwit.tensor

import dimwit.*
import Device.GPU
import org.scalatest.tagobjects.Slow
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import scala.compiletime.testing.typeCheckErrors
import me.shadaj.scalapy.py

class TensorMemorySuite extends AnyFunSpec with Matchers:

  /** Tests to ensure that:
    *   1. Tensor allocations on GPU memory properly throw OutOfMemory exceptions when memory is exhausted.
    *   2. Garbage collection properly frees memory for reuse.
    */

  def createLargeTensor(): Tensor[?, Float] =
    // 10k x 10k Float tensor ~ 400 MB
    Tensor.zeros(Shape(Axis[A] -> 10_000, Axis[B] -> 10_000), VType[Float]).toDevice(GPU)

  def createMediumTensor(): Tensor[?, Float] =
    // 1000 x 1000 Float tensor ~ 4 MB
    Tensor.zeros(Shape(Axis[A] -> 1000, Axis[B] -> 1000), VType[Float]).toDevice(GPU)

  def createSmallTensor(): Tensor[?, Float] =
    // 100 x 100 Float tensor ~ 40 KB
    Tensor.zeros(Shape(Axis[A] -> 100, Axis[B] -> 100), VType[Float]).toDevice(GPU)

  describe("On the GPU"):

    describe("OOM must be thrown when memory is exhausted"):

      it("Referencing infinite large tensors", Slow):
        def oomBlock(): Unit =
          var l = List.empty[Tensor[?, Float]]
          try while true do l = createLargeTensor() :: l
          finally info(s"Allocated ${l.size} 10000x10000 tensors before OOM")
        an[Exception] should be thrownBy oomBlock()

      it("Referencing infinite medium tensors", Slow):
        def oomBlock(): Unit =
          var l = List.empty[Tensor[?, Float]]
          try while true do
              l = createMediumTensor() :: l
          finally info(s"Allocated ${l.size} 1000x1000 tensors before OOM")
        an[Exception] should be thrownBy oomBlock()

      it("Referencing infinite small tensors", Slow):
        def oomBlock(): Unit =
          var l = List.empty[Tensor[?, Float]]
          try while true do
              l = createSmallTensor() :: l
          finally info(s"Allocated ${l.size} 100x100 tensors before OOM")
        an[Exception] should be thrownBy oomBlock()

    describe("No OOM must be thrown when memory is exhausted by temporary tensors"):

      def oomAtN: Int =
        var l = List.empty[Tensor[?, Float]]
        try while true do l = createLargeTensor() :: l
        catch case _: Exception => () // ignore OOM exception
        l.size + 1

      def block(n: Int, createTemporaryTensor: () => Tensor[?, Float]): Unit =
        // Each iteration: Create tensor ready for garbage collection
        for _ <- 0 until n do
          createTemporaryTensor()

      lazy val N = 2 * oomAtN // guaranteed to be above available memory

      it("Allocating too many large tensors", Slow):
        noException should be thrownBy (block(N, createLargeTensor))

      it("Allocating too many medium tensors", Slow):
        noException should be thrownBy (block(N * 100, createMediumTensor))

      it("Allocating too many small tensors", Slow):
        noException should be thrownBy (block(N * 1000, createSmallTensor))
