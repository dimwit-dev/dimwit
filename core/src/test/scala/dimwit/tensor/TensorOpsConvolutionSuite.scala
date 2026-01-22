package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import dimwit.tensor.TensorOps.Convolution.Padding
import org.scalatest.matchers.should.Matchers
import org.scalatest.funspec.AnyFunSpec
import dimwit.stats.Normal

class TensorOpsConvolutionSuite extends AnyFunSpec with Matchers:

  describe("Convolution 1D"):

    it("should perform 1D convolution with correct output shape"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Length] -> 10,
          Axis[InChannels] -> 3
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Length] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 4
        )
      ).fill(1f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv1d(kernel, stride = 1, padding = Padding.SAME))
      val output2 = inputBatched.vmap(Axis[Batch])(input => input.conv1d(kernel, stride = Axis[Length] -> 1, padding = Padding.SAME))

      // Output shape should be (batch=2, length=10, out_channels=4)
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Length]) shouldBe 10
      output.shape(Axis[OutChannels]) shouldBe 4

    it("should perform 1D convolution with stride > 1"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Length] -> 10,
          Axis[InChannels] -> 1
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Length] -> 3,
          Axis[InChannels] -> 1,
          Axis[OutChannels] -> 1
        )
      ).fill(1f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv1d(kernel, stride = 2, padding = Padding.VALID))

      output.shape(Axis[Batch]) shouldBe 1
      output.shape(Axis[OutChannels]) shouldBe 1
      // With VALID padding, output length should be reduced
      output.shape(Axis[Length]) should be < 10

  describe("Convolution 2D"):

    it("should perform 2D convolution with correct output shape"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input: (batch=2, height=8, width=8, in_channels=3)
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[InChannels] -> 3
        )
      ).fill(1f)

      // Kernel: (height=3, width=3, in_channels=3, out_channels=16)
      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv2d(kernel, stride = 1, padding = Padding.SAME))

      // Output shape should be (batch=2, height=8, width=8, out_channels=16)
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 16

    it("should perform 2D convolution with stride=2"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 16,
          Axis[Width] -> 16,
          Axis[InChannels] -> 3
        )
      ).fill(1.0f)

      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 8
        )
      ).fill(1.0f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv2d(kernel, stride = 2, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 1
      // With stride=2 and SAME padding, spatial dims should be halved
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 8

    it("should perform 2D convolution with stride=(1, 2)"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 16,
          Axis[Width] -> 16,
          Axis[InChannels] -> 3
        )
      ).fill(1.0f)

      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 4
        )
      ).fill(1.0f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv2d(kernel, stride = (Axis[Height] -> 1, Axis[Width] -> 2), padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 1
      output.shape(Axis[Height]) shouldBe 16
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 4

    it("should compute correct convolution values for 2D case"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Create a simple 3x3 input with known values (1 batch, 1 channel)
      // Pattern:
      // 1 2 3
      // 4 5 6
      // 7 8 9
      val inputShape = Shape(
        Axis[Batch] -> 1,
        Axis[Height] -> 3,
        Axis[Width] -> 3,
        Axis[InChannels] -> 1
      )
      val inputData = Array(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f)
      val inputBatched = Tensor(inputShape).fromArray(inputData)

      // Create a 2x2 kernel that sums all values (all ones)
      // When convolved, each output element will be the sum of a 2x2 window
      val kernelShape = Shape(
        Axis[Height] -> 2,
        Axis[Width] -> 2,
        Axis[InChannels] -> 1,
        Axis[OutChannels] -> 1
      )
      val kernelData = Array(1.0f, 1.0f, 1.0f, 1.0f)
      val kernel = Tensor(kernelShape).fromArray(kernelData)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv2d(kernel, stride = 1, padding = Padding.VALID))

      // With VALID padding and 2x2 kernel, output should be 2x2
      // Output values:
      // Top-left:    1+2+4+5 = 12
      // Top-right:   2+3+5+6 = 16
      // Bottom-left: 4+5+7+8 = 24
      // Bottom-right: 5+6+8+9 = 28
      output.shape(Axis[Batch]).shouldBe(1)
      output.shape(Axis[Height]).shouldBe(2)
      output.shape(Axis[Width]).shouldBe(2)
      output.shape(Axis[OutChannels]).shouldBe(1)

      val expectedShape = Shape(
        Axis[Batch] -> 1,
        Axis[Height] -> 2,
        Axis[Width] -> 2,
        Axis[OutChannels] -> 1
      )
      val expectedData = Array(12.0f, 16.0f, 24.0f, 28.0f)
      val expected = Tensor(expectedShape).fromArray(expectedData)

      (output === expected).item.shouldBe(true)

  describe("Convolution validation"):

    it("should require matching spatial labels"):
      trait A derives Label
      trait B derives Label
      trait In derives Label
      trait Out derives Label

      val input = Tensor(Shape(Axis[A] -> 3, Axis[In] -> 3)).fill(1.0f)
      // Kernel with different spatial label B instead of A won't compile
      val kernel = Tensor(Shape(Axis[B] -> 2, Axis[In] -> 3, Axis[Out] -> 1)).fill(1.0f)

      // This won't compile due to spatial label mismatch
      assertDoesNotCompile("input.conv(Axis[In], Axis[Out])(kernel)")

  describe("Transpose Convolution 1D"):

    it("should perform 1D transpose convolution with correct output shape"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input has OutChannels (transpose conv maps from output space to input space)
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Length] -> 10,
          Axis[OutChannels] -> 4
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Length] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 4
        )
      ).fill(1f)

      // transposeConv: OutChannels input -> InChannels output
      val output = inputBatched.vmap(Axis[Batch])(input => input.transposeConv1d(kernel, stride = 1, padding = Padding.SAME))

      // Output shape should be (batch=2, length=10, in_channels=3) with stride=1 and SAME padding
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Length]) shouldBe 10
      output.shape(Axis[InChannels]) shouldBe 3

    it("should perform 1D transpose convolution with stride > 1 (upsampling)"):
      trait Batch derives Label
      trait Length derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input has OutChannels
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Length] -> 5,
          Axis[OutChannels] -> 1
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Length] -> 3,
          Axis[InChannels] -> 1,
          Axis[OutChannels] -> 1
        )
      ).fill(1f)

      // With stride=2, output should be approximately 2x the input spatial dimension
      // transposeConv: OutChannels -> InChannels
      val output = inputBatched.vmap(Axis[Batch])(input => input.transposeConv1d(kernel, stride = 2, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 1
      output.shape(Axis[InChannels]) shouldBe 1
      // With SAME padding and stride=2, output length should be ~2x input
      output.shape(Axis[Length]) shouldBe 10

  describe("Transpose Convolution 2D"):

    it("should perform 2D transpose convolution with correct output shape"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input: (batch=2, height=8, width=8, out_channels=16) - transposeConv input space
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      // Kernel: (height=3, width=3, in_channels=3, out_channels=16)
      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      // transposeConv: OutChannels -> InChannels
      val output = inputBatched.vmap(Axis[Batch])(input => input.transposeConv2d(kernel, stride = 1, padding = Padding.SAME))

      // Output shape should be (batch=2, height=8, width=8, in_channels=3) with stride=1
      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[InChannels]) shouldBe 3

    it("should perform 2D transpose convolution with stride=2 (upsampling)"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input has OutChannels
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[OutChannels] -> 8
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 8
        )
      ).fill(1f)

      // transposeConv: OutChannels -> InChannels
      val output = inputBatched.vmap(Axis[Batch])(input => input.transposeConv2d(kernel, stride = 2, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 1
      // With stride=2 and SAME padding, spatial dims should be doubled
      output.shape(Axis[Height]) shouldBe 16
      output.shape(Axis[Width]) shouldBe 16
      output.shape(Axis[InChannels]) shouldBe 3

    it("should allow explicit output shape specification"):
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input has OutChannels
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 4,
          Axis[Width] -> 4,
          Axis[OutChannels] -> 1
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 1,
          Axis[OutChannels] -> 1
        )
      ).fill(1f)

      // With stride=2 and VALID padding, output size is determined by JAX
      // transposeConv: OutChannels -> InChannels
      val output = inputBatched.vmap(Axis[Batch])(input =>
        input.transposeConv2d(
          kernel,
          stride = 2,
          padding = Padding.VALID
        )
      )

      output.shape(Axis[Batch]) shouldBe 1
      // VALID padding with stride 2: output spatial dims are approximately 2*input - kernel + 1
      // For 4x4 input, 3x3 kernel, stride 2: roughly 9x9
      output.shape(Axis[Height]) should be >= 8
      output.shape(Axis[Width]) should be >= 8
      output.shape(Axis[InChannels]) shouldBe 1

  describe("Conv/TransposeConv Integration"):

    it("should satisfy dotproduct equality: <conv(x, k), y> = <x, transposeConv2d(y, k)>"):
      // This test verifies the mathematical property that transpose convolution
      // is the adjoint (transpose) of convolution in the sense of inner products
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      import dimwit.random.Random

      val key = Random.Key(42)

      // Create input tensor x
      val xBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[InChannels] -> 3
        )
      ).sample(key)

      // Create kernel k
      val kernel = Normal.standardNormal(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 5
        )
      ).sample(key)

      // Compute conv(x, k): produces output with OutChannels
      val convOutput = xBatched.vmap(Axis[Batch])(x => x.conv2d(kernel, stride = 1, padding = Padding.SAME))

      // Create y with same shape as conv output (has OutChannels)
      val yBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[OutChannels] -> 5
        )
      ).sample(key)

      // Compute transposeConv2d(y, k)
      // transposeConv takes OutChannels input and produces InChannels output
      val transposeConvOutput = yBatched.vmap(Axis[Batch])(y => y.transposeConv2d(kernel, stride = 1, padding = Padding.SAME))

      // Verify shapes: transposeConvOutput should match x's shape
      transposeConvOutput.shape shouldBe xBatched.shape

      // Compute inner products:
      // dotLeft = <conv(x, k), y> = sum(conv(x,k) * y)
      // dotRight = <x, transposeConv2d(y, k)> = sum(x * transposeConv2d(y,k))
      val dotLeft = (convOutput * yBatched).sum.item
      val dotRight = (xBatched * transposeConvOutput).sum.item

      // They should be equal (within numerical tolerance)
      Math.abs(dotLeft - dotRight) should be < 1e-3f

    it("should satisfy dotproduct equality with stride=2"):
      // Verify the adjoint property also holds with strided convolution
      trait Batch derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      import dimwit.random.Random

      val key = Random.Key(43)

      // Input x: 16x16
      val xBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 16,
          Axis[Width] -> 16,
          Axis[InChannels] -> 2
        )
      ).sample(key)

      val kernel = Normal.standardNormal(
        Shape(
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 2,
          Axis[OutChannels] -> 4
        )
      ).sample(key)

      // conv with stride=2 produces 8x8 output with OutChannels
      val convOutput = xBatched.vmap(Axis[Batch])(x => x.conv2d(kernel, stride = 2, padding = Padding.SAME))

      // y has same shape as conv output (8x8 with OutChannels)
      val yBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[OutChannels] -> 4
        )
      ).sample(key)

      // transposeConv with stride=2 upsamples back to 16x16
      // Takes OutChannels input, produces InChannels output
      val transposeConvOutput = yBatched.vmap(Axis[Batch])(y => y.transposeConv2d(kernel, stride = 2, padding = Padding.SAME))

      // Verify shapes: transposeConv output should match x's shape
      convOutput.shape(Axis[Height]) shouldBe 8
      transposeConvOutput.shape shouldBe xBatched.shape

      // Verify dotproduct equality: <conv(x,k), y> = <x, transposeConv2d(y,k)>
      val dotLeft = (convOutput * yBatched).sum.item
      val dotRight = (xBatched * transposeConvOutput).sum.item

      Math.abs(dotLeft - dotRight) should be < 1e-3f

  describe("Convolution 3D"):

    it("should perform 3D convolution with correct output shape"):
      trait Batch derives Label
      trait Depth derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input: (batch=2, depth=8, height=8, width=8, in_channels=3)
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Depth] -> 8,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[InChannels] -> 3
        )
      ).fill(1f)

      // Kernel: (depth=3, height=3, width=3, in_channels=3, out_channels=16)
      val kernel = Tensor(
        Shape(
          Axis[Depth] -> 3,
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv3d(kernel, stride = 1, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Depth]) shouldBe 8
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 16

    it("should perform 3D convolution with stride=2"):
      trait Batch derives Label
      trait Depth derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 1,
          Axis[Depth] -> 16,
          Axis[Height] -> 16,
          Axis[Width] -> 16,
          Axis[InChannels] -> 2
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Depth] -> 3,
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 2,
          Axis[OutChannels] -> 4
        )
      ).fill(1f)

      val output = inputBatched.vmap(Axis[Batch])(input => input.conv3d(kernel, stride = 2, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 1
      // With stride=2 and SAME padding, spatial dims should be halved
      output.shape(Axis[Depth]) shouldBe 8
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[OutChannels]) shouldBe 4

  describe("Transpose Convolution 3D"):

    it("should perform 3D transpose convolution with correct output shape"):
      trait Batch derives Label
      trait Depth derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      // Input: (batch=2, depth=8, height=8, width=8, out_channels=16)
      val inputBatched = Tensor(
        Shape(
          Axis[Batch] -> 2,
          Axis[Depth] -> 8,
          Axis[Height] -> 8,
          Axis[Width] -> 8,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      val kernel = Tensor(
        Shape(
          Axis[Depth] -> 3,
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 3,
          Axis[OutChannels] -> 16
        )
      ).fill(1f)

      // transposeConv: OutChannels -> InChannels
      val output = inputBatched.vmap(Axis[Batch])(input => input.transposeConv3d(kernel, stride = 1, padding = Padding.SAME))

      output.shape(Axis[Batch]) shouldBe 2
      output.shape(Axis[Depth]) shouldBe 8
      output.shape(Axis[Height]) shouldBe 8
      output.shape(Axis[Width]) shouldBe 8
      output.shape(Axis[InChannels]) shouldBe 3

    it("should satisfy dotproduct equality: <conv3d(x, k), y> = <x, transposeConv3d(y, k)>"):
      trait Batch derives Label
      trait Depth derives Label
      trait Height derives Label
      trait Width derives Label
      trait InChannels derives Label
      trait OutChannels derives Label

      import dimwit.random.Random

      val key = Random.Key(44)

      // Create input x
      val xBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Depth] -> 6,
          Axis[Height] -> 6,
          Axis[Width] -> 6,
          Axis[InChannels] -> 2
        )
      ).sample(key)

      // Create kernel k
      val kernel = Normal.standardNormal(
        Shape(
          Axis[Depth] -> 3,
          Axis[Height] -> 3,
          Axis[Width] -> 3,
          Axis[InChannels] -> 2,
          Axis[OutChannels] -> 4
        )
      ).sample(key)

      // Forward: conv3d(x, k)
      val convOutput = xBatched.vmap(Axis[Batch])(x => x.conv3d(kernel, stride = 1, padding = Padding.SAME))

      // Create y with same shape as convOutput (has OutChannels)
      val yBatched = Normal.standardNormal(
        Shape(
          Axis[Batch] -> 1,
          Axis[Depth] -> 6,
          Axis[Height] -> 6,
          Axis[Width] -> 6,
          Axis[OutChannels] -> 4
        )
      ).sample(key)

      // Backward/Transpose: transposeConv3d(y, k)
      val transposeConvOutput = yBatched.vmap(Axis[Batch])(y => y.transposeConv3d(kernel, stride = 1, padding = Padding.SAME))

      // Verify shapes
      transposeConvOutput.shape shouldBe xBatched.shape

      // Check adjoint property
      val dotLeft = (convOutput * yBatched).sum.item
      val dotRight = (xBatched * transposeConvOutput).sum.item

      Math.abs(dotLeft - dotRight) should be < 1e-3f
