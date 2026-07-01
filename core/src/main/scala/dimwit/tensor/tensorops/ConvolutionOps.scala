package dimwit.tensor.tensorops

import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.AxisExtent
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.Tensor
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.TensorOps.swap
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Writer

/** Provides extension methods for convolution operations on tensors.
  * Convolution operations are restricted to 1D, 2D and 3D convolutions,
  * and support both standard and transposed convolutions.
  */
object ConvolutionOps:

  /** Padding options for convolution operations.
    * SAME: Output size is the same as input size (with appropriate padding).
    * VALID: No padding, output size is reduced based on kernel size.
    *
    * Refer to JAX documentation for more details on padding behavior.
    * https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.conv_general_dilated.html
    */
  enum Padding:
    case SAME, VALID

  type Stride1[S1] = AxisExtent[S1]

  extension [S1: Label, InChannel: Label, V: IsFloating](input: Tensor[S1 *: InChannel *: EmptyTuple, V])

    /** Computes the 1D convolution of this tensor with the specified kernel tensor.
      *
      * @param kernel - The convolution kernel
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the convolution operation.
      */
    def conv1d[OutChannel: Label](
        kernel: Tensor[(S1, InChannel, OutChannel), V],
        stride: Stride1[S1] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: OutChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[InChannel]) == kernel.shape(Axis[InChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[InChannel])} channels, kernel expects ${kernel.shape(Axis[InChannel])} channels"
      )
      val strides = stride match
        case s: Int             => Seq(s)
        case ae: AxisExtent[S1] => Seq(ae.size)
      // JAX requires input and kernel to have same rank, so we must add (and remove) dummy dim to input.
      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_general_dilated(
        lhs = batchInput,
        rhs = kernel.jaxValue,
        window_strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NHC", "HIO", "NHC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)

  extension [S1: Label, OutChannel: Label, V: IsFloating](input: Tensor[S1 *: OutChannel *: EmptyTuple, V])

    /** Computes the transposed 1D convolution of
      * this tensor with the specified kernel tensor.
      * @param kernel - The convolution kernel
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the transposed convolution operation.
      */
    def transposeConv1d[InChannel: Label](
        kernel: Tensor[S1 *: InChannel *: OutChannel *: EmptyTuple, V],
        stride: Stride1[S1] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: InChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[OutChannel]) == kernel.shape(Axis[OutChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[OutChannel])} channels (OutChannel), kernel expects ${kernel.shape(Axis[OutChannel])}"
      )
      val strides = stride match
        case s: Int             => Seq(s)
        case ex: AxisExtent[S1] => Seq(ex.size)

      // kernel -> kernal adjoint: swap in/out channels and flip spatial dims
      var kernelAdjoint = kernel.swap(Axis[InChannel], Axis[OutChannel]).jaxValue
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 0) // flip S1

      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_transpose(
        lhs = batchInput,
        rhs = kernelAdjoint,
        strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NHC", "HIO", "NHC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)

  type Stride2[S1, S2] = (AxisExtent[S1], AxisExtent[S2])

  extension [S1: Label, S2: Label, InChannel: Label, V: IsFloating](input: Tensor[S1 *: S2 *: InChannel *: EmptyTuple, V])

    /** Computes the 2D convolution of this tensor with the specified kernel tensor.
      *
      * @param kernel - The convolution kernel tensor with shape (S1, S2, InChannel, OutChannel).
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the convolution operation.
      */
    def conv2d[OutChannel: Label](
        kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, V],
        stride: Stride2[S1, S2] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[InChannel]) == kernel.shape(Axis[InChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[InChannel])} channels, kernel expects ${kernel.shape(Axis[InChannel])} channels"
      )
      val strides = stride match
        case s: Int     => Seq(s, s)
        case (ae1, ae2) => Seq(ae1.size, ae2.size)
      // JAX requires input and kernel to have same rank, so we must add (and remove) dummy dim to input.
      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_general_dilated(
        lhs = batchInput,
        rhs = kernel.jaxValue,
        window_strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NHWC", "HWIO", "NHWC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)

  extension [S1: Label, S2: Label, OutChannel: Label, V: IsFloating](input: Tensor[S1 *: S2 *: OutChannel *: EmptyTuple, V])

    /** Computes the transposed 2D convolution of this tensor with the specified kernel tensor.
      *
      * @param kernel - The convolution kernel tensor with shape (S1, S2, InChannel, OutChannel).
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the transposed convolution operation.
      */
    def transposeConv2d[InChannel: Label](
        kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, V],
        stride: Stride2[S1, S2] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: S2 *: InChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[OutChannel]) == kernel.shape(Axis[OutChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[OutChannel])} channels (OutChannel), kernel expects ${kernel.shape(Axis[OutChannel])}"
      )

      // JAX requires input and kernel to have same rank. Add dummy batch dim if needed.
      val strides = stride match
        case s: Int     => Seq(s, s)
        case (ae1, ae2) => Seq(ae1.size, ae2.size)

      // kernel -> kernal adjoint: swap in/out channels and flip spatial dims
      var kernelAdjoint = kernel.swap(Axis[InChannel], Axis[OutChannel]).jaxValue
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 0) // flip S1
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 1) // flip S2

      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_transpose(
        lhs = batchInput,
        rhs = kernelAdjoint,
        strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NHWC", "HWIO", "NHWC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)

  type Stride3[S1, S2, S3] = (AxisExtent[S1], AxisExtent[S2], AxisExtent[S3])

  extension [S1: Label, S2: Label, S3: Label, InChannel: Label, V: IsFloating](input: Tensor[S1 *: S2 *: S3 *: InChannel *: EmptyTuple, V])

    /** Computes the 3D convolution of this tensor with the specified kernel tensor.
      *
      * @param kernel - The convolution kernel tensor
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the convolution operation.
      */
    def conv3d[OutChannel: Label](
        kernel: Tensor[S1 *: S2 *: S3 *: InChannel *: OutChannel *: EmptyTuple, V],
        stride: Stride3[S1, S2, S3] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: S2 *: S3 *: OutChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[InChannel]) == kernel.shape(Axis[InChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[InChannel])} channels, kernel expects ${kernel.shape(Axis[InChannel])} channels"
      )
      val strides = stride match
        case s: Int             => Seq(s, s, s)
        case (dim1, dim2, dim3) => Seq(dim1.size, dim2.size, dim3.size)

      // JAX requires input and kernel to have same rank, so we must add (and remove) dummy dim to input.
      // 3D Layout: NDHWC (Batch, Depth, Height, Width, Channel)
      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_general_dilated(
        lhs = batchInput,
        rhs = kernel.jaxValue,
        window_strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NDHWC", "DHWIO", "NDHWC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)

  extension [S1: Label, S2: Label, S3: Label, OutChannel: Label, V: IsFloating](input: Tensor[S1 *: S2 *: S3 *: OutChannel *: EmptyTuple, V])

    /** Computes the transposed 3D convolution of this tensor with the specified kernel tensor.
      *
      * @param kernel - The convolution kernel tensor
      * @param stride - Stride for the convolution.
      * @param padding - Padding mode for the convolution.
      * @return A new tensor representing the result of the transposed convolution operation.
      */
    def transposeConv3d[InChannel: Label](
        kernel: Tensor[S1 *: S2 *: S3 *: InChannel *: OutChannel *: EmptyTuple, V],
        stride: Stride3[S1, S2, S3] | Int = 1,
        padding: Padding = Padding.SAME
    ): Tensor[S1 *: S2 *: S3 *: InChannel *: EmptyTuple, V] =
      require(
        input.shape(Axis[OutChannel]) == kernel.shape(Axis[OutChannel]),
        s"Input channels mismatch: input has ${input.shape(Axis[OutChannel])} channels (OutChannel), kernel expects ${kernel.shape(Axis[OutChannel])}"
      )

      val strides = stride match
        case s: Int          => Seq(s, s, s)
        case (ae1, ae2, ae3) => Seq(ae1.size, ae2.size, ae3.size)

      // kernel -> kernel adjoint: swap in/out channels and flip all spatial dims
      var kernelAdjoint = kernel.swap(Axis[InChannel], Axis[OutChannel]).jaxValue
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 0) // flip S1 (Depth)
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 1) // flip S2 (Height)
      kernelAdjoint = Jax.jnp.flip(kernelAdjoint, axis = 2) // flip S3 (Width)

      val batchInput = Jax.jnp.expand_dims(input.jaxValue, axis = 0) // add dummy dim
      val convResult = Jax.lax.conv_transpose(
        lhs = batchInput,
        rhs = kernelAdjoint,
        strides = strides.toPythonProxy,
        padding = padding.toString,
        dimension_numbers = py.Dynamic.global.tuple(Seq("NDHWC", "DHWIO", "NDHWC").toPythonProxy)
      )
      val unbatchedRes = Jax.jnp.squeeze(convResult, axis = 0) // remove dummy dim
      Tensor(unbatchedRes)
