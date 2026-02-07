package dimwit.tensor

import scala.annotation.targetName
import scala.annotation.implicitNotFound
import scala.util.NotGiven

import dimwit.jax.{Jax, Einops}
import dimwit.tensor.{Label, Labels}
import dimwit.tensor.TupleHelpers.{Subset, StrictSubset, PrimeConcat}
import dimwit.tensor.ShapeTypeHelpers.{AxisRemover, AxesRemover, SharedAxisRemover, AxisReplacer, AxesConditionalRemover, WrapAxes, UnwrapAxes}
import dimwit.{~, `|*|`, `|+|`}

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import me.shadaj.scalapy.readwrite.Writer
import me.shadaj.scalapy.readwrite.Reader

import scala.compiletime.ops.int.<=
import dimwit.tensor.TupleHelpers.{ValidationResult, CanForm, IsPermutation, ComputeMissing, CheckValid, AllOk, MissingAxis}
import dimwit.tensor.ShapeTypeHelpers.UnwrapDims
import dimwit.tensor.ShapeTypeHelpers.DimExtractor
import dimwit.tensor.ShapeTypeHelpers.AxisReplacerAll
import dimwit.tensor.ShapeTypeHelpers.AxisIndex
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.ShapeTypeHelpers.AxesMerger
import dimwit.OnError

import Tuple.:*
import Tuple.++
import dimwit.tensor.ShapeTypeHelpers.MergeLabels

object TensorOps:

  import TensorOpsUtil.*

  @implicitNotFound("Operation only valid for Numeric (Int or Float) tensors.")
  sealed trait IsNumber[V]

  // -----------------------------------------------------------
  // Typeclasses to steer operation availability to prevent runtime errors
  // -----------------------------------------------------------

  @implicitNotFound("Operation only valid for Int or Float tensors.")
  object IsNumber:
    given [V](using ev1: IsFloating[V]): IsNumber[V] = ev1
    given [V](using ev2: IsInteger[V]): IsNumber[V] = ev2

  @implicitNotFound("Operation only valid for Floating tensors.")
  trait IsFloating[V] extends IsNumber[V]
  object IsFloating:
    given IsFloating[Float] with {}

  @implicitNotFound("Operation only valid for Integer tensors.")
  trait IsInteger[V] extends IsNumber[V]
  object IsInteger:
    given IsInteger[Int] with {}

  @implicitNotFound("Operation only valid for Boolean tensors.")
  sealed trait IsBoolean[V]
  object IsBoolean:
    given IsBoolean[Boolean] with {}

  // -----------------------------------------------------------
  // 1. Elementwise Operations (The Field)
  // Preserves Shape: T -> T
  // -----------------------------------------------------------
  object Elementwise:

    // ---------------------------------------------------------
    // General operations
    // ---------------------------------------------------------

    def maximum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))
    def minimum[T <: Tuple: Labels, V](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple: Labels, V](t: Tensor[T, V])

      // --- Comparison ---
      def <(other: Tensor[T, V]): Tensor[T, Boolean] = Tensor(Jax.jnp.less(t.jaxValue, other.jaxValue))
      def <=(other: Tensor[T, V]): Tensor[T, Boolean] = Tensor(Jax.jnp.less_equal(t.jaxValue, other.jaxValue))
      def >(other: Tensor[T, V]): Tensor[T, Boolean] = Tensor(Jax.jnp.greater(t.jaxValue, other.jaxValue))
      def >=(other: Tensor[T, V]): Tensor[T, Boolean] = Tensor(Jax.jnp.greater_equal(t.jaxValue, other.jaxValue))
      def ===(other: Tensor[T, V]): Tensor0[Boolean] = Tensor0(Jax.jnp.array_equal(t.jaxValue, other.jaxValue))

      def elementEquals(other: Tensor[T, V]): Tensor[T, Boolean] =
        require(t.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${t.shape.dimensions} vs ${other.shape.dimensions}")
        Tensor(jaxValue = Jax.jnp.equal(t.jaxValue, other.jaxValue))

      def asBoolean: Tensor[T, Boolean] = t.asType(VType[Boolean])
      def asInt: Tensor[T, Int] = t.asType(VType[Int])
      def asFloat: Tensor[T, Float] = t.asType(VType[Float])

    // ---------------------------------------------------------
    // IsNumber operations (IsFloat or IsInt)
    // ---------------------------------------------------------

    def add[T <: Tuple: Labels, T1 <: T, T2 <: T, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))
    def addScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))

    def negate[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))
    def subtract[T <: Tuple: Labels, T1 <: T, T2 <: T, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t1.jaxValue, t2.jaxValue))
    def subtractScalar[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, t2.jaxValue))

    def multiply[T <: Tuple: Labels, T1 <: T, T2 <: T, V: IsNumber](t1: Tensor[T1, V], t2: Tensor[T2, V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))
    def multiplyScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple: Labels, T1 <: T, T2 <: T, V: IsNumber](t: Tensor[T1, V])

      def +(other: Tensor[T2, V]): Tensor[T, V] = add(t, other)
      def -(other: Tensor[T2, V]): Tensor[T, V] = subtract(t, other)
      def *(other: Tensor[T2, V]): Tensor[T, V] = multiply(t, other)

    extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

      def +![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(add)

      def unary_- : Tensor[T, V] = negate(t)
      def -![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(subtract)

      def *![O <: Tuple](other: Tensor[O, V])(using bc: Broadcast[T, O, V]): Tensor[bc.Out, V] = bc.applyTo(t, other)(multiply)
      def scale(other: Tensor0[V]): Tensor[T, V] = multiplyScalar(t, other)

      def abs: Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))
      def sign: Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))
      def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))
      def pow(n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

    // ---------------------------------------------------------
    // IsFloat operations
    // ---------------------------------------------------------

    def divide[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))
    def divideScalar[T <: Tuple: Labels, V: IsFloating](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

      def /(other: Tensor[T, V]): Tensor[T, V] = divide(t, other)
      def /![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = join.applyTo(t, other)(divide)

      def sqrt: Tensor[T, V] = Tensor(Jax.jnp.sqrt(t.jaxValue))
      def exp: Tensor[T, V] = Tensor(Jax.jnp.exp(t.jaxValue))
      def log: Tensor[T, V] = Tensor(Jax.jnp.log(t.jaxValue))
      def sin: Tensor[T, V] = Tensor(Jax.jnp.sin(t.jaxValue))
      def cos: Tensor[T, V] = Tensor(Jax.jnp.cos(t.jaxValue))
      def tanh: Tensor[T, V] = Tensor(Jax.jnp.tanh(t.jaxValue))

      def approxEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor0[Boolean] = approxElementEquals(other, tolerance).all
      def approxElementEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor[T, Boolean] =
        Tensor(
          Jax.jnp.allclose(
            t.jaxValue,
            other.jaxValue,
            atol = tolerance,
            rtol = tolerance
          )
        )

    // ---------------------------------------------------------
    // IsBoolean operations
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V: IsBoolean](t: Tensor[T, V])

      def all: Tensor0[Boolean] = Tensor0(Jax.jnp.all(t.jaxValue))
      def any: Tensor0[Boolean] = Tensor0(Jax.jnp.any(t.jaxValue))

      def unary_! : Tensor[T, Boolean] = Tensor(Jax.jnp.logical_not(t.jaxValue))

  end Elementwise

  // -----------------------------------------------------------
  // 2. Reduction Operations (The Monoid)
  // Reduces Rank: T -> T - {Axis}
  // -----------------------------------------------------------
  object Reduction:

    // ---------------------------------------------------------
    // IsNumber operations (IsFloat or IsInt)
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

      // --- Sum ---
      def sum: Tensor0[V] = Tensor0(Jax.jnp.sum(t.jaxValue))
      def sum[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.index))
      def sum[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Max ---
      def max: Tensor0[V] = Tensor0(Jax.jnp.max(t.jaxValue))
      def max[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.index))
      def max[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Min ---
      def min: Tensor0[V] = Tensor0(Jax.jnp.min(t.jaxValue))
      def min[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.index))
      def min[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Argmax ---
      def argmax: Tensor0[Int] = Tensor0(Jax.jnp.argmax(t.jaxValue))
      def argmax[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, Int] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.index))
      def argmax[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, Int] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Argmin ---
      def argmin: Tensor0[Int] = Tensor0(Jax.jnp.argmin(t.jaxValue))
      def argmin[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, Int] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.index))
      def argmin[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, Int] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = ev.indices.toPythonProxy))

    // ---------------------------------------------------------
    // IsFloat operations (IsFloat or IsInt)
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

      // --- Mean ---
      def mean: Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))
      def mean[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.index))
      def mean[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Std ---
      def std: Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))
      def std[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.index))
      def std[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Quantile ---
      def quantile(q: Float): Tensor0[V] = Tensor0(Jax.jnp.quantile(t.jaxValue, q))
      def quantile[L: Label, R <: Tuple](q: Float, axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.index))
      def quantile[Inputs <: Tuple, R <: Tuple](q: Float, axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.quantile(t.jaxValue, q, axis = ev.indices.toPythonProxy))

      // --- Median ---
      def median: Tensor0[V] = Tensor0(Jax.jnp.median(t.jaxValue))
      def median[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.index))
      def median[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.median(t.jaxValue, axis = ev.indices.toPythonProxy))

  end Reduction

  object Contraction:

    extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])

      def outerProduct[OtherShape <: Tuple: Labels, Out <: Tuple](other: Tensor[OtherShape, V])(using
          primeConcat: PrimeConcat.Aux[T, OtherShape, Out],
          labels: Labels[Out]
      ): Tensor[Out, V] =
        Tensor(
          // Jax outer product flattens, reshape required
          Jax.jnp.reshape(
            Jax.jnp.outer(tensor.jaxValue, other.jaxValue),
            (tensor.shape.dimensions ++ other.shape.dimensions).toPythonProxy
          )
        )

      def dot[
          ContractAxis,
          OtherShape <: Tuple,
          R1 <: Tuple,
          R2 <: Tuple,
          Out <: Tuple
      ](axis: Axis[ContractAxis])(other: Tensor[OtherShape, V])(using
          ev: AxisRemover[T, ContractAxis, R1],
          evOther: AxisRemover[OtherShape, ContractAxis, R2],
          primeConcat: PrimeConcat.Aux[R1, R2, Out],
          labelsOut: Labels[Out]
      ): Tensor[Out, V] =
        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

        Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))

      @targetName("dotOn")
      def dot[
          ContractAxisA,
          ContractAxisB,
          OtherShape <: Tuple,
          R1 <: Tuple,
          R2 <: Tuple,
          Out <: Tuple
      ](axis: Axis[ContractAxisA ~ ContractAxisB])(other: Tensor[OtherShape, V])(using
          ev: AxisRemover[T, ContractAxisA, R1],
          evOther: AxisRemover[OtherShape, ContractAxisB, R2],
          primeConcat: PrimeConcat.Aux[R1, R2, Out],
          outLabels: Labels[Out]
      ): Tensor[Out, V] =
        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(ev.index).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(evOther.index).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

        Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))

  end Contraction

  object Convolution:

    enum Padding:
      case SAME, VALID

    type Stride1[S1] = AxisExtent[S1]

    extension [S1: Label, InChannel: Label, V: IsFloating](input: Tensor[S1 *: InChannel *: EmptyTuple, V])

      def conv1d[OutChannel: Label](
          kernel: Tensor[S1 *: InChannel *: OutChannel *: EmptyTuple, Float],
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

      def transposeConv1d[InChannel: Label](
          kernel: Tensor[S1 *: InChannel *: OutChannel *: EmptyTuple, Float],
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

      def conv2d[OutChannel: Label](
          kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, Float],
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

      def transposeConv2d[InChannel: Label](
          kernel: Tensor[S1 *: S2 *: InChannel *: OutChannel *: EmptyTuple, Float],
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

      def conv3d[OutChannel: Label](
          kernel: Tensor[S1 *: S2 *: S3 *: InChannel *: OutChannel *: EmptyTuple, Float],
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

      def transposeConv3d[InChannel: Label](
          kernel: Tensor[S1 *: S2 *: S3 *: InChannel *: OutChannel *: EmptyTuple, Float],
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
  end Convolution

  object LinearAlgebra:

    // ---------------------------------------------------------
    // General operations
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V](t: Tensor[T, V])

      def diagonal[L1: Label, L2: Label, R <: Tuple](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
          ev: AxesRemover[T, (L1, L2), R],
          labels: Labels[R]
      ): Tensor[R *: L1 *: EmptyTuple, V] =
        Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

    extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

      def diagonal: Tensor1[L1, V] = t.diagonal(0)
      def diagonal(offset: Int): Tensor1[L1, V] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

    // ---------------------------------------------------------
    // IsNumber operations (IsFloat or IsInt)
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

      def trace[L1: Label, L2: Label, R <: Tuple](axis1: Axis[L1], axis2: Axis[L2], offset: Int = 0)(using
          ev: AxesRemover[T, (L1, L2), R],
          labels: Labels[R]
      ): Tensor[R, V] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = ev.indices(0), axis2 = ev.indices(1)))

    extension [L1: Label, L2: Label, V: IsNumber](t: Tensor2[L1, L2, V])

      def trace: Tensor0[V] = t.trace(0)
      def trace(offset: Int): Tensor0[V] = Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))

    // ---------------------------------------------------------
    // IsFloat operations
    // ---------------------------------------------------------

    extension [T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])

      def norm: Tensor0[V] = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))
      def inv: Tensor[T, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))
      def det[L1: Label, L2: Label, R <: Tuple](axis1: Axis[L1], axis2: Axis[L2])(using
          ev: AxesRemover[T, (L1, L2), R],
          labels: Labels[R]
      ): Tensor[R, V] =
        // JAX det only works on the last two axes (-2, -1). We must move the user's selected axes to the end.
        val moved = Jax.jnp.moveaxis(
          t.jaxValue,
          source = ev.indices.toPythonProxy,
          destination = Seq(-2, -1).toPythonProxy
        )
        Tensor(Jax.jnp.linalg.det(moved))

    extension [L1: Label, L2: Label, V: IsFloating](t: Tensor2[L1, L2, V])

      def det: Tensor0[V] = Tensor0(Jax.jnp.linalg.det(t.jaxValue))

  end LinearAlgebra

  // -----------------------------------------------------------
  // 4. Structural Operations (Isomorphisms)
  // Permutations and Views: T1 -> T2 (Size(T1) == Size(T2))
  // -----------------------------------------------------------
  object Structural:

    private object Util:

      type InsertBefore[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail  => B *: A *: tail
        case h *: tail  => h *: InsertBefore[tail, A, B]

      type InsertAfter[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail  => A *: B *: tail
        case h *: tail  => h *: InsertAfter[tail, A, B]

      type SliceIndex = Int | List[Int] | Range | Tensor0[Int]
      type ExtractLabel[X] = X match
        case AxisAtIndex[l]       => l
        case AxisAtRange[l]       => l
        case AxisAtIndices[l]     => l
        case AxisAtTensorIndex[l] => l
      type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

      trait SliceLabelExtractor[Inputs <: Tuple, Out <: Tuple]

      object SliceLabelExtractor:

        given empty: SliceLabelExtractor[EmptyTuple, EmptyTuple] =
          new SliceLabelExtractor[EmptyTuple, EmptyTuple] {}

        // New givens for AxisSelector types
        given consAxisAtIndex[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[AxisAtIndex[L] *: Tail, L *: TailOut] =
          new SliceLabelExtractor[AxisAtIndex[L] *: Tail, L *: TailOut] {}

        given consAxisAtRange[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[AxisAtRange[L] *: Tail, TailOut] =
          new SliceLabelExtractor[AxisAtRange[L] *: Tail, TailOut] {}

        given consAxisAtIndices[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[AxisAtIndices[L] *: Tail, TailOut] =
          new SliceLabelExtractor[AxisAtIndices[L] *: Tail, TailOut] {}

        given consAxisAtTensorIndex[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[AxisAtTensorIndex[L] *: Tail, L *: TailOut] =
          new SliceLabelExtractor[AxisAtTensorIndex[L] *: Tail, L *: TailOut] {}

        // Keep backward compatibility with tuple syntax
        given consInt[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] =
          new SliceLabelExtractor[(Axis[L], Int) *: Tail, L *: TailOut] {}

        given consTensor0Int[L, Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[(Axis[L], Tensor0[Int]) *: Tail, L *: TailOut] =
          new SliceLabelExtractor[(Axis[L], Tensor0[Int]) *: Tail, L *: TailOut] {}

        given consSeq[L, SeqT <: Seq[Int], Tail <: Tuple, TailOut <: Tuple](using
            tailExt: SliceLabelExtractor[Tail, TailOut]
        ): SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] =
          new SliceLabelExtractor[(Axis[L], SeqT) *: Tail, TailOut] {}

      type Swap[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => EmptyTuple
        case A *: tail  => B *: Swap[tail, A, B]
        case B *: tail  => A *: Swap[tail, A, B]
        case h *: tail  => h *: Swap[tail, A, B]

      @implicitNotFound("The axis ${L} is already present in the tensor shape ${T}.")
      trait AxisAbsent[T, L]
      object AxisAbsent:
        given [T <: Tuple, L](using NotGiven[Tuple.Contains[T, L] =:= true]): AxisAbsent[T, L] = new AxisAbsent[T, L] {}

    import Util.*

    object TensorWhere:
      def where[T <: Tuple: Labels, V](
          condition: Tensor[T, Boolean],
          x: Tensor[T, V],
          y: Tensor[T, V]
      ): Tensor[T, V] =
        Tensor(Jax.jnp.where(condition.jaxValue, x.jaxValue, y.jaxValue))

    export TensorWhere.where

    def triu[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
      Tensor(Jax.jnp.triu(tensor.jaxValue, k = kthDiagonal))

    def tril[T <: Tuple: Labels, V](tensor: Tensor[T, V], kthDiagonal: Int = 0): Tensor[T, V] =
      Tensor(Jax.jnp.tril(tensor.jaxValue, k = kthDiagonal))

    def stack[L: Label, T <: Tuple: Labels, V](
        tensors: Seq[Tensor[T, V]],
        newAxis: Axis[L]
    ): Tensor[L *: T, V] =
      require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = 0)
      Tensor(stackedJaxValue)

    def stack[NewL, L, T <: Tuple: Labels, V](
        tensors: Seq[Tensor[T, V]],
        newAxis: Axis[NewL],
        afterAxis: Axis[L]
    )(using
        newLabel: Label[NewL],
        axisIndex: AxisIndex[T, L]
    ): Tensor[InsertAfter[T, L, NewL], V] =
      require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
      val axisIdx = axisIndex.index + 1 // we are inserting after the given axis, so shift by 1
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = axisIdx)
      val names = summon[Labels[T]].names
      val newNames = names.take(axisIdx) ++ Seq(newLabel.name) ++ names.drop(axisIdx)
      given Labels[InsertAfter[T, L, NewL]] with
        val names = newNames.toSeq
      Tensor(stackedJaxValue)

    def concatenate[L: Label, T <: Tuple: Labels, V](
        tensors: Seq[Tensor[T, V]],
        concatAxis: Axis[L]
    )(using
        axisIndex: AxisIndex[T, L]
    ): Tensor[T, V] =
      require(tensors.nonEmpty, "Cannot concatenate an empty sequence of tensors")
      val axisIdx = axisIndex.index
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
      Tensor(concatenatedJaxValue)

    def concatenate[L: Label, T <: Tuple: Labels, V](
        t1: Tensor[T, V],
        t2: Tensor[T, V],
        concatAxis: Axis[L]
    )(using
        axisIndex: AxisIndex[T, L]
    ): Tensor[T, V] = concatenate(Seq(t1, t2), concatAxis)

    trait ValidConcat[T1 <: Tuple, T2 <: Tuple]:
      type Out <: Tuple
      def index: Int

    object ValidConcat:
      type Aux[T1 <: Tuple, T2 <: Tuple, O <: Tuple] = ValidConcat[T1, T2] { type Out = O }

      given recursive[H, T1Tail <: Tuple, T2Tail <: Tuple, OutTail <: Tuple](using
          next: ValidConcat.Aux[T1Tail, T2Tail, OutTail]
      ): ValidConcat[H *: T1Tail, H *: T2Tail] with
        type Out = H *: OutTail
        def index: Int = next.index + 1

      given concatAxis[H1, H2, Tail <: Tuple](using
          isDifferent: NotGiven[H1 =:= H2]
      ): ValidConcat[H1 *: Tail, H2 *: Tail] with
        type Out = (H1 |+| H2) *: Tail
        def index: Int = 0

    def concatenate[T1 <: Tuple, T2 <: Tuple, V, R <: Tuple](
        t1: Tensor[T1, V],
        t2: Tensor[T2, V]
    )(using
        canConcat: ValidConcat.Aux[T1, T2, R],
        label: Labels[R]
    ): Tensor[R, V] =
      val jaxValues = List(t1.jaxValue, t2.jaxValue).toPythonProxy
      Tensor(Jax.jnp.concatenate(jaxValues, axis = canConcat.index))

    trait Deconcatenator[L]:
      type Components <: Tuple
      def labels: List[Label[?]]

    object Deconcatenator extends DeconcatenatorLowPriority:
      type Aux[L, C <: Tuple] = Deconcatenator[L] { type Components = C }

      given recursive[A, B, CA <: Tuple, CB <: Tuple](using
          da: Aux[A, CA],
          db: Aux[B, CB]
      ): Aux[A |+| B, Tuple.Concat[CA, CB]] =
        new Deconcatenator[A |+| B]:
          type Components = Tuple.Concat[CA, CB]
          def labels = da.labels ++ db.labels

    trait DeconcatenatorLowPriority:
      given base[L](using l: Label[L]): Deconcatenator.Aux[L, L *: EmptyTuple] =
        new Deconcatenator[L]:
          type Components = L *: EmptyTuple
          def labels = List(l)

    trait TensorTupleMaker[Components <: Tuple, FullShape <: Tuple, SplitAxis, V]:
      type Out <: Tuple
      def apply(arrays: Seq[Jax.PyDynamic], compLabels: List[Label[?]], originalLabels: Seq[String], splitIndex: Int): Out

    object TensorTupleMaker:
      type Aux[C <: Tuple, F <: Tuple, S, V, O <: Tuple] =
        TensorTupleMaker[C, F, S, V] { type Out = O }

      given empty[F <: Tuple, S, V]: Aux[EmptyTuple, F, S, V, EmptyTuple] =
        new TensorTupleMaker[EmptyTuple, F, S, V]:
          type Out = EmptyTuple
          def apply(a: Seq[Jax.PyDynamic], c: List[Label[?]], o: Seq[String], i: Int) = EmptyTuple

      given cons[Head, Tail <: Tuple, F <: Tuple, S, V, NewShape <: Tuple](using
          replacer: TupleHelpers.Replacer[F, S, Head] { type Out = NewShape },
          tailMaker: TensorTupleMaker[Tail, F, S, V]
      ): Aux[Head *: Tail, F, S, V, Tensor[NewShape, V] *: tailMaker.Out] =

        new TensorTupleMaker[Head *: Tail, F, S, V]:
          type Out = Tensor[NewShape, V] *: tailMaker.Out

          def apply(arrays: Seq[Jax.PyDynamic], compLabels: List[Label[?]], originalLabels: Seq[String], splitIndex: Int): Out =
            val currentArr = arrays.head
            val currentLabel = compLabels.head
            val newNames = originalLabels.updated(splitIndex, currentLabel.name).toList
            val newLabelsWitness = new Labels[NewShape]:
              val names = newNames
            val headTensor = Tensor[NewShape, V](currentArr)(using newLabelsWitness)
            headTensor *: tailMaker(arrays.tail, compLabels.tail, originalLabels, splitIndex)

    extension [T <: Tuple, V](tensor: Tensor[T, V])

      def deconcatenate[L, Dims <: Tuple, Comps <: Tuple, Result](
          axis: Axis[L],
          dims: Dims
      )(using
          labels: Labels[T],
          axisIndex: AxisIndex[T, L],
          decon: Deconcatenator.Aux[L, Comps],
          extractor: DimExtractor[Dims],
          maker: TensorTupleMaker[Comps, T, L, V]
      ): maker.Out =
        val orderedSizes = dims.toList.asInstanceOf[List[Any]].map {
          case ae: AxisExtent[?] => ae.size
          case _                 => throw new IllegalArgumentException("Invalid dims format - expected AxisExtent")
        }

        require(orderedSizes.size == decon.labels.size, s"Provided ${orderedSizes.size} sizes but axis has ${decon.labels.size} components")

        val splitIndices = orderedSizes.scanLeft(0)(_ + _).tail.init
        val pyIndices = me.shadaj.scalapy.py.Dynamic.global.list(splitIndices.toPythonProxy)
        val splitArrays = Jax.jnp.split(tensor.jaxValue, pyIndices, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
        val originalNames = summon[Labels[T]].names.toSeq

        maker.apply(splitArrays, decon.labels, originalNames, axisIndex.index)

      private def calcPyIndices[Inputs <: Tuple](
          inputs: Inputs,
          targetDims: List[Int]
      ) =

        val PySlice = py.Dynamic.global.slice
        val Colon = PySlice(py.None)
        val rank = tensor.shape.rank
        val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

        val inputList = inputs.toList.asInstanceOf[List[Any]]

        targetDims.zip(inputList).foreach { case (dimIndex, input) =>
          val dimSize = tensor.shape.dimensions(dimIndex)
          input match
            // New AxisSelector types
            case AxisAtIndex(_, idx) =>
              indicesBuffer(dimIndex) = py.Any.from(idx)
            case AxisAtRange(_, range) =>
              indicesBuffer(dimIndex) = PySlice(range.head, range.last + 1, range.step)
            case AxisAtIndices(_, indices) =>
              indicesBuffer(dimIndex) = indices.map(py.Any.from).toPythonProxy
            case AxisAtTensorIndex(_, tensorIdx) =>
              indicesBuffer(dimIndex) = tensorIdx.jaxValue
            // Backward compatibility with tuples
            case (_, sliceIndex) =>
              sliceIndex match
                case sliceSeq: List[Int] @unchecked =>
                  indicesBuffer(dimIndex) = sliceSeq.map(py.Any.from).toPythonProxy
                case range: Range @unchecked =>
                  indicesBuffer(dimIndex) = PySlice(range.head, range.last + 1, range.step)
                case idx: Int =>
                  indicesBuffer(dimIndex) = py.Any.from(idx)
                case tensorId: Tensor0[Int] @unchecked =>
                  indicesBuffer(dimIndex) = tensorId.jaxValue
        }

        Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)

      def unflatten[SplitL, NewT <: Tuple, R <: Tuple](
          splitAxis: Axis[SplitL],
          newShape: Shape[NewT]
      )(using
          ev: AxisReplacerAll.Aux[T, SplitL, NewT, R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val before = tensor.shape.dimensions.take(ev.index)
        val after = tensor.shape.dimensions.drop(ev.index + 1)
        val fullNewShape = before ++ newShape.dimensions ++ after
        Tensor(
          Jax.jnp.reshape(
            tensor.jaxValue,
            py.Dynamic.global.tuple(
              fullNewShape.map(py.Any.from).toPythonProxy
            )
          )
        )

      def unflatten[NewT <: Tuple: Labels](
          newShape: Shape[NewT]
      )(using
          @implicitNotFound("unflatten without axis can only be used on Tensor1 types.")
          ev: T <:< Tuple1[Any] // <--- Ensures this only works on Tensor1
      ): Tensor[NewT, V] =
        val fullNewShape = newShape.dimensions
        Tensor(
          Jax.jnp.reshape(
            tensor.jaxValue,
            py.Dynamic.global.tuple(
              fullNewShape.map(py.Any.from).toPythonProxy
            )
          )
        )

      def transpose[NewOrder <: Tuple, Status <: ValidationResult](newOrder: NewOrder)(using
          ev: AxisIndices[T, UnwrapAxes[NewOrder]],
          newLabels: Labels[UnwrapAxes[NewOrder]]
      )(using
          allAxesEv: IsPermutation[T, UnwrapAxes[NewOrder]]
      ): Tensor[UnwrapAxes[NewOrder], V] =
        val indices = ev.indices
        Tensor(Jax.jnp.transpose(tensor.jaxValue, indices.toPythonProxy))

      // merge all tensor axes to a single vector axis
      def flatten(using labels: Labels[T]): Tensor1[MergeLabels[T], V] =
        given Labels[Tuple1[MergeLabels[T]]] with
          def names = List(summon[Labels[T]].names.mkString("*"))
        Tensor(Jax.jnp.ravel(tensor.jaxValue))

      def flatten[ToMerge <: Tuple, R <: Tuple](
          axes: ToMerge
      )(using
          merger: AxesMerger.Aux[T, UnwrapAxes[ToMerge], R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val permuted = Jax.jnp.transpose(tensor.jaxValue, merger.permutation.toPythonProxy)

        val originalDims = tensor.shape.dimensions
        val mergedSize = merger.mergeIndices.map(originalDims).product

        val remainingDims = originalDims.zipWithIndex
          .filterNot((d, i) => merger.mergeIndices.contains(i))
          .map(_._1)

        val newDimensions = remainingDims.patch(merger.mergedIndex, Seq(mergedSize), 0)

        Tensor(Jax.jnp.reshape(permuted, newDimensions.toPythonProxy))

      def chunk[splitL: Label](splitAxis: Axis[splitL], chunkSize: Int)(using
          labels: Labels[T],
          axisIndex: AxisIndex[T, splitL]
      ): Seq[Tensor[T, V]] =
        val res = Jax.jnp.split(tensor.jaxValue, chunkSize, axis = axisIndex.index).as[Seq[Jax.PyDynamic]]
        res.map(x => Tensor[T, V](x))

      def slice[Inputs <: Tuple, LabelsToRemove <: Tuple, R <: Tuple](
          inputs: Inputs
      )(using
          sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Inputs], R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
        Tensor(tensor.jaxValue.bracketAccess(pyIndices))

      // Convenience overload for AxisAtIndex
      def slice[L, LabelsToRemove <: Tuple, R <: Tuple](
          selector: AxisAtIndex[L]
      )(using
          sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndex[L]], LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndex[L]]], R],
          labels: Labels[R]
      ): Tensor[R, V] = slice(Tuple1(selector))

      // Convenience overload for AxisAtRange
      def slice[L, LabelsToRemove <: Tuple, R <: Tuple](
          selector: AxisAtRange[L]
      )(using
          sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtRange[L]], LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtRange[L]]], R],
          labels: Labels[R]
      ): Tensor[R, V] = slice(Tuple1(selector))

      // Convenience overload for AxisAtIndices
      def slice[L, LabelsToRemove <: Tuple, R <: Tuple](
          selector: AxisAtIndices[L]
      )(using
          sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtIndices[L]], LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtIndices[L]]], R],
          labels: Labels[R]
      ): Tensor[R, V] = slice(Tuple1(selector))

      // Convenience overload for AxisAtTensorIndex
      def slice[L, LabelsToRemove <: Tuple, R <: Tuple](
          selector: AxisAtTensorIndex[L]
      )(using
          sliceExtractor: SliceLabelExtractor[Tuple1[AxisAtTensorIndex[L]], LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[AxisAtTensorIndex[L]]], R],
          labels: Labels[R]
      ): Tensor[R, V] = slice(Tuple1(selector))

      def take[L1, L2: Label, R <: Tuple](
          axis: Axis[L1]
      )(
          indices: Tensor1[L2, Int]
      )(using
          ev: AxisRemover[T, L1, R],
          labels: Labels[R]
      ): Tensor[Tuple.Concat[Tuple1[L2], R], V] =
        val result = Jax.jnp.take(tensor.jaxValue, indices.jaxValue, axis = ev.index)
        Tensor(result)

      def set[Inputs <: Tuple, R <: Tuple](
          inputs: Inputs
      )(using
          labels: Labels[T],
          axesIndices: AxisIndices[T, ExtractLabels[Inputs]]
      )(value: Tensor[R, V]): Tensor[T, V] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices.indices)
        val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value.jaxValue)
        Tensor(result)

      def set[L, I, LabelsToRemove <: Tuple, R <: Tuple](
          axisWithSliceIndex: (Axis[L], I)
      )(using
          labels: Labels[T],
          axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]]
      )(value: Tensor[R, V]): Tensor[T, V] = set(Tuple1(axisWithSliceIndex))(value)

      // Convenience overload for AxisSelector
      def set[L, R <: Tuple](
          selector: AxisSelector[L]
      )(using
          labels: Labels[T],
          axesIndices: AxisIndices[T, ExtractLabels[Tuple1[AxisSelector[L]]]]
      )(value: Tensor[R, V]): Tensor[T, V] = set(Tuple1(selector))(value)

      def rearrange[Axes <: Tuple, Status <: ValidationResult](newOrder: Axes)(using
          Labels[UnwrapAxes[Axes]]
      )(using
          computer: ComputeMissing[UnwrapAxes[Axes], T, EmptyTuple, Status],
          guard: CheckValid[Status]
      ): Tensor[UnwrapAxes[Axes], V] =
        rearrange[Axes, EmptyTuple, Status](newOrder, EmptyTuple)

      // Convenience overload for 1 dims (to support error messages with single axis)
      inline def rearrange[Axes <: Tuple, L1, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Tuple1[AxisExtent[L1]]], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[Tuple1[AxisExtent[L1]]]): Tensor[UnwrapAxes[Axes], V] =
        rearrange(newOrder, Tuple1(d1))

      // Convenience overload for 2 dims
      inline def rearrange[Axes <: Tuple, L1, L2, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2])]): Tensor[UnwrapAxes[Axes], V] =
        rearrange(newOrder, (d1, d2))

      // Convenience overload for 3 dims
      inline def rearrange[Axes <: Tuple, L1, L2, L3, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3])]): Tensor[UnwrapAxes[Axes], V] =
        rearrange(newOrder, (d1, d2, d3))

      // Convenience overload for 4 dims
      inline def rearrange[Axes <: Tuple, L1, L2, L3, L4, Status <: ValidationResult](newOrder: Axes, d1: AxisExtent[L1], d2: AxisExtent[L2], d3: AxisExtent[L3], d4: AxisExtent[L4])(using computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])], Status], guard: CheckValid[Status])(using newLabels: Labels[UnwrapAxes[Axes]], extractor: DimExtractor[(AxisExtent[L1], AxisExtent[L2], AxisExtent[L3], AxisExtent[L4])]): Tensor[UnwrapAxes[Axes], V] =
        rearrange(newOrder, (d1, d2, d3, d4))

      def rearrange[Axes <: Tuple, Dims <: Tuple, Status <: ValidationResult](
          newOrder: Axes,
          dims: Dims
      )(using
          computer: ComputeMissing[UnwrapAxes[Axes], T, UnwrapDims[Dims], Status],
          guard: CheckValid[Status]
      )(using
          newLabels: Labels[UnwrapAxes[Axes]],
          extractor: DimExtractor[Dims]
      ): Tensor[UnwrapAxes[Axes], V] =
        def cleanPatternPrime(pattern: String): String =
          // Support dimwit.Prime by replacing ' with "Prime"
          pattern.replaceAll(
            "'",
            "Prime"
          )
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          def cleanPatternStar(pattern: String): String =
            // to replace all a*b*c in pattern with (a b c), example:
            // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
            val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(
              pattern,
              _.group(1).split("\\*").mkString("(", " ", ")")
            )
          def cleanPatternPlus(pattern: String): String =
            // Support dimwit.|+| by replacing + with underlines
            val regex = raw"([a-zA-Z0-9_]+(\+[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(
              pattern,
              _.group(1).replace("+", "_")
            )
          def cleanPattern(pattern: String): String =
            cleanPatternPlus(cleanPatternStar(cleanPatternPrime(pattern)))
          s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
        val fromPattern = tensor.shape.labels.mkString(" ")
        val toPattern = newLabels.names.mkString(" ")
        val pattern = createEinopsPattern(fromPattern, toPattern)
        val dimSizesMap = extractor.extract(dims)
        val cleanDimSizesMap = dimSizesMap.map { case (k, v) =>
          val newKey = cleanPatternPrime(k)
          (newKey, v)
        }
        Tensor(
          Einops.rearrange(
            tensor.jaxValue,
            pattern,
            kwargsMap = cleanDimSizesMap
          )
        )

      def broadcastTo[O <: Tuple: Labels](newShape: Shape[O])(using
          labels: Labels[T],
          ev: StrictSubset[T, O]
      ): Tensor[O, V] =
        /* Disallow implicit broadcasting where an *existing* axis changes size (implicitly).
         * dimwit broadcasting only adds missing axes, never changes existing ones.
         * 
         * This is a required check to prevent implicit broadcasting across dimwit.
         * If this check is not explicitly present, Jax.jnp.broadcast_to would implicit broadcast.*/
        def disallowImplicitShapeBroadcasting(): Unit =
          val tAxesDims = tensor.axes.zip(tensor.shape.dimensions).toMap
          val newShapeAxesDims = newShape.labels.zip(newShape.dimensions).toMap
          tensor.axes.foreach(axisName =>
            require(
              tAxesDims(axisName) == newShapeAxesDims(axisName),
              s"Broadcasting only adds missing axes. Present axes must have the same size. Axis ${axisName} has size ${tAxesDims(axisName)} in the current tensor but size ${newShapeAxesDims(axisName)} in the target shape."
            )
          )

        disallowImplicitShapeBroadcasting() // Make dimwit coders, good coders :)

        val t = tensor

        val currentNames = summon[Labels[T]].names
        val targetNames = summon[Labels[O]].names

        val targetOrder = targetNames.filter(currentNames.contains)
        val permutation = targetOrder.map(n => currentNames.indexOf(n))

        val alignedJax =
          if permutation != currentNames.indices.toList then Jax.jnp.transpose(t.jaxValue, permutation.toPythonProxy)
          else t.jaxValue

        val currentShapeMap = currentNames.zip(t.shape.dimensions).toMap

        val intermediateShape = targetNames.map { name =>
          currentShapeMap.getOrElse(name, 1)
        }

        val reshapedJax = Jax.jnp.reshape(alignedJax, intermediateShape.toPythonProxy)
        Tensor(Jax.jnp.broadcast_to(reshapedJax, newShape.dimensions.toPythonProxy))

      def relabel[OldLabel: Label, NewLabel: Label](
          rename: (Axis[OldLabel], Axis[NewLabel])
      )(using
          ev: AxisReplacer[T, OldLabel, NewLabel],
          newLabels: Labels[ev.NewShape]
      ): Tensor[ev.NewShape, V] = Tensor(tensor.jaxValue)

      def retag[newT <: Tuple](using newLabels: Labels[newT]): Tensor[newT, V] =
        Tensor(tensor.jaxValue)(using newLabels)

      def relabelAll[newT <: Tuple](
          newAxes: newT
      )(using
          newLabels: Labels[UnwrapAxes[newT]],
          @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
          evSameSize: Tuple.Size[newT] =:= Tuple.Size[T]
      ): Tensor[UnwrapAxes[newT], V] = Tensor[UnwrapAxes[newT], V](tensor.jaxValue)

      def swap[L1: Label, L2: Label](
          axis1: Axis[L1],
          axis2: Axis[L2]
      )(using
          labels: Labels[T],
          axisIndex1: AxisIndex[T, L1],
          axisIndex2: AxisIndex[T, L2]
      ): Tensor[Swap[T, L1, L2], V] =
        given Labels[Swap[T, L1, L2]] with
          def names =
            val originalNames = summon[Labels[T]].names
            val ax1Name = summon[Label[L1]].name
            val ax2Name = summon[Label[L2]].name
            originalNames.map {
              case n if n == ax1Name => ax2Name
              case n if n == ax2Name => ax1Name
              case n                 => n
            }
        Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.index, axisIndex2.index))

      def appendAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[T, Tuple1[L]], V] =
        val newShape = tensor.shape.dimensions :+ 1
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def prependAxis[L: Label](axis: Axis[L])(using labels: Labels[T], ev: AxisAbsent[T, L]): Tensor[Tuple.Concat[Tuple1[L], T], V] =
        val newShape = 1 +: tensor.shape.dimensions
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def squeeze[L: Label, R <: Tuple](axis: Axis[L])(using
          ev: AxisRemover[T, L, R],
          labels: Labels[R]
      ): Tensor[R, V] =
        require(
          tensor.shape.dimensions(ev.index) == 1,
          s"Cannot squeeze axis ${summon[Label[L]].name} of size ${tensor.shape.dimensions(ev.index)}"
        )
        Tensor(Jax.jnp.squeeze(tensor.jaxValue, axis = ev.index))

  end Structural

  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------
  object Functional:

    object ZipVmap:

      type TensorsOf[Shapes <: Tuple, Values <: Tuple] <: Tuple = (Shapes, Values) match
        case (EmptyTuple, EmptyTuple)                             => EmptyTuple
        case ((shapeHead *: shapeTail), (valueHead *: valueTail)) => Tensor[shapeHead, valueHead] *: TensorsOf[shapeTail, valueTail]

      type ExtractShape[T] = T match
        case Tensor[s, v] => s

      type ExtractValue[T] = T match
        case Tensor[s, v] => v

      type ShapesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractShape]
      type ValuesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractValue]

      def zipvmap[L: Label, Inputs <: Tuple, OutShape <: Tuple: Labels, R <: Tuple, OutV](
          axis: Axis[L]
      )(
          tensors: Inputs // This is a Tuple of Tensors
      )(using
          ev: SharedAxisRemover[ShapesOf[Inputs], L, R]
      )(
          f: TensorsOf[R, ValuesOf[Inputs]] => Tensor[OutShape, OutV]
      ): Tensor[L *: OutShape, OutV] =
        // allows us to ignore labels for the intermediate sliced tensors
        val dummyLabels = new Labels[Nothing]:
          val names = Nil

        val fpy = (args: py.Dynamic) =>
          OnError.traceStack:
            val tensorList = args.as[Seq[py.Dynamic]].zipWithIndex.map: (jaxArr, i) =>
              Tensor(jaxArr)(using dummyLabels)

            val inputTuple = Tuple.fromArray(tensorList.toArray)
            val result = f(inputTuple.asInstanceOf[TensorsOf[R, ValuesOf[Inputs]]])
            result.jaxValue

        val jaxInputs = py.Dynamic.global.tuple(tensors.toArray.map(_.asInstanceOf[Tensor[?, ?]].jaxValue).toPythonProxy)
        val indicesAsTuple = py.Dynamic.global.tuple(ev.indices.toPythonProxy)
        val jaxResult = Jax.jax_helper.zipvmap(
          fpy,
          indicesAsTuple
        )(jaxInputs)

        Tensor(jaxResult)

    export ZipVmap.zipvmap

    extension [T <: Tuple: Labels, V](t: Tensor[T, V])

      def vmap[VmapAxis: Label, OuterShape <: Tuple: Labels, R <: Tuple, V2](
          axis: Axis[VmapAxis]
      )(using
          ev: AxisRemover[T, VmapAxis, R]
      )(
          f: Tensor[R, V] => Tensor[OuterShape, V2]
      )(using
          labels: Labels[R]
      ): Tensor[VmapAxis *: OuterShape, V2] =
        val fpy = (jxpr: Jax.PyDynamic) =>
          OnError.traceStack:
            val innerTensor = Tensor[R, V](jxpr)
            val result = f(innerTensor)
            result.jaxValue

        Tensor(Jax.jax_helper.vmap(fpy, ev.index)(t.jaxValue))

      def vapply[L: Label, NewL, R <: Tuple](
          axis: Axis[L]
      )(
          f: Tensor[Tuple1[L], V] => Tensor[Tuple1[NewL], V]
      )(using
          ev: AxisReplacer.Aux[T, L, NewL, R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val fpy = (jxpr: Jax.PyDynamic) =>
          OnError.traceStack:
            val inputTensor = Tensor[Tuple1[L], V](jxpr)
            val result = f(inputTensor)
            result.jaxValue

        Tensor(
          Jax.jnp.apply_along_axis(
            fpy,
            ev.index,
            t.jaxValue
          )
        )

      def vreduce[L: Label, R <: Tuple](
          axis: Axis[L]
      )(
          f: Tensor[Tuple1[L], V] => Tensor0[V]
      )(using
          ev: AxisRemover[T, L, R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val fpy = (jxpr: Jax.PyDynamic) =>
          OnError.traceStack:
            val inputTensor = Tensor[Tuple1[L], V](jxpr)
            val result = f(inputTensor)
            result.jaxValue

        Tensor(
          Jax.jnp.apply_along_axis(
            fpy,
            ev.index,
            t.jaxValue
          )
        )

  end Functional

  export Elementwise.*
  export Reduction.*
  export Contraction.*
  export Convolution.*
  export LinearAlgebra.*
  export Structural.*
  export Functional.*

  // -----------------------------------------------------------
  // Common specialized operation names
  // -----------------------------------------------------------
  object Tensor0Ops:
    extension [V: Reader](scalar: Tensor0[V])

      def item: V =
        require(
          !scalar.isTracer,
          """
          | Cannot convert a JAX Tracer to a scalar value. Tensor0 is part of a JAX computation graph (e.g., inside vmap or a jitted function).
          | Common mistakes leading to this error:
          |   - calling .slice(t0.item) rather than .slice(t0); breaking the computation graph unintentionally.
          |""".stripMargin
        )
        scalar.jaxValue.item().as[V]

  object ValueOps:

    import Elementwise.+!

    extension [V: IsNumber: Writer](scalar: V)

      def +![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] =
        given ExecutionType[V] = ExecutionTypeFor[V](t.dtype)
        Tensor0(scalar).broadcastTo(t.shape) + t
      def -![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] =
        given ExecutionType[V] = ExecutionTypeFor[V](t.dtype)
        Tensor0(scalar).broadcastTo(t.shape) - t
      def *![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] =
        given ExecutionType[V] = ExecutionTypeFor[V](t.dtype)
        Tensor0(scalar).broadcastTo(t.shape) * t

    extension [V: IsFloating: Writer](scalar: V)

      def /![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] =
        given ExecutionType[V] = ExecutionTypeFor[V](t.dtype)
        Tensor0(scalar).broadcastTo(t.shape) / t

  object Tensor1Ops:

    extension [L, V](t: Tensor1[L, V])

      def relabelTo[NewL: Label](newAxis: Axis[NewL]): Tensor1[NewL, V] = Tensor[Tuple1[NewL], V](t.jaxValue)

  object Tensor2Ops:

    extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

      // Support .transpose without arguments for 2D tensors while keeping (not shadowing) the general .transpose with arguments
      def transpose: Tensor2[L2, L1, V] = t.transpose(Axis[L2], Axis[L1])
      def transpose(axis2: Axis[L2], axis1: Axis[L1]): Tensor2[L2, L1, V] = TensorOps.Structural.transpose(t)(axis2, axis1)

  export Tensor0Ops.*
  export ValueOps.*
  export Tensor1Ops.*
  export Tensor2Ops.*

end TensorOps

object TensorOpsUtil:

  import TensorOps.Structural.broadcastTo

  @implicitNotFound("Cannot broadcast tensors of shapes ${T1} and ${T2}. If same shape no broadcasting allowed!")
  sealed trait Broadcast[T1 <: Tuple, T2 <: Tuple, V]:
    type Out <: Tuple
    given labelsOut: Labels[Out]
    def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]): (Tensor[Out, V], Tensor[Out, V])
    def applyTo[V2](t1: Tensor[T1, V], t2: Tensor[T2, V])(f: (Tensor[Out, V], Tensor[Out, V]) => Tensor[Out, V2]): Tensor[Out, V2] =
      val (bt1, bt2) = broadcast(t1, t2)
      f(bt1, bt2)

  object Broadcast extends BroadcastLowPriority:

    given broadcastLeft[T1 <: Tuple: Labels, T2 <: Tuple: Labels, V](using
        StrictSubset[T2, T1]
    ): Broadcast[T1, T2, V] with
      type Out = T1
      val labelsOut = summon[Labels[T1]]
      def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]) =
        (t1, t2.broadcastTo[T1](t1.shape))

  trait BroadcastLowPriority:
    given broadcastRight[T1 <: Tuple: Labels, T2 <: Tuple: Labels, V](using
        StrictSubset[T1, T2]
    ): Broadcast[T1, T2, V] with
      type Out = T2
      val labelsOut = summon[Labels[T2]]
      def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]) =
        (t1.broadcastTo[T2](t2.shape), t2)

end TensorOpsUtil
