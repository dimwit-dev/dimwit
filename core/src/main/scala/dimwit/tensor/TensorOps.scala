package dimwit.tensor

import scala.annotation.targetName
import scala.annotation.implicitNotFound
import scala.util.NotGiven

import dimwit.jax.{Jax, Einops}
import dimwit.tensor.{Label, Labels}
import dimwit.tensor.Axis.UnwrapAxes
import dimwit.tensor.TupleHelpers.{Subset, StrictSubset, PrimeConcat}
import dimwit.tensor.ShapeTypeHelpers.{AxisRemover, AxesRemover, SharedAxisRemover, AxisReplacer, AxesConditionalRemover}
import dimwit.{~, `|*|`}

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import me.shadaj.scalapy.readwrite.Writer
import me.shadaj.scalapy.readwrite.Reader

object TensorOps:

  import TensorOpsUtil.*

  @implicitNotFound("Operation only valid for Numeric (Int or Float) tensors.")
  sealed trait IsNumber[V]

  // -----------------------------------------------------------
  // Typeclasses to steer operation availability to prevent runtime errors
  // -----------------------------------------------------------

  @implicitNotFound("Operation only valid for Int or Float tensors.")
  object IsNumber:
    given [V](using ev1: IsFloat[V]): IsNumber[V] = ev1
    given [V](using ev2: IsInt[V]): IsNumber[V] = ev2

  @implicitNotFound("Operation only valid for Float tensors.")
  trait IsFloat[V] extends IsNumber[V]
  object IsFloat:
    given IsFloat[Float] with {}

  @implicitNotFound("Operation only valid for Int tensors.")
  trait IsInt[V] extends IsNumber[V]
  object IsInt:
    given IsInt[Int] with {}

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

    def add[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))
    def addScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.add(t1.jaxValue, t2.jaxValue))

    def negate[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))
    def subtract[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t1.jaxValue, t2.jaxValue))
    def subtractScalar[T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, t2.jaxValue))

    def multiply[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))
    def multiplyScalar[T <: Tuple: Labels, V: IsNumber](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])

      def +(other: Tensor[T, V]): Tensor[T, V] = add(t, other)
      def +![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = join.applyTo(t, other)(add)

      def unary_- : Tensor[T, V] = negate(t)
      def -(other: Tensor[T, V]): Tensor[T, V] = subtract(t, other)
      def -![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = join.applyTo(t, other)(subtract)

      def *(other: Tensor[T, V]): Tensor[T, V] = multiply(t, other)
      def *![O <: Tuple](other: Tensor[O, V])(using join: Broadcast[T, O, V]): Tensor[join.Out, V] = join.applyTo(t, other)(multiply)
      def scale(other: Tensor0[V]): Tensor[T, V] = multiplyScalar(t, other)

      def abs: Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))
      def sign: Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))
      def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))
      def pow(n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))

    // ---------------------------------------------------------
    // IsFloat operations
    // ---------------------------------------------------------

    def divide[T <: Tuple: Labels, V: IsFloat](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))
    def divideScalar[T <: Tuple: Labels, V: IsFloat](t1: Tensor[T, V], t2: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple: Labels, V: IsFloat](t: Tensor[T, V])

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

    extension [T <: Tuple: Labels, V: IsFloat](t: Tensor[T, V])

      // --- Mean ---
      def mean: Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))
      def mean[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.index))
      def mean[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = ev.indices.toPythonProxy))

      // --- Std ---
      def std: Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))
      def std[L: Label, R <: Tuple](axis: Axis[L])(using ev: AxisRemover[T, L, R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.index))
      def std[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using ev: AxesRemover[T, UnwrapAxes[Inputs], R], l: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = ev.indices.toPythonProxy))

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

      def contract[
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

      @targetName("contractOn")
      def contract[
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

    extension [T <: Tuple: Labels, V: IsFloat](t: Tensor[T, V])

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

    extension [L1: Label, L2: Label, V: IsFloat](t: Tensor2[L1, L2, V])

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
        case (Axis[l], SliceIndex) => l
      type ExtractLabels[Inputs <: Tuple] = Tuple.Map[Inputs, ExtractLabel]

      trait SliceLabelExtractor[Inputs <: Tuple, Out <: Tuple]

      object SliceLabelExtractor:

        given empty: SliceLabelExtractor[EmptyTuple, EmptyTuple] =
          new SliceLabelExtractor[EmptyTuple, EmptyTuple] {}

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

      type TupleReduce[T <: Tuple, Op[_ <: String, _ <: String]] = T match
        case EmptyTuple      => ""
        case h *: EmptyTuple => h
        case h *: t          => Op[h, TupleReduce[t, Op]]

      type TupleUnion[T <: Tuple] = T match
        case EmptyTuple      => EmptyTuple
        case h *: EmptyTuple => h
        case h *: t          => h | TupleUnion[t]

      type FoldLeft[T <: Tuple, Z, F[_, _]] = T match
        case EmptyTuple => Z
        case h *: t     => FoldLeft[t, F[Z, h], F]

      trait DimExtractor[T]:
        def extract(t: T): Map[String, Int]

      object DimExtractor:
        given DimExtractor[EmptyTuple] with
          def extract(t: EmptyTuple) = Map.empty

        given [L, Tail <: Tuple](using
            label: Label[L],
            tailExtractor: DimExtractor[Tail]
        ): DimExtractor[(Axis[L], Int) *: Tail] with
          def extract(t: (Axis[L], Int) *: Tail) =
            val (_, size) = t.head
            Map(label.name -> size) ++ tailExtractor.extract(t.tail)

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
      val axisIdx = axisIndex.value + 1 // we are inserting after the given axis, so shift by 1
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
      val axisIdx = axisIndex.value
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
      Tensor(concatenatedJaxValue)

    extension [T <: Tuple: Labels, V](tensor: Tensor[T, V])

      private def calcPyIndices[Inputs <: Tuple](
          inputs: Inputs,
          targetDims: List[Int]
      ) =

        val PySlice = py.Dynamic.global.slice
        val Colon = PySlice(py.None)
        val rank = tensor.shape.rank
        val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

        val inputList = inputs.toList.asInstanceOf[List[(Any, Any)]]

        targetDims.zip(inputList).foreach { case (dimIndex, (_, sliceIndex)) =>
          val dimSize = tensor.shape.dimensions(dimIndex)
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

      def split[newL, splitL](newAxis: Axis[newL], splitAxis: Axis[splitL], interval: Int)(using
          newLabel: Label[newL],
          axisIndex: AxisIndex[T, splitL]
      ): Tensor[InsertBefore[T, splitL, newL], V] =
        val splitIdx = axisIndex.value
        val names = summon[Labels[T]].names
        val newNames = names.take(splitIdx) ++ Seq(newLabel.name) ++ names.drop(splitIdx)
        given Labels[InsertBefore[T, splitL, newL]] with
          val names = newNames.toSeq
        val (before, after) = tensor.shape.dimensions.splitAt(splitIdx)
        val newShape = before ++ Seq(interval, after.head / interval) ++ after.drop(1)
        Tensor(
          Jax.jnp.reshape(
            tensor.jaxValue,
            Jax.Dynamic.global.tuple(
              newShape.map(py.Any.from).toPythonProxy
            )
          )
        )

      def chunk[splitL: Label](splitAxis: Axis[splitL], interval: Int)(using
          axisIndex: AxisIndex[T, splitL]
      ): Seq[Tensor[T, V]] =
        val res = Jax.jnp.split(tensor.jaxValue, interval, axis = axisIndex.value).as[Seq[Jax.PyDynamic]]
        res.map(x => Tensor[T, V](x))

      def tile = ???
      def repeat = ???

      def slice[Inputs <: Tuple, LabelsToRemove <: Tuple, R <: Tuple](
          inputs: Inputs
      )(using
          sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Inputs], R],
          labels: Labels[R]
      ): Tensor[R, V] =
        val pyIndices = tensor.calcPyIndices(inputs, ev.indices)
        Tensor(tensor.jaxValue.bracketAccess(pyIndices))

      def slice[L, I, LabelsToRemove <: Tuple, R <: Tuple](
          axisWithSliceIndex: (Axis[L], I)
      )(using
          sliceExtractor: SliceLabelExtractor[Tuple1[(Axis[L], I)], LabelsToRemove],
          ev: AxesConditionalRemover[T, LabelsToRemove, ExtractLabels[Tuple1[(Axis[L], I)]], R],
          labels: Labels[R]
      ): Tensor[R, V] = slice(Tuple1(axisWithSliceIndex))

      def gather[L](
          indices: Tensor1[L, Int]
      )(using
          axesIndex: AxisIndex[T, L]
      ): Tensor[T, V] =
        Tensor(Jax.jnp.take(tensor.jaxValue, indices.jaxValue, axis = axesIndex.value))

      def set[Inputs <: Tuple, R <: Tuple](
          inputs: Inputs
      )(using
          axesIndices: AxisIndices[T, ExtractLabels[Inputs]]
      )(value: Tensor[R, V]): Tensor[T, V] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices.values)
        val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value.jaxValue)
        Tensor(result)

      def set[L, I, LabelsToRemove <: Tuple, R <: Tuple](
          axisWithSliceIndex: (Axis[L], I)
      )(using
          axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]]
      )(value: Tensor[R, V]): Tensor[T, V] = set(Tuple1(axisWithSliceIndex))(value)

      def rearrange[Axes <: Tuple](newOrder: Axes)(using Labels[UnwrapAxes[Axes]]): Tensor[UnwrapAxes[Axes], V] =
        rearrange[Axes, EmptyTuple](newOrder, EmptyTuple)

      def rearrange[Axes <: Tuple, Dims <: Tuple](
          newOrder: Axes,
          dims: Dims
      )(using
          newLabels: Labels[UnwrapAxes[Axes]],
          extractor: DimExtractor[Dims]
      ): Tensor[UnwrapAxes[Axes], V] =
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          def cleanPattern(pattern: String): String =
            // to replace all a*b*c in pattern with (a b c), example:
            // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
            val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(
              pattern,
              m =>
                val group = m.group(1)
                val replaced = group.split("\\*").mkString("(", " ", ")")
                replaced
            )
          s"${cleanPattern(fromPattern)} -> ${cleanPattern(toPattern)}"
        val fromPattern = tensor.shape.labels.mkString(" ")
        val toPattern = newLabels.names.mkString(" ")
        val pattern = createEinopsPattern(fromPattern, toPattern)
        val dimSizesMap = extractor.extract(dims)
        Tensor(
          Einops.rearrange(
            tensor.jaxValue,
            pattern,
            kwargsMap = dimSizesMap
          )
        )

      def lift[O <: Tuple: Labels](newShape: Shape[O])(using
          ev: StrictSubset[T, O] // Ensures T's axes are all present in O
      ): Tensor[O, V] =
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
        Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.value, axisIndex2.value))

      def ravel: Tensor1[FoldLeft[Tuple.Tail[T], Tuple.Head[T], |*|], V] =
        given Labels[Tuple1[FoldLeft[Tuple.Tail[T], Tuple.Head[T], |*|]]] with
          def names = List(summon[Labels[T]].names.mkString("*"))
        Tensor(Jax.jnp.ravel(tensor.jaxValue))

      def appendAxis[L: Label](axis: Axis[L])(using AxisAbsent[T, L]): Tensor[Tuple.Concat[T, Tuple1[L]], V] =
        import Labels.ForConcat.given
        val newShape = tensor.shape.dimensions :+ 1
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def prependAxis[L: Label](axis: Axis[L])(using AxisAbsent[T, L]): Tensor[Tuple.Concat[Tuple1[L], T], V] =
        import Labels.ForConcat.given
        val newShape = 1 +: tensor.shape.dimensions
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def squeeze[L: Label, R <: Tuple](axis: Axis[L])(using
          ev: AxisRemover[T, L, R],
          labels: Labels[R]
      ): Tensor[R, V] =
        require(
          tensor.shape.dimensions(ev.index) == 1,
          s"Cannot squeeze axis ${axis} of size ${tensor.shape.dimensions(ev.index)}"
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
          val names = "dummy" :: Nil

        val fpy = (args: py.Dynamic) =>
          val tensorList = args.as[Seq[py.Dynamic]].zipWithIndex.map { (jaxArr, i) =>
            Tensor(jaxArr)(using dummyLabels)
          }

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
  export LinearAlgebra.*
  export Structural.*
  export Functional.*

  // -----------------------------------------------------------
  // Common specialized operation names
  // -----------------------------------------------------------
  object Tensor0Ops:
    extension [V: Reader](scalar: Tensor0[V])

      def item: V = scalar.jaxValue.item().as[V]

  object ValueOps:

    extension [V: IsNumber: Writer](scalar: V)

      def +![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] = addScalar(t, Tensor0.const(t.vtype)(scalar))
      def -![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] = subtractScalar(t, Tensor0.const(t.vtype)(scalar))
      def *![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] = multiplyScalar(t, Tensor0.const(t.vtype)(scalar))

    extension [V: IsFloat: Writer](scalar: V)

      def /![T <: Tuple: Labels](t: Tensor[T, V]): Tensor[T, V] = divideScalar(t, Tensor0.const(t.vtype)(scalar))

  object Tensor1Ops:

    extension [L: Label, V](t: Tensor1[L, V])

      def dot(other: Tensor1[L, V]): Tensor0[V] = t.innerDot(other)
      def innerDot(other: Tensor1[L, V]): Tensor0[V] = t.contract(Axis[L])(other)
      def outerDot[OtherLabel: Label](other: Tensor1[OtherLabel, V]): Tensor2[L, OtherLabel, V] = t.outerProduct(other)
      def relabelTo[NewL: Label](newAxis: Axis[NewL]): Tensor1[NewL, V] = Tensor[Tuple1[NewL], V](t.jaxValue)

  object Tensor2Ops:

    extension [L1: Label, L2: Label, V](t: Tensor2[L1, L2, V])

      def transpose: Tensor2[L2, L1, V] = t.rearrange((Axis[L2], Axis[L1]))

      @targetName("tensor2MatmulTensor2")
      def matmul[L3: Label](other: Tensor2[L2, L3, V])(using
          ev: AxisRemover[(L1, L2), L2, Tuple1[L1]],
          evOther: AxisRemover[(L2, L3), L2, Tuple1[L3]]
      ): Tensor2[L1, L3, V] = t.contract(Axis[L2])(other)

      @targetName("tensor2MatmulTensor1")
      def matmul(other: Tensor1[L2, V])(using
          ev: AxisRemover[(L1, L2), L2, Tuple1[L1]],
          evOther: AxisRemover[Tuple1[L2], L2, EmptyTuple]
      ): Tensor[Tuple1[L1], V] = t.contract(Axis[L2])(other)

  export Tensor0Ops.*
  export ValueOps.*
  export Tensor1Ops.*
  export Tensor2Ops.*

end TensorOps

object TensorOpsUtil:

  import TensorOps.Structural.lift

  sealed trait Broadcast[T1 <: Tuple, T2 <: Tuple, V]:
    type Out <: Tuple
    given labelsOut: Labels[Out]
    def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]): (Tensor[Out, V], Tensor[Out, V])
    def applyTo[V2](t1: Tensor[T1, V], t2: Tensor[T2, V])(f: (Tensor[Out, V], Tensor[Out, V]) => Tensor[Out, V2]): Tensor[Out, V2] =
      val (bt1, bt2) = broadcast(t1, t2)
      f(bt1, bt2)

  object Broadcast extends BroadcastLowPriority:
    given identity[T <: Tuple: Labels, V]: Broadcast[T, T, V] with
      type Out = T
      val labelsOut = summon[Labels[T]]
      def broadcast(t1: Tensor[T, V], t2: Tensor[T, V]) = (t1, t2)

    given broadcastLeft[T1 <: Tuple: Labels, T2 <: Tuple: Labels, V](using
        StrictSubset[T2, T1]
    ): Broadcast[T1, T2, V] with
      type Out = T1
      val labelsOut = summon[Labels[T1]]
      def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]) =
        (t1, t2.lift[T1](t1.shape))

  trait BroadcastLowPriority:
    given broadcastRight[T1 <: Tuple: Labels, T2 <: Tuple: Labels, V](using
        StrictSubset[T1, T2]
    ): Broadcast[T1, T2, V] with
      type Out = T2
      val labelsOut = summon[Labels[T2]]
      def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]) =
        (t1.lift[T2](t2.shape), t2)

end TensorOpsUtil
