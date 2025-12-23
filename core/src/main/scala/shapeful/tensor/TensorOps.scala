package shapeful.tensor

import shapeful.jax.{Jax, Einops}
import scala.annotation.targetName
import scala.annotation.implicitNotFound
import shapeful.tensor.TupleHelpers.{Subset, Remover, RemoverAll, Replacer}
import shapeful.tensor.{Label, Labels}
import shapeful.tensor.Axis.UnwrapAxes
import scala.util.NotGiven
import scala.collection.View.Empty
import shapeful.Bool
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import shapeful.tensor.TupleHelpers.PrimeConcat
import shapeful.{~, `|*|`}


trait Broadcast[T1 <: Tuple, T2 <: Tuple, Out <: Tuple, V : Value]:
  def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]): (Tensor[Out, V], Tensor[Out, V])

object Broadcast:
  
  import shapeful.tensor.TensorOps.Structural.lift

  given identity[T <: Tuple, V : Value]: Broadcast[T, T, T, V] with
    def broadcast(t1: Tensor[T, V], t2: Tensor[T, V]): (Tensor[T, V], Tensor[T, V]) = (t1, t2)
  
  given broadcastToLeft[T1 <: Tuple : Labels, T2 <: Tuple : Labels, V : Value](using
    ev: Subset[T1, T2]
  ): Broadcast[T1, T2, T1, V] with
    def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]): (Tensor[T1, V], Tensor[T1, V]) =
      val liftedT2 = t2.lift[T1](t1.shape)
      (t1, liftedT2)

  given broadcastToRight[T1 <: Tuple : Labels, T2 <: Tuple : Labels, V : Value](using
    ev: Subset[T2, T1],
  ): Broadcast[T1, T2, T2, V] with
    def broadcast(t1: Tensor[T1, V], t2: Tensor[T2, V]): (Tensor[T2, V], Tensor[T2, V]) =
      val liftedT1 = t1.lift[T2](t2.shape)
      (liftedT1, t2)

object TensorOps:

  // -----------------------------------------------------------
  // 1. Elementwise Operations (The Field)
  // Preserves Shape: T -> T
  // -----------------------------------------------------------
  object Elementwise:

    def maximum[T <: Tuple : Labels, V : Value](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.maximum(t1.jaxValue, t2.jaxValue))
    def minimum[T <: Tuple : Labels, V : Value](t1: Tensor[T, V], t2: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.minimum(t1.jaxValue, t2.jaxValue))

    extension [T <: Tuple : Labels, V : Value](t: Tensor[T, V])
      
      def +(other: Tensor[T, V]): Tensor[T, V] = t.add(other)
      def :+[O <: Tuple](other: Tensor[O, V])(using broadcaster: Broadcast[T, O, T, V]): Tensor[T, V] = broadcaster.broadcast(t, other) match { case (l, r) => l.add(r) }
      def +:[O <: Tuple : Labels](other: Tensor[O, V])(using broadcaster: Broadcast[O, T, O, V]): Tensor[O, V] = broadcaster.broadcast(other, t) match { case (r, l) => l.add(r) }
      private def add(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.add(t.jaxValue, other.jaxValue))
      
      def unary_- : Tensor[T, V] = Tensor(Jax.jnp.negative(t.jaxValue))
      def -(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))
      def :-[O <: Tuple](other: Tensor[O, V])(using broadcaster: Broadcast[T, O, T, V]): Tensor[T, V] = broadcaster.broadcast(t, other) match { case (l, r) => l.subtract(r) }
      def -:[O <: Tuple : Labels](other: Tensor[O, V])(using broadcaster: Broadcast[O, T, O, V]): Tensor[O, V] = broadcaster.broadcast(other, t) match { case (r, l) => l.subtract(r) }
      private def subtract(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.subtract(t.jaxValue, other.jaxValue))

      def *(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      def scale(other: Tensor0[V]) = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      def :*[O <: Tuple](other: Tensor[O, V])(using broadcaster: Broadcast[T, O, T, V]): Tensor[T, V] = broadcaster.broadcast(t, other) match { case (l, r) => l.multiply(r) }
      def *:[O <: Tuple : Labels](other: Tensor[O, V])(using broadcaster: Broadcast[O, T, O, V]): Tensor[O, V] = broadcaster.broadcast(other, t) match { case (r, l) => l.multiply(r) }
      private def multiply(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.multiply(t.jaxValue, other.jaxValue))
      
      def /(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))
      def :/[O <: Tuple](other: Tensor[O, V])(using broadcaster: Broadcast[T, O, T, V]): Tensor[T, V] = broadcaster.broadcast(t, other) match { case (l, r) => l.divide(r) }
      def /:[O <: Tuple : Labels](other: Tensor[O, V])(using broadcaster: Broadcast[O, T, O, V]): Tensor[O, V] = broadcaster.broadcast(other, t) match { case (r, l) => l.divide(r) }
      private def divide(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.divide(t.jaxValue, other.jaxValue))

      // --- Unary Math ---
      def abs: Tensor[T, V] = Tensor(Jax.jnp.abs(t.jaxValue))
      def sign: Tensor[T, V] = Tensor(Jax.jnp.sign(t.jaxValue))
      def pow(n: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.power(t.jaxValue, n.jaxValue))
      def sqrt: Tensor[T, V] = Tensor(Jax.jnp.sqrt(t.jaxValue))
      def exp: Tensor[T, V] = Tensor(Jax.jnp.exp(t.jaxValue))
      def log: Tensor[T, V] = Tensor(Jax.jnp.log(t.jaxValue))
      def sin: Tensor[T, V] = Tensor(Jax.jnp.sin(t.jaxValue))
      def cos: Tensor[T, V] = Tensor(Jax.jnp.cos(t.jaxValue))
      def tanh: Tensor[T, V] = Tensor(Jax.jnp.tanh(t.jaxValue))

      // --- Clipping ---
      def clip(min: Float, max: Float): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min, max))
      def clip(min: Tensor0[V], max: Tensor0[V]): Tensor[T, V] = Tensor(Jax.jnp.clip(t.jaxValue, min.jaxValue, max.jaxValue))

      // --- Comparison ---
      def <(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.less(t.jaxValue, other.jaxValue))
      def <=(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.less_equal(t.jaxValue, other.jaxValue))
      def >(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.greater(t.jaxValue, other.jaxValue))
      def >=(other: Tensor[T, V]): Tensor[T, V] = Tensor(Jax.jnp.greater_equal(t.jaxValue, other.jaxValue))
      
      def elementEquals(other: Tensor[T, V]): Tensor[T, V] =
        require(t.shape.dimensions == other.shape.dimensions, s"Shape mismatch: ${t.shape.dimensions} vs ${other.shape.dimensions}")
        Tensor(jaxValue = Jax.jnp.equal(t.jaxValue, other.jaxValue))

      def all: Boolean = Tensor0(Jax.jnp.all(t.jaxValue)).toBool
      def any: Boolean = Tensor0(Jax.jnp.any(t.jaxValue)).toBool

      def approxEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Boolean = approxElementEquals(other, tolerance).all
      def approxElementEquals(other: Tensor[T, V], tolerance: Float = 1e-6f): Tensor[T, V] =
        Tensor(Jax.jnp.allclose(
          t.jaxValue,
          other.jaxValue,
          atol = tolerance,
          rtol = tolerance
        ))
  
  end Elementwise

  // -----------------------------------------------------------
  // 2. Reduction Operations (The Monoid)
  // Reduces Rank: T -> T - {Axis}
  // -----------------------------------------------------------
  object Reduction:

    extension [T <: Tuple : Labels, V : Value](t: Tensor[T, V])
      // --- Sum ---
      def sum: Tensor0[V] = Tensor0(Jax.jnp.sum(t.jaxValue))
      def sum[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = axisIndex.value))
      def sum[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using remover: RemoverAll.Aux[T, UnwrapAxes[Inputs], R], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.sum(t.jaxValue, axis = axesIndices.values.toPythonProxy))

      // --- Mean ---
      def mean: Tensor0[V] = Tensor0(Jax.jnp.mean(t.jaxValue))
      def mean[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = axisIndex.value))
      def mean[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using remover: RemoverAll.Aux[T, UnwrapAxes[Inputs], R], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.mean(t.jaxValue, axis = axesIndices.values.toPythonProxy))

      // --- Std ---
      def std: Tensor0[V] = Tensor0(Jax.jnp.std(t.jaxValue))
      def std[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = axisIndex.value))
      def std[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using remover: RemoverAll.Aux[T, UnwrapAxes[Inputs], R], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.std(t.jaxValue, axis = axesIndices.values.toPythonProxy))

      // --- Max ---
      def max: Tensor0[V] = Tensor0(Jax.jnp.max(t.jaxValue))
      def max[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = axisIndex.value))
      def max[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using remover: RemoverAll.Aux[T, UnwrapAxes[Inputs], R], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.max(t.jaxValue, axis = axesIndices.values.toPythonProxy))

      // --- Min ---
      def min: Tensor0[V] = Tensor0(Jax.jnp.min(t.jaxValue))
      def min[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = axisIndex.value))
      def min[Inputs <: Tuple, R <: Tuple](axes: Inputs)(using remover: RemoverAll.Aux[T, UnwrapAxes[Inputs], R], axesIndices: AxisIndices[T, UnwrapAxes[Inputs]], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.min(t.jaxValue, axis = axesIndices.values.toPythonProxy))

      // --- Argmax ---
      def argmax: Tensor0[V] = Tensor0(Jax.jnp.argmax(t.jaxValue))
      def argmax[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.argmax(t.jaxValue, axis = axisIndex.value))

      // --- Argmin ---
      def argmin: Tensor0[V] = Tensor0(Jax.jnp.argmin(t.jaxValue))
      def argmin[L : Label, R <: Tuple](axis: Axis[L])(using axisIndex: AxisIndex[T, L], remover: Remover.Aux[T, L, R], labels: Labels[R]): Tensor[R, V] = Tensor(Jax.jnp.argmin(t.jaxValue, axis = axisIndex.value))    
  end Reduction

  object Contraction:

    extension [T <: Tuple : Labels, V : Value](tensor: Tensor[T, V])
      
      def outerProduct[OtherShape <: Tuple : Labels, Out <: Tuple](other: Tensor[OtherShape, V])(
        using primeConcat: PrimeConcat.Aux[T, OtherShape, Out],
      ): Tensor[Out, V] =
        import Labels.ForPrimeConcat.given
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
      ]
      (axis: Axis[ContractAxis])
      (other: Tensor[OtherShape, V])(using
        remover: RemoverAll.Aux[T, ContractAxis *: EmptyTuple, R1],
        otherRemover: RemoverAll.Aux[OtherShape, ContractAxis *: EmptyTuple, R2],
        axisIndex: AxisIndex[T, ContractAxis],
        otherAxisIndex: AxisIndex[OtherShape, ContractAxis],
        primeConcat: PrimeConcat.Aux[R1, R2, Out],
        r1Labels: Labels[R1],
        r2Labels: Labels[R2],
      ): Tensor[Out, V] =
        import Labels.ForPrimeConcat.given
        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(axisIndex.value).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
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
      ]
      (axis: Axis[ContractAxisA ~ ContractAxisB])
      (other: Tensor[OtherShape, V])(using
        remover: RemoverAll.Aux[T, ContractAxisA *: EmptyTuple, R1],
        otherRemover: RemoverAll.Aux[OtherShape, ContractAxisB *: EmptyTuple, R2],
        axisIndex: AxisIndex[T, ContractAxisA],
        otherAxisIndex: AxisIndex[OtherShape, ContractAxisB],
        primeConcat: PrimeConcat.Aux[R1, R2, Out],
        outLabels: Labels[Out],
      ): Tensor[Out, V] =
        import Labels.ForPrimeConcat.given
        val axesTuple1 = Jax.Dynamic.global.tuple(Seq(axisIndex.value).toPythonProxy)
        val axesTuple2 = Jax.Dynamic.global.tuple(Seq(otherAxisIndex.value).toPythonProxy)
        val axesPair = Jax.Dynamic.global.tuple(Seq(axesTuple1, axesTuple2).toPythonProxy)

        Tensor(Jax.jnp.tensordot(tensor.jaxValue, other.jaxValue, axes = axesPair))
  
  end Contraction

  object LinearAlgebra:
    
    extension [T <: Tuple : Labels, V : Value](t: Tensor[T, V])
      def norm: Tensor0[V] = Tensor0(Jax.jnp.linalg.norm(t.jaxValue))
      def inv: Tensor[T, V] = Tensor(Jax.jnp.linalg.inv(t.jaxValue))
      
      def det[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2])(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out, V] = 
        // JAX det only works on the last two axes (-2, -1). We must move the user's selected axes to the end.
        val moved = Jax.jnp.moveaxis(
          t.jaxValue, 
          source = Seq(idx1.value, idx2.value).toPythonProxy, 
          destination = Seq(-2, -1).toPythonProxy
        )
        Tensor(Jax.jnp.linalg.det(moved))
      
      def trace[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int=0)(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out, V] = Tensor(Jax.jnp.trace(t.jaxValue, offset = offset, axis1 = idx1.value, axis2 = idx2.value))

      def diagonal[L1 : Label, L2 : Label](axis1: Axis[L1], axis2: Axis[L2], offset: Int=0)(using 
        idx1: AxisIndex[T, L1], 
        idx2: AxisIndex[T, L2],
        remover: RemoverAll[T, (L1, L2)],
        namesOf: Labels[remover.Out] 
      ): Tensor[remover.Out *: L1 *: EmptyTuple, V] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset, axis1 = idx1.value, axis2 = idx2.value))

    extension [L1 : Label, L2 : Label, V : Value](t: Tensor2[L1, L2, V])
      def det: Tensor0[V] = Tensor0(Jax.jnp.linalg.det(t.jaxValue))
      def trace: Tensor0[V] = t.trace(0)
      def trace(offset: Int): Tensor0[V] = Tensor0(Jax.jnp.trace(t.jaxValue, offset = offset))
      def diagonal: Tensor1[L1, V] = t.diagonal(0)
      def diagonal(offset: Int): Tensor1[L1, V] = Tensor(Jax.jnp.diagonal(t.jaxValue, offset = offset))

  end LinearAlgebra

  // -----------------------------------------------------------
  // 4. Structural Operations (Isomorphisms)
  // Permutations and Views: T1 -> T2 (Size(T1) == Size(T2))
  // -----------------------------------------------------------
  object Structural:
    
    private object Util:
      
      type InsertBefore[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail => B *: A *: tail
        case h *: tail => h *: InsertBefore[tail, A, B]

      type InsertAfter[T <: Tuple, A, B] <: Tuple = T match
        case EmptyTuple => B *: EmptyTuple
        case A *: tail => A *: B *: tail
        case h *: tail => h *: InsertAfter[tail, A, B]

      type SliceIndex = Int | List[Int] | Range
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
        case EmptyTuple => ""
        case h *: EmptyTuple => h
        case h *: t => Op[h, TupleReduce[t, Op]]

      type TupleCombine[T <: Tuple] = T match
        case EmptyTuple => EmptyTuple
        case h *: EmptyTuple => h
        case h *: (h2 *: EmptyTuple) => h |*| h2
        case h *: t => h |*| TupleCombine[t]

      type TupleUnion[T <: Tuple] = T match
        case EmptyTuple => EmptyTuple
        case h *: EmptyTuple => h
        case h *: t => h | TupleUnion[t]

      type JoinNames[T <: Tuple] = TupleCombine[T]

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
      def where[T <: Tuple : Labels, V : Value](
        condition: Tensor[T, Bool],
        x: Tensor[T, V],
        y: Tensor[T, V]
      ): Tensor[T, V] =
        Tensor(Jax.jnp.where(condition.jaxValue, x.jaxValue, y.jaxValue))
    
    export TensorWhere.where

    def stack[L : Label, T <: Tuple : Labels, V : Value](
      tensors: Seq[Tensor[T, V]], 
      newAxis: Axis[L],
    ): Tensor[L *: T, V] = 
      require(tensors.nonEmpty, "Cannot stack an empty sequence of tensors")
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val stackedJaxValue = Jax.jnp.stack(jaxValuesSeq, axis = 0)
      Tensor(stackedJaxValue)

    def stack[NewL, L, T <: Tuple : Labels, V : Value](
      tensors: Seq[Tensor[T, V]], 
      newAxis: Axis[NewL],
      afterAxis: Axis[L],
    )(using 
      newLabel: Label[NewL],
      axisIndex: AxisIndex[T, L],
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

    def concatenate[L : Label, T <: Tuple : Labels, V : Value](
      tensors: Seq[Tensor[T, V]], 
      concatAxis: Axis[L],
    )(
      using axisIndex: AxisIndex[T, L],
    ): Tensor[T, V] =
      require(tensors.nonEmpty, "Cannot concatenate an empty sequence of tensors")
      val axisIdx = axisIndex.value
      val jaxValuesSeq = tensors.map(_.jaxValue).toPythonProxy
      val concatenatedJaxValue = Jax.jnp.concatenate(jaxValuesSeq, axis = axisIdx)
      Tensor(concatenatedJaxValue)

    extension [T <: Tuple : Labels, V : Value](tensor: Tensor[T, V])

      private def calcPyIndices[Inputs <: Tuple](
          inputs: Inputs,
          axesIndices: AxisIndices[T, ExtractLabels[Inputs]]
      ) = 

        val PySlice = py.Dynamic.global.slice
        val Colon = PySlice(py.None)
        val rank = tensor.shape.rank
        val indicesBuffer = collection.mutable.ArrayBuffer.fill[py.Any](rank)(Colon)

        val inputList = inputs.toList.asInstanceOf[List[(Any, Any)]]
        val targetDims: List[Int] = axesIndices.values

        targetDims.zip(inputList).foreach { 
          case (dimIndex, (_, sliceIndex)) =>
            val dimSize = tensor.shape.dimensions(dimIndex)
            sliceIndex match {
              case sliceSeq: List[Int] @unchecked => 
                indicesBuffer(dimIndex) = sliceSeq.map(py.Any.from).toPythonProxy
              case range: Range @unchecked => 
                indicesBuffer(dimIndex) = PySlice(range.head, range.last+1, range.step)
              case idx: Int =>
                indicesBuffer(dimIndex) = py.Any.from(idx)
            }
        }
        
        Jax.Dynamic.global.tuple(indicesBuffer.toSeq.toPythonProxy)

      def split[newL, splitL](newAxis: Axis[newL], splitAxis: Axis[splitL], interval: Int)(using 
        newLabel: Label[newL],
        axisIndex: AxisIndex[T, splitL],
      ): Tensor[InsertBefore[T, splitL, newL], V] = 
        val splitIdx = axisIndex.value
        val names = summon[Labels[T]].names
        val newNames = names.take(splitIdx) ++ Seq(newLabel.name) ++ names.drop(splitIdx)
        given Labels[InsertBefore[T, splitL, newL]] with
          val names = newNames.toSeq
        val (before, after) = tensor.shape.dimensions.splitAt(splitIdx)
        val newShape = before ++ Seq(interval, after.head / interval) ++ after.drop(1)
        println(newShape)
        Tensor(
          Jax.jnp.reshape(
            tensor.jaxValue,
            Jax.Dynamic.global.tuple(
              newShape.map(py.Any.from).toPythonProxy
            )
          )
        )

      def chunk[splitL : Label](splitAxis: Axis[splitL], interval: Int)(using 
        axisIndex: AxisIndex[T, splitL],
      ): Seq[Tensor[T, V]] = 
        val res = Jax.jnp.split(tensor.jaxValue, interval, axis = axisIndex.value).as[Seq[Jax.PyDynamic]]
        res.map(x => Tensor[T, V](x))

      def tile = ???
      def repeat = ???

      def slice[Inputs <: Tuple, LabelsToRemove <: Tuple, R <: Tuple](
        inputs: Inputs,
      )(using 
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        remover: RemoverAll.Aux[T, LabelsToRemove, R],
        axesIndices: AxisIndices[T, ExtractLabels[Inputs]],
        labels: Labels[R],
      ): Tensor[R, V] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices)
        Tensor(tensor.jaxValue.bracketAccess(pyIndices))

      def slice[L, I, LabelsToRemove <: Tuple, R <: Tuple](
        axisWithSliceIndex: (Axis[L], I)
      )(using 
        sliceExtractor: SliceLabelExtractor[Tuple1[(Axis[L], I)], LabelsToRemove],
        remover: RemoverAll.Aux[T, LabelsToRemove, R],
        axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]],
        labels: Labels[R],
      ): Tensor[R, V] = slice(Tuple1(axisWithSliceIndex))

      def set[Inputs <: Tuple, LabelsToRemove <: Tuple, R <: Tuple](
        inputs: Inputs
      )(using 
        sliceExtractor: SliceLabelExtractor[Inputs, LabelsToRemove],
        remover: RemoverAll.Aux[T, LabelsToRemove, R],
        axesIndices: AxisIndices[T, ExtractLabels[Inputs]],
        labels: Labels[R],
      )(value: Tensor[R, V]): Tensor[T, V] =
        val pyIndices = tensor.calcPyIndices(inputs, axesIndices)
        val result = tensor.jaxValue.at.bracketAccess(pyIndices).set(value.jaxValue)
        Tensor(result)

      def set[L, I, LabelsToRemove <: Tuple, R <: Tuple](
        axisWithSliceIndex: (Axis[L], I)
      )(using 
        sliceExtractor: SliceLabelExtractor[Tuple1[(Axis[L], I)], LabelsToRemove],
        remover: RemoverAll.Aux[T, LabelsToRemove, R],
        axesIndices: AxisIndices[T, ExtractLabels[Tuple1[(Axis[L], I)]]],
        labels: Labels[R]
      )(value: Tensor[R, V]): Tensor[T, V] = set(Tuple1(axisWithSliceIndex))(value)

      def rearrange[Axes <: Tuple](newOrder: Axes)(using Labels[UnwrapAxes[Axes]]): Tensor[UnwrapAxes[Axes], V] = 
        rearrange[Axes, EmptyTuple](newOrder, EmptyTuple)

      def rearrange[Axes <: Tuple, Dims <: Tuple](
          newOrder: Axes,
          dims: Dims,
      )(using 
        newLabels: Labels[UnwrapAxes[Axes]],
        extractor: DimExtractor[Dims],
      ): Tensor[UnwrapAxes[Axes], V] =
        def createEinopsPattern(fromPattern: String, toPattern: String): String =
          def cleanPattern(pattern: String): String =
            // to replace all a*b*c in pattern with (a b c), example:
            // "a*b*c d e f*g h" -> "(a b c) d e (f g) h"
            val regex = raw"([a-zA-Z0-9_]+(\*[a-zA-Z0-9_]+)+)".r
            regex.replaceAllIn(pattern, m => {
              val group = m.group(1)
              val replaced = group.split("\\*").mkString("(", " ", ")")
              replaced
            })
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

      def lift[O <: Tuple : Labels](newShape: Shape[O])(
        using ev: Subset[O, T] // Ensures T's axes are all present in O
      ): Tensor[O, V] =
        val t = tensor
        
        val currentNames = summon[Labels[T]].names
        val targetNames = summon[Labels[O]].names
        
        val targetOrder = targetNames.filter(currentNames.contains)
        val permutation = targetOrder.map(n => currentNames.indexOf(n))
        
        val alignedJax = if (permutation != currentNames.indices.toList) {
          Jax.jnp.transpose(t.jaxValue, permutation.toPythonProxy)
        } else {
          t.jaxValue
        }

        val currentShapeMap = currentNames.zip(t.shape.dimensions).toMap
        
        val intermediateShape = targetNames.map { name =>
          currentShapeMap.getOrElse(name, 1)
        }
        
        val reshapedJax = Jax.jnp.reshape(alignedJax, intermediateShape.toPythonProxy)
        Tensor(Jax.jnp.broadcast_to(reshapedJax, newShape.dimensions.toPythonProxy))

      def relabel[OldLabel : Label, NewLabel : Label](
        rename: (Axis[OldLabel], Axis[NewLabel]),
      )(using
        replacer: Replacer[T, OldLabel, NewLabel]
      ): Tensor[replacer.Out, V] = Tensor(tensor.jaxValue)

      def retag[newT <: Tuple](using newLabels: Labels[newT]): Tensor[newT, V] = 
        Tensor(tensor.jaxValue)(using newLabels)

      def relabelAll[newT <: Tuple](
        newAxes: newT,
      )(using 
        newLabels: Labels[UnwrapAxes[newT]],
        @implicitNotFound("Cannot convert tensor of shape ${T} to shape ${newT} due to size mismatch.")
        evSameSize: Tuple.Size[newT] =:= Tuple.Size[T],
      ): Tensor[UnwrapAxes[newT], V] = Tensor[UnwrapAxes[newT], V](tensor.jaxValue)
  
      def swap[L1 : Label, L2 : Label](
        axis1: Axis[L1],
        axis2: Axis[L2],
      )(using
        axisIndex1: AxisIndex[T, L1],
        axisIndex2: AxisIndex[T, L2],
      ): Tensor[Swap[T, L1, L2], V] =
        given Labels[Swap[T, L1, L2]] with
          def names = 
            val originalNames = summon[Labels[T]].names
            val ax1Name = summon[Label[L1]].name
            val ax2Name = summon[Label[L2]].name
            originalNames.map {
              case n if n == ax1Name => ax2Name
              case n if n == ax2Name => ax1Name
              case n => n
            }
        Tensor(Jax.jnp.swapaxes(tensor.jaxValue, axisIndex1.value, axisIndex2.value))

      def ravel: Tensor1[JoinNames[T], V] = 
        given Labels[Tuple1[JoinNames[T]]] with
          def names = List(summon[Labels[T]].names.mkString("*"))
        Tensor(Jax.jnp.ravel(tensor.jaxValue))

      def appendAxis[L : Label](axis: Axis[L])(using AxisAbsent[T, L]): Tensor[Tuple.Concat[T, Tuple1[L]], V] =
        import Labels.ForConcat.given
        val newShape = tensor.shape.dimensions :+ 1
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def prependAxis[L : Label](axis: Axis[L])(using AxisAbsent[T, L]): Tensor[Tuple.Concat[Tuple1[L], T], V] =
        import Labels.ForConcat.given
        val newShape = 1 +: tensor.shape.dimensions
        Tensor(Jax.jnp.reshape(tensor.jaxValue, newShape.toPythonProxy))

      def squeeze[L : Label, R <: Tuple](axis: Axis[L])(using 
        remover: RemoverAll.Aux[T, Tuple1[L], R],
        axisIndex: AxisIndex[T, L],
        labels: Labels[R],
      ): Tensor[R, V] =
        require(
          tensor.shape.dimensions(axisIndex.value) == 1, 
          s"Cannot squeeze axis ${axis} of size ${tensor.shape.dimensions(axisIndex.value)}"
        )
        Tensor(Jax.jnp.squeeze(tensor.jaxValue, axis = axisIndex.value))

  end Structural

  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------
  object Functional:

    object ZipVmap:

      type TensorsOf[Shapes <: Tuple, V] <: Tuple = Shapes match
        case EmptyTuple => EmptyTuple
        case head *: tail => head match
          case Tuple => Tensor[head, V] *: TensorsOf[tail, V]

      type ExtractShape[T] = T match
        case Tensor[s, v] => s
      
      type ExtractValue[T] = T match
        case Tensor[s, v] => v

      type ShapesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractShape]

      trait Zipper[Shapes <: Tuple, L, SlicedShapes <: Tuple, V]:
          def dimSize(tensors: TensorsOf[Shapes, V], axis: Axis[L]): Int
          def sliceAll(tensors: TensorsOf[Shapes, V], axis: Axis[L], idx: Int): TensorsOf[SlicedShapes, V]
        
      object Zipper:

        given empty[L, V]: Zipper[EmptyTuple, L, EmptyTuple, V] = new Zipper[EmptyTuple, L, EmptyTuple, V]:
          def dimSize(t: EmptyTuple, axis: Axis[L]) = 0
          def sliceAll(t: EmptyTuple, axis: Axis[L], idx: Int) = EmptyTuple
        
        given cons[
              HeadShape <: Tuple,
              TailShapes <: Tuple, 
              L, 
              HeadSliced <: Tuple, 
              TailSliced <: Tuple,
              V
            ](using
              remover: RemoverAll.Aux[HeadShape, L *: EmptyTuple, HeadSliced],
              axisIndex: AxisIndex[HeadShape, L],
              tailZipper: Zipper[TailShapes, L, TailSliced, V],
              labels1: Labels[HeadShape],
              labels2: Labels[HeadSliced],
              value: Value[V],
            ): Zipper[HeadShape *: TailShapes, L, HeadSliced *: TailSliced, V] = 
              new Zipper[HeadShape *: TailShapes, L, HeadSliced *: TailSliced, V]:

                def dimSize(tensors: TensorsOf[HeadShape *: TailShapes, V], axis: Axis[L]): Int =
                  val head = tensors.asInstanceOf[Tensor[HeadShape, V] *: Tuple].head
                  head.shape.dimensions(axisIndex.value)

                def sliceAll(tensors: TensorsOf[HeadShape *: TailShapes, V], axis: Axis[L], idx: Int): TensorsOf[HeadSliced *: TailSliced, V] =
                  val tuple = tensors.asInstanceOf[Tensor[HeadShape, V] *: TensorsOf[TailShapes, V]]
                  val slicedHead = tuple.head.slice(axis -> idx)(using labels1, value)
                  val slicedTail = tailZipper.sliceAll(tuple.tail, axis, idx)
                  (slicedHead *: slicedTail).asInstanceOf[TensorsOf[HeadSliced *: TailSliced, V]]
        
      case class ZipResult[L : Label, Shapes <: Tuple, SlicedShapes <: Tuple, V : Value](
          axis: Axis[L],
          tensors: TensorsOf[Shapes, V]
        )(using zipper: Zipper[Shapes, L, SlicedShapes, V]):

          def vmap[OutShape <: Tuple : Labels](
            f: TensorsOf[SlicedShapes, V] => Tensor[OutShape, V]
          ): Tensor[L *: OutShape, V] =

            val size = zipper.dimSize(tensors, axis)

            // Use Python Lambda for speed (optional optimization, but good for stack safety)
            // or standard mapping if you prefer your current implementation.
            // Your current Scala-side map implementation:
            val results = (0 until size).map { i =>
              val slicedTuple = zipper.sliceAll(tensors, axis, i)
              f(slicedTuple) // f now receives Tensors with concrete Shape types!
            }

            Structural.stack(results, axis)

      def zip[L : Label, Inputs <: Tuple, Sliced <: Tuple, V : Value](
        axis: Axis[L]
      )(
        tensors: Inputs
      )(using 
        zipper: Zipper[ShapesOf[Inputs], L, Sliced, V]
      ): ZipResult[L, ShapesOf[Inputs], Sliced, V] = 
        ZipResult(axis, tensors.asInstanceOf[TensorsOf[ShapesOf[Inputs], V]])

      def zipvmap[
          L : Label, 
          Inputs <: NonEmptyTuple, 
          OutShape <: Tuple : Labels,
          Sliced <: Tuple,
      ](
          axis: Axis[L]
      )(
          tensors: Inputs
      )(using 
          zipper: Zipper[ShapesOf[Inputs], L, Sliced, ExtractValue[Tuple.Head[Inputs]]]
      )(
          f: TensorsOf[Sliced, ExtractValue[Tuple.Head[Inputs]]] => Tensor[OutShape, ExtractValue[Tuple.Head[Inputs]]]
      )(using Value[ExtractValue[Tuple.Head[Inputs]]]): Tensor[L *: OutShape, ExtractValue[Tuple.Head[Inputs]]] = 
          zip(axis)(tensors).vmap(f)
        
    export ZipVmap.zipvmap

    extension [T <: Tuple : Labels, V : Value](t: Tensor[T, V])

      def vmap[VmapAxis : Label, OuterShape <: Tuple : Labels, R <: Tuple](
        axis: Axis[VmapAxis]
      )(using
        remover: Remover.Aux[T, VmapAxis, R],
        vmapAxisIndex: AxisIndex[T, VmapAxis],
      )(
          f: Tensor[R, V] => Tensor[OuterShape, V]
      )(using
        labels: Labels[R]
      ): Tensor[VmapAxis *: OuterShape, V] =
        val fpy = (jxpr: Jax.PyDynamic) =>
            val innerTensor = Tensor[R, V](jxpr)
            println(("A", innerTensor.shape))
            val result = f(innerTensor)
            result.jaxValue

        Tensor(Jax.jax_helper.vmap(fpy, vmapAxisIndex.value)(t.jaxValue))

      def vapply[L : Label, OutAxis : Label](
        axis: Axis[L]
      )(using
        axisIndex: AxisIndex[T, L],
        replacer: Replacer[T, L, OutAxis],
      )(
        f: Tensor[Tuple1[L], V] => Tensor[Tuple1[OutAxis], V]
      ): Tensor[replacer.Out, V] = 
        val fpy = (jxpr: Jax.PyDynamic) =>
          val inputTensor = Tensor[Tuple1[L], V](jxpr)
          val result = f(inputTensor)
          result.jaxValue

        Tensor(Jax.jnp.apply_along_axis(
          fpy, 
          axisIndex.value, 
          t.jaxValue
        ))

      def vreduce[L : Label, R <: Tuple](
        axis: Axis[L]
      )(
        f: Tensor[Tuple1[L], V] => Tensor0[V]
      )(using
        axisIndex: AxisIndex[T, L],
        remover: Remover.Aux[T, L, R],
        labels: Labels[R]
      ): Tensor[remover.Out, V] = 
        val fpy = (jxpr: Jax.PyDynamic) =>
          val inputTensor = Tensor[Tuple1[L], V](jxpr)
          val result = f(inputTensor)
          result.jaxValue

        Tensor(Jax.jnp.apply_along_axis(
          fpy, 
          axisIndex.value, 
          t.jaxValue
        ))

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
  object ScalarOps:
    extension [V : Value](t: Tensor0[V])
      def toInt: Int = t.jaxValue.item().as[Int]
      def toFloat: Float = t.jaxValue.item().as[Float]
      def toBool: Boolean = t.jaxValue.item().as[Boolean]

      @targetName("tensor0Pow")
      def pow(exponent: Tensor0[V]): Tensor0[V] = Tensor0(Jax.jnp.pow(t.jaxValue, exponent.jaxValue))

  object VectorOps:
    extension [L : Label, V : Value](t: Tensor1[L, V])
      def dot(other: Tensor1[L, V]): Tensor0[V] = t.innerDot(other)
      def innerDot(other: Tensor1[L, V]): Tensor0[V] = t.contract(Axis[L])(other)
      def outerDot[OtherLabel : Label](other: Tensor1[OtherLabel, V]): Tensor2[L, OtherLabel, V] = 
        val result = t.outerProduct(other)
        result

      def relabelTo[NewL : Label](newAxis: Axis[NewL]): Tensor1[NewL, V] = Tensor[Tuple1[NewL], V](t.jaxValue)

  object MatrixOps:
    extension [L1 : Label, L2 : Label, V : Value](t: Tensor2[L1, L2, V])
      def transpose: Tensor2[L2, L1, V] = t.rearrange((Axis[L2], Axis[L1]))

      @targetName("tensor2MatmulTensor2")
      def matmul[L3 : Label](other: Tensor2[L2, L3, V])(using 
        remover: Remover.Aux[(L1, L2), L2, Tuple1[L1]],
        otherRemover: Remover.Aux[(L2, L3), L2, Tuple1[L3]],
        idx1: AxisIndex[(L1, L2), L2],
        idx2: AxisIndex[(L2, L3), L2],
      ): Tensor2[L1, L3, V] =
        val result = t.contract(Axis[L2])(other)
        result

      @targetName("tensor2MatmulTensor1")
      def matmul(other: Tensor1[L2, V])(using 
        remover: Remover.Aux[(L1, L2), L2, Tuple1[L1]],
        otherRemover: Remover.Aux[Tuple1[L2], L2, EmptyTuple],
        idx1: AxisIndex[(L1, L2), L2],
        idx2: AxisIndex[Tuple1[L2], L2],
      ): Tensor[Tuple1[L1], V] =
        val result = t.contract(Axis[L2])(other)
        result

  export ScalarOps.*
  export VectorOps.*
  export MatrixOps.*

end TensorOps
