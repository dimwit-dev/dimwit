
import scala.annotation.targetName
package object shapeful:

  import scala.compiletime.ops.string.+

  object StringLabelMath:
    infix type *[A <: String, B <: String] = A + "*" + B

  trait Prime[T]
  object Prime:
    given[L](using label: Label[L]): Label[Prime[L]] with
      val name: String = s"${label.name}'"

    type RemovePrimes[T <: Tuple] <: Tuple = T match
      case EmptyTuple => EmptyTuple
      case Prime[l] *: tail => l *: RemovePrimes[tail]
      case h *: tail => h *: RemovePrimes[tail]

    extension[T <: Tuple : Labels, V : Value](tensor: Tensor[T, V])
      def dropPrimes: Tensor[RemovePrimes[T], V] =
        given newLabels: Labels[RemovePrimes[T]] with
          val names: List[String] = 
            val oldLabels = summon[Labels[T]]
            oldLabels.names.toList.map(_.replace("'", ""))
        Tensor.fromPy[RemovePrimes[T], V](tensor.jaxValue)

  @targetName("On") 
  infix trait ~[A, B]
  object `~`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A ~ B] with
      val name: String = s"${labelA.name}_on_${labelB.name}"
  
  @targetName("Combined")
  infix trait |*|[A, B]
  object `|*|`:
    given [A, B](using labelA: Label[A], labelB: Label[B]): Label[A |*| B] with
      val name: String = s"${labelA.name}*${labelB.name}"

  // Export tensor and related types
  export shapeful.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Tensor3}
  export shapeful.tensor.{Shape, Shape0, Shape1, Shape2, Shape3}
  export shapeful.tensor.{DType, Device}
  export shapeful.tensor.{Label, Labels, Axis, AxisIndex, AxisIndices, Dim}
  export shapeful.tensor.Value
  
  // Opaque types for DTypes - clean imports and display without .type suffix
  opaque type Float32 = DType.Float32.type
  object Float32:
    given Value[Float32] = summon[Value[DType.Float32.type]]
  
  opaque type Float64 = DType.Float64.type
  object Float64:
    given Value[Float64] = summon[Value[DType.Float64.type]]
  
  opaque type Int32 = DType.Int32.type
  object Int32:
    given Value[Int32] = summon[Value[DType.Int32.type]]
  
  opaque type Int64 = DType.Int64.type
  object Int64:
    given Value[Int64] = summon[Value[DType.Int64.type]]
  
  opaque type Int16 = DType.Int16.type
  object Int16:
    given Value[Int16] = summon[Value[DType.Int16.type]]
  
  opaque type Int8 = DType.Int8.type
  object Int8:
    given Value[Int8] = summon[Value[DType.Int8.type]]
  
  opaque type UInt32 = DType.UInt32.type
  object UInt32:
    given Value[UInt32] = summon[Value[DType.UInt32.type]]
  
  opaque type UInt16 = DType.UInt16.type
  object UInt16:
    given Value[UInt16] = summon[Value[DType.UInt16.type]]
  
  opaque type UInt8 = DType.UInt8.type
  object UInt8:
    given Value[UInt8] = summon[Value[DType.UInt8.type]]
  
  opaque type Bool = DType.Bool.type
  object Bool:
    given Value[Bool] = summon[Value[DType.Bool.type]]
  
  // Export type helpers
  export shapeful.tensor.Axis.UnwrapAxes
  export shapeful.tensor.TupleHelpers.*
  export shapeful.tensor.Broadcast
  export Prime.*
  
  // Export operations
  export shapeful.tensor.TensorOps.*
  
  // Export automatic differentiation
  export shapeful.autodiff.{Autodiff, TensorTree, ToPyTree}

  // Export Just-in-Time compilation
//  export shapeful.jax.Jit.{jit, jit2}

