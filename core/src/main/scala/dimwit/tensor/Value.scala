package dimwit.tensor

trait ExecutionType[V]:
  def dtype: DType

object ExecutionType:

  given floatValue: ExecutionType[Float] with
    def dtype: DType = DType.Float32
  given intValue: ExecutionType[Int] with
    def dtype: DType = DType.Int32
  given booleanValue: ExecutionType[Boolean] with
    def dtype: DType = DType.Bool

object VType:
  def apply[V](tensor: Tensor[?, V]): VType[V] = new OfImpl[V](tensor.dtype)
  def apply[A: ExecutionType]: VType[A] = new OfImpl[A](summon[ExecutionType[A]].dtype)

sealed trait VType[A]:
  def dtype: DType

class OfImpl[A](val dtype: DType) extends VType[A]
