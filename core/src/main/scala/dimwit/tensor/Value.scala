package dimwit.tensor

import dimwit.stats.Prob
import dimwit.stats.LogProb
import scala.compiletime.ops.double
import java.nio.ByteBuffer

trait ExecutionType[V]:
  def dtype: DType

object ExecutionType:

  def apply[V](using executionType: ExecutionType[V]): ExecutionType[V] = executionType

  given floatValue: ExecutionType[Float] with
    def dtype: DType = DType.Float32

  given intValue: ExecutionType[Int] with
    def dtype: DType = DType.Int32

  given booleanValue: ExecutionType[Boolean] with
    def dtype: DType = DType.Bool

  given byteValue: ExecutionType[Byte] with
    def dtype: DType = DType.Int8

  given doubleValue: ExecutionType[Double] with
    def dtype: DType = DType.Float64

  given prob: ExecutionType[Prob] with
    def dtype: DType = summon[ExecutionType[Float]].dtype

  given logProb: ExecutionType[LogProb] with
    def dtype: DType = summon[ExecutionType[Float]].dtype

object VType:
  def apply[V](tensor: Tensor[?, V]): VType[V] = new OfImpl[V](tensor.dtype)
  def apply[A: ExecutionType]: VType[A] = new OfImpl[A](summon[ExecutionType[A]].dtype)

sealed trait VType[A]:
  def dtype: DType

class OfImpl[A](val dtype: DType) extends VType[A]

case class ExecutionTypeFor[V](dtype: DType) extends ExecutionType[V]
