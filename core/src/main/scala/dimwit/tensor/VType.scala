package dimwit.tensor

import dimwit.tensor.TensorOps.HasDType

object VType:
  def apply[V](tensor: Tensor[?, V]): VType[V] = VTypeImpl[V](tensor.dtype)
  def apply[A: HasDType]: VType[A] = VTypeImpl[A](summon[HasDType[A]].dtype)

sealed trait VType[A]:
  def dtype: DType

private case class VTypeImpl[A](override val dtype: DType) extends VType[A]
