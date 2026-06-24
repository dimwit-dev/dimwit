package dimwit.tensor.tensorops

import scala.annotation.implicitNotFound
import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.TupleHelpers.StrictSubset

object TensorOpsUtil:

  import dimwit.tensor.TensorOps.broadcastTo

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
