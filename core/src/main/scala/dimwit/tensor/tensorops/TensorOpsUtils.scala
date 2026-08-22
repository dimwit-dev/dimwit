package dimwit.tensor.tensorops

import dimwit.tensor.Labels
import dimwit.tensor.Tensor
import dimwit.tensor.TupleHelpers.StrictSubset

import scala.annotation.implicitNotFound

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

  /** Broadcasts three tensors to their common shape, which is the one of the three shapes containing all
    * axes of the other two. As for [[Broadcast]] at least one of the tensors has to be broadcast.
    */
  @implicitNotFound(
    "Cannot broadcast tensors of shapes ${T1}, ${T2} and ${T3}. One of them must contain all axes of the other two. If all same shape no broadcasting allowed!"
  )
  sealed trait Broadcast3[T1 <: Tuple, T2 <: Tuple, T3 <: Tuple, V]:
    type Out <: Tuple
    given labelsOut: Labels[Out]
    def broadcast[V1](t1: Tensor[T1, V1], t2: Tensor[T2, V], t3: Tensor[T3, V]): (Tensor[Out, V1], Tensor[Out, V], Tensor[Out, V])

  object Broadcast3 extends Broadcast3LowPriority:

    /** `t2` and `t3` broadcast against each other, `t1` already has their common shape. */
    given valuesBroadcast[O <: Tuple, T2 <: Tuple, T3 <: Tuple, V](using
        bc: Broadcast[T2, T3, V] { type Out = O }
    ): Broadcast3[O, T2, T3, V] with
      type Out = O
      val labelsOut = bc.labelsOut
      def broadcast[V1](t1: Tensor[O, V1], t2: Tensor[T2, V], t3: Tensor[T3, V]) =
        val (bt2, bt3) = bc.broadcast(t2, t3)
        (t1, bt2, bt3)

    /** `t2` and `t3` broadcast against each other, `t1` is broadcast to their common shape. */
    given conditionAndValuesBroadcast[T1 <: Tuple: Labels, T2 <: Tuple, T3 <: Tuple, O <: Tuple, V](using
        bc: Broadcast[T2, T3, V] { type Out = O },
        ev: StrictSubset[T1, O]
    ): Broadcast3[T1, T2, T3, V] with
      type Out = O
      val labelsOut = bc.labelsOut
      def broadcast[V1](t1: Tensor[T1, V1], t2: Tensor[T2, V], t3: Tensor[T3, V]) =
        given Labels[O] = bc.labelsOut
        val (bt2, bt3) = bc.broadcast(t2, t3)
        (t1.broadcastTo[O](bt2.shape), bt2, bt3)

    /** `t2` and `t3` have the same shape, only `t1` is broadcast to it. */
    given conditionBroadcast[T1 <: Tuple: Labels, T <: Tuple: Labels, V](using
        ev: StrictSubset[T1, T]
    ): Broadcast3[T1, T, T, V] with
      type Out = T
      val labelsOut = summon[Labels[T]]
      def broadcast[V1](t1: Tensor[T1, V1], t2: Tensor[T, V], t3: Tensor[T, V]) =
        (t1.broadcastTo[T](t2.shape), t2, t3)

  trait Broadcast3LowPriority:

    /** `t2` and `t3` are both broadcast to the shape of `t1`. */
    given valuesBroadcastToFirst[T1 <: Tuple: Labels, T2 <: Tuple: Labels, T3 <: Tuple: Labels, V](using
        ev2: StrictSubset[T2, T1],
        ev3: StrictSubset[T3, T1]
    ): Broadcast3[T1, T2, T3, V] with
      type Out = T1
      val labelsOut = summon[Labels[T1]]
      def broadcast[V1](t1: Tensor[T1, V1], t2: Tensor[T2, V], t3: Tensor[T3, V]) =
        (t1, t2.broadcastTo[T1](t1.shape), t3.broadcastTo[T1](t1.shape))

end TensorOpsUtil
