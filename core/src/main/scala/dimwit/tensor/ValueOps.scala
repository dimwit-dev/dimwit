package dimwit.tensor

import dimwit.tensor.TensorOps.IsNumber
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.tensorops.ElementWiseOps.add
import dimwit.tensor.tensorops.ElementWiseOps.subtract
import dimwit.tensor.tensorops.ElementWiseOps.multiply
import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast
import dimwit.tensor.tensorops.ElementWiseOps.divide

object ValueOps:

  extension [V: IsNumber](t: Tensor0[V])

    def +(t2: Tensor0[V]): Tensor0[V] = TensorOps.add(t, t2)
    def -(t2: Tensor0[V]): Tensor0[V] = TensorOps.subtract(t, t2)
    def *(t2: Tensor0[V]): Tensor0[V] = TensorOps.multiply(t, t2)

  extension [V: IsFloating](t: Tensor0[V])

    def /(scalar: Tensor0[V]): Tensor0[V] = TensorOps.divide(t, scalar)

  extension (scalar: Float)

    def +[V: IsNumber](t: Tensor0[V]): Tensor0[V] = add(Tensor0.likeDType(t)(scalar), t)
    def +![T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])(using bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = bc.applyTo(Tensor0.likeDType(t)(scalar), t)(add)

    def -[V: IsNumber](t: Tensor0[V]): Tensor0[V] = subtract(Tensor0.likeDType(t)(scalar), t)
    def -![T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])(using bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = bc.applyTo(Tensor0.likeDType(t)(scalar), t)(subtract)

    def *[V: IsNumber](t: Tensor0[V]): Tensor0[V] = multiply(Tensor0.likeDType(t)(scalar), t)
    def *![T <: Tuple: Labels, V: IsNumber](t: Tensor[T, V])(using bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = bc.applyTo(Tensor0.likeDType(t)(scalar), t)(multiply)

  extension (scalar: Float)

    def /[V: IsFloating](t: Tensor0[V]): Tensor0[V] = divide(Tensor0.likeDType(t)(scalar), t)
    def /![T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V])(using bc: Broadcast[EmptyTuple, T, V]): Tensor[bc.Out, V] = bc.applyTo(Tensor0.likeDType(t)(scalar), t)(divide)
