package nn

import dimwit.*
import dimwit.jax.Jax

object ActivationFunctions:

  def sigmoid[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    Tensor(Jax.jnn.sigmoid(t.jaxValue))

  def relu[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    Tensor(Jax.jnn.relu(t.jaxValue))

  def gelu[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    Tensor(Jax.jnn.gelu(t.jaxValue))

  def softmax[L: Label, V](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnn.softmax(t.jaxValue, axis = 0))
