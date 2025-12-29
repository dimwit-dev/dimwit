package nn

import dimwit.*
import dimwit.jax.Jax

object ActivationFunctions:

  // TODO rewrite relu, sigmoid to JAX

  def sigmoid[T <: Tuple: Labels](t: Tensor[T, Float]): Tensor[T, Float] =
    val ones = Tensor.ones(t.shape, t.vtype)
    val minust = t.scale(-Tensor0.one(t.vtype))
    ones / (ones + (minust).exp)

  def relu[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    val zeros = Tensor.zeros(t.shape, t.vtype)
    maximum(t, zeros)

  def gelu[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    Tensor(Jax.jnn.gelu(t.jaxValue))

  def softmax[L: Label, V](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor(Jax.jnn.softmax(t.jaxValue, axis = 0))
