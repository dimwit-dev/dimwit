package dimwit.nn

import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.python.PyBridge.toPyTensor
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensor.*

object ActivationFunctions:

  def sigmoid[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] =
    val x = Jax.jnn.sigmoid
    liftPyTensor(Jax.jnn.sigmoid(toPyTensor(t)))

  def relu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] =
    liftPyTensor(Jax.jnn.relu(toPyTensor(t)))

  def gelu[T <: Tuple: Labels, V: IsFloating](t: Tensor[T, V]): Tensor[T, V] =
    liftPyTensor(Jax.jnn.gelu(toPyTensor(t)))

  def softmax[L: Label, V: IsFloating](t: Tensor1[L, V]): Tensor1[L, V] =
    liftPyTensor(Jax.jnn.softmax(toPyTensor(t), axis = 0))
