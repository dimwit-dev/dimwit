package shapeful.jax

import shapeful.tensor.{Tensor, Shape, Labels}
import shapeful.jax.{Jax, JaxDType}
import shapeful.autodiff.ToPyTree
import me.shadaj.scalapy.py
import shapeful.tensor.Value

object Jit:

  def jit[PyTree: ToPyTree, OutT <: Tuple : Labels, V : Value](
    f: PyTree => Tensor[OutT, V]
  ): PyTree => Tensor[OutT, V] =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[PyTree].fromPyTree(pyTreePy)
      val result = f(pyTree)
      result.jaxValue

    // Apply JIT compilation
    val jitted = Jax.jax_helper.jit_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: PyTree) =>
      val pyTreePy = ToPyTree[PyTree].toPyTree(pyTree)
      val resultJax = jitted(pyTreePy)
      Tensor.fromPy[OutT, V](resultJax)

  def jit2[PyTree: ToPyTree, OutT <: Tuple : Labels](
    f: PyTree => PyTree
  ): PyTree => PyTree =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[PyTree].fromPyTree(pyTreePy)
      val result = f(pyTree)
      val tt = ToPyTree[PyTree].toPyTree(result)
      tt

    // Apply JIT compilation
    val jitted = Jax.jax_helper.jit_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: PyTree) =>
      val pyTreePy = ToPyTree[PyTree].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[PyTree].fromPyTree(res)
