package dimwit.jax

import dimwit.tensor.{Tensor, Shape, Labels}
import dimwit.jax.{Jax, JaxDType}
import dimwit.autodiff.ToPyTree
import me.shadaj.scalapy.py

object Jit:

  def jit[InPyTree: ToPyTree, OutPyTree: ToPyTree](
      f: InPyTree => OutPyTree
  ): InPyTree => OutPyTree =

    // Python function that accepts a pytree
    val fpy = (pyTreePy: Jax.PyDynamic) =>
      val pyTree = ToPyTree[InPyTree].fromPyTree(pyTreePy)
      val result = f(pyTree)
      val tt = ToPyTree[OutPyTree].toPyTree(result)
      tt

    // Apply JIT compilation
    val jitted = Jax.jax_helper.jit_fn(fpy)

    // Return a function that converts Scala types to pytree and applies jitted function
    (pyTree: InPyTree) =>
      val pyTreePy = ToPyTree[InPyTree].toPyTree(pyTree)
      val res = jitted(pyTreePy)
      ToPyTree[OutPyTree].fromPyTree(res)
