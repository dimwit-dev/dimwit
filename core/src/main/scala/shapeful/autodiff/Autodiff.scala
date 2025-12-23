package shapeful.autodiff

import shapeful.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Shape, AxisIndices, Value}
import shapeful.jax.Jax
import me.shadaj.scalapy.py

object Autodiff:

  type Gradient[In, Out] = Out match
    case EmptyTuple => EmptyTuple
    case h *: t => Gradient[In, h] *: Gradient[In, t]
    case Tensor[outS, v] => GradientTensorVsInput[In, outS, v]
    case _ => EmptyTuple

  type GradientTensorVsInput[In, OutShape <: Tuple, V] = In match
    case EmptyTuple => EmptyTuple
    case h *: t => GradientTensorVsInput[h, OutShape, V] *: GradientTensorVsInput[t, OutShape, V]
    case Tensor[inS, v2] => Tensor[Tuple.Concat[OutShape, inS], V]

  def grad[Input, V : Value](f: Input => Tensor0[V])(using 
    inTree: ToPyTree[Input],
    outTree: ToPyTree[Tensor0[V]],
  ): Input => Input =

    val fpy = (jxpr: py.Dynamic) =>
      val x = inTree.fromPyTree(jxpr)
      outTree.toPyTree(f(x))

    val gpy = Jax.jax_helper.grad(fpy)

    (params: Input) =>
      val xpy = inTree.toPyTree(params)
      val pygrad = gpy(xpy)
      inTree.fromPyTree(pygrad).asInstanceOf[Input]

  def jacobian[In, Out](f: In => Out)(using 
    inTree: ToPyTree[In],
    outTree: ToPyTree[Out],
    gradTree: ToPyTree[Gradient[In, Out]] // Compiler infers this!
  ): In => Gradient[In, Out] =

    val fpy = (jxpr: py.Dynamic) =>
      val x = inTree.fromPyTree(jxpr)
      outTree.toPyTree(f(x))

    val jpy = Jax.jax_helper.jacobian(fpy)

    (params: In) =>
      val xpy = inTree.toPyTree(params)
      val res = jpy(xpy)
      gradTree.fromPyTree(res)

  def jacRev[In, Out](f: In => Out)(using 
    inTree: ToPyTree[In], outTree: ToPyTree[Out], gradTree: ToPyTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) => outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacrev(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))

  def jacFwd[In, Out](f: In => Out)(using 
    inTree: ToPyTree[In], outTree: ToPyTree[Out], gradTree: ToPyTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) => outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacfwd(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))