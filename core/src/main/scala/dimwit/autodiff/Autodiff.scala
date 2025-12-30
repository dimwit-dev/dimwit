package dimwit.autodiff

import dimwit.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Shape, AxisIndices}
import dimwit.jax.Jax
import me.shadaj.scalapy.py
import dimwit.tensor.Label

object Autodiff:

  type Gradient[In, Out] = Out match
    case EmptyTuple      => EmptyTuple
    case h *: t          => Gradient[In, h] *: Gradient[In, t]
    case Tensor[outS, v] => GradientTensorVsInput[In, outS, v]
    case _               => EmptyTuple

  type GradientTensorVsInput[In, OutShape <: Tuple, V] = In match
    case EmptyTuple      => EmptyTuple
    case h *: t          => GradientTensorVsInput[h, OutShape, V] *: GradientTensorVsInput[t, OutShape, V]
    case Tensor[inS, v2] => Tensor[Tuple.Concat[OutShape, inS], V]

  // TODO replace with TupledFunction when available (no longer experimental)
  def grad[T1, T2, V](f: (T1, T2) => Tensor0[V])(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], outTree: ToPyTree[Tensor0[V]]): (T1, T2) => (T1, T2) = (t1, t2) => grad(f.tupled)((t1, t2))
  def grad[T1, T2, T3, V](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: ToPyTree[T1], t2Tree: ToPyTree[T2], t3Tree: ToPyTree[T3], outTree: ToPyTree[Tensor0[V]]): (T1, T2, T3) => (T1, T2, T3) = (t1, t2, t3) => grad(f.tupled)((t1, t2, t3))

  def grad[Input, V](f: Input => Tensor0[V])(using
      inTree: ToPyTree[Input],
      outTree: ToPyTree[Tensor0[V]]
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
      inTree: ToPyTree[In],
      outTree: ToPyTree[Out],
      gradTree: ToPyTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) => outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacrev(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))

  def jacFwd[In, Out](f: In => Out)(using
      inTree: ToPyTree[In],
      outTree: ToPyTree[Out],
      gradTree: ToPyTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) => outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacfwd(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))
