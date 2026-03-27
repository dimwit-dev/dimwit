package dimwit.autodiff

import dimwit.OnError
import dimwit.tensor.{Tensor, Tensor0, Tensor1, Tensor2, Shape}
import dimwit.tensor.ShapeTypeHelpers.AxisIndices
import dimwit.tensor.TupleHelpers.PrimeConcatType
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
    case Tensor[inS, v2] => Tensor[PrimeConcatType[OutShape, inS], V]

  // TODO replace with TupledFunction when available (no longer experimental)
  def grad[T1, T2, V](f: (T1, T2) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], outTree: TensorTree[Tensor0[V]]): (T1, T2) => Grad[(T1, T2)] = (t1, t2) => grad(f.tupled)((t1, t2))
  def grad[T1, T2, T3, V](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], t3Tree: TensorTree[T3], outTree: TensorTree[Tensor0[V]]): (T1, T2, T3) => Grad[(T1, T2, T3)] = (t1, t2, t3) => grad(f.tupled)((t1, t2, t3))

  def grad[Input, V](f: Input => Tensor0[V])(using
      inTree: TensorTree[Input],
      outTree: TensorTree[Tensor0[V]]
  ): Input => Grad[Input] =

    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromPyTree(jxpr)
        outTree.toPyTree(f(x))

    val gpy = Jax.jax_helper.grad(fpy)

    (params: Input) =>
      val pyParams = inTree.toPyTree(params)
      val pyGrad = gpy(pyParams)
      Grad(inTree.fromPyTree(pyGrad).asInstanceOf[Input])

  def valueAndGrad[T1, T2, V](f: (T1, T2) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], outTree: TensorTree[Tensor0[V]]): (T1, T2) => (Tensor0[V], Grad[(T1, T2)]) = (t1, t2) => valueAndGrad(f.tupled)((t1, t2))
  def valueAndGrad[T1, T2, T3, V](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], t3Tree: TensorTree[T3], outTree: TensorTree[Tensor0[V]]): (T1, T2, T3) => (Tensor0[V], Grad[(T1, T2, T3)]) = (t1, t2, t3) => valueAndGrad(f.tupled)((t1, t2, t3))

  def valueAndGrad[Input, V](f: Input => Tensor0[V])(using
      inTree: TensorTree[Input],
      outTree: TensorTree[Tensor0[V]]
  ): Input => (Tensor0[V], Grad[Input]) =

    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromPyTree(jxpr)
        outTree.toPyTree(f(x))

    val gpy = Jax.jax_helper.value_and_grad(fpy)

    (params: Input) =>
      val pyParams = inTree.toPyTree(params)
      val r = gpy(pyParams)
      val pyValue = r.bracketAccess(0)
      val pyGrad = r.bracketAccess(1)
      (Tensor(pyValue), Grad(inTree.fromPyTree(pyGrad).asInstanceOf[Input]))

  def jacobian[In, Out](f: In => Out)(using
      inTree: TensorTree[In],
      outTree: TensorTree[Out],
      gradTree: TensorTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =

    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromPyTree(jxpr)
        outTree.toPyTree(f(x))

    val jpy = Jax.jax_helper.jacobian(fpy)

    (params: In) =>
      val xpy = inTree.toPyTree(params)
      val res = jpy(xpy)
      gradTree.fromPyTree(res)

  def jacRev[In, Out](f: In => Out)(using
      inTree: TensorTree[In],
      outTree: TensorTree[Out],
      gradTree: TensorTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacrev(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))

  def jacFwd[In, Out](f: In => Out)(using
      inTree: TensorTree[In],
      outTree: TensorTree[Out],
      gradTree: TensorTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacfwd(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))
