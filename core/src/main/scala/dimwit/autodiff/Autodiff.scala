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
  def grad[T1, T2, V](f: (T1, T2) => Tensor0[V])(using t1Tree: ToFloatTensorTree[T1], t2Tree: ToFloatTensorTree[T2], outTree: ToTensorTree[Tensor0[V]]): (T1, T2) => Grad[(T1, T2)] = (t1, t2) => grad(f.tupled)((t1, t2))
  def grad[T1, T2, T3, V](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: ToFloatTensorTree[T1], t2Tree: ToFloatTensorTree[T2], t3Tree: ToFloatTensorTree[T3], outTree: ToTensorTree[Tensor0[V]]): (T1, T2, T3) => Grad[(T1, T2, T3)] = (t1, t2, t3) => grad(f.tupled)((t1, t2, t3))

  def grad[Input, V](f: Input => Tensor0[V])(using
      inTree: ToFloatTensorTree[Input],
      outTree: ToTensorTree[Tensor0[V]]
  ): Input => Grad[Input] =

    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromTensorTree(TensorTree[Input](jxpr))
        outTree.toTensorTree(f(x)).pyTree

    val gpy = Jax.jax_helper.grad(fpy)

    (params: Input) =>
      val xpy = inTree.toTensorTree(params).pyTree
      val pygrad = gpy(xpy)
      Grad(inTree.fromTensorTree(TensorTree[Input](pygrad)).asInstanceOf[Input])

  def jacobian[In, Out](f: In => Out)(using
      inTree: ToFloatTensorTree[In],
      outTree: ToTensorTree[Out],
      gradTree: ToTensorTree[Gradient[In, Out]] // Compiler infers this!
  ): In => Gradient[In, Out] =

    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromTensorTree(TensorTree[In](jxpr))
        outTree.toTensorTree(f(x)).pyTree

    val jpy = Jax.jax_helper.jacobian(fpy)

    (params: In) =>
      val xpy = inTree.toTensorTree(params).pyTree
      val res = jpy(xpy)
      gradTree.fromTensorTree(TensorTree[Gradient[In, Out]](res))

  def jacRev[In, Out](f: In => Out)(using
      inTree: ToFloatTensorTree[In],
      outTree: ToTensorTree[Out],
      gradTree: ToTensorTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toTensorTree(f(inTree.fromTensorTree(TensorTree[In](jxpr)))).pyTree
    val jpy = Jax.jax_helper.jacrev(fpy)
    (params: In) => gradTree.fromTensorTree(TensorTree[Gradient[In, Out]](jpy(inTree.toTensorTree(params).pyTree)))

  def jacFwd[In, Out](f: In => Out)(using
      inTree: ToFloatTensorTree[In],
      outTree: ToTensorTree[Out],
      gradTree: ToTensorTree[Gradient[In, Out]]
  ): In => Gradient[In, Out] =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toTensorTree(f(inTree.fromTensorTree(TensorTree[In](jxpr)))).pyTree
    val jpy = Jax.jax_helper.jacfwd(fpy)
    (params: In) => gradTree.fromTensorTree(TensorTree[Gradient[In, Out]](jpy(inTree.toTensorTree(params).pyTree)))
