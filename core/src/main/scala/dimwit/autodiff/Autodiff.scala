package dimwit.autodiff

import dimwit.python.PyIndex.itemAt
import dimwit.OnError
import dimwit.jax.Jax
import dimwit.prime.PrimeConcat
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.TensorOps.IsFloating
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py

import scala.NamedTuple.NamedTuple
import scala.annotation.implicitNotFound
import scala.deriving.Mirror

object Autodiff:

  /** The derivative of a function `In => Out`: the structure of `Out`, with every
    * tensor in it replaced by its derivative with respect to the whole of `In`.
    *
    * Instances are open, so this stays in step with [[TensorTree]] - a structure
    * a user has given a `TensorTree` can be given a `Gradient` too.
    */
  @implicitNotFound(
    "Cannot express the derivative of ${Out} with respect to ${In}. Both must be built from tensors, tuples, named tuples or case classes with a TensorTree instance"
  )
  trait Gradient[In, Out]:
    type Result

  object Gradient extends GradientLowPriority:

    type Aux[In, Out, R] = Gradient[In, Out] { type Result = R }

    private[autodiff] def instance[In, Out, R]: Aux[In, Out, R] =
      new Gradient[In, Out]:
        type Result = R

    /** An output tensor is differentiated against every tensor in the input. */
    given tensor[In, OutShape <: Tuple, V, R](using
        vsInput: GradientTensorVsInput.Aux[In, OutShape, V, R]
    ): Aux[In, Tensor[OutShape, V], R] = instance

    given emptyTuple[In]: Aux[In, EmptyTuple, EmptyTuple] = instance

    given consTuple[In, H, HR, T <: Tuple, TR <: Tuple](using
        head: Aux[In, H, HR],
        tail: Aux[In, T, TR]
    ): Aux[In, H *: T, HR *: TR] = instance

    given namedTuple[In, N <: Tuple, Vs <: Tuple, R <: Tuple](using
        values: Aux[In, Vs, R]
    ): Aux[In, NamedTuple[N, Vs], NamedTuple[N, R]] = instance

  trait GradientLowPriority:
    /** A case class output becomes a named tuple of its field derivatives,
      * keeping the field names. Lower priority than the tuple cases, since
      * tuples are Products too.
      */
    given product[In, P <: Product, Names <: Tuple, Elems <: Tuple, R <: Tuple](using
        m: Mirror.ProductOf[P] { type MirroredElemLabels = Names; type MirroredElemTypes = Elems },
        elems: Gradient.Aux[In, Elems, R]
    ): Gradient.Aux[In, P, NamedTuple[Names, R]] = Gradient.instance

  /** The derivative of one output tensor of shape `OutShape` with respect to the
    * whole input structure `In`. Mirrors [[Gradient]], recursing on the input.
    */
  trait GradientTensorVsInput[In, OutShape <: Tuple, V]:
    type Result

  object GradientTensorVsInput extends GradientTensorVsInputLowPriority:

    type Aux[In, OutShape <: Tuple, V, R] = GradientTensorVsInput[In, OutShape, V] { type Result = R }

    private[autodiff] def instance[In, OutShape <: Tuple, V, R]: Aux[In, OutShape, V, R] =
      new GradientTensorVsInput[In, OutShape, V]:
        type Result = R

    /** Output axes first, then the input axes, primed where they collide. */
    given tensor[InShape <: Tuple, InV, OutShape <: Tuple, V, O <: Tuple](using
        concat: PrimeConcat.Aux[OutShape, InShape, O]
    ): Aux[Tensor[InShape, InV], OutShape, V, Tensor[O, V]] = instance

    given emptyTuple[OutShape <: Tuple, V]: Aux[EmptyTuple, OutShape, V, EmptyTuple] = instance

    given consTuple[H, HR, T <: Tuple, TR <: Tuple, OutShape <: Tuple, V](using
        head: Aux[H, OutShape, V, HR],
        tail: Aux[T, OutShape, V, TR]
    ): Aux[H *: T, OutShape, V, HR *: TR] = instance

    given namedTuple[N <: Tuple, Vs <: Tuple, R <: Tuple, OutShape <: Tuple, V](using
        values: Aux[Vs, OutShape, V, R]
    ): Aux[NamedTuple[N, Vs], OutShape, V, NamedTuple[N, R]] = instance

  trait GradientTensorVsInputLowPriority:
    given product[P <: Product, Names <: Tuple, Elems <: Tuple, R <: Tuple, OutShape <: Tuple, V](using
        m: Mirror.ProductOf[P] { type MirroredElemLabels = Names; type MirroredElemTypes = Elems },
        elems: GradientTensorVsInput.Aux[Elems, OutShape, V, R]
    ): GradientTensorVsInput.Aux[P, OutShape, V, NamedTuple[Names, R]] = GradientTensorVsInput.instance

  // TODO replace with TupledFunction when available (no longer experimental)
  def grad[T1, T2, V: IsFloating](f: (T1, T2) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], outTree: TensorTree[Tensor0[V]]): (T1, T2) => Grad[(T1, T2)] = (t1, t2) => grad(f.tupled)((t1, t2))
  def grad[T1, T2, T3, V: IsFloating](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], t3Tree: TensorTree[T3], outTree: TensorTree[Tensor0[V]]): (T1, T2, T3) => Grad[(T1, T2, T3)] = (t1, t2, t3) => grad(f.tupled)((t1, t2, t3))

  def grad[Input, V: IsFloating](f: Input => Tensor0[V])(using
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

  def valueAndGrad[T1, T2, V: IsFloating](f: (T1, T2) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], outTree: TensorTree[Tensor0[V]]): (T1, T2) => (Tensor0[V], Grad[(T1, T2)]) = (t1, t2) => valueAndGrad(f.tupled)((t1, t2))
  def valueAndGrad[T1, T2, T3, V: IsFloating](f: (T1, T2, T3) => Tensor0[V])(using t1Tree: TensorTree[T1], t2Tree: TensorTree[T2], t3Tree: TensorTree[T3], outTree: TensorTree[Tensor0[V]]): (T1, T2, T3) => (Tensor0[V], Grad[(T1, T2, T3)]) = (t1, t2, t3) => valueAndGrad(f.tupled)((t1, t2, t3))

  def valueAndGrad[Input, V: IsFloating](f: Input => Tensor0[V])(using
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
      val pyValue = r.itemAt(0)
      val pyGrad = r.itemAt(1)
      (Tensor(pyValue), Grad(inTree.fromPyTree(pyGrad).asInstanceOf[Input]))

  def jacobian[In, Out](f: In => Out)(using
      inTree: TensorTree[In],
      outTree: TensorTree[Out],
      gradient: Gradient[In, Out],
      gradTree: TensorTree[gradient.Result]
  ): In => gradient.Result =

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
      gradient: Gradient[In, Out],
      gradTree: TensorTree[gradient.Result]
  ): In => gradient.Result =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacrev(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))

  def jacFwd[In, Out](f: In => Out)(using
      inTree: TensorTree[In],
      outTree: TensorTree[Out],
      gradient: Gradient[In, Out],
      gradTree: TensorTree[gradient.Result]
  ): In => gradient.Result =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        outTree.toPyTree(f(inTree.fromPyTree(jxpr)))
    val jpy = Jax.jax_helper.jacfwd(fpy)
    (params: In) => gradTree.fromPyTree(jpy(inTree.toPyTree(params)))

  def hessian[In, V: IsFloating](f: In => Tensor0[V])(using
      inTree: TensorTree[In],
      outTree: TensorTree[Tensor0[V]],
      hess: Gradient[In, In],
      hessTree: TensorTree[hess.Result]
  ): In => hess.Result =
    val fpy = (jxpr: py.Dynamic) =>
      OnError.traceStack:
        val x = inTree.fromPyTree(jxpr)
        outTree.toPyTree(f(x))

    val hpy = Jax.jax_helper.hessian(fpy)

    (params: In) =>
      val xpy = inTree.toPyTree(params)
      val res = hpy(xpy)
      hessTree.fromPyTree(res)
