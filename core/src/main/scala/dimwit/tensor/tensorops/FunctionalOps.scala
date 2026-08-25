package dimwit.tensor.tensorops

import dimwit.OnError
import dimwit.jax.Jax
import dimwit.tensor.Axis
import dimwit.tensor.Label
import dimwit.tensor.Labels
import dimwit.tensor.LabelsImpl
import dimwit.tensor.ShapeTypeHelpers.AxisRemover
import dimwit.tensor.ShapeTypeHelpers.AxisReplacer
import dimwit.tensor.ShapeTypeHelpers.SharedAxisRemover
import dimwit.tensor.Tensor
import dimwit.tensor.Tensor0
import dimwit.tensor.tensorops.FunctionalOps.ZipVmap.TensorsOf
import dimwit.tensortree.TensorTree
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Reader
import me.shadaj.scalapy.readwrite.Writer

import scala.NamedTuple.NamedTuple
import scala.annotation.implicitNotFound

object FunctionalOps:

  /** Prepends the axis `L` to every tensor of the tensor tree `FOut`: the result
    * type of a `vmap`/`zipvmap` whose body returned `FOut`.
    *
    * Instances cover tensors, tuples, named tuples and any nesting of those.
    */
  @implicitNotFound(
    "Cannot prepend Axis[${L}] to ${FOut}. A vmap/zipvmap body may return a Tensor, a tuple, a named tuple, or any nesting of those"
  )
  trait PrependAxis[L, FOut]:
    type Out

  object PrependAxis:

    type Aux[L, FOut, Out0] = PrependAxis[L, FOut] { type Out = Out0 }

    private def instance[L, FOut, Out0]: Aux[L, FOut, Out0] =
      new PrependAxis[L, FOut]:
        type Out = Out0

    given tensor[L, Shape <: Tuple, V]: Aux[L, Tensor[Shape, V], Tensor[L *: Shape, V]] = instance

    given emptyTuple[L]: Aux[L, EmptyTuple, EmptyTuple] = instance

    given consTuple[L, H, HOut, T <: Tuple, TOut <: Tuple](using
        head: Aux[L, H, HOut],
        tail: Aux[L, T, TOut]
    ): Aux[L, H *: T, HOut *: TOut] = instance

    given namedTuple[L, N <: Tuple, V <: Tuple, VOut <: Tuple](using
        values: Aux[L, V, VOut]
    ): Aux[L, NamedTuple[N, V], NamedTuple[N, VOut]] = instance

  object ZipVmap:

    type TensorsOf[Shapes <: Tuple, Values <: Tuple] <: Tuple = (Shapes, Values) match
      case (EmptyTuple, EmptyTuple)                             => EmptyTuple
      case ((shapeHead *: shapeTail), (valueHead *: valueTail)) => Tensor[shapeHead, valueHead] *: TensorsOf[shapeTail, valueTail]

    type ExtractShape[T] = T match
      case Tensor[s, v] => s

    type ExtractValue[T] = T match
      case Tensor[s, v] => v

    type ShapesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractShape]
    type ValuesOf[Tensors <: Tuple] = Tuple.Map[Tensors, ExtractValue]

    /** Zips the given given tensors along the specified axis
      * and applies the function `f` to the zipped tensors.
      *
      * @param axis The axis along which to zip the tensors.
      * @param tensors A tuple of tensors to be zipped.
      * @param f A function that takes a tuple of tensors (with the specified axis removed) and returns a new tensor.
      * @return A new tensor resulting from applying `f`
      *
      * Example usage:
      * {{{
      * val tensor1: Tensor2[A, B, Int] = ...
      * val tensor2: Tensor2[A, B, Int] = ...
      * val result: Tensor2[A, C, Int] = ZipVmap.zipvmap(Axis[A])(tensor1, tensor2) { case (t1, t2) =>
      *   // Perform operations on t1 and t2, which are tensors with axis A removed, and return a new tensor
      *   ...
      * }
      */
    def zipvmap[L: Label, Inputs <: Tuple, FOut](
        axis: Axis[L]
    )(
        tensors: Inputs
    )(using
        ev: SharedAxisRemover[ShapesOf[Inputs], L]
    )(
        f: TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]] => FOut
    )(using
        prependAxis: PrependAxis[L, FOut],
        toPyTree: TensorTree[FOut],
        fromPyTree: TensorTree[prependAxis.Out]
    ): prependAxis.Out =
      val fpy = (args: py.Dynamic) =>
        OnError.traceStack:
          val tensorList = args.as[Seq[py.Dynamic]].zip(ev.shapesLabels).map: (jaxArr, labels) =>
            Tensor(jaxArr)(using LabelsImpl(labels))

          val inputTuple = Tuple.fromArray(tensorList.toArray)
          val result = f(inputTuple.asInstanceOf[TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]]])
          toPyTree.toPyTree(result)

      val jaxInputs = py.Dynamic.global.tuple(tensors.toArray.map(_.asInstanceOf[Tensor[?, ?]].jaxValue).toPythonProxy)
      val indicesAsTuple = py.Dynamic.global.tuple(ev.indices.toPythonProxy)

      val jaxResult = Jax.jax_helper.zipvmap(
        fpy,
        indicesAsTuple
      )(jaxInputs)

      fromPyTree.fromPyTree(jaxResult)
  export ZipVmap.zipvmap

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    /** Zips the current tensor with another tensor along the specified axis
      * and applies the function `f` to the zipped tensors.
      *
      * @param axis The axis along which to zip the tensors.
      * @param other The other tensor to be zipped with the current tensor.
      * @param f A function that takes a tuple of tensors (with the specified axis removed) and returns a new tensor.
      * @return A new tensor resulting from applying `f` to the zipped tensors.
      */
    def zipvmap[L: Label, T2 <: Tuple, V2, FOut](axis: Axis[L])(
        other: Tensor[T2, V2]
    )(using
        ev: SharedAxisRemover[(T, T2), L]
    )(
        f: TensorsOf[ev.RemainingAxes, (V, V2)] => FOut
    )(using
        prependAxis: PrependAxis[L, FOut],
        toPyTree: TensorTree[FOut],
        fromPyTree: TensorTree[prependAxis.Out]
    ): prependAxis.Out =
      ZipVmap.zipvmap(axis)(t, other)(f)

    /** Vectorized mapping over a specified axis of the tensor.
      *
      * @param axis The axis along which to apply the function `f`.
      * @param f A function that takes a tensor with the specified axis removed and returns a new tensor.
      * @return A new tensor resulting from applying `f` to each slice along the specified axis.
      */
    def vmap[VmapAxis: Label, FOut](
        axis: Axis[VmapAxis]
    )(using
        ev: AxisRemover[T, VmapAxis]
    )(
        f: Tensor[ev.RemainingAxes, V] => FOut
    )(using
        prependAxis: PrependAxis[VmapAxis, FOut],
        toPyTree: TensorTree[FOut],
        fromPyTree: TensorTree[prependAxis.Out],
        labels: Labels[ev.RemainingAxes]
    ): prependAxis.Out =
      val fpy = (jxpr: Jax.PyDynamic) =>
        OnError.traceStack:
          val innerTensor = Tensor[ev.RemainingAxes, V](jxpr)
          val result = f(innerTensor)
          toPyTree.toPyTree(result)
      fromPyTree.fromPyTree(Jax.jax_helper.vmap(fpy, ev.index)(t.jaxValue))

    /** Apply a function independently to each 1D slice along a labeled axis.
      *
      *  @param axis The axis along which to apply the function `f`.
      *  @param f A function f that is applied to each L-axis slice; it may rename that axis to NewL and change the element type.
      *
      *  @return A new tensor resulting from applying `f` to each slice along the specified axis.
      */
    def vapply[L: Label, NewL, R <: Tuple, NewV](
        axis: Axis[L]
    )(
        f: Tensor[Tuple1[L], V] => Tensor[Tuple1[NewL], NewV]
    )(using
        ev: AxisReplacer.Aux[T, L, NewL, R],
        labels: Labels[R]
    ): Tensor[R, NewV] =
      val fpy = (jxpr: Jax.PyDynamic) =>
        OnError.traceStack:
          val inputTensor = Tensor[Tuple1[L], V](jxpr)
          val result = f(inputTensor)
          result.jaxValue

      Tensor(
        Jax.jnp.apply_along_axis(
          fpy,
          ev.index,
          t.jaxValue
        )
      )

    /** Reduce a tensor along a labeled axis by applying a function to each 1D slice.
      *
      * @param axis The axis along which to reduce the tensor.
      * @param f A function f that is applied to each L-axis slice; it must return a scalar (Tensor0).
      *
      * @return A new tensor resulting from reducing the specified axis.
      */
    def vreduce[L: Label](
        axis: Axis[L]
    )(
        f: Tensor[Tuple1[L], V] => Tensor0[V]
    )(using
        ev: AxisRemover[T, L],
        labels: Labels[ev.RemainingAxes]
    ): Tensor[ev.RemainingAxes, V] =
      val fpy = (jxpr: Jax.PyDynamic) =>
        OnError.traceStack:
          val inputTensor = Tensor[Tuple1[L], V](jxpr)
          val result = f(inputTensor)
          result.jaxValue

      Tensor(
        Jax.jnp.apply_along_axis(
          fpy,
          ev.index,
          t.jaxValue
        )
      )
