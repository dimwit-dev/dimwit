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
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.readwrite.Reader
import me.shadaj.scalapy.readwrite.Writer

object FunctionalOps:
  // -----------------------------------------------------------
  // 5. Functional Operations (Higher Order)
  // Lifting functions over axes
  // -----------------------------------------------------------

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

    def zipvmap[L: Label, Inputs <: Tuple, OutShape <: Tuple: Labels, OutV](
        axis: Axis[L]
    )(
        tensors: Inputs // This is a Tuple of Tensors
    )(using
        ev: SharedAxisRemover[ShapesOf[Inputs], L]
    )(
        f: TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]] => Tensor[OutShape, OutV]
    ): Tensor[L *: OutShape, OutV] =
      val fpy = (args: py.Dynamic) =>
        OnError.traceStack:
          val tensorList = args.as[Seq[py.Dynamic]].zip(ev.shapesLabels).map: (jaxArr, labels) =>
            Tensor(jaxArr)(using LabelsImpl(labels))

          val inputTuple = Tuple.fromArray(tensorList.toArray)
          val result = f(inputTuple.asInstanceOf[TensorsOf[ev.RemainingAxes, ValuesOf[Inputs]]])
          result.jaxValue

      val jaxInputs = py.Dynamic.global.tuple(tensors.toArray.map(_.asInstanceOf[Tensor[?, ?]].jaxValue).toPythonProxy)
      val indicesAsTuple = py.Dynamic.global.tuple(ev.indices.toPythonProxy)
      val jaxResult = Jax.jax_helper.zipvmap(
        fpy,
        indicesAsTuple
      )(jaxInputs)

      Tensor(jaxResult)

  export ZipVmap.zipvmap

  extension [T <: Tuple: Labels, V](t: Tensor[T, V])

    def vmap[VmapAxis: Label, OuterShape <: Tuple: Labels, V2](
        axis: Axis[VmapAxis]
    )(using
        ev: AxisRemover[T, VmapAxis]
    )(
        f: Tensor[ev.RemainingAxes, V] => Tensor[OuterShape, V2]
    )(using
        labels: Labels[ev.RemainingAxes]
    ): Tensor[VmapAxis *: OuterShape, V2] =
      val fpy = (jxpr: Jax.PyDynamic) =>
        OnError.traceStack:
          val innerTensor = Tensor[ev.RemainingAxes, V](jxpr)
          val result = f(innerTensor)
          result.jaxValue

      Tensor(Jax.jax_helper.vmap(fpy, ev.index)(t.jaxValue))

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

    def zipvmap[L: Label, T2 <: Tuple, OutShape <: Tuple: Labels, OutV](axis: Axis[L])(
        other: Tensor[T2, V]
    )(using
        ev: SharedAxisRemover[(T, T2), L]
    )(
        f: TensorsOf[ev.RemainingAxes, (V, V)] => Tensor[OutShape, OutV]
    ): Tensor[L *: OutShape, OutV] =
      ZipVmap.zipvmap(axis)(t, other)(f)

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
