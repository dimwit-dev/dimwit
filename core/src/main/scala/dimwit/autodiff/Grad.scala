package dimwit.autodiff

import dimwit._
import dimwit.jax.Jax

import scala.deriving.Mirror

/** Type-level tag marking a parameter structure as gradients.
  *
  * Similar to Prime[T] for tensor labels, but for entire parameter trees.
  * Grad[Params] has identical structure to Params but is a distinct type,
  * providing compile-time safety against confusing parameters with gradients.
  *
  * Example:
  * {{{
  * val grads: Grad[Params] = Autodiff.grad(loss)(params)
  * optimizer.update(grads, state, params)  // Type-safe!
  * optimizer.update(params, state, grads)  // Compile error!
  * }}}
  */
opaque type Grad[T] = T

object Grad:
  /** Tag a value as gradients */
  inline def apply[T](t: T): Grad[T] = t

  /** Extract underlying value (rarely needed in user code) */
  extension [T](g: Grad[T])
    inline def value: T = g

  // Automatic TensorTree instance - gradients have same tree structure as params
  given [T](using ev: TensorTree[T]): TensorTree[Grad[T]] with
    def map(g: Grad[T], f: [U <: Tuple, V] => Labels[U] ?=> Tensor[U, V] => Tensor[U, V]): Grad[T] =
      Grad(ev.map(g, f))

    def mapWithName(g: Grad[T], f: [U <: Tuple, V] => Labels[U] ?=> ((String, Tensor[U, V]) => Tensor[U, V]), path: String = ""): Grad[T] =
      Grad(ev.mapWithName(g, f, path))

    def foreach(g: Grad[T], f: [U <: Tuple, V] => Labels[U] ?=> (Tensor[U, V] => Unit)): Unit =
      ev.foreach(g, f)

    def foreachWithName(g: Grad[T], f: [U <: Tuple, V] => Labels[U] ?=> ((String, Tensor[U, V]) => Unit), path: String = ""): Unit =
      ev.foreachWithName(g, f, path)

    def zipMap(g1: Grad[T], g2: Grad[T], f: [U <: Tuple, V] => Labels[U] ?=> (Tensor[U, V], Tensor[U, V]) => Tensor[U, V]): Grad[T] =
      Grad(ev.zipMap(g1, g2, f))

    def mapLeaves[A](p: Grad[T], f: [T <: Tuple, V] => (x: Labels[T]) ?=> (t: Tensor[T, V]) => A): Iterator[A] = ev.mapLeaves(p, f)

    def toPyTree(g: Grad[T]): Jax.PyAny = ev.toPyTree(g)

    def fromPyTree(pyVal: Jax.PyAny): Grad[T] = Grad(ev.fromPyTree(pyVal))

  // FloatTree witness for gradient math (++, --, scale, etc.)
  // given [T, V: IsFloating](using FloatTree[T, V]): FloatTree[Grad[T], V] with {}

  // Bridge extension so we can call .asFloats directly on Grad[Params[V]]
  extension [F[_], V](g: Grad[F[V]])(using
      tt: TensorTree[F[V]],
      ft: FloatTree[F[V], V],
      isF: IsFloating[V]
  )
    def asFloats[NewV: IsFloating](vtype: VType[NewV])(using m: Mirror.ProductOf[F[NewV]]): Grad[F[NewV]] =
      Grad(dimwit.FloatTree.ops.asFloats(g.value)(vtype))
