package dimwit.autodiff

import dimwit.*
import dimwit.jax.Jax

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

  // Automatic PyTree instance - gradients have same PyTree structure as params
  given [T](using ev: ToPyTree[T]): ToPyTree[Grad[T]] with
    def toPyTree(g: Grad[T]): Jax.PyAny = ev.toPyTree(g)
    def fromPyTree(p: Jax.PyAny): Grad[T] = ev.fromPyTree(p)

  // Automatic TensorTree instance - gradients have same tree structure
  given [T](using ev: FloatTensorTree[T]): FloatTensorTree[Grad[T]] with
    def map(g: Grad[T], f: [U <: Tuple] => Labels[U] ?=> Tensor[U, Float] => Tensor[U, Float]): Grad[T] =
      Grad(ev.map(g, f))

    def zipMap(g1: Grad[T], g2: Grad[T], f: [U <: Tuple] => Labels[U] ?=> (Tensor[U, Float], Tensor[U, Float]) => Tensor[U, Float]): Grad[T] =
      Grad(ev.zipMap(g1, g2, f))

    def zipMap(g1: Grad[T], g2: Grad[T], g3: Grad[T], f: [U <: Tuple] => Labels[U] ?=> (Tensor[U, Float], Tensor[U, Float], Tensor[U, Float]) => Tensor[U, Float]): Grad[T] =
      Grad(ev.zipMap(g1, g2, g3, f))
