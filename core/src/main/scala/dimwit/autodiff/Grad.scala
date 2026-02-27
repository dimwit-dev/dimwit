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

  // Automatic ToTensorTree instance - gradients have same PyTree structure as params
  given [T](using ev: ToTensorTree[T]): ToTensorTree[Grad[T]] with
    def toTensorTree(g: Grad[T]): TensorTree[Grad[T]] = TensorTree[Grad[T]](ev.toTensorTree(g).pyTree)
    def fromTensorTree(t: TensorTree[Grad[T]]): Grad[T] = ev.fromTensorTree(TensorTree[T](t.pyTree))

  // Automatic ToFloatTensorTree instance - gradients have same all-float proof as params
  // This also provides ToTensorTree[Grad[T]] via inheritance when T: ToFloatTensorTree
  given [T](using ev: ToFloatTensorTree[T]): ToFloatTensorTree[Grad[T]] with
    def toTensorTree(g: Grad[T]): TensorTree[Grad[T]] = TensorTree[Grad[T]](ev.toTensorTree(g).pyTree)
    def fromTensorTree(t: TensorTree[Grad[T]]): Grad[T] = ev.fromTensorTree(TensorTree[T](t.pyTree))
