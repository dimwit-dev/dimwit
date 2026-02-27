package dimwit.autodiff

import dimwit.tensor.*
import dimwit.tensor.TensorOps.*
import dimwit.jax.Jax
import me.shadaj.scalapy.py

/** Concrete, reified tensor tree backed by a JAX pytree.
  *
  * `TensorTree[P]` wraps a `Jax.PyAny` (a JAX pytree) with a phantom type `P`
  * that tracks the Scala structure it was created from. All arithmetic operations
  * are implemented via `jax.tree_util.tree_map`, avoiding Scala-side structural
  * traversal.
  *
  * Create via `TensorTree.from(params)` and reconstruct via `.toScala`.
  *
  * Arithmetic methods require a `ToFloatTensorTree[P]` evidence to ensure all
  * leaves are Float tensors (compile-time safety).
  */
class TensorTree[P] private[dimwit] (private[dimwit] val pyTree: Jax.PyAny):

  /** Reconstruct the typed Scala value from this pytree. */
  def toScala(using tt: ToTensorTree[P]): P = tt.fromTensorTree(this)

  // --- Binary tree-vs-tree operations (require all-float proof) ---

  def ++(other: TensorTree[P])(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic, b: py.Dynamic) => Jax.jnp.add(a, b),
      pyTree,
      other.pyTree
    ))

  def --(other: TensorTree[P])(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic, b: py.Dynamic) => Jax.jnp.subtract(a, b),
      pyTree,
      other.pyTree
    ))

  def **(other: TensorTree[P])(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic, b: py.Dynamic) => Jax.jnp.multiply(a, b),
      pyTree,
      other.pyTree
    ))

  def `//`(other: TensorTree[P])(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic, b: py.Dynamic) => Jax.jnp.divide(a, b),
      pyTree,
      other.pyTree
    ))

  // --- Broadcast operations (tree vs scalar) ---

  def ++!(scalar: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    val s = scalar.jaxValue
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.add(a, s),
      pyTree
    ))

  def --!(scalar: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    val s = scalar.jaxValue
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.subtract(a, s),
      pyTree
    ))

  def **!(scalar: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    val s = scalar.jaxValue
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.multiply(a, s),
      pyTree
    ))

  def `//!`(scalar: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    val s = scalar.jaxValue
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.divide(a, s),
      pyTree
    ))

  // --- Unary / math operations ---

  def sqrt(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.sqrt(a),
      pyTree
    ))

  def pow(exponent: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    val e = exponent.jaxValue
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.power(a, e),
      pyTree
    ))

  def scale(scalar: Tensor0[Float])(using ToFloatTensorTree[P]): TensorTree[P] =
    this **! scalar

  def sign(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.sign(a),
      pyTree
    ))

  def fillZeros(using ToFloatTensorTree[P]): TensorTree[P] =
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.zeros_like(a),
      pyTree
    ))

  def fillCopy(value: Float)(using ToFloatTensorTree[P]): TensorTree[P] =
    val v = Jax.jnp.float32(value)
    TensorTree(Jax.jax.tree_util.tree_map(
      (a: py.Dynamic) => Jax.jnp.full_like(a, v),
      pyTree
    ))

  override def toString: String = s"TensorTree($pyTree)"

object TensorTree:
  /** Create a TensorTree from a typed Scala value. */
  def from[P](p: P)(using tt: ToTensorTree[P]): TensorTree[P] =
    tt.toTensorTree(p)

  /** Wrap a raw pytree with the phantom type P. For internal use. */
  private[dimwit] def apply[P](raw: Jax.PyAny): TensorTree[P] =
    new TensorTree[P](raw)

  /** Wrap a raw Python pytree as a TensorTree. Unsafe: the caller must ensure type compatibility. */
  def unsafeWrap[P](raw: py.Any): TensorTree[P] =
    new TensorTree[P](raw.as[Jax.PyAny])

  /** Extract the underlying raw Python pytree. */
  def unsafeRaw[P](tree: TensorTree[P]): py.Any =
    tree.pyTree.as[py.Any]
