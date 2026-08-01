package dimwit.tensortree

import dimwit.jax.Jax
import me.shadaj.scalapy.py

import java.nio.file.Path

/** Provides an interface for reading and writing tensor trees to and from disk in various formats.
  */
trait TensorTreeFormat:
  def write[P](p: P, path: Path)(using tt: TensorTree[P]): Unit
  def read[P](path: Path)(using tt: TensorTree[P]): P

object TensorTreeFormat:

  /** Pickle format for saving and loading tensor trees.
    *
    * Arrays are moved off-device to numpy before pickling, and re-materialized
    * as JAX arrays on load, so files remain portable across host/GPU/TPU.
    * `jax.tree_util.tree_map` handles the tuple/list/None nesting natively, so
    * no recursive walk is needed here.
    */
  object Pickle extends TensorTreeFormat:
    private lazy val pickle = py.module("pickle")
    private lazy val builtins = py.module("builtins")

    def write[P](p: P, path: Path)(using tt: TensorTree[P]): Unit =
      val toHost = (x: Jax.PyDynamic) => Jax.np.asarray(Jax.jax.device_get(x))
      val numpyTree = Jax.jax.tree_util.tree_map(toHost, tt.toPyTree(p))
      val file = builtins.open(path.toAbsolutePath().toString(), "wb").as[py.Dynamic]
      try pickle.dump(numpyTree, file)
      finally file.close()

    def read[P](path: Path)(using tt: TensorTree[P]): P =
      val file = builtins.open(path.toAbsolutePath().toString(), "rb").as[py.Dynamic]
      val numpyTree =
        try pickle.load(file).as[py.Dynamic]
        finally file.close()
      val toDevice = (x: Jax.PyDynamic) => Jax.jnp.asarray(x)
      tt.fromPyTree(Jax.jax.tree_util.tree_map(toDevice, numpyTree))
