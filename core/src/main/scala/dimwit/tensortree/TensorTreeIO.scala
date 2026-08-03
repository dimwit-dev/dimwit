package dimwit.tensortree

import java.nio.file.Path

/** Provides methods to save and load tensor trees to and from disk.
  * The default format is pickle, but other formats can be specified if needed.
  */
object TensorTreeIO:

  /** Saves a tensor tree to disk in the specified format.
    *
    * @param p The tensor tree structure (e.g., model parameters) to be saved.
    * @param path The file path where the tensor tree will be saved.
    * @param format The format in which to save the tensor tree (default is pickle).
    */
  def save[P](p: P, path: Path, format: TensorTreeFormat = TensorTreeFormat.Pickle)(using tt: TensorTree[P]): Unit =
    format.write(p, path)

  /** Loads a tensor tree from disk in the specified format.
    *
    * @tparam P The type of the tensor tree structure to be loaded.
    * @param path The file path from which to load the tensor tree.
    * @param format The format in which to load the tensor tree (default is pickle).
    * @return The loaded tensor tree structure.
    */
  def load[P](path: Path, format: TensorTreeFormat = TensorTreeFormat.Pickle)(using tt: TensorTree[P]): P =
    format.read(path)
