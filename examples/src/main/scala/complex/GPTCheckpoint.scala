package src.main.scala.complex

import dimwit.autodiff.{ToTensorTree, TensorTree}
import me.shadaj.scalapy.py

def safePyTree[T: ToTensorTree](value: T, path: String): Unit =
  val pickle = py.module("pickle")
  val pyTree = TensorTree.unsafeRaw(ToTensorTree[T].toTensorTree(value))
  val file = py.Dynamic.global.open(path, "wb")
  try
    pickle.dump(pyTree, file, protocol = 5)
    println(s"Saved checkpoint: $path")
  finally
    file.close()

def loadPyTree[T: ToTensorTree](path: String): T =
  val pickle = py.module("pickle")
  val file = py.Dynamic.global.open(path, "rb")

  try
    val pyTree = pickle.load(file)
    ToTensorTree[T].fromTensorTree(TensorTree.unsafeWrap[T](pyTree))
  finally
    file.close()
