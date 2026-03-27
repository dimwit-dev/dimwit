package src.main.scala.complex

import dimwit.autodiff.TensorTree
import me.shadaj.scalapy.py

def safePyTree[T: TensorTree](value: T, path: String): Unit =
  val pickle = py.module("pickle")
  val pyTree = TensorTree[T].toPyTree(value)
  val file = py.Dynamic.global.open(path, "wb")
  try
    pickle.dump(pyTree, file, protocol = 5)
    println(s"Saved checkpoint: $path")
  finally
    file.close()

def loadPyTree[T: TensorTree](path: String): T =
  val pickle = py.module("pickle")
  val file = py.Dynamic.global.open(path, "rb")

  try
    val pyTree = pickle.load(file)
    TensorTree[T].fromPyTree(pyTree)
  finally
    file.close()
