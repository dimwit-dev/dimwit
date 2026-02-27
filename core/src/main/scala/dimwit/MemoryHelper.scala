package dimwit

import me.shadaj.scalapy.py
import dimwit.autodiff.{ToTensorTree, TensorTree}

private[dimwit] object MemoryHelper:

  def withLocalCleanup(f: => Unit): Unit =
    py.local:
      f

  def withLocalCleanup[A: ToTensorTree](f: => A): A =
    val lifeRaft = me.shadaj.scalapy.py.Dynamic.global.list()
    py.local:
      val res = f
      val pyRes = ToTensorTree[A].toTensorTree(res).pyTree
      lifeRaft.append(pyRes)
    val res = lifeRaft.pop()
    ToTensorTree[A].fromTensorTree(TensorTree[A](res))
