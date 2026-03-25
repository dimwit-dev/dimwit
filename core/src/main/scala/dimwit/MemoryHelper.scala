package dimwit

import me.shadaj.scalapy.py
import dimwit.autodiff.TensorTree

private[dimwit] object MemoryHelper:

  def withLocalCleanup(f: => Unit): Unit =
    py.local:
      f

  def withLocalCleanup[A: TensorTree](f: => A): A =
    val lifeRaft = me.shadaj.scalapy.py.Dynamic.global.list()
    py.local:
      val res = f
      val pyRes = summon[TensorTree[A]].toPyTree(res)
      lifeRaft.append(pyRes)
    val res = lifeRaft.pop()
    summon[TensorTree[A]].fromPyTree(res)
