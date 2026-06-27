package dimwit

import dimwit.autodiff.TensorTree
import me.shadaj.scalapy.py

private[dimwit] object MemoryHelper:

  private[dimwit] def withLocalCleanupImpl(f: => Unit): Unit =
    py.local:
      f

  private[dimwit] def withLocalCleanupImpl[A: TensorTree](f: => A): A =
    val lifeRaft = me.shadaj.scalapy.py.Dynamic.global.list()
    py.local:
      val res = f
      val pyRes = TensorTree[A].toPyTree(res)
      lifeRaft.append(pyRes)
    val res = lifeRaft.pop()
    TensorTree[A].fromPyTree(res)
