package dimwit.python

import me.shadaj.scalapy.py

/** Leak-free replacement for ScalaPy's `bracketAccess`.
  *
  * `py.Dynamic.bracketAccess` goes through `CPythonInterpreter.selectBracket`,
  * which wraps `PyObject_GetItem` — a function that returns a *new* reference —
  * in `PyValue.fromBorrowed`, and that takes a second reference. Only one of the
  * two is ever released, so every element read this way keeps its Python object
  * alive for the rest of the process; for a JAX array that also pins its device
  * buffer.
  *
  * Reading through `__getitem__` uses the ordinary attribute-call path
  * (`PyObject_GetAttrString` + `PyValue.fromNew`), whose refcounting is
  * balanced, so the element is released as soon as Scala drops it.
  *
  * This matters most in the training loop: every leaf of a jitted function's
  * result is read out of the returned pytree, so a leak here grows with the
  * number of steps and makes each `gc.collect()` progressively slower.
  */
private[dimwit] object PyIndex:

  extension (value: py.Dynamic)
    /** `value[index]`, without leaking a reference to the element. */
    def itemAt(index: Int): py.Dynamic =
      value.applyDynamic("__getitem__")(index)

    /** `value[key]`, without leaking a reference to the element. */
    def itemAt(key: py.Any): py.Dynamic =
      value.applyDynamic("__getitem__")(key)
