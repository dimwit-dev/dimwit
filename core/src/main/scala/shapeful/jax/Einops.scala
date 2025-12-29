package shapeful.jax

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import me.shadaj.scalapy.py.PyQuote

object Einops:

  type PyAny = py.Any
  type PyDynamic = py.Dynamic

  private val kwargUnpacker = py.eval("lambda f, x, p, k: f(x, p, **k)")

  def rearrange(
      x: PyDynamic,
      pattern: String,
      kwargsMap: Map[String, Int] = Map()
  ): PyDynamic =
    val einops = py.module("einops")
    val convertedSeq = kwargsMap.map { case (k, v: Int) =>
      k -> py.Any.from(v)
    }.toSeq
    val pyKwargs = py.Dynamic.global.dict(convertedSeq.toPythonProxy)
    kwargUnpacker(einops.rearrange, x, pattern, pyKwargs).as[py.Dynamic]
