package examples

def timed[A](template: String)(block: => A): A =
  val t0 = System.currentTimeMillis()
  val result = block
  println(s"$template took ${System.currentTimeMillis() - t0} ms")
  result
