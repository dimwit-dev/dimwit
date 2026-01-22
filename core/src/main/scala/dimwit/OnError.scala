package dimwit

import java.io.StringWriter
import java.io.PrintWriter

object OnError:

  private val TraceHeader = "🛑 SCALA ERROR"

  private val niceErrorMessage: Boolean = sys.env.getOrElse("DimWitNiceErrorMessages", "enabled") == "enabled"

  /** In tracer blocks (e.g., jit, grad, vmap) the original scala error stack is lost in PythonException.
    * This utility captures the scala stack trace at the first exception and re-throws a cleaned up version.
    *
    * @param block
    * @return
    */
  def traceStack[T](block: => T): T =
    try
      block
    catch
      case e: Throwable if e.getMessage != null && e.getMessage.contains(TraceHeader) =>
        // Already traced => re-throw original error
        throw e
      case e: Throwable if niceErrorMessage =>
        // First time error - capture and re-throw with cleaned stack trace
        val sw = new StringWriter()
        e.printStackTrace(new PrintWriter(sw))
        val cleanTrace = sw.toString.linesIterator
          .filterNot(_.contains("me.shadaj.scalapy"))
          .filterNot(_.contains("com.sun.jna."))
          .filterNot(_.contains("java.base/jdk.internal."))
          .filterNot(_.contains("java.base/java.lang"))
          .mkString("\n")
        val Prefix = f"\n****\n* $TraceHeader:\n***\n"
        val Postfix = "\n****\n* END SCALA ERROR\n***\n"
        throw RuntimeException(s"$Prefix $cleanTrace $Postfix")
      case e => throw e
