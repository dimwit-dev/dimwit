package dimwit.python

import me.shadaj.scalapy.py

import scala.sys.process.Process
import scala.sys.process.ProcessLogger

/** Manages Python environment setup for DimWit.
  *
  * Handles extraction of Python helper modules from JAR resources and configuration of Python path for ScalaPy integration.
  */
object PythonSetup:

  /** Configures the JVM system properties that ScalaPy/JNA need to locate the Python shared library.
    *
    * This must run before any `py.*` call (i.e. before ScalaPy's own class initialiser).
    *
    * Respects three env-var overrides:
    *   - DIMWIT_SKIP_SYNC      — skip uv sync and manage Python environment manually (overrides performUvSync argument)
    *   - DIMWIT_PYTHON_PATH     — use a specific Python interpreter
    *   - DIMWIT_PYTHON_LIBRARY  — use a specific shared-library path
    */
  def configureScalaPy(performUvSync: Boolean): Unit =
    val skipSync = sys.env.get("DIMWIT_SKIP_SYNC").exists(v => v == "true" || v == "1")
    if performUvSync && !skipSync then
      val uvLogger = ProcessLogger(_ => (), _ => ())
      if Process(Seq("uv", "sync")).!(uvLogger) != 0 then
        throw new RuntimeException(
          """[dimwit] uv sync failed. Ensure uv is installed (https://docs.astral.sh/uv/) and
            |that a pyproject.toml with JAX dependencies exists in your project, for example:
            |
            |  [project]
            |  name = "my-project"
            |  version = "0.1.0"
            |  requires-python = ">=3.11"
            |  dependencies = [
            |      "jax[cpu]>=0.4",      # or jax[cuda12], jax[tpu]
            |      "einops>=0.8",
            |  ]
            |
            |Set DIMWIT_SKIP_SYNC=true to manage the Python environment yourself.""".stripMargin
        )

    val python = scala.sys.env.getOrElse(
      "DIMWIT_PYTHON_PATH",
      try
        val out = new StringBuilder
        val logger = ProcessLogger(
          line =>
            out.append(line); out.append("\n")
          ,
          _ => ()
        )
        val rc = Process(Seq("uv", "run", "python", "-c", "import sys; print(sys.executable)")).!(logger)
        if rc != 0 then throw new RuntimeException(s"uv run exited with code $rc")
        out.toString.trim
      catch
        case e: RuntimeException => throw new RuntimeException(
            "[dimwit] Could not resolve Python interpreter. " +
              "Set DIMWIT_PYTHON_PATH or check your uv setup.",
            e
          )
    )

    val library = scala.sys.env.getOrElse(
      "DIMWIT_PYTHON_LIBRARY",
      try
        Process(Seq(
          python,
          "-c",
          "import sys, os, sysconfig as sc, ctypes.util as cu; v = sys.version_info\n" +
            "if sys.platform == 'win32': print(os.path.join(sys.base_prefix, 'python{}{}.dll'.format(v.major, v.minor)))\n" +
            "else:\n" +
            "    lib = os.path.join(sc.get_config_var('LIBDIR') or '', sc.get_config_var('INSTSONAME') or sc.get_config_var('LDLIBRARY') or '')\n" +
            "    print(cu.find_library('python{}.{}'.format(v.major, v.minor)) if not os.path.isfile(lib) else lib)"
        )).!!.trim
      catch
        case e: Exception =>
          throw new RuntimeException(
            s"[dimwit] Could not locate Python shared library for interpreter '$python'. " +
              "Set DIMWIT_PYTHON_LIBRARY to the full path of libpython (e.g. /usr/lib/libpython3.11.so.1.0).",
            e
          )
    )

    System.setProperty("scalapy.python.programname", python)
    System.setProperty("scalapy.python.library", library)

  /** Initialize Python environment by extracting helper modules and configuring Python path.
    *
    * This method:
    *   - Configures ScalaPy JVM properties (must happen before any py.* call)
    *   - Extracts jax_helper.py from JAR resources to a temporary directory
    *   - Adds the temp directory to Python's sys.path
    *   - Registers shutdown hook for cleanup
    *   - Falls back to development paths if running from source
    *
    * Called lazily on first access to any JAX module — safe to call multiple times.
    */
  lazy val initialize: Unit =

    lazy val sys = py.module("sys")

    // Extract jax_helper.py from JAR resources to a temporary directory
    val resourcePath = "/python/jax_helper.py"
    val resourceStream = getClass.getResourceAsStream(resourcePath)

    if resourceStream != null then
      try
        val tempDir = java.nio.file.Files.createTempDirectory("dimwit-python")
        val targetFile = tempDir.resolve("jax_helper.py")
        java.nio.file.Files.copy(
          resourceStream,
          targetFile,
          java.nio.file.StandardCopyOption.REPLACE_EXISTING
        )

        // Add the temp directory to Python path
        sys.path.append(tempDir.toAbsolutePath.toString)

        // Register shutdown hook to clean up temp directory
        Runtime.getRuntime.addShutdownHook(new Thread(() =>
          try
            java.nio.file.Files
              .walk(tempDir)
              .sorted(java.util.Comparator.reverseOrder())
              .forEach(java.nio.file.Files.delete)
          catch case _: Exception => () // Ignore cleanup errors
        ))
      finally resourceStream.close()
    else
      // Fallback to legacy path for development/local usage
      sys.path.append("./core/src/main/resources/python")
      sys.path.append("./src/main/resources/python")
