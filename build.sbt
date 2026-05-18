import ai.kien.python.Python
import scala.sys.process._

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / organization := "ch.contrafactus"

// Add resolver for snapshot dependencies
ThisBuild / resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

addCommandAlias("testAndCoverage", "; clean; coverage; test; coverageReport")

lazy val root = (project in file("."))
  .aggregate(core, optimizer, examples)
  .settings(
    name := "dimwit-root"
  )

lazy val uvPython: String =
  sys.env.getOrElse(
    "DIMWIT_PYTHON_PATH",
    Seq("uv", "run", "--no-sync", "python", "-c", "import sys; print(sys.executable)").!!.trim
  )
lazy val python = Python(uvPython)
lazy val scalapyJavaOptions = python.scalapyProperties.get.map { case (k, v) => s"-D$k=$v" }.toSeq

lazy val core = (project in file("core"))
  .settings(
    name := "dimwit-core",
    libraryDependencies ++= Seq(
      "dev.scalapy" %% "scalapy-core" % "0.5.3",
      "org.scalatest" %% "scalatest" % "3.2.19" % Test,
      "org.scalacheck" %% "scalacheck" % "1.18.0" % Test,
      "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0" % Test
    ),
    fork := true,
    javaOptions ++= scalapyJavaOptions,
    Test / envVars += "DIMWIT_SKIP_SYNC" -> "true",
    coverageMinimumStmtTotal := 80,
    coverageFailOnMinimum := false,
    coverageHighlighting := true,
    Compile / packageSrc / publishArtifact := true,
    Compile / packageDoc / publishArtifact := true
  )

lazy val optimizer = (project in file("optimizer"))
  .settings(
    name := "dimwit-optimizer"
  )
  .dependsOn(core)

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .dependsOn(optimizer)
  .settings(
    name := "dimwit-examples",
    // Examples use the same Scala version and dependencies as main project
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "toolkit" % "0.1.7",
      "dev.scalapy" %% "scalapy-core" % "0.5.3"
    ),
    fork := true,
    // Don't publish examples
    publish := {},
    publishLocal := {},
    publishArtifact := false,
    // Examples source directory
    Compile / scalaSource := baseDirectory.value,
    Compile / resourceDirectory := baseDirectory.value / "src" / "main" / "resources",
    scalafmtFailOnErrors := false,
    javaOptions ++= {
      if (sys.props("os.name").toLowerCase.contains("mac")) {
        Seq("-XstartOnFirstThread") // For MacOS to run Python with GUI support
      } else {
        Seq.empty
      }
    }
  )

// Processes files in /mdocs that need to be copied to the root (e.g. README.md)
lazy val docsRoot = (project in file(".dimwit-docs-root"))
  .enablePlugins(MdocPlugin)
  .dependsOn(core, optimizer)
  .settings(
    name := "dimwit-docs-root",
    mdocIn := (ThisBuild / baseDirectory).value / "mdocs",
    mdocOut := (ThisBuild / baseDirectory).value,
    mdocExtraArguments := Seq("--no-link-hygiene"),
    mdocVariables := Map(
      "VERSION" -> version.value
    ),
    fork := true,
    javaOptions ++= scalapyJavaOptions,
    envVars := (ThisBuild / envVars).value
  )

// Processes all other docs in /mdocs/docs/ → output to docs/
lazy val docs = (project in file(".dimwit-docs"))
  .enablePlugins(MdocPlugin)
  .dependsOn(core, optimizer)
  .settings(
    name := "dimwit-docs",
    mdocIn := (ThisBuild / baseDirectory).value / "mdocs/docs",
    mdocOut := (ThisBuild / baseDirectory).value / "docs",
    mdocExtraArguments := Seq("--no-link-hygiene"),
    mdocVariables := Map(
      "VERSION" -> version.value
    ),
    fork := true,
    javaOptions ++= scalapyJavaOptions,
    envVars := (ThisBuild / envVars).value
  )
