import ai.kien.python.Python
import scala.sys.process._

ThisBuild / version := "0.2-SNAPSHOT"
ThisBuild / scalaVersion := "3.9.0"
ThisBuild / organization := "ch.contrafactus"

// Add resolver for snapshot dependencies
ThisBuild / resolvers += Resolver.sonatypeCentralSnapshots
ThisBuild / versionScheme := Some("early-semver")
// Publishing to Sonatype Central
ThisBuild / sonatypeCredentialHost := "central.sonatype.com"
ThisBuild / publishTo := {
  if (isSnapshot.value)
    Some("central-snapshots" at "https://central.sonatype.com/repository/maven-snapshots/")
  else
    sonatypePublishToBundle.value
}
ThisBuild / publishMavenStyle := true
ThisBuild / homepage := Some(url("https://github.com/dimwit-dev/dimwit"))
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/dimwit-dev/dimwit"),
    "scm:git@github.com:dimwit-dev/dimwit.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "dimwit-dev",
    name = "DimWit Contributors",
    email = "",
    url = url("https://github.com/dimwit-dev")
  )
)

lazy val commonScalacOptions = Seq(
  "-deprecation",
  "-unchecked",
  "-Wunused:imports",
  "-explain-cyclic"
)

// Setup for Scalafix and SemanticDB
inThisBuild(Seq(
  semanticdbEnabled := true,
  semanticdbVersion := scalafixSemanticdb.revision
))

ThisBuild / scalafixDependencies +=
  "com.github.liancheng" %% "organize-imports" % "0.6.0"

addCommandAlias("testAndCoverage", "; clean; coverage; test; coverageReport")

// alias for uploading core to Sonatype Central
addCommandAlias("sonaUploadCore", "; project core; sonatypeCentralUpload; project root")

lazy val root = (project in file("."))
  .aggregate(core, examples)
  .settings(
    name := "dimwit-root",
    publish / skip := true,
    publishLocal / skip := true,
    publishArtifact := false
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
    Compile / packageDoc / publishArtifact := true,
    scalacOptions ++= commonScalacOptions
  )

// Examples subproject
lazy val examples = (project in file("examples"))
  .dependsOn(core)
  .settings(
    name := "dimwit-examples",
    // Examples use the same Scala version and dependencies as main project
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "toolkit" % "0.1.7",
      "dev.scalapy" %% "scalapy-core" % "0.5.3"
    ),
    fork := true,
    // Don't publish examples
    publish / skip := true,
    publishLocal / skip := true,
    publishArtifact := false,
    scalacOptions ++= commonScalacOptions,
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
  .dependsOn(core)
  .settings(
    name := "dimwit-docs-root",
    publish / skip := true,
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
  .dependsOn(core)
  .settings(
    name := "dimwit-docs",
    publish / skip := true,
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
