package examples.dataset

import dimwit.Conversions.given
import dimwit.*
import me.shadaj.scalapy.py

import java.io.RandomAccessFile
import scala.util.Try

object MNISTLoader:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label

  private val pythonLoader = py.eval("lambda b64, shape: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=__import__('numpy').uint8).reshape(shape).astype(__import__('numpy').int32))")

  def loadImages[S <: Sample: Label](filename: String, maxImages: Option[Int] = None): Tensor3[S, Height, Width, UInt8] =
    val file = new RandomAccessFile(filename, "r")
    try
      val magic = file.readInt()
      if magic != 2051 then throw new IllegalArgumentException(s"Invalid magic: $magic")

      val totalImages = file.readInt()
      val rows = file.readInt()
      val cols = file.readInt()

      val numImages = maxImages.map(math.min(_, totalImages)).getOrElse(totalImages)
      val totalPixels = numImages * rows * cols

      println(s"Scala-Loading $numImages images (${rows}x${cols}) from $filename...")

      val pixels = new Array[Byte](totalPixels)
      file.readFully(pixels)

      val shape = Shape(Axis[S] -> numImages, Axis[Height] -> rows, Axis[Width] -> cols)
      Tensor(shape, VType[UInt8]).fromArray(pixels)

    finally
      file.close()

  def loadLabels[S <: Sample: Label](filename: String, maxLabels: Option[Int] = None): Tensor1[S, Int8] =
    val file = new RandomAccessFile(filename, "r")
    try
      val magic = file.readInt()
      if magic != 2049 then throw new IllegalArgumentException(s"Invalid magic for labels: $magic (expected 2049)")

      val totalLabels = file.readInt()
      val numLabels = maxLabels.map(math.min(_, totalLabels)).getOrElse(totalLabels)

      println(s"JAX-Loading $numLabels labels from $filename...")

      val labels = new Array[Byte](numLabels)
      file.readFully(labels)

      val shape = Shape(Axis[S] -> numLabels)
      Tensor(shape).fromArray(labels)

    finally
      file.close()

  private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(S, Height, Width), Float32], Tensor1[S, Int8]]] =
    Try:
      val images = loadImages[S](imagesFile, maxSamples)
      val labels = loadLabels[S](labelsFile, maxSamples)
      require(images.shape(Axis[S]) == labels.shape(Axis[S]), s"Number of images and labels must match")
      val imagesFloat = images.asFloat32 /! 255.0f
      (imagesFloat, labels)

  def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float32], Tensor1[TrainSample, Int8]]] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset[TrainSample](imagesFile, labelsFile, maxSamples)

  def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TestSample, Height, Width), Float32], Tensor1[TestSample, Int8]]] =
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset[TestSample](imagesFile, labelsFile, maxSamples)
