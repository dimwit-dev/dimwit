package examples.dataset

import dimwit.*
import dimwit.Conversions.given

import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import java.io.RandomAccessFile
import scala.util.Try

object MNISTLoader:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label

  private val pythonLoader = py.eval("lambda b64, shape: __import__('jax').numpy.array(__import__('numpy').frombuffer(__import__('base64').b64decode(b64), dtype=__import__('numpy').uint8).reshape(shape).astype(__import__('numpy').int32))")

  def loadImages[S <: Sample: Label](filename: String, maxImages: Option[Int] = None): Tensor3[S, Height, Width, Int] =
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
      Tensor.fromArray(shape)(pixels)

    finally
      file.close()

  def loadLabels[S <: Sample: Label](filename: String, maxLabels: Option[Int] = None): Tensor1[S, Int] =
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
      Tensor.fromArray(shape)(labels)

    finally
      file.close()

  private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(S, Height, Width), Float], Tensor1[S, Int]]] =
    Try:
      val images = loadImages[S](imagesFile, maxSamples)
      val labels = loadLabels[S](labelsFile, maxSamples)
      require(images.shape(Axis[S]) == labels.shape(Axis[S]), s"Number of images and labels must match")
      val imagesFloat = images.asFloat /! 255.0f
      (imagesFloat, labels)

  def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float], Tensor1[TrainSample, Int]]] =
    val imagesFile = s"$dataDir/train-images-idx3-ubyte"
    val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
    createDataset[TrainSample](imagesFile, labelsFile, maxSamples)

  def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TestSample, Height, Width), Float], Tensor1[TestSample, Int]]] =
    val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
    val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
    createDataset[TestSample](imagesFile, labelsFile, maxSamples)
