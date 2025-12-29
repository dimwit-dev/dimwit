package examples.basic

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import dimwit.random.Random

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Try
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}

def binaryCrossEntropy[L: Label](
    logits: Tensor1[L, Float],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val stableExp = (logits :- maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label)
  -(targetLogit - logSumExp)

object MLPClassifierMNist:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label
  trait Hidden derives Label
  trait Output derives Label

  object MLP:
    case class Params(
        layer1: LinearLayer.Params[Height |*| Width, Hidden],
        layer2: LinearLayer.Params[Hidden, Output]
    )

    object Params:
      def apply(
          layer1Dim: Dim[Height |*| Width],
          layer2Dim: Dim[Hidden],
          outputDim: Dim[Output]
      )(
          paramKey: Random.Key
      ): Params =
        val (key1, key2) = paramKey.split2()
        Params(
          layer1 = LinearLayer.Params(key1)(layer1Dim, layer2Dim),
          layer2 = LinearLayer.Params(key2)(layer2Dim, outputDim)
        )

  case class MLP(params: MLP.Params) extends Function[Tensor2[Height, Width, Float], Tensor0[Int]]:

    private val layer1 = LinearLayer(params.layer1)
    private val layer2 = LinearLayer(params.layer2)

    def logits(
        image: Tensor2[Height, Width, Float]
    ): Tensor1[Output, Float] =
      val hidden = relu(layer1(image.ravel))
      layer2(hidden)

    override def apply(image: Tensor2[Height, Width, Float]): Tensor0[Int] = logits(image).argmax(Axis[Output])

  object MNISTLoader:

    private val device = Device.GPU

    private def readInt(dis: DataInputStream): Int = dis.readInt()
    private def loadImagePixels[S <: Sample: Label](filename: String, maxImages: Option[Int] = None): Try[Tensor3[S, Height, Width, Float]] =
      Try {
        val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
        try
          val magic = readInt(dis)
          if magic != 2051 then throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")

          val totalImages = readInt(dis)
          val rows = readInt(dis)
          val cols = readInt(dis)

          val numImages = maxImages.map(max => math.min(max, totalImages)).getOrElse(totalImages)
          println(s"Loading $numImages of $totalImages images (${rows}x${cols}) from $filename into memory as Tensor3")

          // Read all pixel data at once
          val totalPixels = numImages * rows * cols
          val pixelBytes = new Array[Byte](totalPixels)
          dis.readFully(pixelBytes)
          // Convert bytes to floats with vectorized operation
          val allPixels = pixelBytes.map(b => (b & 0xff) / 255.0f)
          val shape = Shape(Axis[S] -> numImages, Axis[Height] -> rows, Axis[Width] -> cols)
          val tensor = Tensor.fromArray(shape, VType[Float])(allPixels)
          tensor.toDevice(device)
        finally dis.close()
      }

    private def loadLabelsArray[S <: Sample: Label](filename: String, maxLabels: Option[Int] = None): Try[Tensor1[S, Int]] = Try {
      val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
      try
        val magic = readInt(dis)
        if magic != 2049 then throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")

        val totalLabels = readInt(dis)
        val numLabels = maxLabels.map(max => math.min(max, totalLabels)).getOrElse(totalLabels)
        println(s"Loading $numLabels of $totalLabels labels from $filename into memory as Tensor1")

        val labels = Array.ofDim[Int](numLabels)
        for i <- 0.until(numLabels) do labels(i) = dis.readUnsignedByte()

        // Create Tensor1 from labels - specify the label type correctly
        val tensor = Tensor1.fromArray(Axis[S], VType[Int])(labels)
        tensor.toDevice(device)
      finally dis.close()
    }

    private def createDataset[S <: Sample: Label](imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(S, Height, Width), Float], Tensor1[S, Int]]] =
      for
        imagePixels <- loadImagePixels[S](imagesFile, maxSamples)
        labels <- loadLabelsArray[S](labelsFile, maxSamples)
      yield
        val numImages = imagePixels.shape(Axis[S])
        val numLabels = labels.shape.size
        if numImages != numLabels then throw new IllegalArgumentException(s"Mismatch: $numImages images vs $numLabels labels")
        println(s"Created in-memory MNIST dataset with $numImages images")
        (imagePixels, labels)

    def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float], Tensor1[TrainSample, Int]]] =
      val imagesFile = s"$dataDir/train-images-idx3-ubyte"
      val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
      createDataset[TrainSample](imagesFile, labelsFile, maxSamples)

    def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TestSample, Height, Width), Float], Tensor1[TestSample, Int]]] =
      val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
      val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
      createDataset[TestSample](imagesFile, labelsFile, maxSamples)

  def main(args: Array[String]): Unit =

    val learningRate = 5e-2f
    val numSamples = 5120 // 59904
    val numTestSamples = 1024 // 9728
    val batchSize = 512
    val numEpochs = 100
    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width), Float], batchLabels: Tensor1[TrainSample, Int])(
        params: MLP.Params
    ): Tensor0[Float] =
      val model = MLP(params)
      val batchSize = batchImages.shape(Axis[TrainSample])
      val losses = (0 until batchSize)
        .map: idx =>
          val image = batchImages.slice(Axis[TrainSample] -> idx)
          val label = batchLabels.slice(Axis[TrainSample] -> idx)
          val logits = model.logits(image)
          binaryCrossEntropy(logits, label)
        .reduce(_ + _)
      losses / batchSize.toFloat

    val initParams = MLP.Params(
      Axis[Height |*| Width] -> 28 * 28,
      Axis[Hidden] -> 128,
      Axis[Output] -> 10
    )(initKey)

    def accuracy[Sample: Label](
        predictions: Tensor1[Sample, Int],
        targets: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val matches = zipvmap(Axis[Sample])(predictions, targets)(_ === _)
      matches.mean

    def gradientStep(
        imageBatch: Tensor[(TrainSample, Height, Width), Float],
        labelBatch: Tensor1[TrainSample, Int],
        params: MLP.Params
    ): MLP.Params =
      val lossBatch = batchLoss(imageBatch, labelBatch)
      val df = Autodiff.grad(lossBatch)
      GradientDescent(df, learningRate).step(params)
    val jitStep = jit(gradientStep.tupled)
    def miniBatchGradientDescent(
        imageBatches: Seq[Tensor[(TrainSample, Height, Width), Float]],
        labelBatches: Seq[Tensor1[TrainSample, Int]]
    )(
        params: MLP.Params
    ): MLP.Params =
      imageBatches
        .zip(labelBatches)
        .foldLeft(params):
          case (currentParams, (imageBatch, labelBatch)) =>
            jitStep(imageBatch, labelBatch, currentParams)

    def timed[A](template: String)(block: => A): A =
      val t0 = System.currentTimeMillis()
      val result = block
      println(s"$template took ${System.currentTimeMillis() - t0} ms")
      result

    val trainMiniBatchGradientDescent = miniBatchGradientDescent(
      trainX.chunk(Axis[TrainSample], batchSize),
      trainY.chunk(Axis[TrainSample], batchSize)
    )
    val trainTrajectory = Iterator.iterate(initParams)(currentParams =>
      timed("Training"):
        trainMiniBatchGradientDescent(currentParams)
    )
    def evaluate(
        params: MLP.Params,
        dataX: Tensor[(Sample, Height, Width), Float],
        dataY: Tensor1[Sample, Int]
    ): Tensor0[Float] =
      val model = MLP(params)
      val predictions = dataX.vmap(Axis[Sample])(model)
      accuracy(predictions, dataY)
    val jitEvaluate = jit(evaluate.tupled)
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val testAccuracy = jitEvaluate(params, testX, testY)
            val trainAccuracy = jitEvaluate(params, trainX, trainY)
            println(
              List(
                s"Epoch $epoch",
                f"Test accuracy: ${testAccuracy.item * 100}%.2f%%",
                f"Train accuracy: ${trainAccuracy.item * 100}%.2f%%"
              ).mkString(", ")
            )
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")
