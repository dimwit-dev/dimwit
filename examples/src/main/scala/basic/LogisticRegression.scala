package examples.basic

import dimwit.*
import dimwit.Conversions.given
import nn.*
import nn.ActivationFunctions.{sigmoid, relu}
import dimwit.random.Random
import dimwit.stats.Normal

object LogisticRegression:

  // Define labels for tensor axes
  trait Sample derives Label
  trait Feature derives Label

  // Define a binary logistic regression model
  case class BinaryLogisticRegression(
      params: BinaryLogisticRegression.Params
  ) extends Function[Tensor1[Feature, Float], Tensor0[Boolean]]:

    def logits(input: Tensor1[Feature, Float]): Tensor0[Float] =
      params.weights.dot(Axis[Feature])(input) + params.bias

    def probits(input: Tensor1[Feature, Float]): Tensor0[Float] =
      sigmoid(logits(input))

    def apply(input: Tensor1[Feature, Float]): Tensor0[Boolean] =
      logits(input) >= Tensor0(0f)

  // Parameters are, by convention, defined in the companion object
  object BinaryLogisticRegression:
    case class Params(
        weights: Tensor1[Feature, Float],
        bias: Tensor0[Float]
    ) derives ToPyTree, FloatTensorTree

    // The loss is a simple binary cross-entropy loss
    def loss(data: Tensor2[Sample, Feature, Float], labels: Tensor1[Sample, Boolean])(params: BinaryLogisticRegression.Params)
        : Tensor0[Float] =

      // Create the model with the given parameters
      val model = BinaryLogisticRegression(params)

      // Compute the logistic loss for the model over the dataset
      val losses = zipvmap(Axis[Sample])(data, labels.asFloat):
        case (sample, label) =>
          val logits = model.logits(sample)
          relu(logits) - logits * label + ((-logits.abs).exp + 1f).log
      losses.mean

  def main(args: Array[String]): Unit =

    // we need two keys. One for initializing parameters,
    // the other for shuffling data
    val (initKey, shuffleKey) = Random.Key(42).split2()

    // Load and preprocess the penguin dataset
    val df = PenguinCSV
      .parse("./data/penguins.csv")
      .filter(row => row.species != 2)

    val featureData = df.map { row =>
      Array(
        row.flipper_length_mm.toFloat,
        row.bill_length_mm.toFloat,
        row.bill_depth_mm.toFloat,
        row.body_mass_g.toFloat
      )
    }.toArray

    val labelData = df.map(_.species).toArray.map {
      case 1 => true
      case 0 => false
    }

    val numSamples = featureData.length
    val numFeatures = featureData.head.length

    // Convert the data into tensors
    val dataInitial = Tensor2(Axis[Sample], Axis[Feature]).fromArray(featureData)
    val labelsInitial = Tensor1(Axis[Sample]).fromArray(labelData)

    // Create a permutation to shuffle data and
    // split the permutation indices into training and validation
    val perm = Random.permutation(dataInitial.shape.extent(Axis[Sample]))(shuffleKey)
    val testSplitRatio = 0.4f
    val splitIndex = (numSamples * testSplitRatio).toInt
    val trainPerm = perm.slice(Axis[Sample].at(0 until splitIndex))
    val testPerm = perm.slice(Axis[Sample].at(splitIndex until numSamples))

    // Use the permutations to get our training and validation data
    val trainingDataUnnormalized = dataInitial.take(Axis[Sample])(trainPerm)
    val valDataUnnormalized = dataInitial.take(Axis[Sample])(testPerm)
    val trainLabels = labelsInitial.take(Axis[Sample])(trainPerm)
    val valLabels = labelsInitial.take(Axis[Sample])(testPerm)

    def calcMeanAndStd(t: Tensor2[Sample, Feature, Float]): (Tensor1[Feature, Float], Tensor1[Feature, Float]) =
      val mean = t.vmap(Axis[Feature])(_.mean)
      val std = zipvmap(Axis[Feature])(t, mean):
        case (x, m) =>
          val epsilon = 1e-6f
          (x -! m).pow(2f).mean.sqrt + epsilon
      (mean, std)

    def standardizeData(mean: Tensor1[Feature, Float], std: Tensor1[Feature, Float])(data: Tensor2[Sample, Feature, Float])
        : Tensor2[Sample, Feature, Float] =
      (data -! mean) /! std

    // Standardize the training and validation data
    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardizeData(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardizeData(trainMean, trainStd)(valDataUnnormalized)

    // Initialize model parameters
    val initParams = BinaryLogisticRegression.Params(
      weights = Normal.standardNormal(Shape(Axis[Feature] -> numFeatures)).sample(initKey) *! 0.01f,
      bias = Tensor0(0f)
    )
    // Setting up the loss functions and optimizer.
    // We always jit the loss functions for performance.
    val trainLoss = jit(BinaryLogisticRegression.loss(trainingData, trainLabels))
    val valLoss = jit(BinaryLogisticRegression.loss(valData, valLabels))
    val learningRate = 5e-1f
    val gd = GradientDescent(learningRate)

    // Training loop
    val numiterations = 1000
    val trainTrajectory = gd.iterate(initParams)(Autodiff.grad(trainLoss))
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, index) =>
          val model = BinaryLogisticRegression(params)
          val trainPreds = trainingData.vmap(Axis[Sample])(model)
          val valPreds = valData.vmap(Axis[Sample])(model)
          println(
            List(
              "epoch: " + index,
              "trainAcc: " + (1f - (trainPreds.asFloat - trainLabels.asFloat).abs.mean),
              "valAcc: " + (1f - (valPreds.asFloat - valLabels.asFloat).abs.mean)
            ).mkString(", ")
          )
      .map((params, _) => params)
      .drop(numiterations - 1) // we are only interested in the final parameters
      .next() // consume the value we are actually interested in

    val finalModel = BinaryLogisticRegression(finalParams)
    val predictions = trainingData.vmap(Axis[Sample])(finalModel.probits)
    println(predictions)
    val predictionClasses = trainingData.vmap(Axis[Sample])(x => finalModel(x))

    println("\nTraining complete. Optimized parameters:" + finalParams)

object PenguinCSV:
  case class Row(
      species: Int,
      bill_length_mm: Double,
      bill_depth_mm: Double,
      flipper_length_mm: Double,
      body_mass_g: Double
  )

  def parse(path: String): Seq[Row] =
    val source = scala.io.Source.fromFile(path)
    try
      val lines = source.getLines().toSeq
      lines
        .drop(1)
        .map { line =>
          val parts = line.split(",")
          Row(
            species = parts(1).toInt,
            bill_length_mm = parts(2).toDouble,
            bill_depth_mm = parts(3).toDouble,
            flipper_length_mm = parts(4).toDouble,
            body_mass_g = parts(5).toDouble
          )
        }
        .toSeq
    finally source.close()
