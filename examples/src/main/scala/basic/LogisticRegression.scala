package examples.basic

import dimwit.*
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import dimwit.Conversions.given
import dimwit.random.Random

object LogisticRegression:

  trait Sample derives Label
  trait Feature derives Label

  object BinaryLogisticRegression:
    case class Params(
        linearMap: LinearMap.Params[Feature]
    )

  case class BinaryLogisticRegression(
      params: BinaryLogisticRegression.Params
  ) extends Function[Tensor1[Feature, Float], Tensor0[Boolean]]:
    private val linear = LinearMap(params.linearMap)
    def logits(input: Tensor1[Feature, Float]): Tensor0[Float] = linear(input)
    def probits(input: Tensor1[Feature, Float]): Tensor0[Float] = sigmoid(logits(input))
    def apply(input: Tensor1[Feature, Float]): Tensor0[Boolean] = logits(input) >= Tensor0(0f)

  def main(args: Array[String]): Unit =

    val df = PenguinCSV
      .parse("./data/penguins.csv")
      .filter(row => row.species != 2)

    val dfShuffled = scala.util.Random.shuffle(df)

    val featureData = dfShuffled.map { row =>
      Array(
        row.flipper_length_mm.toFloat,
        row.bill_length_mm.toFloat,
        row.bill_depth_mm.toFloat,
        row.body_mass_g.toFloat
      )
    }.toArray
    val labelData = dfShuffled.map(_.species).toArray.map {
      case 1 => true
      case 0 => false
    }

    val dataUnnormalized = Tensor2.fromArray(Axis[Sample], Axis[Feature], VType[Float])(featureData)
    val dataLabels = Tensor1.fromArray(Axis[Sample], VType[Boolean])(labelData)

    // TODO implement split
    val (trainingDataUnnormalized, valDataUnnormalized) = (dataUnnormalized, dataUnnormalized)
    val (trainLabels, valLabels) = (dataLabels, dataLabels)

    def calcMeanAndStd(t: Tensor2[Sample, Feature, Float]): (Tensor1[Feature, Float], Tensor1[Feature, Float]) =
      val mean = t.vmap(Axis[Feature])(_.mean)
      val std = zipvmap(Axis[Feature])(t, mean):
        case (x, m) =>
          val epsilon = 1e-6f
          (x :- m).pow(2f).mean.sqrt + epsilon
          // x.vmap(Axis[Sample])(xi => (xi - m).pow(2)).mean.sqrt + epsilon
      (mean, std)

    def standardizeData(mean: Tensor1[Feature, Float], std: Tensor1[Feature, Float])(data: Tensor2[Sample, Feature, Float]): Tensor2[Sample, Feature, Float] =
      data.vapply(Axis[Feature])(feature => (feature - mean) / std)
      // (data :- mean) :/ std

    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardizeData(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardizeData(trainMean, trainStd)(valDataUnnormalized)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(data: Tensor2[Sample, Feature, Float])(params: BinaryLogisticRegression.Params): Tensor0[Float] =
      val model = BinaryLogisticRegression(params)
      val losses = zipvmap(Axis[Sample])(data, trainLabels.toFloat):
        case (sample, label) =>
          val logits = model.logits(sample)
          relu(logits) - logits * label + ((-logits.abs).exp + 1f).log
      losses.mean

    val initParams = BinaryLogisticRegression.Params(
      LinearMap.Params(initKey)(dataUnnormalized.dim(Axis[Feature]))
    )

    val trainLoss = loss(trainingData)
    val valLoss = loss(valData)
    val learningRate = 3e-1f
    val xxx = summon[FloatTensorTree[BinaryLogisticRegression.Params]]
    val gd = GradientDescent(Autodiff.grad(trainLoss), learningRate)

    val trainTrajectory = Iterator.iterate(initParams)(gd.step)
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, index) =>
          val model = BinaryLogisticRegression(params)
          val trainPreds = trainingData.vmap(Axis[Sample])(model)
          val valPreds = valData.vmap(Axis[Sample])(model)
          println(
            List(
              "epoch: " + index,
              "trainAcc: " + (1f - (trainPreds.toInt - trainLabels.toInt).abs.mean),
              "valAcc: " + (1f - (valPreds.toInt - valLabels.toInt).abs.mean)
            ).mkString(", ")
          )
      .map((params, _) => params)
      .drop(2500)
      .next()

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
