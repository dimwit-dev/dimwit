package examples.basic.vae

import examples.basic.timed

import dimwit.*
import dimwit.Conversions.given
import dimwit.stats.Normal
import dimwit.random.Random
import Encoder.EncoderParams
import Decoder.DecoderParams
import examples.basic.MNISTLoader
import examples.basic.MNISTLoader.{Sample, TrainSample, TestSample, Height, Width}
import nn.LinearLayer
import nn.ActivationFunctions.relu
import nn.GradientDescent
import dimwit.jax.Jax
import nn.ActivationFunctions.sigmoid
import dimwit.random.Random.Key

type SourceFeature = Height |*| Width
type ReconstructedFeature = Height |*| Width

trait EHidden1 derives Label
trait EHidden2 derives Label

trait Latent derives Label
trait MeanLatent extends Latent derives Label
trait LogVarLatent extends Latent derives Label

trait DHidden1 derives Label
trait DHidden2 derives Label

trait Batch derives Label

type FTensor1[T] = Tensor1[T, Float]

class Encoder(p: EncoderParams):
  def apply(v: FTensor1[Height |*| Width]): (FTensor1[MeanLatent], FTensor1[LogVarLatent]) =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val meanLayer = LinearLayer(p.meanLayer)
    val logVarLayer = LinearLayer(p.logVarLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val mean = meanLayer(h2)
    val logVar = logVarLayer(h2).clip(-10f, 10f)

    (mean, logVar)

object Encoder:
  case class EncoderParams(
      layer1: LinearLayer.Params[Height |*| Width, EHidden1],
      layer2: LinearLayer.Params[EHidden1, EHidden2],
      meanLayer: LinearLayer.Params[EHidden2, MeanLatent],
      logVarLayer: LinearLayer.Params[EHidden2, LogVarLatent]
  )

class Decoder(p: DecoderParams):
  def apply(v: FTensor1[Latent]): FTensor1[ReconstructedFeature] =
    val layer1 = LinearLayer(p.layer1)
    val layer2 = LinearLayer(p.layer2)
    val outputLayer = LinearLayer(p.outputLayer)

    val h1 = relu(layer1(v))
    val h2 = relu(layer2(h1))
    val reconstructed = sigmoid(outputLayer(h2))

    reconstructed

object Decoder:
  case class DecoderParams(
      layer1: LinearLayer.Params[Latent, DHidden1],
      layer2: LinearLayer.Params[DHidden1, DHidden2],
      outputLayer: LinearLayer.Params[DHidden2, ReconstructedFeature]
  )

def reparametrize(mean: FTensor1[Latent], logVar: FTensor1[Latent], key: Random.Key): FTensor1[Latent] =
  val std = (logVar *! 0.5f).exp
  Normal(mean, std).sample(key)

case class VAE(params: VAE.Params):

  val encoder = Encoder(params.encoderParams)
  val decoder = Decoder(params.decoderParams)

  def apply(v: FTensor1[SourceFeature], key: Random.Key): (FTensor1[ReconstructedFeature], FTensor1[Latent], FTensor1[Latent]) =
    val (mean, logVar) = encoder(v)
    val latent = reparametrize(mean, logVar, key)
    val reconstructed = decoder(latent)
    (reconstructed, mean, logVar)

  def loss(original: FTensor1[Height |*| Width], key: Random.Key): Tensor0[Float] =
    val (reconstructed, mean, logVar) = apply(original, key)
    val eps = 1e-5f
    val reconLoss = -((original * (reconstructed +! eps).log) + ((1f -! original) * (1f -! reconstructed +! eps).log)).sum
    val kldLoss = -0.5f * (1f +! logVar - mean.pow(2f) - logVar.exp).sum

    reconLoss + kldLoss

object VAE:
  case class Params(
      encoderParams: Encoder.EncoderParams,
      decoderParams: Decoder.DecoderParams
  )
  object Params:
    def apply(params: VAE.Params)(key: Random.Key): Params =
      Params(
        params.encoderParams,
        params.decoderParams
      )

object VAEExample:

  def main(args: Array[String]): Unit =

    val learningRate = 5e-4f

    val numTestSamples = 9728
    val batchSize = 256
    val numSamples = 59904
    val numEpochs = 120
    val latentDim = 20

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(numTestSamples)).get

    /*
     * Initialize the model parameters
     * */
    val initKeys = initKey.split(7)
    val encoderParams = Encoder.EncoderParams(
      LinearLayer.Params[Height |*| Width, EHidden1](initKeys(0))(
        Axis[Height |*| Width] -> (28 * 28),
        Axis[EHidden1] -> 512
      ),
      LinearLayer.Params[EHidden1, EHidden2](initKeys(1))(
        Axis[EHidden1] -> 512,
        Axis[EHidden2] -> 256
      ),
      LinearLayer.Params[EHidden2, MeanLatent](initKeys(2))(
        Axis[EHidden2] -> 256,
        Axis[MeanLatent] -> latentDim
      ),
      LinearLayer.Params[EHidden2, LogVarLatent](initKeys(3))(
        Axis[EHidden2] -> 256,
        Axis[LogVarLatent] -> latentDim
      )
    )
    val decoderParams = Decoder.DecoderParams(
      LinearLayer.Params[Latent, DHidden1](initKeys(4))(
        Axis[Latent] -> 20,
        Axis[DHidden1] -> 256
      ),
      LinearLayer.Params[DHidden1, DHidden2](initKeys(5))(
        Axis[DHidden1] -> 256,
        Axis[DHidden2] -> 512
      ),
      LinearLayer.Params[DHidden2, ReconstructedFeature](initKeys(6))(
        Axis[DHidden2] -> 512,
        Axis[ReconstructedFeature] -> (28 * 28)
      )
    )

    def batchLoss(key: Random.Key, trainData: Tensor3[Sample, Height, Width, Float])(params: VAE.Params): Tensor0[Float] =
      val vae = VAE(params)
      val keys = key.splitToTensor(Axis[Sample], trainData.shape.dim(Axis[Sample])._2)
      zipvmap(Axis[Sample])(trainData, keys) { (sample, key) =>
        vae.loss(sample.ravel, Key(key.jaxValue))
      }.mean

    val batches = trainX.chunk(Axis[TrainSample], numSamples / batchSize)
    def trainBatch(trainKey: Random.Key, batch: Tensor3[Sample, Height, Width, Float], params: VAE.Params): VAE.Params =
      val df = Autodiff.grad(batchLoss(trainKey, batch))
      GradientDescent(df, learningRate).step(params)

    val jittedTrainBatch = jit(trainBatch, Map("donate_argnums" -> Tuple1(2)))

    def trainEpoch(key: Random.Key, epoch: Int, params: VAE.Params): VAE.Params =
      val batchKeys = key.split(batches.size)
      batches.zip(batchKeys).foldLeft(params):
        case (batchParams, (batch, key)) =>
          jittedTrainBatch(key, batch, batchParams)

    // run the loop
    val keysForEpochs = dataKey.split(numEpochs)

    val initialParams = FloatTensorTree[VAE.Params].map(
      VAE.Params(encoderParams, decoderParams),
      [T <: Tuple] => (n: Labels[T]) ?=> (t: Tensor[T, Float]) => t *! 0.1f
    )

    val trainedParams = (0 until numEpochs).foldLeft(initialParams):
      case (params, epoch) =>
        timed(s"Evaluation $epoch/$numEpochs"):
          val lossValue = batchLoss(keysForEpochs(epoch), testX)(params)
          println(s"Test loss in epoch $epoch: $lossValue")
        timed(s"Training $epoch/$numEpochs"):
          dimwit.gc()
          trainEpoch(keysForEpochs(epoch), epoch, params)

    /*
     * Evaluation
     * */
    val vae = VAE(trainedParams)

    val reconstructed = testX
      .slice(Axis[TestSample] -> (0 until 64))
      .vmap(Axis[TestSample]): sample =>
        val (mean, logVar) = vae.encoder(sample.ravel)
        val latent = reparametrize(mean, logVar, dataKey) // TODo Key management
        vae.decoder(latent)
      .relabel(Axis[TestSample] -> Axis[Prime[Height] |*| Prime[Width]])

    plotImg(
      reconstructed
        .rearrange(
          (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
          (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 28)
        )
    )

    /*
     * Sampling from the latent space
     */
    val stdNormal = Normal.standardNormal(Shape(Axis[Latent] -> latentDim))
    val sampled = dataKey.splitvmap(Axis[Prime[Height] |*| Prime[Width]], 64): key =>
      val z = stdNormal.sample(key)
      vae.decoder(z)

    plotImg(
      // rearrange to 8x8 grid of 28x28 images
      sampled.rearrange(
        (Axis[Prime[Height] |*| Height], Axis[Prime[Width] |*| Width]),
        (Axis[Prime[Height]] -> 8, Axis[Prime[Width]] -> 8, Axis[Height] -> 28, Axis[Width] -> 8)
      )
    )

def plotImg[H, W](img2d: Tensor2[H, W, Float]): Unit =
  import me.shadaj.scalapy.py
  val matplotlib = py.module("matplotlib")
  matplotlib.use("macosx")
  val plt = py.module("matplotlib.pyplot")
  plt.imshow(img2d.jaxValue, cmap = "gray")
  plt.show()
