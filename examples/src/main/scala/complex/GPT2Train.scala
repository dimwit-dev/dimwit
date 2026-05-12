package examples.complex.nanoGPT

import dimwit.*
import dimwit.Conversions.given

import nn.ActivationFunctions.*
import dimwit.random.Random
import dimwit.stats.Normal
import nn.AdamW
import nn.Adam
import nn.Loss
import examples.timed
import dimwit.python.PythonSetup
import src.main.scala.complex.safePyTree

import java.io.RandomAccessFile
import java.nio.channels.FileChannel
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import dimwit.jax.Jax
import dimwit.tensor.DType
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters
import src.main.scala.complex.loadPyTree
import dimwit.stats.Categorical
import dimwit.stats.Uniform

object Config:
  inline val numIterations = 60_000
  inline val trainLogInterval = 10
  inline val evalLogInterval = 250
  inline val numberOfEvalIterations = 200
  inline val vocabSize = 65
  inline val learningRate = 1e-3f
  inline val beta1 = 0.9f
  inline val beta2 = 0.99f
  inline val batchSize = 64
  inline val contextLength = 256
  inline val numberOfLayers = 6
  inline val numberOfHeads = 6
  inline val extentEmbedding = 384
  inline val dropout = 0.2

  private inline def validateConfig: Unit =
    inline if extentEmbedding % numberOfHeads != 0 then
      import scala.compiletime.{error, constValue}
      import scala.compiletime.ops.int.ToString
      scala.compiletime.error(
        "Config Error: 'extentEmbedding' must be divisible by 'numberOfHeads', but got extentEmbedding = " + constValue[ToString[extentEmbedding.type]] + " and numberOfHeads = " + constValue[ToString[numberOfHeads.type]]
      )

  validateConfig

import Config.*

// assert(extentEmbedding % numberOfHeads == 0, "Embedding size must be divisible by number of heads")
val headAxisExtent = Axis[Head] -> numberOfHeads
val headKeyAxisExtent = Axis[HeadKey] -> extentEmbedding / numberOfHeads
val headQueryAxisExtent = Axis[HeadQuery] -> extentEmbedding / numberOfHeads
val headValueAxisExtent = Axis[HeadValue] -> extentEmbedding / numberOfHeads
val embeddingAxisExtent = Axis[Embedding] -> extentEmbedding
val embeddingMixedAxisExtent = Axis[EmbeddingMixed] -> extentEmbedding * 4
val vocabAxisExtent = Axis[Vocab] -> vocabSize
val contextAxisExtent = Axis[Context] -> contextLength

// Dimensions
trait Vocab derives Label
trait Embedding derives Label
trait Context derives Label
trait EmbeddingMixed derives Label

trait Batch derives Label

case class LayerNormalizationParams(
    weight: Tensor1[Embedding, Float32],
    bias: Tensor1[Embedding, Float32]
)

case class LinearLayerParams[In, Out](
    weight: Tensor2[In, Out, Float32],
    bias: Tensor1[Out, Float32]
)

case class ProjectionLayerParams[In, Out](
    weight: Tensor2[In, Out, Float32]
)

trait Head derives Label
trait HeadKey derives Label
trait HeadQuery derives Label
trait HeadValue derives Label

case class HeadsParams[Kind](val weights: Tensor3[Head, Embedding, Kind, Float32], val bias: Tensor2[Head, Kind, Float32])

case class MultiHeadAttentionParams(
    wq: HeadsParams[HeadQuery],
    wk: HeadsParams[HeadKey],
    wv: HeadsParams[HeadValue],
    proj: LinearLayerParams[Head |*| HeadValue, Embedding]
) derives TensorTree

case class EmbeddingMixerParams(
    c_fc: LinearLayerParams[Embedding, EmbeddingMixed],
    c_proj: LinearLayerParams[EmbeddingMixed, Embedding]
) derives TensorTree

case class TransformerLayerParams(
    ln1: LayerNormalizationParams,
    attn: MultiHeadAttentionParams,
    ln2: LayerNormalizationParams,
    embeddingMixer: EmbeddingMixerParams
) derives TensorTree

case class GPT2Params(
    vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float32],
    positionalEmbeddings: Tensor2[Context, Embedding, Float32],
    layers: List[TransformerLayerParams],
    outputNormalization: LayerNormalizationParams
) derives TensorTree

object GPT2Params:

  def init(initKey: Random.Key): GPT2Params =
    def initLayerNormalizationParams(): LayerNormalizationParams =
      LayerNormalizationParams(
        weight = Tensor(Shape(embeddingAxisExtent)).fill(1f),
        bias = Tensor(Shape(embeddingAxisExtent)).fill(0f)
      )
    def initMutliHeadAttentionParams(key: Random.Key): MultiHeadAttentionParams =
      MultiHeadAttentionParams(
        wq = HeadsParams(
          weights = Normal.standardIsotropic(Shape(headAxisExtent, embeddingAxisExtent, headQueryAxisExtent), scale = 0.02f).sample(key),
          bias = Tensor(Shape(headAxisExtent, headQueryAxisExtent)).fill(0f)
        ),
        wk = HeadsParams(
          weights = Normal.standardIsotropic(Shape(headAxisExtent, embeddingAxisExtent, headKeyAxisExtent), scale = 0.02f).sample(key),
          bias = Tensor(Shape(headAxisExtent, headKeyAxisExtent)).fill(0f)
        ),
        wv = HeadsParams(
          weights = Normal.standardIsotropic(Shape(headAxisExtent, embeddingAxisExtent, headValueAxisExtent), scale = 0.02f).sample(key),
          bias = Tensor(Shape(headAxisExtent, headValueAxisExtent)).fill(0f)
        ),
        proj = LinearLayerParams(
          weight = Normal.standardIsotropic(Shape(headAxisExtent * headValueAxisExtent, embeddingAxisExtent), scale = 0.02f).sample(key),
          bias = Tensor(Shape(embeddingAxisExtent)).fill(0f)
        )
      )
    def initEmbeddingMixerParams(key: Random.Key): EmbeddingMixerParams =
      val (fcKey, projKey) = key.split2()
      EmbeddingMixerParams(
        c_fc = LinearLayerParams(
          weight = Normal.standardIsotropic(Shape(embeddingAxisExtent, embeddingMixedAxisExtent), scale = 0.02f).sample(fcKey),
          bias = Tensor(Shape(embeddingMixedAxisExtent)).fill(0f)
        ),
        c_proj = LinearLayerParams(
          weight = Normal.standardIsotropic(Shape(embeddingMixedAxisExtent, embeddingAxisExtent), scale = 0.02f).sample(projKey),
          bias = Tensor(Shape(embeddingAxisExtent)).fill(0f)
        )
      )
    def initTransformerLayerParams(key: Random.Key): TransformerLayerParams =
      val (attnKey, mixKey) = key.split2()
      TransformerLayerParams(
        ln1 = initLayerNormalizationParams(),
        attn = initMutliHeadAttentionParams(attnKey),
        ln2 = initLayerNormalizationParams(),
        embeddingMixer = initEmbeddingMixerParams(mixKey)
      )
    val keys = initKey.split(4)
    val layerKeys = keys(2).split(numberOfLayers)
    GPT2Params(
      vocabularyEmbeddings = Normal.standardIsotropic(Shape(vocabAxisExtent, embeddingAxisExtent), scale = 0.02f).sample(keys(0)),
      positionalEmbeddings = Normal.standardIsotropic(Shape(contextAxisExtent, embeddingAxisExtent), scale = 0.02f).sample(keys(1)),
      layers = layerKeys.map(initTransformerLayerParams).toList,
      outputNormalization = initLayerNormalizationParams()
    )

case class GPT2(params: GPT2Params) extends (Tensor1[Context, Int32] => Tensor1[Context, Int32]):

  private case class LinearLayer[In: Label, Out: Label](params: LinearLayerParams[In, Out]) extends (Tensor1[In, Float32] => Tensor1[Out, Float32]):
    override def apply(x: Tensor1[In, Float32]): Tensor1[Out, Float32] =
      x.dot(Axis[In])(params.weight) + params.bias

  private case class EmbeddingMixer(params: EmbeddingMixerParams) extends (Tensor2[Context, Embedding, Float32] => Tensor2[Context, Embedding, Float32]):
    private val hiddenLayer = LinearLayer(params.c_fc)
    private val outputLayer = LinearLayer(params.c_proj)
    // TODO add dropout

    def apply(in: Tensor2[Context, Embedding, Float32]): Tensor2[Context, Embedding, Float32] =
      in.vmap(Axis[Context])(x =>
        val hidden = gelu(hiddenLayer(x))
        outputLayer(hidden)
      )

  private case class ProjectionLayer[In: Label, Out: Label](params: ProjectionLayerParams[In, Out]) extends (Tensor1[In, Float32] => Tensor1[Out, Float32]):
    def apply(x: Tensor1[In, Float32]): Tensor1[Out, Float32] =
      x.dot(Axis[In])(params.weight)

  private case class MultiHeadAttention(params: MultiHeadAttentionParams) extends (Tensor2[Context, Embedding, Float32] => Tensor2[Context, Embedding, Float32]):

    private val projection = LinearLayer(params.proj)

    def apply(x: Tensor2[Context, Embedding, Float32]): Tensor2[Context, Embedding, Float32] =
      val heads = zipvmap(Axis[Head])(params.wq.weights, params.wq.bias, params.wk.weights, params.wk.bias, params.wv.weights, params.wv.bias):
        attention.tupled(_)(x)
      heads.vmap(Axis[Context])(heads => projection(heads.flatten))

    private def attention(
        wq: Tensor2[Embedding, HeadQuery, Float32],
        wqBias: Tensor1[HeadQuery, Float32],
        wk: Tensor2[Embedding, HeadKey, Float32],
        wkBias: Tensor1[HeadKey, Float32],
        wv: Tensor2[Embedding, HeadValue, Float32],
        wvBias: Tensor1[HeadValue, Float32]
    )(x: Tensor2[Context, Embedding, Float32]): Tensor2[Context, HeadValue, Float32] =

      trait AttnWeights derives Label

      def causalMasking(attnScores: Tensor2[Context, Prime[Context], Float32]): Tensor2[Context, Prime[Context], Float32] =
        val ctxLength = attnScores.shape(Axis[Context])
        val causalMask = tril(Tensor(Shape((Axis[Context] -> ctxLength, Axis[Prime[Context]] -> ctxLength))).fill(true))
        where(causalMask, attnScores, Tensor.like(attnScores).fill(Float.NegativeInfinity))

      val queries = x.dot(Axis[Embedding])(wq) +! wqBias
      val keys = x.dot(Axis[Embedding])(wk) +! wkBias
      val values = x.dot(Axis[Embedding])(wv) +! wvBias
      val dk = Tensor0(Math.sqrt(keys.shape(Axis[HeadKey])).toFloat)
      val attnScores = (queries.dot(Axis[HeadQuery ~ HeadKey])(keys) /! dk)
      val attnWeights = causalMasking(attnScores)
        .vmap(Axis[Context])(attnScore => softmax(attnScore).relabelTo(Axis[AttnWeights]))
      val res = attnWeights.dot(Axis[AttnWeights ~ Context])(values)
      res

  private case class LayerNorm(params: LayerNormalizationParams) extends (Tensor1[Embedding, Float32] => Tensor1[Embedding, Float32]):

    private def standardize(x: Tensor1[Embedding, Float32]): Tensor1[Embedding, Float32] =
      val x0 = x -! x.mean
      val variance = x0.pow(2).mean
      val epsilon = 1e-6f
      x0 /! (variance + epsilon).sqrt

    def apply(x: Tensor1[Embedding, Float32]): Tensor1[Embedding, Float32] =
      standardize(x) * params.weight + params.bias

  private case class TransformerLayer(params: TransformerLayerParams) extends (Tensor2[Context, Embedding, Float32] => Tensor2[Context, Embedding, Float32]):
    private val embeddingMixer = EmbeddingMixer(params.embeddingMixer)
    private val multiHeadAttention = MultiHeadAttention(params.attn)
    private val preNormalization = LayerNorm(params.ln1)
    private val postNormalization = LayerNorm(params.ln2)

    def apply(t: Tensor2[Context, Embedding, Float32]): Tensor2[Context, Embedding, Float32] =
      var x = t
      x = x + multiHeadAttention(x.vmap(Axis[Context])(preNormalization))
      x = x + embeddingMixer(x.vmap(Axis[Context])(postNormalization))
      x

  private case class TransformerBlock(layers: List[TransformerLayer]) extends (Tensor2[Context, Embedding, Float32] => Tensor2[Context, Embedding, Float32]):
    override def apply(t: Tensor2[Context, Embedding, Float32]): Tensor2[Context, Embedding, Float32] =
      layers.foldLeft(t):
        case (t, layer) => layer(t)

  case class Embedder(vocabularyEmbeddings: Tensor2[Vocab, Embedding, Float32], positionalEmbeddings: Tensor2[Context, Embedding, Float32]):

    def apply(tokens: Tensor1[Context, Int32]): Tensor2[Context, Embedding, Float32] =
      val embeddings = vocabularyEmbeddings.take(Axis[Vocab])(tokens)
      embeddings + positionalEmbeddings

  case class OutputLayer(normalization: LayerNormalizationParams, projectionParams: ProjectionLayerParams[Embedding, Vocab]) extends (Tensor1[Embedding, Float32] => Tensor1[Vocab, Float32]):
    private val normalizationLayer = LayerNorm(normalization)
    private val projection = ProjectionLayer(projectionParams)
    override def apply(x: Tensor1[Embedding, Float32]): Tensor1[Vocab, Float32] =
      projection(normalizationLayer(x))

  private val embedder = Embedder(params.vocabularyEmbeddings, params.positionalEmbeddings)
  private val transformerBlock = TransformerBlock(params.layers.map(TransformerLayer(_)))
  private val outputLayer = OutputLayer(
    params.outputNormalization,
    ProjectionLayerParams(params.vocabularyEmbeddings.transpose) // Tying output weights with input embeddings
  )

  def logits(inputTokens: Tensor1[Context, Int32]): Tensor2[Context, Vocab, Float32] =
    val startEmbeddings = embedder(inputTokens)
    val endEmbeddings = transformerBlock(startEmbeddings)
    endEmbeddings.vmap(Axis[Context])(x => outputLayer(x))

  def probits(inputTokens: Tensor1[Context, Int32]): Tensor2[Context, Vocab, Float32] =
    val x = logits(inputTokens)
    val res = x.vapply(Axis[Vocab])(softmax)
    return res

  def apply(inputTokens: Tensor1[Context, Int32]): Tensor1[Context, Int32] =
    val x = probits(inputTokens)
    val res = x.argmax(Axis[Vocab])
    return res

@main def train(): Unit =

  import Tensor0.given

  dimwit.initialize()

  trait Data derives Label

  PythonSetup.initialize
  lazy val np = py.module("numpy")
  case class Sample(
      input: Tensor2[Batch, Context, Int32],
      labels: Tensor2[Batch, Context, Int32]
  )
  def createDataset(key: Random.Key, pathToBinaryFile: String): Iterator[Sample] =
    val data = dimwit.python.PyBridge.liftPyTensor1(Axis[Data], VType[Int32])(Jax.jnp.asarray(np.memmap(pathToBinaryFile, dtype = np.uint16, mode = "r")))
    def sliceContextBlockAt(idx: Tensor0[Int32]): Tensor1[Context, Int32] =
      data
        .dynamicSlice(idx, contextLength)
        .relabelTo(Axis[Context])
    val numDataPoints = data.shape(Axis[Data])
    val lastValidIdx = numDataPoints - contextLength
    val batchIndicesDist = IndependentDistribution.fromUnivariate(shape = Shape1(Axis[Batch] -> batchSize), Uniform(min = 0, max = lastValidIdx))
    Iterator.unfold(key):
      case key =>
        val randomBatchIndex = batchIndicesDist.sample(key)
        val x = randomBatchIndex.vmap(Axis[Batch])(index => sliceContextBlockAt(index))
        val y = randomBatchIndex.vmap(Axis[Batch])(index => sliceContextBlockAt(index + 1))
        Some(Sample(x, y), key.next)

  def createTrainDataset(key: Random.Key): Iterator[Sample] =
    val pathToTrainBinaryFile = "data/nanoGPT/shakespeare_char/train.bin"
    createDataset(key, pathToTrainBinaryFile)
  def createValDataset(key: Random.Key): Iterator[Sample] =
    val pathToValBinaryFile = "data/nanoGPT/shakespeare_char/val.bin"
    createDataset(key, pathToValBinaryFile)

  val initParams = GPT2Params.init(Random.Key(42))

  import Tensor0.given
  val adam = Adam(learningRate = learningRate, b1 = beta1, b2 = beta2, epsilon = 1e-8f)
  val adamW = AdamW(adam, weightDecayFactor = 1e-1f)
  type AdamWState = adamW.State[GPT2Params]

  case class TrainingState(
      params: GPT2Params,
      adamWState: AdamWState,
      loss: Tensor0[Float32]
  )

  def batchLoss(input: Tensor2[Batch, Context, Int32], labels: Tensor2[Batch, Context, Int32])(params: GPT2Params): Tensor0[Float32] =
    val model = GPT2(params)
    val logits = input.vmap(Axis[Batch])(model.logits)
    val lossPerSample = zipvmap(Axis[Batch])(labels, logits): (labels, logits) =>
      val lossPerContextPosition = zipvmap(Axis[Context])(labels, logits): (label, logits) =>
        Loss.crossEntropy(logits = logits, label = label)
      lossPerContextPosition.mean
    lossPerSample.mean

  def gradientStep(
      input: Tensor2[Batch, Context, Int32],
      labels: Tensor2[Batch, Context, Int32],
      state: TrainingState
  ): TrainingState =
    val lossBatch = batchLoss(input, labels)
    val grads = Autodiff.grad(lossBatch)(state.params)
    val loss = lossBatch(state.params) // TODO move to gradAndValue
    val (params, adamWState) = adamW.update(grads, state.params, state.adamWState)
    TrainingState(params = params, adamWState = adamWState, loss = loss)
  val jitStep = jitDonatingUnsafe(gradientStep)

  def evaluate(input: Tensor2[Batch, Context, Int32], labels: Tensor2[Batch, Context, Int32], params: GPT2Params): Tensor0[Float32] =
    batchLoss(input, labels)(params)
  // val evalF = jit(evaluate)
  val evalF = eagerCleanup(evaluate)

  def miniBatchGradientDescent(
      samples: Iterator[Sample],
      startState: TrainingState
  ): Iterator[TrainingState] =
    samples.scanLeft(startState):
      case (state, sample) =>
        dimwit.gc()
        jitStep(sample.input, sample.labels, state)

  val trainSampleStream = createTrainDataset(Random.Key(42))
  val valSampleStream = createValDataset(Random.Key(42))
  val initState = TrainingState(initParams, adamW.init(initParams), Tensor0(-1f))
  val trainTrajectory = miniBatchGradientDescent(trainSampleStream, initState)
  val finalState = trainTrajectory.zipWithIndex
    .drop(1)
    .tapEach:
      case (state, iter) =>
        if iter % trainLogInterval == 0 then
          println(
            List(
              s"iter $iter",
              f"loss: ${state.loss.item}%.2f"
            ).mkString(", ")
          )
    .tapEach:
      case (state, iter) =>
        if iter % evalLogInterval == 0 then
          val valLossStream = valSampleStream.map: sample =>
            evalF(sample.input, sample.labels, state.params).item // evalF is new
          val avgValLoss = valLossStream.take(numberOfEvalIterations).sum / numberOfEvalIterations
          println(f"Evaluation at iter $iter: validation loss: $avgValLoss%.2f")
          safePyTree(state.params, f"gpt2_params_iter_$iter.pkl")
          // dimwit.gc()
          // Thread.sleep(100)
    .drop(numIterations - 1) // iterate to final iteration
    .next()

@main def inference(): Unit =
  // 1. Setup
  PythonSetup.initialize
  val checkpointPath = "gpt2_params_iter_1000.pkl"

  // 2. Load Weights
  println(s"Loading model from $checkpointPath...")
  val state: GPT2Params = loadPyTree[GPT2Params](checkpointPath)
  val model = GPT2(state)

  // 3. Define Prompt
  // val promptText = "To be, or not to be, that is the question:"
  val promptText = "Romeo, Romeo, wherefore art thou"
  println(s"Prompt: $promptText")

  // 4. Run Generation
  val result = InferenceUtil.generate(model, promptText, maxNewTokens = 100)

  println("-" * 50)
  println("Full Generated Text:")
  println(result)

object InferenceUtil:
  // Standard characters from the shakespeare_char dataset (Vocab Size = 65)
  val chars = "\n" + " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  println(s"Vocab Size: ${chars.length}")
  val charToInt = chars.zipWithIndex.toMap
  val intToChar = chars.zipWithIndex.map(_.swap).toMap

  def encode(s: String): List[Int] = s.map(c => charToInt.getOrElse(c, 0)).toList
  def decode(l: List[Int]): String = l.map(i => intToChar.getOrElse(i, ' ')).mkString

  def generate(
      model: GPT2,
      prompt: String,
      maxNewTokens: Int,
      temperatur: Float = 1.0f
  ): String =

    var currentTokens = encode(prompt)

    val ctxExtent = Axis[Context] -> Config.contextLength

    println(s"Generating $maxNewTokens tokens, starting at prompt length ${currentTokens.length}...")

    val sampleKey = Random.Key.fromTime()

    for i <- 0 until maxNewTokens do
      val window =
        if currentTokens.length > Config.contextLength
        then currentTokens.takeRight(Config.contextLength)
        else currentTokens

      val effectiveLength = window.length

      val inputTensor = Tensor(Shape(ctxExtent)).fill(0)

      val paddedData = window ++ List.fill(Config.contextLength - effectiveLength)(0)

      val inputData = paddedData.toArray
      val currentBatch = Tensor(Shape(ctxExtent)).fromArray(inputData)
      val logits = model.logits(currentBatch)

      val lastTokenIndex = effectiveLength - 1
      val nextTokenLogits = logits.slice(Axis[Context].at(lastTokenIndex))
      val nextTokenId =
        if temperatur == 0f
        then nextTokenLogits.argmax(Axis[Vocab]).item
        else Categorical.fromFloat(softmax(nextTokenLogits /! temperatur)).sample(sampleKey).item

      currentTokens = currentTokens :+ nextTokenId

      println((nextTokenId, decode(List(nextTokenId))))

    println()
    decode(currentTokens)
