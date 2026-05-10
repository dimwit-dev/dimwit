package dimwit.tensor

import dimwit.*
import scala.collection.View.Empty

class TensorCovarianceSuite extends DimwitTest:

  it("Shape type hierarchy example: Generic function with upper-bounded type parameter"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label
    def genericFunction[T <: Parent: Label](t: Tensor1[T, Float32]): Tensor1[T, Float32] = t + t
    val child1: Tensor1[Child1, Float32] = Tensor(Shape1(Axis[Child1] -> 4)).fill(1f)
    val child2: Tensor1[Child2, Float32] = Tensor(Shape1(Axis[Child2] -> 4)).fill(1f)

    "genericFunction(child1)" should compile
    "genericFunction(child2)" should compile
    "genericFunction(noChild)" shouldNot compile

  it("Value-types example: Logits cannot be added to Probabilities"):
    trait Classes derives Label

    object MLContext:
      opaque type Logit = Float32
      opaque type Prob = Float32

      def createLogits[L: Label](s: Shape1[L]): Tensor1[L, Logit] = Tensor(s).fill(0f)
      def createProbs[L: Label](s: Shape1[L]): Tensor1[L, Prob] = Tensor(s).fill(0f)

      // Operation restricted only to Logit 'land'
      def combineLogits[L: Label](a: Tensor1[L, Logit], b: Tensor1[L, Logit]): Tensor1[L, Logit] = a + b
      def combineProbs[L: Label](a: Tensor1[L, Prob], b: Tensor1[L, Prob]): Tensor1[L, Prob] = a * b
      def toProbs[L: Label](logits: Tensor1[L, Logit]): Tensor1[L, Prob] = logits.vmap(Axis[L]) { l => 1.0f / (1.0f + -l.exp) }

    val shape = Shape1(Axis[Classes] -> 10)
    val logits = MLContext.createLogits(shape)
    val probs = MLContext.createProbs(shape)
    val rawFloats = Tensor(shape).fill(1f)

    "MLContext.combineLogits(logits, logits)" should compile
    "MLContext.combineProbs(probs, probs)" should compile
    "MLContext.combineLogits(logits, probs)" shouldNot compile
    "MLContext.combineProbs(logits, probs)" shouldNot compile
    "MLContext.combineLogits(logits, rawFloats)" shouldNot compile
    "MLContext.combineProbs(probs, rawFloats)" shouldNot compile
