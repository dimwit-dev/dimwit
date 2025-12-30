package dimwit.tensor

import dimwit.*
import dimwit.Conversions.given
import org.scalatest.propspec.AnyPropSpec
import org.scalatest.matchers.should.Matchers
import org.scalatestplus.scalacheck.ScalaCheckPropertyChecks
import scala.collection.View.Empty

class TensorCovarianceSuite extends AnyPropSpec with ScalaCheckPropertyChecks with Matchers:

  property("Shape type hierarchy example: Concrete function with supertype parameter"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label
    trait NoChild derives Label
    def concreteFunction(t: Tensor1[Parent, Float]): Tensor1[Parent, Float] = t + t
    val child1: Tensor1[Child1, Float] = Tensor.ones(Shape1(Axis[Child1] -> 4), VType[Float])
    val child2: Tensor1[Child2, Float] = Tensor.ones(Shape1(Axis[Child2] -> 4), VType[Float])
    val noChild: Tensor1[NoChild, Float] = Tensor.ones(Shape1(Axis[NoChild] -> 4), VType[Float])

    "concreteFunction(child1)" should compile
    "concreteFunction(child2)" should compile
    "concreteFunction(noChild)" shouldNot compile

  property("Shape type hierarchy example: Generic function with upper-bounded type parameter"):
    trait Parent derives Label
    trait Child1 extends Parent derives Label
    trait Child2 extends Parent derives Label
    def genericFunction[T <: Parent: Label](t: Tensor1[T, Float]): Tensor1[T, Float] = t + t
    val child1: Tensor1[Child1, Float] = Tensor.ones(Shape1(Axis[Child1] -> 4), VType[Float])
    val child2: Tensor1[Child2, Float] = Tensor.ones(Shape1(Axis[Child2] -> 4), VType[Float])

    "genericFunction(child1)" should compile
    "genericFunction(child2)" should compile
    "genericFunction(noChild)" shouldNot compile

  property("Value-types example: Logits cannot be added to Probabilities"):
    trait Classes derives Label

    object MLContext:
      opaque type Logit = Float
      opaque type Prob = Float

      def createLogits[L: Label](s: Shape1[L]): Tensor1[L, Logit] = Tensor.zeros(s, VType[Logit])
      def createProbs[L: Label](s: Shape1[L]): Tensor1[L, Prob] = Tensor.zeros(s, VType[Prob])

      // Operation restricted only to Logit 'land'
      def combineLogits[L: Label](a: Tensor1[L, Logit], b: Tensor1[L, Logit]): Tensor1[L, Logit] = a + b
      def combineProbs[L: Label](a: Tensor1[L, Prob], b: Tensor1[L, Prob]): Tensor1[L, Prob] = a * b
      def toProbs[L: Label](logits: Tensor1[L, Logit]): Tensor1[L, Prob] = logits.vmap(Axis[L]) { l => 1.0f / (1.0f + -l.exp) }

    val shape = Shape1(Axis[Classes] -> 10)
    val logits = MLContext.createLogits(shape)
    val probs = MLContext.createProbs(shape)
    val rawFloats = Tensor.ones(shape, VType[Float])

    "MLContext.combineLogits(logits, logits)" should compile
    "MLContext.combineProbs(probs, probs)" should compile
    "MLContext.combineLogits(logits, probs)" shouldNot compile
    "MLContext.combineProbs(logits, probs)" shouldNot compile
    "MLContext.combineLogits(logits, rawFloats)" shouldNot compile
    "MLContext.combineProbs(probs, rawFloats)" shouldNot compile
