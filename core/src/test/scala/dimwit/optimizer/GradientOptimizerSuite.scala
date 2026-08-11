package dimwit.optimizer

import dimwit.*
import dimwit.Conversions.given

class GradientOptimizerSuite extends DimwitTest:

  describe("GradientDescent"):
    it("should converge towards the minimum of f(x) = (x+1)^2 at x = -1"):
      val optimizer = GradientDescent(learningRate = 0.1)
      val minX = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(1000).next()
      minX.item shouldBe -1.0f +- 0.1f

    it("should compute the step (single step)"):
      val optimizer = GradientDescent(learningRate = 0.1)
      val initParams = Tensor0(2.0f)
      val initState = optimizer.init(initParams)

      val grad = Grad(Tensor0(6.0f))
      val (nextParams, nextState) = optimizer.update(grad, initParams, initState)

      nextParams.item shouldBe 1.4f +- 1e-5f

  describe("Adam"):
    it("should converge towards the minimum of f(x) = (x+1)^2 at x = -1"):
      val optimizer = Adam(learningRate = 0.1f)
      val minX = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(1000).next()
      minX.item shouldBe -1.0f +- 0.1f

    it("should compute the exact momentum and velocity updates (single step)"):
      val optimizer = Adam(learningRate = 0.1f, beta1 = 0.9f, beta2 = 0.999f)
      val initParams = Tensor0(2.0f)
      val initState = optimizer.init(initParams)

      val grad = Grad(Tensor0(6.0f))
      val (nextParams, nextState) = optimizer.update(grad, initParams, initState)

      nextParams.item shouldBe 1.9f +- 1e-5f
      nextState.momentums.item shouldBe 0.6f +- 1e-5f
      nextState.velocities.item shouldBe 0.036f +- 1e-5f
      nextState.beta1t.item shouldBe 0.9f +- 1e-5f
      nextState.beta2t.item shouldBe 0.999f +- 1e-5

  describe("AdamW"):
    it("should converge towards the minimum of f(x) = (x+1)^2 at x = -1"):
      val optimizer = AdamW(Adam(learningRate = 0.1f), weightDecayFactor = 0.1f)
      val minX = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(1000).next()
      minX.item shouldBe -1.0f +- 0.1f

    it("should apply decoupled weight decay (single step)"):
      val adam = Adam(learningRate = 0.1f)
      val adamW = AdamW(adam, weightDecayFactor = 0.1f)

      val initParams = Tensor0(2.0f)
      val grad = Grad(Tensor0(6.0f))
      val (adamParams, _) = adam.update(grad, initParams, adam.init(initParams))
      val (adamWParams, _) = adamW.update(grad, initParams, adamW.init(initParams))

      adamWParams.item shouldBe (adamParams.item - 0.02f) +- 1e-5f

  describe("Lion"):
    it("should converge towards the minimum of f(x) = (x+1)^2"):
      val optimizer = Lion(learningRate = 0.1f)
      val minX = optimizer.iterate(Tensor0(2.0f))(x => Grad(2 * (x + 1))).drop(1000).next()
      minX.item shouldBe -1.0f +- 0.1f

    it("should compute the exact sign-based update and momentum (single step)"):
      val optimizer = Lion(learningRate = 0.1f, beta1 = 0.9f, beta2 = 0.99f)
      val initParams = Tensor0(2.0f)
      val initMomentum = optimizer.init(initParams)

      val grad = Grad(Tensor0(6.0f))
      val (nextParams, nextState) = optimizer.update(grad, initParams, initMomentum)

      nextParams.item shouldBe 1.9f +- 1e-5f
      nextState.momentums.item shouldBe 0.06f +- 1e-5f
      nextState.step.item shouldBe 2
