package examples.api

import shapeful.* 

@main
def autoDiffAPI(): Unit =
  val AB = Tensor.of[Float32].ones(Shape(
    Axis["A"] -> 10,
    Axis["B"] -> 5,
  ))
  val AC = Tensor.of[Float32].ones(Shape(
      Axis["A"]-> 10,
      Axis["C"] -> 5
  ))
  val ABCD = Tensor.of[Float32].ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5,
  ))
  {
    def f(x: Tensor1["A", Float32]): Tensor0[Float32] = x.sum
    val df = Autodiff.grad(f)
    val delta = df(Tensor1.of[Float32](Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    type ParamsTuple = (Tensor2["A", "B", Float32], Tensor1["C", Float32])
    def f(params: ParamsTuple): Tensor0[Float32] =
      params._1.sum + params._2.sum
    val df = Autodiff.grad(f)
    val delta = df((
      Tensor2.of[Float32](Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1.of[Float32](Axis["C"], Array.fill(5)(1.0f))
    ))
    println((delta._1.shape, delta._2.shape))
  }
  {
    case class Params(
      a: Tensor2["A", "B", Float32], 
      b: Tensor1["C", Float32],
    ) derives TensorTree
    def f(params: Params): Tensor0[Float32] =
      params.a.sum + params.b.sum
    val df = Autodiff.grad(f)
    val delta = df(Params(
      Tensor2.of[Float32](Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1.of[Float32](Axis["C"], Array.fill(5)(1.0f))
    ))
    println(delta)
  }
  {
    def f(x: Tensor1["A", Float32]): Tensor1["A", Float32] = x
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1.of[Float32](Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float32]) = x.outerProduct(x)
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1.of[Float32](Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float32], Tensor1["C", Float32])
    def f(x: ParamsTuple): Tensor1["A", Float32] = x._1.slice(Axis["B"] -> 0)
    val df = Autodiff.jacobian(f)
    val delta = df((
      Tensor2.of[Float32](Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1.of[Float32](Axis["C"], Array.fill(5)(1.0f))
    ))
    println((delta._1.shape, delta._2.shape))
  }
  {
    println("Hessian")
    def f(x: Tensor1["A", Float32]): Tensor0[Float32] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(Tensor1.of[Float32](Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float32]): Tensor0[Float32] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val dddf = Autodiff.jacobian(ddf)
    val ddddf = Autodiff.jacobian(dddf)
    val delta = ddddf(Tensor1.of[Float32](Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float32], Tensor1["C", Float32])
    def f(x: ParamsTuple): Tensor0[Float32] = x._1.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf((
      Tensor2.of[Float32](Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1.of[Float32](Axis["C"], Array.fill(5)(1.0f))
    ))
    // TODO Is this actually correct, check it!
    println((
      (delta._1._1.shape, delta._1._2.shape),
      (delta._2._1.shape, delta._2._2.shape)
    ))
  }