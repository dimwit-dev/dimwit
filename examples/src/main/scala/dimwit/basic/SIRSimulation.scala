package dimwit.examples.basic

import dimwit.*

/** A simple SIR (Susceptible-Infectious-Recovered) simulation.
  */
object SIRSimulation:

  trait Time derives Label
  trait Group derives Label

  // Explicit state representation replaces the Compartment dimension
  case class SIRState(
      S: Tensor1[Group, Float32],
      I: Tensor1[Group, Float32],
      R: Tensor1[Group, Float32]
  ):
    lazy val N: Tensor1[Group, Float32] = S + I + R

  /** One step of the simulation, according to the SIR model equations
    *
    * @param state The current state of the system
    * @param beta The infection rate matrix (with entry beta[h, g] controlling how strongly infectious
    * individuals in group h infect susceptible individuals in group g)
    * @param gamma The recovery rate
    * @param dt The time step
    * @return The next state of the system
    */
  def step(
      beta: Tensor2[Group, Prime[Group], Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32]
  )(state: SIRState): SIRState =
    import state.{S, I, R, N}

    val infectiousFraction = I / N
    val force = infectiousFraction.dot(Axis[Group])(beta)
    val transmissions = S * force.dropPrimes

    val recoveries = I *! gamma

    // compute next state
    val SNext = S - transmissions *! dt
    val INext = I + (transmissions - recoveries) *! dt
    val RNext = R + recoveries *! dt

    SIRState(SNext, INext, RNext)

  /** run n steps of the simulation, starting from the initial state
    *
    * @param initial The initial state of the system
    * @param beta @see [[step]]
    * @param gamma @see [[step]]
    * @param dt @see [[step]]
    * @param nSteps The number of steps to simulate
    * @return The trajectory of the system over time
    */
  def simulate(
      initial: SIRState,
      beta: Tensor2[Group, Prime[Group], Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32]
  ): LazyList[SIRState] =
    LazyList.iterate(initial)(step(beta, gamma, dt))

  @main def runSIRSimulation(): Unit =
    dimwit.initialize()

    val groupDim = Axis[Group] -> 3

    // Setup initial state of the system

    val initialS = Tensor(Shape(groupDim)).fromFunction(index =>
      index(Axis[Group]) match
        case 0 => 990f
        case 1 => 1995f
        case 2 => 1500f
    )
    val initialI = Tensor(Shape(groupDim)).fromFunction(index =>
      index(Axis[Group]) match
        case 0 => 10f
        case 1 => 5f
        case 2 => 0f
    )
    val initialR = Tensor(Shape(groupDim)).fromFunction(_ => 0f)
    val initial = SIRState(initialS, initialI, initialR)

    // Setup parameters for the simulation

    /*
     * beta(h, g) controls how strongly infectious individuals in group h
     * infect susceptible individuals in group g.
     */
    val beta: Tensor2[Group, Prime[Group], Float32] =
      Tensor(Shape(groupDim, Axis[Prime[Group]] -> groupDim.size)).fromFunction(index =>
        (index(Axis[Group]), index(Axis[Prime[Group]])) match
          // infectious children -> susceptible children/adults/elderly
          case (0, 0) => 0.40f
          case (0, 1) => 0.20f
          case (0, 2) => 0.10f

          // infectious adults -> susceptible children/adults/elderly
          case (1, 0) => 0.20f
          case (1, 1) => 0.30f
          case (1, 2) => 0.15f

          // infectious elderly -> susceptible children/adults/elderly
          case (2, 0) => 0.10f
          case (2, 1) => 0.15f
          case (2, 2) => 0.20f

          case (_, _) => throw new IllegalArgumentException("Invalid group indices")
      )
    val gamma = Tensor0(0.1f)
    val dt = Tensor0(0.1f)

    // Run the simulation

    val nSteps = 160
    val trajectory =
      SIRSimulation.simulate(
        initial = initial,
        beta = beta,
        gamma = gamma,
        dt = dt
      ).take(nSteps + 1).toList

    // Report the results

    val infectedOverTime: Tensor1[Time, Float32] =
      stack(trajectory.map(_.I), Axis[Time]).sum(Axis[Group])
    println(s"I(0)   = ${infectedOverTime.slice(Axis[Time].at(0))}")
    println(s"I(mid) = ${infectedOverTime.slice(Axis[Time].at(nSteps / 2))}")
    println(s"I(end) = ${infectedOverTime.slice(Axis[Time].at(nSteps))}")

    val finalInfectedByGroup: Tensor1[Group, Float32] =
      stack(trajectory.map(_.I), Axis[Time]).slice(Axis[Time].at(nSteps))
    println(s"Final infected by group: $finalInfectedByGroup")
