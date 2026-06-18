package src.main.scala.basic

import dimwit.*
import dimwit.Conversions.given
import dimwit.autodiff.*

/** A simple SIR (Susceptible-Infectious-Recovered) simulation.
  */
object SIRSimulation:

  trait Time derives Label
  trait InfectiousGroup derives Label
  trait SusceptibleGroup derives Label
  trait Compartment derives Label

  // We store all the compartments (S, I, R) in a separate tensor dimension (Compartment) and encode them
  // using different IDs.
  val SIndex = 0
  val IIndex = 1
  val RIndex = 2

  /**    One step of the simulation, according to the SIR model equations
    *
    *    @param state The current state of the system, with shape [SusceptibleGroup, Compartment]
    *    @param beta The infection rate matrix (with entry beta[h, g] controlling how strongly infectious
    *      individuals in group h infect susceptible individuals in group g)
    *    @param gamma The recovery rate
    *    @param dt The time step
    *    @return The next state of the system
    */
  def step(
      state: Tensor2[SusceptibleGroup, Compartment, Float32],
      beta: Tensor2[InfectiousGroup, SusceptibleGroup, Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32]
  ): Tensor2[SusceptibleGroup, Compartment, Float32] =

    val S = state.slice(Axis[Compartment].at(SIndex))
    val I = state.slice(Axis[Compartment].at(IIndex))
    val R = state.slice(Axis[Compartment].at(RIndex))

    val N = S + I + R // All individuals in the population

    val infectiousFraction = (I / N).relabel(Axis[SusceptibleGroup].as(Axis[InfectiousGroup]))
    val force = infectiousFraction.dot(Axis[InfectiousGroup])(beta)
    val newInfections = S * force

    val recoveries = I *! gamma

    // compute next state
    val SNext = S - newInfections *! dt
    val INext = I + (newInfections - recoveries) *! dt
    val RNext = R + recoveries *! dt

    stack(Seq(SNext, INext, RNext), Axis[Compartment]).transpose

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
      initial: Tensor2[SusceptibleGroup, Compartment, Float32],
      beta: Tensor2[InfectiousGroup, SusceptibleGroup, Float32],
      gamma: Tensor0[Float32],
      dt: Tensor0[Float32],
      nSteps: Int
  ): Tensor3[Time, SusceptibleGroup, Compartment, Float32] =

    val states: IndexedSeq[Tensor2[SusceptibleGroup, Compartment, Float32]] =
      (0 until nSteps).scanLeft(initial): (state, _) =>
        step(state, beta, gamma, dt)

    stack(states, Axis[Time])

  @main def runSIRSimulation(): Unit =
    dimwit.initialize()

    val susceptibleGroupDim = Axis[SusceptibleGroup] -> 3
    val infectiousGroupDim = Axis[InfectiousGroup] -> 3
    val compartmentDim = Axis[Compartment] -> 3

    /*
     * Three groups, coded by (0, 1, 2), which is e.g.
     * children, adults, and elderly.
     */
    val initial: Tensor2[SusceptibleGroup, Compartment, Float32] =
      Tensor(Shape(Axis[SusceptibleGroup] -> 3, Axis[Compartment] -> 3)).fromFunction(index =>
        (index(Axis[SusceptibleGroup]), index(Axis[Compartment])) match
          // children
          case (0, 0) => 990f
          case (0, 1) => 10f
          case (0, 2) => 0f

          // adults
          case (1, 0) => 1995f
          case (1, 1) => 5f
          case (1, 2) => 0f

          // elderly
          case (2, 0) => 1500f
          case (2, 1) => 0f
          case (2, 2) => 0f

          case _ => 0f
      )

    /*
     *
     * beta(h, g) controls how strongly infectious individuals in group h
     * infect susceptible individuals in group g.
     */
    val beta: Tensor2[InfectiousGroup, SusceptibleGroup, Float32] =
      Tensor(Shape(Axis[InfectiousGroup] -> 3, Axis[SusceptibleGroup] -> 3)).fromFunction(index =>
        (index(Axis[InfectiousGroup]), index(Axis[SusceptibleGroup])) match // infectious children -> susceptible children/adults/elderly
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
          case _      => 0f
      )

    val gamma = Tensor0(0.1f)
    val dt = Tensor0(0.1f)
    val nSteps = 160

    val trajectory =
      SIRSimulation.simulate(
        initial = initial,
        beta = beta,
        gamma = gamma,
        dt = dt,
        nSteps = nSteps
      )

    /*
     * Total infected population over time:
     *
     *   I_total(t) = sum_g I_g(t)
     */
    val infectedOverTime: Tensor1[Time, Float32] =
      trajectory
        .slice(Axis[Compartment].at(SIRSimulation.IIndex))
        .sum(Axis[SusceptibleGroup])

    println(s"I(0)   = ${infectedOverTime.slice(Axis[Time].at(0))}")
    println(s"I(mid) = ${infectedOverTime.slice(Axis[Time].at(nSteps / 2))}")
    println(s"I(end) = ${infectedOverTime.slice(Axis[Time].at(nSteps))}")

    /*
     * Infected population in each group at final time:
     */
    val finalInfectedByGroup: Tensor1[SusceptibleGroup, Float32] =
      trajectory
        .slice(Axis[Time].at(nSteps))
        .slice(Axis[Compartment].at(SIRSimulation.IIndex))

    println(s"Final infected by group: $finalInfectedByGroup")
