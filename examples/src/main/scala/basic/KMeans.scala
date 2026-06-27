package examples.basic.kmeans

import dimwit.Conversions.given
import dimwit.*
import dimwit.random.Random
import dimwit.stats.Normal

trait Point derives Label // N data points to cluster
trait Dim derives Label // D feature dimensions (coordinates of each point)
trait Cluster derives Label // K centroids — the result of clustering

trait CenterBasedClustering(
    val nClusters: Int
):

  /** Computes the distance between a point and a cluster center.
    * @param point A single data point
    * @param clusterCenter A single cluster center
    * @return The distance between the point and the cluster center.
    */
  def distance(point: Tensor1[Dim, Float32], clusterCenter: Tensor1[Dim, Float32]): Tensor0[Float32]

  /** Computes the new center given the clusterPoints
    * @param clusterPoints Points assigned to a cluster, NaN for points not assigned to this cluster.
    * @return The new cluster center computed from the assigned points.
    */
  def calcCenter(clusterPoints: Tensor2[Point, Dim, Float32]): Tensor1[Dim, Float32]

  /** Compute the pairwise distances between each point and each center.
    * @param points The data points
    * @param center The cluster centers
    * @return A tensor of shape [Point, Cluster] containing the distances from each point to each cluster center.
    */
  def pairwiseDistances(
      points: Tensor2[Point, Dim, Float32],
      centers: Tensor2[Cluster, Dim, Float32]
  ): Tensor2[Point, Cluster, Float32] =
    points.vmap(Axis[Point]): point =>
      centers.vmap(Axis[Cluster]): center =>
        distance(point, center)

  /** Assign each point to the nearest center.
    * @param points The data points.
    * @param center The cluster centers.
    */
  def assignPoints(
      points: Tensor2[Point, Dim, Float32],
      centers: Tensor2[Cluster, Dim, Float32]
  ): Tensor1[Point, Int32] =
    val distances = pairwiseDistances(points, centers)
    distances.argmin(Axis[Cluster])

  /** Update the centers by computing the mean of the points assigned to each cluster.
    *
    * @param points The data points
    * @param assignments The cluster index assigned to each point
    * @return The new cluster centers
    */
  private def calcCenters(
      points: Tensor2[Point, Dim, Float32],
      assignments: Tensor1[Point, Int32]
  ): Tensor2[Cluster, Dim, Float32] =
    val centers = (0 until nClusters).map: clusterIndex =>
      val clusterMembershipMask = assignments.elementEquals(Tensor.like(assignments).fill(clusterIndex))
      calcCenter(where(clusterMembershipMask.broadcastTo(points.shape), points, Tensor.like(points).fill(Float.NaN)))

    stack(centers, Axis[Cluster])

  /** Perform one iteration of the EM algorithm: assign points to nearest centers (E-step) and then update centers (M-step).
    * @param points The data points
    * @param centers The current cluster centers
    * @return The updated cluster centers
    */
  private def emStep(
      points: Tensor2[Point, Dim, Float32],
      centers: Tensor2[Cluster, Dim, Float32]
  ): Tensor2[Cluster, Dim, Float32] =
    val assignments = assignPoints(points, centers) // E-step: Assign points to nearest centers
    calcCenters(points, assignments) // M-step: Update centers based on assignments

  /** Run the trajectory of the K-Means algorithm
    * @param points The data points
    * @param centers The initial cluster centers (starting point)
    * @return Cluster centers at each iteration, ending with the final converged centers.
    */
  def run(
      points: Tensor2[Point, Dim, Float32],
      centers: Tensor2[Cluster, Dim, Float32]
  ): LazyList[Tensor2[Cluster, Dim, Float32]] =
    val nextCenters = emStep(points, centers)
    val converged = nextCenters.elementEquals(centers).all.item
    centers #:: (if converged then LazyList.empty else run(points, nextCenters))

  /** Main entry point to run K-Means clustering.
    * @param points The data points
    * @param key A random key for random center initialization
    * @param maxIterations Maximum number of iterations to run (in case of non-convergence)
    * @return The final cluster centers (after convergence or reaching max iterations)
    */
  def apply(
      points: Tensor2[Point, Dim, Float32],
      key: Random.Key,
      maxIterations: Int
  ): Tensor2[Cluster, Dim, Float32] =
    val steps = run(points, initializeCenters(points, key))
    steps.take(maxIterations).last

  /** Initialize cluster centers by randomly selecting points from the dataset.
    * @param points The data points
    * @param key A random key for random selection
    * @return Initial cluster centers randomly chosen from the data points
    */
  private def initializeCenters(
      points: Tensor2[Point, Dim, Float32],
      key: Random.Key
  ): Tensor2[Cluster, Dim, Float32] =
    val perm = Random.permutation(points.shape.extent(Axis[Point]))(key)
    points.take(Axis[Point])(perm.slice(Axis[Point].at(0 until nClusters))).relabel(Axis[Point].as(Axis[Cluster]))

/** Example implementation of the K-Means clustering algorithm, showcasing the use
  * of Named Tensors for clear and type-safe code.
  */
class KMeans(nClusters: Int) extends CenterBasedClustering(nClusters):

  override def distance(point: Tensor1[Dim, Float32], centroid: Tensor1[Dim, Float32]): Tensor0[Float32] =
    (point - centroid).pow(2).sum

  override def calcCenter(clusterPoints: Tensor2[Point, Dim, Float32]): Tensor1[Dim, Float32] =
    clusterPoints.nanmean(Axis[Point])

class KMedians(nClusters: Int) extends CenterBasedClustering(nClusters):

  override def distance(point: Tensor1[Dim, Float32], centroid: Tensor1[Dim, Float32]): Tensor0[Float32] =
    (point - centroid).abs.sum

  override def calcCenter(clusterPoints: Tensor2[Point, Dim, Float32]): Tensor1[Dim, Float32] =
    clusterPoints.nanmedian(Axis[Point])

object KMeans:

  //
  // Example usage
  //
  @main def run(): Unit =

    val n = 50 // points per cluster
    val k = 3 // clusters to discover

    val trueCenters = Array(
      Array(-3.0f, 0.0f), // cloud 0: lower-left
      Array(3.0f, 0.0f), // cloud 1: lower-right
      Array(0.0f, 4.0f) // cloud 2: top-center
    )

    def evaluate(alg: CenterBasedClustering, name: String, points: Tensor2[Point, Dim, Float32], trainKey: Random.Key): Unit =
      val centers = alg(points, trainKey, maxIterations = 50)
      val assignments = alg.assignPoints(points, centers)

      val inertia = alg.pairwiseDistances(points, centers).vmap(Axis[Point])(_.min).sum.item

      println(f"Inertia (within-cluster sum of distances): $inertia%.2f\n")

      // ── Report discovered clusters ─────────────────────────────────────────────
      val nTotal = n * k
      println(s"Discovered centers (${name}) vs true centers:")
      println(f"  ${"cluster"}%-9s  ${"centroid (x,y)"}%-22s  ${"true center"}%-16s  size")
      println("  " + "─" * 65)

      (0 until k).foreach: c =>
        // Extract the c-th centroid row: slice along the named Cluster axis.
        val center = centers.slice(Axis[Cluster].at(c))
        val cx = center.slice(Axis[Dim].at(0)).item
        val cy = center.slice(Axis[Dim].at(1)).item

        // Cluster size: count points whose assignment equals c.
        // elementEquals produces [Point, Bool]; asFloat32 then sum gives a count.
        val clusterSize =
          assignments
            .elementEquals(Tensor(Shape(Axis[Point] -> nTotal)).fill(c))
            .asFloat32.sum.item.toInt

        val tc = trueCenters(c)
        println(f"  cluster $c%-7d  ($cx%+6.2f, $cy%+6.2f)  (${tc(0)}%+6.2f, ${tc(1)}%+6.2f)  $clusterSize")

    def prepareData(dataKey: Random.Key): Tensor2[Point, Dim, Float32] =
      def cloud(center: Array[Float], key: Random.Key): Tensor2[Point, Dim, Float32] =
        val noise = Normal.standardNormal(Shape(Axis[Point] -> n, Axis[Dim] -> center.length)).sample(key)
        Tensor1(Axis[Dim]).fromArray(center) +! noise

      val (key0, key1, key2) = dataKey.splitToTuple(3)
      concatenate(
        List(
          cloud(trueCenters(0), key0),
          cloud(trueCenters(1), key1),
          cloud(trueCenters(2), key2)
        ),
        Axis[Point]
      )

    dimwit.initialize()

    val rootKey = Random.Key(1337)
    val (dataKey, trainKey) = rootKey.split2()

    val points = prepareData(dataKey)

    println(s"Dataset: ${n * k} points, 2 dimensions, $k true clusters\n")
    val (kmeansKey, kmediansKey) = trainKey.split2()
    evaluate(new KMeans(k), "K-Means", points, kmeansKey)
    evaluate(new KMedians(k), "K-Median", points, kmediansKey)
