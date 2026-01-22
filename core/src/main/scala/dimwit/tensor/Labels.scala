package dimwit.tensor

import scala.compiletime.*
import scala.quoted.*
import Tuple.:*

@scala.annotation.implicitNotFound("""
An axis label ${T} was given or inferred, which does not have a Label instance.
Ensure that all axis types ${T} are defined with 'derives Label' (e.g. 'trait T derives Label')
""")
trait Label[T]:
  def name: String

object Label:
  inline def derived[T]: Label[T] = ${ derivedMacro[T] }

  private def derivedMacro[T: Type](using Quotes): Expr[Label[T]] =
    import quotes.reflect.*
    val tpe = TypeRepr.of[T]
    val simpleName = tpe.typeSymbol.name
    '{
      new Label[T]:
        def name: String = ${ Expr(simpleName) }
    }

@scala.annotation.implicitNotFound("""
A tuple of axis labels ${T} was given or inferred that does not have a valid Labels instance. 

Ensure that all of the types in the tuple have a 'derives Label' clause.
""")
trait Labels[T]:
  def names: List[String]

private class LabelsImpl[T](val names: List[String]) extends Labels[T]

private trait LabelsLowPriority

object Labels extends LabelsLowPriority:

  given namesOfEmpty: Labels[EmptyTuple] = new LabelsImpl[EmptyTuple](Nil)

  given lift[A](using v: Label[A]): Labels[A] = new LabelsImpl[A](List(v.name))

  given [A, B](using a: Labels[A], b: Labels[B]): Labels[(A, B)] = new LabelsImpl[(A, B)](a.names ++ b.names)
  given [A, B, C](using a: Labels[A], b: Labels[B], c: Labels[C]): Labels[(A, B, C)] = new LabelsImpl[(A, B, C)](a.names ++ b.names ++ c.names)
  given [A, B, C, D](using a: Labels[A], b: Labels[B], c: Labels[C], d: Labels[D]): Labels[(A, B, C, D)] = new LabelsImpl[(A, B, C, D)](a.names ++ b.names ++ c.names ++ d.names)
  given [A, B, C, D, E](using a: Labels[A], b: Labels[B], c: Labels[C], d: Labels[D], e: Labels[E]): Labels[(A, B, C, D, E)] = new LabelsImpl[(A, B, C, D, E)](a.names ++ b.names ++ c.names ++ d.names ++ e.names)
  given [A, B, C, D, E, F](using a: Labels[A], b: Labels[B], c: Labels[C], d: Labels[D], e: Labels[E], f: Labels[F]): Labels[(A, B, C, D, E, F)] = new LabelsImpl[(A, B, C, D, E, F)](a.names ++ b.names ++ c.names ++ d.names ++ e.names ++ f.names)

  given concat[head, tail <: Tuple](using
      v: Label[head],
      t: Labels[tail]
  ): Labels[head *: tail] = new LabelsImpl[head *: tail](
    v.name :: t.names
  )

  given append[head, tail <: Tuple](using
      v: Label[head],
      t: Labels[tail]
  ): Labels[tail :* head] = new LabelsImpl[tail :* head](
    t.names :+ v.name
  )

  object ForConcat:
    given [T1 <: Tuple, T2 <: Tuple](using
        n1: Labels[T1],
        n2: Labels[T2]
    ): Labels[Tuple.Concat[T1, T2]] = new LabelsImpl(n1.names ++ n2.names)
