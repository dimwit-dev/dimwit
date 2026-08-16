package dimwit.tensor

import scala.quoted.*

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

object Labels extends LabelsLowPriority:

  given emptyTuple: Labels[EmptyTuple] = new LabelsImpl[EmptyTuple](Nil)

  given lift[A](using v: Label[A]): Labels[A] = new LabelsImpl[A](List(v.name))

  given consTuple[H, T <: Tuple](using
      head: Label[H],
      tail: Labels[T]
  ): Labels[H *: T] = new LabelsImpl[H *: T](head.name :: tail.names)

private trait LabelsLowPriority:
  given concatTuple[T1 <: Tuple, T2 <: Tuple](using n1: Labels[T1], n2: Labels[T2]): Labels[Tuple.Concat[T1, T2]] =
    new LabelsImpl(n1.names ++ n2.names)
