package dimwit.tensor

import dimwit.tensor.TupleHelpers.{RemoverAll, Replacer}
import scala.compiletime.*
import scala.quoted.*
import dimwit.tensor.TupleHelpers.PrimeConcat

trait Label[T]:
  def name: String

private trait LabelLowPriority:
  given union[A, B](using a: Label[A], b: Label[B]): Label[A | B] with
    def name: String = a.name + "|" + b.name

object Label extends LabelLowPriority:
  inline def derived[T]: Label[T] = ${ derivedMacro[T] }

  private def derivedMacro[T: Type](using Quotes): Expr[Label[T]] =
    import quotes.reflect.*
    val tpe = TypeRepr.of[T]
    val simpleName = tpe.typeSymbol.name
    '{
      new Label[T]:
        def name: String = ${ Expr(simpleName) }
    }

  given [T](using valueOf: ValueOf[T]): Label[T] with
    def name: String = valueOf.value.toString

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

  given derivedReplacer[T <: Tuple, ToReplace, OutAxis, O <: Tuple](using
      replacer: Replacer[T, ToReplace, OutAxis] { type Out = O },
      labels: Labels[T],
      toReplaceLabels: Label[ToReplace],
      outAxisValue: Label[OutAxis]
  ): Labels[O] =
    val toReplaceNames = List(toReplaceLabels.name)
    LabelsImpl[O](
      labels.names.map { name =>
        if toReplaceNames.contains(name) then outAxisValue.name else name
      }
    )

  object ForConcat:

    given [T1 <: Tuple, T2 <: Tuple](using
        n1: Labels[T1],
        n2: Labels[T2]
    ): Labels[Tuple.Concat[T1, T2]] = new LabelsImpl(n1.names ++ n2.names)

  object ForPrimeConcat:
    given derivePrimeConcat[T1 <: Tuple, T2 <: Tuple](using
        primeConcat: PrimeConcat[T1, T2],
        n1: Labels[T1],
        n2: Labels[T2]
    ): Labels[primeConcat.Out] =
      val n1Names = n1.names.toSet
      val newN2Names = n2.names.map { name =>
        if n1Names.contains(name) then s"${name}'" else name
      }
      new LabelsImpl(n1.names ++ newN2Names)
