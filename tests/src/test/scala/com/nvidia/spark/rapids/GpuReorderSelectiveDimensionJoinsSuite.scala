/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nvidia.spark.rapids

import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, AttributeMap}
import org.apache.spark.sql.catalyst.expressions.{EqualTo, Expression, Literal, Or, PredicateHelper}
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.catalyst.optimizer.CollapseProject
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{BROADCAST, ColumnStat, Filter, HintInfo, Join}
import org.apache.spark.sql.catalyst.plans.logical.{JoinHint, LeafNode, LocalRelation, LogicalPlan}
import org.apache.spark.sql.catalyst.plans.logical.{Project, Statistics}
import org.apache.spark.sql.catalyst.rules.RuleExecutor
import org.apache.spark.sql.types.{LongType, StringType}

class GpuReorderSelectiveDimensionJoinsSuite
    extends AnyFunSuite with BeforeAndAfterAll with PredicateHelper {
  private var spark: SparkSession = _
  private val enabled = GpuReorderSelectiveDimensionJoins.enabledKey
  private val legacy = "spark.rapids.sql.optimizer.pushDimensionChainBeforeFact.enabled"

  override def beforeAll(): Unit = {
    super.beforeAll()
    spark = SparkSession.builder().master("local[1]").appName(getClass.getSimpleName)
      .config("spark.ui.enabled", "false")
      .config("spark.sql.cbo.enabled", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "1g")
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.shuffle.partitions", "2")
      .config(enabled, "true")
      .withExtensions(_.injectPostHocResolutionRule(
        GpuReorderSelectiveDimensionJoins.registrar)).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    SparkSession.setActiveSession(spark)
  }

  override def afterAll(): Unit = {
    try {
      if (spark != null) spark.stop()
      SparkSession.clearActiveSession()
      SparkSession.clearDefaultSession()
    } finally {
      super.afterAll()
    }
  }

  private case class Fixture(plan: LogicalPlan, leaves: Seq[ReorderStatLeaf],
      predicates: Seq[Expression], leftSet: Set[String], rightSet: Set[String])

  private def leaf(name: String, count: Long, ndvs: Long*): ReorderStatLeaf = {
    val attrs = ndvs.indices.map(i => AttributeReference(s"${name}_$i", LongType)())
    val stats = AttributeMap(attrs.zip(ndvs).map { case (a, ndv) =>
      a -> ColumnStat(distinctCount = Some(BigInt(ndv)), nullCount = Some(BigInt(0)),
        min = Some(1L), max = Some(ndv), avgLen = Some(8L), maxLen = Some(8L))
    })
    ReorderStatLeaf(name, attrs, Statistics(BigInt(count) * (8 + 8 * attrs.size),
      Some(BigInt(count)), stats))
  }

  private def q5(): Fixture = {
    val a = leaf("a", 230000000, 230000000, 150000000)
    val b = leaf("b", 6000000000L, 1500000000, 10000000)
    val c = leaf("c", 150000000, 150000000, 25)
    val d = leaf("d", 10000000, 10000000, 25)
    val e = leaf("e", 25, 25, 5)
    val f = leaf("f", 5, 5, 5)
    val ls = Seq(a, b, c, d, e, f)
    val ps = Seq(EqualTo(a.output(0), b.output(0)), EqualTo(a.output(1), c.output(0)),
      EqualTo(b.output(1), d.output(0)), EqualTo(c.output(1), d.output(1)),
      EqualTo(d.output(1), e.output(0)), EqualTo(e.output(1), f.output(0)))
    val ef = join(e, Filter(EqualTo(f.output(1), Literal(1L)), f), ps)
    val tree = join(join(join(join(a, b, ps), d, ps), ef, ps), c, ps)
    Fixture(Project(Seq(a.output(0), c.output(1)), tree), ls, ps,
      Set("a", "c"), Set("b", "d", "e", "f"))
  }

  private def q7(): Fixture = {
    val a = leaf("a", 1500000000, 1500000000, 150000000)
    val b = leaf("b", 1800000000, 500000000, 10000000)
    val c = leaf("c", 150000000, 150000000, 25)
    val d = leaf("d", 10000000, 10000000, 25)
    val e = leaf("e", 25, 25, 25)
    val f = leaf("f", 25, 25, 25)
    val pair = Or(And(EqualTo(e.output(1), Literal(1L)), EqualTo(f.output(1), Literal(2L))),
      And(EqualTo(e.output(1), Literal(2L)), EqualTo(f.output(1), Literal(1L))))
    val ps = Seq(EqualTo(a.output(0), b.output(0)), EqualTo(a.output(1), c.output(0)),
      EqualTo(b.output(1), d.output(0)), EqualTo(c.output(1), f.output(0)),
      EqualTo(d.output(1), e.output(0)), pair)
    def filtered(p: LogicalPlan): LogicalPlan = Filter(Or(
      EqualTo(p.output(1), Literal(1L)), EqualTo(p.output(1), Literal(2L))), p)
    val small = join(join(filtered(e), filtered(f), ps), c, ps)
    val tree = join(join(join(b, d, ps), a, ps), small, ps)
    Fixture(Project(Seq(a.output(0), e.output(1), f.output(1)), tree),
      Seq(a, b, c, d, e, f), ps, Set("a", "c", "f"), Set("b", "d", "e"))
  }

  private def join(l: LogicalPlan, r: LogicalPlan, ps: Seq[Expression]): LogicalPlan = {
    val attrs = l.outputSet ++ r.outputSet
    val here = ps.filter(p => p.references.subsetOf(attrs) &&
      !p.references.subsetOf(l.outputSet) && !p.references.subsetOf(r.outputSet))
    Join(l, r, Inner, here.reduceOption(And), JoinHint.NONE)
  }

  private def names(p: LogicalPlan): Set[String] =
    p.collect { case l: ReorderStatLeaf => l.label }.toSet

  private def conditions(p: LogicalPlan): Seq[Expression] = p.collect {
    case j: Join => j.condition.toSeq.flatMap(splitConjunctivePredicates)
  }.flatten

  for ((label, make) <- Seq("single selective chain" -> (() => q5()),
      "two selective branches" -> (() => q7()))) {
    test(s"$label preserves predicates, prepares branches and converges") {
      val f = make()
      val rule = GpuReorderSelectiveDimensionJoins(spark)
      val result = rule(f.plan)
      assert(!result.fastEquals(f.plan), result.treeString)
      assert(result.output == f.plan.output)
      assert(result.resolved && result.missingInput.isEmpty)
      assert(result.exists {
        case j: Join => Set(names(j.left), names(j.right)) == Set(f.leftSet, f.rightSet)
        case _ => false
      }, result.treeString)
      assert(conditions(result).size == f.predicates.size)
      assert(f.predicates.forall(p => conditions(result).count(_.semanticEquals(p)) == 1))
      assert(rule(result).fastEquals(result))
    }
  }

  test("missing key NDVs abstain") {
    val f = q7()
    val noStats = f.plan.transformDown {
      case l: ReorderStatLeaf => l.copy(statistics = l.statistics.copy(attributeStats =
        AttributeMap(l.output.map(a => a -> ColumnStat(nullCount = Some(BigInt(0)))))))
    }
    assert(GpuReorderSelectiveDimensionJoins(spark)(noStats).fastEquals(noStats))
  }

  test("a three-relation graph moves the filtered lookup before the large join") {
    val events = leaf("events", 400000000, 400000000, 1000)
    val details = leaf("details", 400000000, 400000000)
    val lookup = leaf("lookup", 1000, 1000, 100)
    val ps = Seq(EqualTo(events.output(0), details.output(0)),
      EqualTo(events.output(1), lookup.output(0)))
    val selected = Filter(EqualTo(lookup.output(1), Literal(1L)), lookup)
    val original = Project(Seq(events.output(0)), join(join(events, details, ps), selected, ps))
    val result = GpuReorderSelectiveDimensionJoins(spark)(original)
    assert(!result.fastEquals(original))
    assert(result.exists {
      case j: Join => names(j) == Set("events", "lookup")
      case _ => false
    }, result.treeString)
  }

  test("no selective filter and unknown row counts abstain") {
    val original = q5().plan
    val unfiltered = original.transformDown { case f: Filter => f.child }
    val unknown = original.transformDown {
      case l: ReorderStatLeaf => l.copy(statistics = l.statistics.copy(rowCount = None))
    }
    val rule = GpuReorderSelectiveDimensionJoins(spark)
    assert(rule(unfiltered).fastEquals(unfiltered))
    assert(rule(unknown).fastEquals(unknown))
  }

  test("missing variable-width statistics and AQE abstain") {
    val f = q5()
    val payload = AttributeReference("payload", StringType)()
    val unknown = f.plan.transformDown {
      case l: ReorderStatLeaf if l.label == "a" => l.copy(attrs = l.attrs :+ payload)
    }
    val rule = GpuReorderSelectiveDimensionJoins(spark)
    assert(rule(unknown).fastEquals(unknown))
    try {
      spark.conf.set("spark.sql.adaptive.enabled", "true")
      assert(rule(f.plan).fastEquals(f.plan))
    } finally {
      spark.conf.set("spark.sql.adaptive.enabled", "false")
    }
  }

  test("aliases retain output identity and the late batch converges with projection cleanup") {
    val f = q7()
    val original = Project(f.plan.output.map(a => Alias(a, "renamed_" + a.name)()), f.plan)
    val rule = GpuReorderSelectiveDimensionJoins(spark)
    val batch = new RuleExecutor[LogicalPlan] {
      override protected def batches: Seq[Batch] =
        Seq(Batch("late", FixedPoint(10), rule, CollapseProject))
    }
    val result = batch.execute(original)
    assert(result.output == original.output)
    assert(batch.execute(result).fastEquals(result))
    assert(result.exists {
      case j: Join => Set(names(j.left), names(j.right)) == Set(f.leftSet, f.rightSet)
      case _ => false
    }, result.treeString)
  }

  test("legacy reorder takes precedence and disabled new rule leaves the plan alone") {
    val plan = q5().plan
    val rule = GpuReorderSelectiveDimensionJoins(spark)
    try {
      spark.conf.set(legacy, "true")
      assert(rule(plan).fastEquals(plan))
      spark.conf.set(legacy, "false")
      spark.conf.set(enabled, "false")
      assert(rule(plan).fastEquals(plan))
    } finally {
      spark.conf.set(legacy, "false")
      spark.conf.set(enabled, "true")
    }
  }

  test("a hinted join is a boundary") {
    val f = q5()
    val hint = JoinHint(None, Some(HintInfo(strategy = Some(BROADCAST))))
    val hinted = f.plan.transformDown { case j: Join => j.copy(hint = hint) }
    assert(GpuReorderSelectiveDimensionJoins(spark)(hinted).fastEquals(hinted))
  }

  test("first analysis registers once in each session") {
    val session = spark.newSession()
    session.sql("SELECT 1").queryExecution.analyzed
    def count: Int = session.experimental.extraOptimizations.count(
      _.isInstanceOf[GpuReorderSelectiveDimensionJoins])
    assert(count == 1)
    session.sql("SELECT 2").queryExecution.analyzed
    assert(count == 1)
  }

  for ((label, make) <- Seq("composite equality" -> (() => q5()),
      "cross-branch residual" -> (() => q7()))) {
    test(s"CPU results preserve duplicates, nulls and $label") {
      val f = make()
      val rewritten = GpuReorderSelectiveDimensionJoins(spark)(f.plan)
      assert(!rewritten.fastEquals(f.plan))
      def executable(plan: LogicalPlan): LogicalPlan = plan.transformDown {
        case l: ReorderStatLeaf =>
          val data = Seq(InternalRow(1L, 1L), InternalRow(1L, 1L), InternalRow(1L, 2L),
            InternalRow(2L, 1L), InternalRow(2L, 2L), InternalRow(null, 1L),
            InternalRow(1L, null), InternalRow(3L, 3L))
          LocalRelation(l.output, data)
      }
      def collect(plan: LogicalPlan): Seq[String] =
        spark.sessionState.executePlan(executable(plan)).executedPlan.executeCollect()
          .map(_.toString).sorted.toSeq
      try {
        spark.conf.set(enabled, "false")
        val expected = collect(f.plan)
        assert(expected.nonEmpty)
        assert(collect(rewritten) == expected)
      } finally {
        spark.conf.set(enabled, "true")
      }
    }
  }
}

private[rapids] case class ReorderStatLeaf(label: String, attrs: Seq[Attribute],
    statistics: Statistics) extends LeafNode {
  override def output: Seq[Attribute] = attrs
  override def computeStats(): Statistics = statistics
}
