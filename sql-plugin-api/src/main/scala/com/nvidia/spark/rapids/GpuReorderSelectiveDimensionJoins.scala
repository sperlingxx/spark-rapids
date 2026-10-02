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

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, EqualTo, Expression}
import org.apache.spark.sql.catalyst.expressions.{PredicateHelper, SubqueryExpression}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.catalyst.plans.logical.{Filter, Join, JoinHint, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.{ArrayType, BinaryType, MapType, StringType, StructType}

object GpuReorderSelectiveDimensionJoins {
  val enabledKey = "spark.rapids.sql.optimizer.selectiveDimensionReorder.enabled"
  private val legacyKey = "spark.rapids.sql.optimizer.pushDimensionChainBeforeFact.enabled"

  /** Register during analysis, before the optimizer snapshots extraOptimizations. */
  def registrar(spark: SparkSession): Rule[LogicalPlan] = new Rule[LogicalPlan] {
    override def apply(plan: LogicalPlan): LogicalPlan = {
      val experimental = spark.experimental
      experimental.synchronized {
        if (!experimental.extraOptimizations.exists(
            _.isInstanceOf[GpuReorderSelectiveDimensionJoins])) {
          experimental.extraOptimizations = experimental.extraOptimizations :+
            GpuReorderSelectiveDimensionJoins(spark)
        }
      }
      plan
    }
  }
}

/**
 * Prepare selective dimension branches before a large inner join, using only Catalyst statistics.
 *
 * For each anchor equality, keep its endpoints in separate components while greedily joining
 * the other connected inputs. This exposes both single-chain and two-branch reductions without
 * enumerating join permutations or inferring new equalities. Original residuals are retained.
 * Admission compares projected join inputs, not inconsistent estimates of the common output.
 */
case class GpuReorderSelectiveDimensionJoins(spark: SparkSession)
    extends Rule[LogicalPlan] with PredicateHelper with Logging {
  import GpuReorderSelectiveDimensionJoins._

  // Match the existing prototype's bounded region size; never increase Catalyst's search bound.
  private val maxRegionSize = 12
  // BroadcastHashJoin's general hashed-relation row limit (also used by the retained backend).
  private val maxBroadcastRows = BigInt(512000000)

  private case class Work(transfer: BigInt, processing: BigInt, build: BigInt) {
    def +(other: Work): Work = Work(
      transfer + other.transfer, processing + other.processing, build.max(other.build))
    def improves(other: Work): Boolean =
      transfer < other.transfer && processing <= other.processing && build <= other.build
  }
  private val zeroWork = Work(0, 0, 0)
  private case class Component(plan: LogicalPlan, ids: Set[Int], used: Set[Int], work: Work)
  private case class Region(leaves: Seq[LogicalPlan], predicates: Seq[Expression])

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = spark.sessionState.conf
    if (!conf.getConfString(enabledKey, "false").toBoolean || !conf.cboEnabled ||
        conf.adaptiveExecutionEnabled || !plan.resolved || plan.isStreaming) {
      plan
    } else if (conf.getConfString(legacyKey, "false").toBoolean) {
      logDebug("Selective dimension reorder: abstaining because the legacy reorder is enabled")
      plan
    } else {
      replicas.map(n => visit(plan, n)).getOrElse(plan)
    }
  }

  // Fixed executor allocation gives a conservative payload replication count. A local driver
  // receives one copy. Dynamic allocation has no fixed destination count, so abstain.
  private def replicas: Option[Int] = {
    val conf = spark.sparkContext.getConf
    if (conf.getBoolean("spark.dynamicAllocation.enabled", false)) {
      None
    } else if (spark.sparkContext.isLocal) {
      Some(1)
    } else {
      conf.getOption("spark.executor.instances").map(_.toInt).filter(_ > 0)
    }
  }

  private def visit(plan: LogicalPlan, copies: Int): LogicalPlan = {
    val (target, required, wrap) = plan match {
      case p: Project if p.projectList.forall(_.deterministic) =>
        (p.child, p.references, (child: LogicalPlan) => p.copy(child = child))
      case _ => (plan, plan.outputSet, (child: LogicalPlan) => child)
    }
    extractRegion(target) match {
      case Some(region) if region.leaves.size >= 3 =>
        // A maximal supported region has one owner. Do not optimize overlapping child regions.
        rewrite(target, required, region, copies).map(wrap).getOrElse(plan)
      case _ => plan.mapChildren(child => visit(child, copies))
    }
  }

  /**
   * A Region represents a group of inputs and join predicates that this rule can consider
   * reordering together. It contains the input subplans (leaves) and the original join
   * predicates, without encoding the current join order. A leaf can include a scan's filters
   * and projections; candidate construction treats that entire subplan as one input.
   *
   * For example, suppose A joins B on A.b_id = B.id, and B joins C on B.c_id = C.id:
   * {{{
   *   Original tree                 Possible candidate built later
   *
   *      Join B-C                         Join A-B
   *      /      \                         /      \
   *   Join A-B  Filter C                 A      Join B-C
   *   /     \      |                            /     \
   *  A       B     C                           B    Filter C
   *                                                    |
   *                                                    C
   *
   *   Extracted Region:
   *     inputs:     A, B, Filter(C)
   *     predicates: A.b_id = B.id, B.c_id = C.id
   * }}}
   * The Region lets buildCandidate try filtering B through C before joining it to A.
   * The filter on C stays attached to C, and both join conditions must still be applied.
   *
   * This method extracts that representation from the subtree rooted at root. It determines
   * whether the subtree has a supported shape; it does not change the plan or decide whether
   * reordering will help. Subsequent evidence and cost checks make that decision.
   *
   * Starting at root, follow the inner joins and collect their conditions. A step between joins
   * that only selects existing columns can be passed through because the rule can select those
   * columns again when building a candidate. Keep a table read and its filters or calculated
   * columns together as one input; do not reorder operations inside that input.
   *
   * Return None if this part of the plan contains something the rule cannot safely rearrange,
   * such as an outer join, an explicit join hint, an aggregation, or an expression such as rand()
   * that can change on each evaluation. The caller may then look for smaller groups of joins
   * below that operation.
   *
   * Require at least three inputs and cap their count at the smaller of maxRegionSize and
   * spark.sql.cbo.joinReorder.dp.threshold. That Spark setting limits how many inputs its
   * dynamic programming (DP) search considers when comparing alternative join orders. This rule
   * reuses the input-count limit to bound its own search; it does not use Spark's DP algorithm.
   *
   * Each output column must belong to exactly one input, using Spark's internal column IDs to
   * distinguish columns even when their names are equal. Sort inputs by those IDs so candidate
   * construction starts in a repeatable order.
   */
  private def extractRegion(root: LogicalPlan): Option[Region] = {
    val leaves = ArrayBuffer.empty[LogicalPlan]
    val predicates = ArrayBuffer.empty[Expression]
    def collect(plan: LogicalPlan): Boolean = plan match {
      case Project(columns, child) if columns.forall(_.isInstanceOf[Attribute]) &&
          child.exists(_.isInstanceOf[Join]) => collect(child)
      case Join(left, right, Inner, Some(condition), JoinHint.NONE)
          if condition.deterministic && !SubqueryExpression.hasSubquery(condition) =>
        predicates ++= splitConjunctivePredicates(condition)
        collect(left) && collect(right)
      case other =>
        // Keep scan-local computations/filters opaque. Other operator boundaries are not leaves
        // of this rewrite; visit() may independently consider a supported region beneath them.
        def scanBranch(p: LogicalPlan): Boolean = p match {
          case p: Project => p.projectList.forall(_.deterministic) && scanBranch(p.child)
          case f: Filter => f.condition.deterministic &&
            !SubqueryExpression.hasSubquery(f.condition) && scanBranch(f.child)
          case leaf => leaf.children.isEmpty && !leaf.isStreaming
        }
        if (scanBranch(other)) {
          leaves += other
          true
        } else {
          false
        }
    }
    val limit = math.min(maxRegionSize, spark.sessionState.conf.joinReorderDPThreshold)
    if (collect(root) && leaves.size <= limit && leaves.size >= 3 &&
        leaves.map(_.outputSet).combinations(2).forall(p => p.head.intersect(p.last).isEmpty)) {
      Some(Region(leaves.toSeq.sortBy(_.output.map(_.exprId.id).min), predicates.toSeq))
    } else {
      None
    }
  }

  private def rows(plan: LogicalPlan): Option[BigInt] = plan.stats.rowCount.filter(_ > 0)

  private def hasEvidence(region: Region): Boolean = {
    val keys = region.predicates.flatMap {
      case EqualTo(l: Attribute, r: Attribute) => Seq(l, r)
      case _ => Nil
    }
    region.leaves.forall { leaf =>
      rows(leaf).isDefined && leaf.stats.sizeInBytes > 0 &&
        leaf.output.forall { attr =>
          attr.dataType match {
            case StringType | BinaryType =>
              leaf.stats.attributeStats.get(attr).exists(_.avgLen.exists(_ >= 0))
            case _: ArrayType | _: MapType | _: StructType => false
            case _ => true
          }
        } &&
        keys.filter(leaf.outputSet.contains).forall { key =>
          leaf.stats.attributeStats.get(key).exists { stat =>
            stat.distinctCount.exists(_ > 0) && stat.nullCount.isDefined
          }
        }
    } && region.leaves.exists(_.exists {
      case f: Filter => (rows(f), rows(f.child)) match {
        case (Some(after), Some(before)) => after < before
        case _ => false
      }
      case _ => false
    })
  }

  private def equiConnects(left: LogicalPlan, right: LogicalPlan,
      predicates: Seq[Expression]): Boolean = predicates.exists {
    case EqualTo(l: Attribute, r: Attribute) =>
      (left.outputSet.contains(l) && right.outputSet.contains(r)) ||
        (left.outputSet.contains(r) && right.outputSet.contains(l))
    case _ => false
  }

  /**
   * Estimate one join's work from its projected inputs, using the same model for the original
   * tree and every candidate. Work records transfer bytes, processed input bytes and the largest
   * build payload; these are comparison proxies, not latency or resident hash-table memory.
   *
   * If the smaller input meets Spark's broadcast byte and row limits, charge one build copy per
   * destination and include the replicated build in processing. Otherwise, an equi-join charges
   * both inputs for shuffle and processing. Bound its build by the larger input because GPU
   * planning need not retain the CPU planner's build orientation. Do not charge join output:
   * equivalent trees can receive inconsistent output estimates from Spark.
   *
   * @param copies fixed number of broadcast destinations, or one for local execution
   * @return None for missing positive row counts or a non-broadcast theta join
   */
  private def joinWork(left: LogicalPlan, right: LogicalPlan, equi: Boolean,
      copies: Int): Option[Work] = {
    var work: Option[Work] = None
    rows(left).foreach { lrows =>
      rows(right).foreach { rrows =>
        val lbytes = left.stats.sizeInBytes
        val rbytes = right.stats.sizeInBytes
        val threshold = spark.sessionState.conf.autoBroadcastJoinThreshold
        val small = if (lbytes <= rbytes) (lbytes, lrows) else (rbytes, rrows)
        if (small._1 <= threshold && small._2 < maxBroadcastRows) {
          work = Some(Work(
            small._1 * copies, lbytes + rbytes + small._1 * (copies - 1), small._1))
        } else if (equi) {
          work = Some(Work(lbytes + rbytes, lbytes + rbytes, lbytes.max(rbytes)))
        }
      }
    }
    work
  }

  /**
   * Cost the existing tree within the extracted region. Stop at its opaque leaves: their
   * scan-local work is common to all candidates. Traverse projects without charging them
   * separately, but pass the actual projected join inputs to joinWork.
   *
   * Sum transfer and processing across joins and retain the maximum build payload through
   * Work.+. If either subtree or the current join cannot be costed, return None so rewrite()
   * keeps the original plan rather than comparing a candidate against a partial baseline.
   */
  private def baselineWork(plan: LogicalPlan, region: Region, copies: Int): Option[Work] = {
    if (region.leaves.exists(_.fastEquals(plan))) {
      Some(zeroWork)
    } else {
      plan match {
        case p: Project => baselineWork(p.child, region, copies)
        case Join(left, right, Inner, Some(condition), JoinHint.NONE) =>
          var work: Option[Work] = None
          baselineWork(left, region, copies).foreach { leftWork =>
            baselineWork(right, region, copies).foreach { rightWork =>
              joinWork(left, right,
                equiConnects(left, right, splitConjunctivePredicates(condition)), copies)
                .foreach { here =>
                  work = Some(leftWork + rightWork + here)
                }
            }
          }
          work
        case _ => None
      }
    }
  }

  private def combine(left: Component, right: Component, region: Region,
      required: AttributeSet, copies: Int): Option[Component] = {
    val available = left.plan.outputSet ++ right.plan.outputSet
    val used = left.used ++ right.used
    val here = region.predicates.indices.filter { i =>
      !used.contains(i) && region.predicates(i).references.subsetOf(available)
    }
    val predicates = here.map(region.predicates)
    if (!equiConnects(left.plan, right.plan, predicates)) {
      None
    } else {
      joinWork(left.plan, right.plan, equi = true, copies).map { work =>
        val consumed = used ++ here
        val remaining = region.predicates.indices.filterNot(consumed.contains)
        val needed = required ++ AttributeSet(remaining.flatMap(region.predicates(_).references))
        val join = Join(left.plan, right.plan, Inner, predicates.reduceOption(
          org.apache.spark.sql.catalyst.expressions.And), JoinHint.NONE)
        val projected = Project(join.output.filter(needed.contains), join)
        Component(projected, left.ids ++ right.ids, consumed, left.work + right.work + work)
      }
    }
  }

  /**
   * Build one candidate by greedily merging connected components. The anchor identifies two
   * leaves joined by an existing equality; keep them in separate components until the final
   * merge so filtering branches can be prepared on either side of that join.
   *
   * Each merge consumes newly available predicates and retains columns needed by the caller
   * or unapplied predicates (see combine). Reject dimension-only merges whose estimated output
   * exceeds both inputs, but allow anchor attachments and the final merge to be costed fully.
   * Exclude merges with missing statistics; abandon the candidate if no legal merge remains.
   *
   * @param required attributes needed above the region
   * @param anchor leaf indices that must remain separated until the final join
   * @param prioritizeWork choose by accumulated transfer when true, otherwise by output bytes;
   *                       output bytes and component indices break ties deterministically
   * @return a complete component only if every original predicate has been consumed
   */
  private def buildCandidate(region: Region, required: AttributeSet,
      anchor: (Int, Int), prioritizeWork: Boolean, copies: Int): Option[Component] = {
    var components = region.leaves.zipWithIndex.map { case (leaf, i) =>
      Component(leaf, Set(i), Set.empty, zeroWork)
    }
    while (components.size > 1) {
      val choices = ArrayBuffer.empty[(Int, Int, Component)]
      components.indices.foreach { i =>
        ((i + 1) until components.size).foreach { j =>
          val ids = components(i).ids ++ components(j).ids
          val finalJoin = components.size == 2
          if (finalJoin || !(ids.contains(anchor._1) && ids.contains(anchor._2))) {
            combine(components(i), components(j), region, required, copies).foreach { merged =>
              rows(merged.plan).foreach { out =>
                // Avoid large fanout between dimension sets. Anchor attachments may have
                // slight estimator inflation (e.g. a filtered FK join), so cost them fully.
                if (finalJoin || ids.contains(anchor._1) || ids.contains(anchor._2) ||
                    out <= rows(components(i).plan).get.max(rows(components(j).plan).get)) {
                  choices += ((i, j, merged))
                }
              }
            }
          }
        }
      }
      if (choices.isEmpty) {
        return None
      }
      val (i, j, merged) = choices.minBy { case (l, r, c) =>
        val first = if (prioritizeWork) c.work.transfer else c.plan.stats.sizeInBytes
        (first, c.plan.stats.sizeInBytes, l, r)
      }
      components = components.zipWithIndex.filterNot { case (_, k) => k == i || k == j }
        .map(_._1) :+ merged
      components = components.sortBy(_.ids.min)
    }
    components.headOption.filter(_.used.size == region.predicates.size)
  }

  private def rewrite(original: LogicalPlan, required: AttributeSet,
      region: Region, copies: Int): Option[LogicalPlan] = {
    if (!hasEvidence(region)) {
      None
    } else {
      baselineWork(original, region, copies).flatMap { before =>
        val anchors = ArrayBuffer.empty[(Int, Int)]
        region.leaves.indices.foreach { i =>
          ((i + 1) until region.leaves.size).foreach { j =>
            if (equiConnects(region.leaves(i), region.leaves(j), region.predicates)) {
              anchors += ((i, j))
            }
          }
        }
        val candidates = ArrayBuffer.empty[Component]
        anchors.foreach { anchor =>
          Seq(false, true).foreach { priority =>
            buildCandidate(region, required, anchor, priority, copies).foreach { candidate =>
              if (candidate.work.improves(before)) {
                candidates += candidate
              }
            }
          }
        }
        candidates.sortBy(c => (c.work.transfer, c.work.processing, c.work.build))
          .headOption.flatMap { best =>
            val output = original.output.filter(required.contains)
            val result = best.plan match {
              case p: Project => p.copy(projectList = output)
              case other => Project(output, other)
            }
            if (result.resolved && result.missingInput.isEmpty) {
              logDebug(s"Selective dimension reorder: ${region.leaves.size} inputs, " +
                s"before=$before after=${best.work} copies=$copies")
              Some(result)
            } else {
              None
            }
          }
      }
    }
  }
}
