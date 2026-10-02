# Selective dimension join reordering

`GpuReorderSelectiveDimensionJoins` is an opt-in logical rule for preparing filtered dimension
branches before expensive inner joins. It uses existing Catalyst statistics and predicates; it
has no table-name, query-ID, sidecar-metadata, or transitive-equality matching.

```properties
spark.rapids.sql.optimizer.selectiveDimensionReorder.enabled=true
```

The default is `false`. The initial supported profile requires CBO enabled, AQE disabled, and a
fixed executor count (`spark.executor.instances`, or local execution). Dynamic allocation,
streaming, and the legacy `pushDimensionChainBeforeFact.enabled=true` configuration cause the
rule to abstain. Disabling the new flag restores the prior optimizer behavior without disabling
CBO or removing cached statistics.

## Transformation

For each existing equality edge in a bounded inner-join region, reserve its endpoints for two
separate branches. Greedily prepare connected subtrees before joining the endpoints, comparing
two deterministic priorities: intermediate bytes and accumulated transfer bytes. This can turn
`(A join B) join filtered_D` into `(A join filtered_D) join B`, or prepare filtered branches on
both sides. The relation count is bounded by the smaller of 12 and Spark's existing DP threshold.

Every original join predicate is consumed exactly once, at its first eligible node. Residual
predicates are preserved, including cross-branch OR conditions. Computations local to a scan stay
in that leaf; aggregates, unsupported operators, subqueries, and join hints form boundaries.
Output expression IDs and order are preserved. No join hints or new predicates are introduced.

## Admission

Inputs require row counts and usable join-key NDVs/null counts. Variable-width columns require
average lengths; complex payloads are outside the initial scope. At least one input must have an
estimated selective filter. Missing evidence keeps the original plan.

The comparison charges projected inputs and broadcast replication under the current Spark
threshold. A candidate must reduce estimated transfer without increasing the input-byte
processing proxy or maximum build payload. These are directional work estimates, not latency or
resident hash-table memory guarantees. Common final output is excluded because equivalent trees
can receive inconsistent output estimates from Spark. Existing backend memory limits still apply.

## Optimizer integration

An analysis hook registers the rule once per session in `experimental.extraOptimizations`, before
the optimizer captures that list. The rule itself does not mutate registration or recursively run
the optimizer. It runs after CBO and runtime-filter injection. It does not inject new Bloom filters
or assume subsequent column pruning; it constructs its own internal projections.

The legacy global dimension reorder takes precedence to avoid competing late reorders. The test
suite checks fixed points both directly and with projection cleanup. Arbitrary third-party late
rules are not qualified by these tests.

## Validation scope

`GpuReorderSelectiveDimensionJoinsSuite` covers one/two selective branches, a three-relation
non-TPC-H graph, predicate/output preservation, missing statistics, hints, aliases, registration,
convergence, and CPU multiset equivalence with duplicate and null keys.

The initial performance target is the retained Spark 3.5.3 SF1000 two-GPU Q5/Q7 profile, with
restored statistics and unchanged CBO/star-filter settings. Full upstream issue acceptance still
requires broader query and supported-version qualification; opt-in support does not imply a
default-on rollout.
