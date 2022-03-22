# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from marks import allow_non_gpu
from spark_session import with_cpu_session, with_gpu_session


@allow_non_gpu('FileSourceScanExec')
def test_coalesce_perf():
    from pyspark.sql.functions import col, collect_list, size
    from pyspark.sql import SparkSession

    def gen_data(spark, data_path, n_part=10, rows_per_part=100000):
        from pyspark.sql.types import IntegerType, StructField, StructType

        rdd = spark.sparkContext.parallelize(list(range(n_part)), numSlices=n_part)

        def rand_gen(seed_iter):
            from random import Random
            from pyspark.sql import Row
            rd = Random(next(seed_iter))
            for _ in range(rows_per_part):
                rdVal = rd.randint(0, 10)
                if rdVal < 6:
                    yield Row(a=100)
                elif rdVal < 9:
                    yield Row(a=200)
                else:
                    yield Row(a=300)

        rows = rdd.mapPartitions(rand_gen)
        df = spark.createDataFrame(rows, StructType([StructField('a', IntegerType(), True)]))
        df.write.parquet(data_path)

    path = 'PARQUET_DATA_1234'
    # with_cpu_session(lambda spark: gen_data(spark, path, n_part=1000, rows_per_part=1000000))

    def fn(spark: SparkSession):
        return spark.read.parquet(path) \
            .groupby("a") \
            .agg(collect_list(col('a')).alias('cc')) \
            .selectExpr("a", "size(cc)")

    with_gpu_session(
        lambda spark: fn(spark).collect(),
        conf={'spark.rapids.shuffle.enabled': 'false',
              'spark.rapids.sql.useAsyncShuffleCoalesce': 'true'})
