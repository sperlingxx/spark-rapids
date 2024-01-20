/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "330"}
{"spark": "330cdh"}
{"spark": "330db"}
{"spark": "331"}
{"spark": "332"}
{"spark": "332cdh"}
{"spark": "332db"}
{"spark": "333"}
{"spark": "334"}
{"spark": "340"}
{"spark": "341"}
{"spark": "341db"}
{"spark": "342"}
{"spark": "350"}
{"spark": "351"}
spark-rapids-shim-json-lines ***/
package org.apache.spark.sql.execution.shim;

import org.apache.spark.sql.execution.vectorized.WritableColumnVector;
import org.apache.spark.sql.types.DataType;

import java.nio.ByteBuffer;

public abstract class ShimWritableColumnVector extends WritableColumnVector {

	protected ShimWritableColumnVector(int capacity, DataType dataType) {
		super(capacity, dataType);
	}

	public abstract void putBitMask(int rowId, byte src);

	public abstract ByteBuffer byteBuffer(int rowId, int count);

	@Override
	public void putBooleans(int rowId, byte src) {
		putBitMask(rowId, src);
	}

	@Override
	public ByteBuffer getByteBuffer(int rowId, int count) {
		return byteBuffer(rowId, count);
	}

}
