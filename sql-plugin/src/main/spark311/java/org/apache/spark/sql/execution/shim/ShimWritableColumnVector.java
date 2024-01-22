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
{"spark": "311"}
{"spark": "312"}
{"spark": "313"}
{"spark": "320"}
{"spark": "321"}
{"spark": "321cdh"}
{"spark": "321db"}
{"spark": "322"}
{"spark": "323"}
{"spark": "324"}
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

	public abstract boolean isValid(int rowId);

	@Override
	public boolean isNullAt(int rowId) {
		return !isValid(rowId);
	}

}
