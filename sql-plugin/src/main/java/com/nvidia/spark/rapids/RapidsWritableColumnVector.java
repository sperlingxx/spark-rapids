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

package com.nvidia.spark.rapids;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;

import org.apache.spark.sql.execution.shim.ShimWritableColumnVector;
import org.apache.spark.sql.execution.vectorized.WritableColumnVector;
import org.apache.spark.sql.types.*;
import org.apache.spark.unsafe.types.UTF8String;

public class RapidsWritableColumnVector extends ShimWritableColumnVector {

	private HostMemoryBuffer data;
	private HostMemoryBuffer valid;
	private HostMemoryBuffer offsets;

	public RapidsWritableColumnVector(int capacity, DataType type) {
		super(capacity, type);
		reserveInternal(capacity);
	}

	public HostColumnVector build() {
		List<HostColumnVectorCore> children = new ArrayList<>();
		for (WritableColumnVector child : childColumns) {
			children.add(((RapidsWritableColumnVector) child).build());
		}

		return new HostColumnVector(
				GpuColumnVector.getRapidsType(type),
				elementsAppended, Optional.of((long) numNulls),
				data, valid, offsets, children);
	}

	@Override
	public void putBitMask(int rowId, byte src) {
		data.setByte(rowId, (byte)(src & 1));
		data.setByte(rowId + 1, (byte)(src >>> 1 & 1));
		data.setByte(rowId + 2, (byte)(src >>> 2 & 1));
		data.setByte(rowId + 3, (byte)(src >>> 3 & 1));
		data.setByte(rowId + 4, (byte)(src >>> 4 & 1));
		data.setByte(rowId + 5, (byte)(src >>> 5 & 1));
		data.setByte(rowId + 6, (byte)(src >>> 6 & 1));
		data.setByte(rowId + 7, (byte)(src >>> 7 & 1));
	}

	@Override
	public ByteBuffer byteBuffer(int rowId, int count) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public void putNotNull(int rowId) {
		valid.setByte(rowId, (byte) 1);
	}

	@Override
	public void putNull(int rowId) {
		valid.setByte(rowId, (byte) 0);
		++numNulls;
	}

	@Override
	public void putNulls(int rowId, int count) {
		valid.setMemory(rowId, count, (byte) 0);
		numNulls += count;
	}

	@Override
	public void putNotNulls(int rowId, int count) {
		if (!hasNull()) return;
		valid.setMemory(rowId, count, (byte) 1);
	}

	@Override
	public void putBoolean(int rowId, boolean value) {
		data.setBoolean(rowId, value);
	}

	@Override
	public void putBooleans(int rowId, int count, boolean value) {
		data.setMemory(rowId, count, value ? (byte) 1 : (byte) 0);
	}

	@Override
	public void putByte(int rowId, byte value) {
		data.setByte(rowId, value);
	}

	@Override
	public void putBytes(int rowId, int count, byte value) {
		data.setMemory(rowId, count, value);
	}

	@Override
	public void putBytes(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId, src, srcIndex, count);
	}

	@Override
	public void putShort(int rowId, short value) {
		data.setShort(rowId * 2L, value);
	}

	@Override
	public void putShorts(int rowId, int count, short value) {
		for (int offset = rowId; offset < rowId + count; offset++) {
			data.setShort(offset * 2L, value);
		}
	}

	@Override
	public void putShorts(int rowId, int count, short[] src, int srcIndex) {
		data.setShorts(rowId * 2L, src, srcIndex, count);
	}

	@Override
	public void putShorts(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 2L, src, srcIndex, count * 2L);
	}

	@Override
	public void putInt(int rowId, int value) {
		data.setInt(rowId * 4L, value);
	}

	@Override
	public void putInts(int rowId, int count, int value) {
		for (int offset = rowId; offset < rowId + count; offset++) {
			data.setInt(offset * 4L, value);
		}
	}

	@Override
	public void putInts(int rowId, int count, int[] src, int srcIndex) {
		data.setInts(rowId * 4L, src, srcIndex, count);
	}

	@Override
	public void putInts(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void putIntsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void putLong(int rowId, long value) {
		data.setLong(rowId * 8L, value);
	}

	@Override
	public void putLongs(int rowId, int count, long value) {
		for (int offset = rowId; offset < rowId + count; offset++) {
			data.setLong(offset * 8L, value);
		}
	}

	@Override
	public void putLongs(int rowId, int count, long[] src, int srcIndex) {
		data.setLongs(rowId * 8L, src, srcIndex, count);
	}

	@Override
	public void putLongs(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 8L, src, srcIndex, count * 8L);
	}

	@Override
	public void putLongsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 8L, src, srcIndex, count * 8L);
	}

	@Override
	public void putFloat(int rowId, float value) {
		data.setFloat(rowId * 4L, value);
	}

	@Override
	public void putFloats(int rowId, int count, float value) {
		for (int offset = rowId; offset < rowId + count; offset++) {
			data.setFloat(offset * 4L, value);
		}
	}

	@Override
	public void putFloats(int rowId, int count, float[] src, int srcIndex) {
		data.setFloats(rowId * 4L, src, srcIndex, count);
	}

	@Override
	public void putFloats(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void putFloatsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 4L, src, srcIndex, count * 4L);
	}

	@Override
	public void putDouble(int rowId, double value) {
		data.setDouble(rowId * 8L, value);
	}

	@Override
	public void putDoubles(int rowId, int count, double value) {
		for (int offset = rowId; offset < rowId + count; offset++) {
			data.setDouble(offset * 8L, value);
		}
	}

	@Override
	public void putDoubles(int rowId, int count, double[] src, int srcIndex) {
		data.setDoubles(rowId * 8L, src, srcIndex, count);
	}

	@Override
	public void putDoubles(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 8L, src, srcIndex, count * 8L);
	}

	@Override
	public void putDoublesLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		data.setBytes(rowId * 8L, src, srcIndex, count * 8L);
	}

	@Override
	public void putArray(int rowId, int offset, int length) {
		assert(offset >= 0 &&
				offset + length <= ((RapidsWritableColumnVector) childColumns[0]).capacity);

		offsets.setInt((rowId + 1) * 4L, offset + length);
	}

	@Override
	public int putByteArray(int rowId, byte[] value, int offset, int length) {
		int result = childColumns[0].appendBytes(length, value, offset);
		offsets.setInt((rowId + 1) * 4L, result + length);
		return result;
	}

	@Override
	protected void reserveInternal(int newCap) {
		int oldCapacity = capacity;
		if (isArray() || type instanceof MapType) {
			offsets = moveBuffer(HostMemoryBuffer.allocate((newCap + 1) * 4L), offsets);
			offsets.setInt(0, 0);
		} else if (type instanceof ByteType || type instanceof BooleanType) {
			data = moveBuffer(HostMemoryBuffer.allocate(newCap), data);
		} else if (type instanceof ShortType) {
			data = moveBuffer(HostMemoryBuffer.allocate(newCap * 2L), data);
		} else if (type instanceof IntegerType || type instanceof FloatType ||
				type instanceof DateType || DecimalType.is32BitDecimalType(type) ||
				type instanceof YearMonthIntervalType) {
			data = moveBuffer(HostMemoryBuffer.allocate(newCap * 4L), data);
		} else if (type instanceof LongType || type instanceof DoubleType ||
				DecimalType.is64BitDecimalType(type) || type instanceof TimestampType ||
				type instanceof TimestampNTZType || type instanceof DayTimeIntervalType) {
			data = moveBuffer(HostMemoryBuffer.allocate(newCap * 8L), data);
		} else if (childColumns != null) {
			// Nothing to store.
		} else {
			throw new RuntimeException("Unhandled " + type);
		}

		valid = moveBuffer(HostMemoryBuffer.allocate(newCap), valid);
		valid.setMemory(oldCapacity, newCap - oldCapacity, (byte) 1);

		capacity = newCap;
	}

	@Override
	protected WritableColumnVector reserveNewColumn(int capacity, DataType type) {
		return new RapidsWritableColumnVector(capacity, type);
	}

	private HostMemoryBuffer moveBuffer(HostMemoryBuffer targetBuffer, HostMemoryBuffer buffer) {
		try {
			if (buffer != null) {
				targetBuffer.copyFromHostBuffer(0, buffer, 0, buffer.getLength());
				buffer.close();
			}
			return targetBuffer;
		} catch (Exception e) {
			if (targetBuffer != null) {
				targetBuffer.close();
			}
			throw e;
		}
	}

	@Override
	public int getDictId(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	protected UTF8String getBytesAsUTF8String(int rowId, int count) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public ByteBuffer getByteBuffer(int rowId, int count) {
		return null;
	}

	@Override
	public int getArrayLength(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public int getArrayOffset(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public boolean isNullAt(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public boolean getBoolean(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public byte getByte(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public short getShort(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public int getInt(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public long getLong(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public float getFloat(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public double getDouble(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}
}
