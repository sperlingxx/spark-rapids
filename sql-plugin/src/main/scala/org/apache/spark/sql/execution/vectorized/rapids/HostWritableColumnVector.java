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

package org.apache.spark.sql.execution.vectorized.rapids;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Optional;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;

import com.nvidia.spark.rapids.GpuColumnVector;
import org.apache.spark.sql.types.*;
import org.apache.spark.unsafe.types.UTF8String;

public class HostWritableColumnVector extends WritableColumnVector {

	private HostMemoryBuffer data;
	private HostMemoryBuffer offsets;
	private final BitSet nullMask;

	private int lastRowIndex = -1;

	public HostWritableColumnVector(int capacity, DataType type) {
		super(capacity, type);
		nullMask = new BitSet(capacity);
		this.capacity = 0;
		reserveInternal(capacity);
	}

	public HostColumnVectorCore build(boolean topLevel) {
		DType cudfType;
		if (type instanceof MapType) {
			cudfType = DType.LIST;
		} else {
			cudfType = GpuColumnVector.getRapidsType(type);
		}

		int numRows = (elementsAppended > 0) ? elementsAppended : capacity;

		List<HostColumnVectorCore> children = new ArrayList<>();
		if (type instanceof MapType) {
			List<HostColumnVectorCore> mapChild = new ArrayList<>();
			mapChild.add(((HostWritableColumnVector) childColumns[0]).build(false));
			mapChild.add(((HostWritableColumnVector) childColumns[1]).build(false));
			children.add(
					new HostColumnVectorCore(
							DType.STRUCT,
							mapChild.get(0).getRowCount(),
							Optional.of(0L),
							null, null, null,
							mapChild));
		} else if (cudfType == DType.STRING) {
			data = ((HostWritableColumnVector) childColumns[0]).data;
		} else if (childColumns != null) {
			for (WritableColumnVector child : childColumns) {
				children.add(((HostWritableColumnVector) child).build(false));
			}
		}

		HostMemoryBuffer valid = null;
		if (!nullMask.isEmpty()) {
			nullMask.flip(0, numRows);
			byte[] bitBuffer = nullMask.toByteArray();
			valid = HostMemoryBuffer.allocate(bitBuffer.length);
			valid.setBytes(0, bitBuffer, 0, bitBuffer.length);
			nullMask.clear();
		}

		if (topLevel) {
			return new HostColumnVector(
					cudfType, numRows, Optional.of((long) numNulls), data, valid, offsets, children);
		}
		return new HostColumnVectorCore(
				cudfType, numRows, Optional.of((long) numNulls), data, valid, offsets, children);
	}

	public void reAllocate(int newCapacity) {
		this.capacity = 0;
		this.elementsAppended = 0;
		this.numNulls = 0;
		data = null;
		offsets = null;
		nullMask.clear();
		reserveInternal(newCapacity);

		if (isArray() && (!(type instanceof ArrayType))) {
				newCapacity *= DEFAULT_ARRAY_LENGTH;
		}
		for (WritableColumnVector ch: childColumns) {
			((HostWritableColumnVector) ch).reAllocate(newCapacity);
		}
	}

	@Override
	public void putBooleans(int rowId, byte src) {
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
	public boolean isNullAt(int rowId) {
		return nullMask.get(rowId);
	}

	@Override
	public void putNotNull(int rowId) {
		nullMask.clear(rowId);
	}

	@Override
	public void putNull(int rowId) {
		nullMask.set(rowId);
		++numNulls;
	}

	@Override
	public void putNulls(int rowId, int count) {
		nullMask.set(rowId, rowId + count);
		numNulls += count;
	}

	@Override
	public void putNotNulls(int rowId, int count) {
		nullMask.clear(rowId, rowId + count);
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
		ByteBuffer bb = ByteBuffer.wrap(src).order(ByteOrder.LITTLE_ENDIAN);
		long offset = 4L * rowId;
		for (int i = 0; i < count; ++i, offset += 4) {
			data.setInt(offset, bb.getInt(srcIndex + (4 * i)));
		}
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
		ByteBuffer bb = ByteBuffer.wrap(src).order(ByteOrder.LITTLE_ENDIAN);
		long offset = 8L * rowId;
		for (int i = 0; i < count; ++i, offset += 8) {
			data.setLong(offset, bb.getLong(srcIndex + (8 * i)));
		}
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
		data.setBytes(rowId * 4L, src, srcIndex * 4L, count * 4L);
	}

	@Override
	public void putFloatsLittleEndian(int rowId, int count, byte[] src, int srcIndex) {
		ByteBuffer bb = ByteBuffer.wrap(src).order(ByteOrder.LITTLE_ENDIAN);
		long offset = 4L * rowId;
		for (int i = 0; i < count; ++i, offset += 4) {
			data.setFloat(offset, bb.getFloat(srcIndex + (4 * i)));
		}
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
		ByteBuffer bb = ByteBuffer.wrap(src).order(ByteOrder.LITTLE_ENDIAN);
		long offset = 8L * rowId;
		for (int i = 0; i < count; ++i, offset += 8) {
			data.setDouble(offset, bb.getDouble(srcIndex + (8 * i)));
		}
	}

	@Override
	public void putArray(int rowId, int offset, int length) {
		assert(offset >= 0 && offset + length <= childColumns[0].capacity);
		for (int i = lastRowIndex + 1; i < rowId; ++i) {
			offsets.setInt((i + 1) * 4L, offset);
		}
		offsets.setInt((rowId + 1) * 4L, offset + length);
		lastRowIndex = rowId;
	}

	@Override
	public int putByteArray(int rowId, byte[] value, int offset, int length) {
		int result = arrayData().appendBytes(length, value, offset);
		for (int i = lastRowIndex + 1; i < rowId; ++i) {
			offsets.setInt((i + 1) * 4L, result);
		}
		offsets.setInt((rowId + 1) * 4L, result + length);
		lastRowIndex = rowId;
		return result;
	}

	@Override
	protected void reserveInternal(int newCap) {
		System.err.println("reserve Column(" + type + ") from " + capacity + " to " + newCap);

		if (isArray() || type instanceof MapType) {
			HostMemoryBuffer newOffsets = moveBuffer(HostMemoryBuffer.allocate((newCap + 1) * 4L), offsets);
			if (offsets == null) {
				newOffsets.setInt(0, 0);
			}
			offsets = newOffsets;
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

		capacity = newCap;
	}

	@Override
	protected WritableColumnVector reserveNewColumn(int capacity, DataType type) {
		return new HostWritableColumnVector(capacity, type);
	}

	@Override
	public WritableColumnVector reserveDictionaryIds(int capacity) {
		if (dictionaryIds == null) {
			System.err.println("reserved OnHeap DictIds " + capacity);
			dictionaryIds = new OnHeapColumnVector(capacity, DataTypes.IntegerType);
		} else {
			dictionaryIds.reset();
			dictionaryIds.reserve(capacity);
		}

		return dictionaryIds;
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
	public int getArrayLength(int rowId) {
		throw new UnsupportedOperationException("RapidsWritableColumnVector does NOT support getters");
	}

	@Override
	public int getArrayOffset(int rowId) {
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

	@Override
	public ByteBuffer getByteBuffer(int rowId, int count) {
		byte[] buffer = new byte[count];
		data.getBytes(buffer, count, rowId, count);
		return ByteBuffer.wrap(buffer);
	}
}
