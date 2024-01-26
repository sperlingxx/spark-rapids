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
import java.util.List;
import java.util.Optional;

import ai.rapids.cudf.*;

import com.nvidia.spark.rapids.GpuColumnVector;
import org.apache.spark.sql.types.*;
import org.apache.spark.unsafe.types.UTF8String;

public class HostWritableColumnVector extends WritableColumnVector {

	private HostMemoryBuffer data;
	private HostMemoryBuffer offsets;
	private HostMemoryBuffer valid;
	private int lastRowIndex = -1;

	public HostWritableColumnVector(int capacity, DataType type) {
		super(capacity, type);
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
			HostWritableColumnVector keyCol = (HostWritableColumnVector) childColumns[0];
			if (keyCol.valid != null) {
				keyCol.valid.close();
				keyCol.valid = null;
				keyCol.numNulls = 0;
			}
			mapChild.add(keyCol.build(false));
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

		Optional<Long> nullCnt = valid == null ? Optional.of(0L) : Optional.of((long) numNulls);
		if (topLevel) {
			return new HostColumnVector(cudfType, numRows, nullCnt, data, valid, offsets, children);
		}
		return new HostColumnVectorCore(cudfType, numRows, nullCnt, data, valid, offsets, children);
	}

	public void reAllocate(int newCapacity) {
		this.capacity = 0;
		this.elementsAppended = 0;
		this.numNulls = 0;
		data = null;
		offsets = null;
		valid = null;
		lastRowIndex = -1;
		reserveInternal(newCapacity);

		if (childColumns != null) {
			if (isArray() && (!(type instanceof ArrayType))) {
					newCapacity *= DEFAULT_ARRAY_LENGTH;
			}
			for (WritableColumnVector ch : childColumns) {
				((HostWritableColumnVector) ch).reAllocate(newCapacity);
			}
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
		if (valid == null) {
			return false;
		}
		int b = valid.getByte(rowId / 8);
		int i = b & (1 << (rowId % 8));
		return i == 0;
	}

	@Override
	public void putNotNull(int rowId) {
		/*if (valid != null) {
			long bucket = rowId / 8;
			byte currentByte = valid.getByte(bucket);
			int bitmask = 1 << (rowId % 8);
			valid.setByte(bucket, (byte) (currentByte | bitmask));
		}*/
	}

	@Override
	public void putNull(int rowId) {
		if (valid == null) {
			valid = initNullMask(elementsAppended > 0 ? elementsAppended : capacity, null);
		}
		long bucket = rowId / 8;
		byte currentByte = valid.getByte(bucket);
		int bitmask = 1 << (rowId % 8);
		valid.setByte(bucket, (byte) (currentByte & ~bitmask));
		++numNulls;
	}

	@Override
	public void putNulls(int rowId, int count) {
		if (valid == null) {
			valid = initNullMask(elementsAppended > 0 ? elementsAppended : capacity, null);
		}
		long startBucket = rowId / 8;
		long endBucket = (rowId + count - 1) / 8 + 1;
		// handle head bucket
		int bitmask = 0;
		for (int i = rowId % 8; i < 8; i++) bitmask |= 1 << i;
		valid.setByte(startBucket, (byte) (valid.getByte(startBucket) & ~bitmask));
		// handle tail bucket
		if (startBucket < endBucket - 1) {
			bitmask = 0;
			for (int i = 0; i <= (rowId + count) % 8; i++) bitmask |= 1 << i;
			valid.setByte(endBucket - 1, (byte) (valid.getByte(endBucket - 1) & ~bitmask));
		}
		// handle middle buckets
		if (startBucket < endBucket - 2) {
			valid.setMemory(startBucket + 1, endBucket - startBucket - 2, (byte) 0x00);
		}
		numNulls += count;
	}

	@Override
	public void putNotNulls(int rowId, int count) {
		/*if (!hasNull() || valid == null) return;

		long startBucket = rowId / 8;
		long endBucket = (rowId + count - 1) / 8 + 1;
		// handle head bucket
		int bitmask = 0;
		for (int i = rowId % 8; i < 8; i++) bitmask |= 1 << i;
		valid.setByte(startBucket, (byte) (valid.getByte(startBucket) | bitmask));
		// handle tail bucket
		if (startBucket < endBucket - 1) {
			bitmask = 0;
			for (int i = 0; i <= (rowId + count) % 8; i++) bitmask |= 1 << i;
			valid.setByte(endBucket - 1, (byte) (valid.getByte(endBucket - 1) | bitmask));
		}
		// handle middle buckets
		if (startBucket < endBucket - 2) {
			valid.setMemory(startBucket + 1, endBucket - startBucket - 2, (byte) 0xFF);
		}*/
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
		int realOffset = offsets.getInt((lastRowIndex + 1) * 4L);
		for (int i = lastRowIndex + 1; i < rowId; ++i) {
			offsets.setInt((i + 1) * 4L, realOffset);
		}
		offsets.setInt((rowId + 1) * 4L, realOffset + length);
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
		if (valid != null) {
			valid = initNullMask(newCap, valid);
		}
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

	private HostMemoryBuffer initNullMask(int rowCapacity, HostMemoryBuffer curBuffer) {
		long actualBytes = ((long) rowCapacity + 7) >> 3;
		long paddingBytes = ((actualBytes + 63) >> 6) << 6;
		HostMemoryBuffer newBuffer = HostMemoryBuffer.allocate(paddingBytes);
		long offset = curBuffer == null ? 0 : curBuffer.getLength();
		long length = paddingBytes - offset;
		newBuffer.setMemory(offset, length, (byte) 0xFF);
		if (curBuffer != null) {
			return moveBuffer(newBuffer, curBuffer);
		}
		return newBuffer;
	}

	@Override
	protected WritableColumnVector reserveNewColumn(int capacity, DataType type) {
		return new HostWritableColumnVector(capacity, type);
	}

	@Override
	public WritableColumnVector reserveDictionaryIds(int capacity) {
		if (dictionaryIds == null) {
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
