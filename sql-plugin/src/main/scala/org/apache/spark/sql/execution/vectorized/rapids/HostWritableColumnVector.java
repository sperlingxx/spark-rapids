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

import com.google.crypto.tink.subtle.Random;
import com.nvidia.spark.rapids.GpuColumnVector;
import org.apache.spark.sql.types.*;
import org.apache.spark.unsafe.types.UTF8String;

public class HostWritableColumnVector extends WritableColumnVector {

	private HostMemoryBuffer data;
	private byte[] valids = null;
	// offset for ArrayType|MapType
	private int[] arrayLengths;
	private int[] arrayOffsets;
	private int selectedLength = 0;
	private List<Integer> childrenRanges;
	// offset for StringType
	private HostMemoryBuffer charOffset;
	private int lastCharRowId = -1;

	private int rowCnt;

	public HostWritableColumnVector(int capacity, DataType type) {
		super(capacity, type);
		this.capacity = 0;
		childrenRanges = new ArrayList<>();
		reserveInternal(capacity);
		this.rowCnt = capacity;
	}

	public HostColumnVector build() {
		List<Integer> initRanges = new ArrayList<>();
		initRanges.add(0);
		initRanges.add(rowCnt);
		int rdSeed = Random.randInt(Integer.MAX_VALUE - 1);
		return (HostColumnVector) buildImpl(initRanges, rowCnt, true, rdSeed);
	}

	private HostColumnVectorCore buildImpl(List<Integer> ranges, int rangeLength, boolean topLevel, int rdSeed) {

		DType cudfType = type instanceof MapType ? DType.LIST : GpuColumnVector.getRapidsType(type);

		HostMemoryBuffer offsetBuffer = null;
		HostMemoryBuffer validBuffer = null;
		Optional<Long> nullCnt = Optional.of((long) numNulls);

		if (type instanceof MapType || type instanceof ArrayType) {
			// Merge current range with parent ranges
			gatherNestedRanges(ranges, rangeLength);
			// Convert arrayOffset/arrayLength to cuDF offset buffer
			offsetBuffer = buildOffsetBuffer(rangeLength);
		} else if (type instanceof StringType) {
			// Gather String(Array[Byte]) with potential existed parent ranges
			gatherByteArray(ranges, rangeLength);
			offsetBuffer = charOffset;
		} else if (type instanceof StructType) {
			childrenRanges = ranges;
			selectedLength = rangeLength;
		} else {
			gatherFixedWidthBuffer(ranges, rangeLength, cudfType.getSizeInBytes());
		}

		/*if (type instanceof MapType || type instanceof ArrayType) {
			if (offsetBuffer != null)
				dumpOffsetVector(offsetBuffer, rdSeed);
		}
		if (type instanceof LongType) {
			if (data != null)
				dumpLongVector(data, rdSeed);
		}*/

		if (valids != null) {
			// Truncate valid (temp) array via Ranges
			gatherValidBuffer(ranges, rangeLength);
			// Build bitwise validity mask
			if (!ranges.isEmpty()) {
				validBuffer = buildNullMask(rangeLength);
			}
		}

		// Build child columns recursively
		List<HostColumnVectorCore> children = new ArrayList<>();
		if (childColumns != null) {
			for (WritableColumnVector ch : childColumns) {
				children.add(((HostWritableColumnVector) ch).buildImpl(
						childrenRanges, selectedLength, false, rdSeed));
			}
		}

		// Wrap the level of Struct to adapt cuDF layout for MapType
		// Array[(keyCol, valueCol)] => Array[Struct[keyCol, valueCol]]
		if (type instanceof MapType) {
			HostColumnVectorCore cv = new HostColumnVectorCore(
					DType.STRUCT,
					children.get(0).getRowCount(),
					Optional.of(0L),
					null, null, null, children);
			children = new ArrayList<>();
			children.add(cv);
		}

		if (topLevel) {
			return new HostColumnVector(
					cudfType, rangeLength, nullCnt, data, validBuffer, offsetBuffer, children);
		}
		return new HostColumnVectorCore(
				cudfType, rangeLength, nullCnt, data, validBuffer, offsetBuffer, children);
	}

	private void gatherNestedRanges(List<Integer> ranges, int numRecord) {
		if (ranges.isEmpty()) {
			childrenRanges.clear();
			selectedLength = 0;
			arrayOffsets = null;
			arrayLengths = null;
			return;
		}
		if (ranges.size() == 2 && ranges.get(0) == 0 && ranges.get(1) == rowCnt) {
			return;
		}
		int[] newArrayOffsets = new int[numRecord];
		int[] newArrayLengths = new int[numRecord];
		int dstOffset = 0, newSelectedLength = 0;
		List<Integer> newChildRanges = new ArrayList<>();
		int rangeUb = -1, rangeIndex = -1;
		for (int i = 0; i < ranges.size(); i += 2) {
			for (int j = ranges.get(i); j < ranges.get(i + 1); ++j) {
				newArrayOffsets[dstOffset] = arrayOffsets[j];
				newArrayLengths[dstOffset] = arrayLengths[j];
				dstOffset++;
				newSelectedLength += arrayLengths[j];
				// skip null records because their offsets are not correctly setup
				if (valids != null && valids[j] == (byte) 1) {
					continue;
				}
				if (rangeUb == arrayOffsets[j]) {
					rangeUb += arrayLengths[j];
					newChildRanges.set(rangeIndex, rangeUb);
				} else {
					newChildRanges.add(arrayOffsets[j]);
					rangeUb = arrayOffsets[j] + arrayLengths[j];
					newChildRanges.add(rangeUb);
					rangeIndex += 2;
				}
			}
		}
		arrayOffsets = newArrayOffsets;
		arrayLengths = newArrayLengths;
		selectedLength = newSelectedLength;
		childrenRanges = newChildRanges;
	}

	private void gatherByteArray(List<Integer> ranges, int numRecord) {
		if (ranges.isEmpty()) {
			if (charOffset != null) {
				charOffset.close();
				charOffset = null;
			}
			if (childColumns != null) {
				childColumns[0].close();
				childColumns = null;
			}
			if (data != null) {
				data.close();
				data = null;
			}
			return;
		}

		// padding tail values
		int rangeUb = ranges.get(ranges.size() - 1);
		if (lastCharRowId + 1 < rangeUb) {
			int byteArrayEnd = charOffset.getInt((lastCharRowId + 1) * 4L);
			for (int i = lastCharRowId + 2; i < rangeUb + 1; ++i)
				charOffset.setInt(i * 4L, byteArrayEnd);
		}

		HostMemoryBuffer newCharOffset;
		if (ranges.size() == 2) {
			long offset = ranges.get(0) * 4L;
			long size = (ranges.get(1) + 1) * 4L - offset;
			newCharOffset = charOffset.slice(offset, size);
		} else {
			newCharOffset = HostMemoryBuffer.allocate((numRecord + 1) * 4L);
			long dstOff = 0;
			for (int i = 0; i < ranges.size(); i += 2) {
				long srcOff = ranges.get(i) * 4L;
				long len = ranges.get(i + 1) * 4L - srcOff;
				newCharOffset.copyFromHostBuffer(dstOff, charOffset, srcOff, len);
				dstOff += len;
			}
			long charOffsetEnd = ranges.get(ranges.size() - 1) * 4L;
			newCharOffset.setInt(dstOff, charOffset.getInt(charOffsetEnd));
		}

		charOffset.close();
		charOffset = newCharOffset;

		data = ((HostWritableColumnVector) childColumns[0]).data;
		childColumns = null;
		int byteArrayEnd = charOffset.getInt(charOffset.getLength() - 4);
		HostMemoryBuffer newData = data.slice(0, byteArrayEnd);
		data.close();
		data = newData;
	}

	private void gatherFixedWidthBuffer(List<Integer> ranges, int numRecord, long rcdWidth) {
		if (ranges.isEmpty()) {
			if (data != null) {
				data.close();
				data = null;
			}
			return;
		}

		HostMemoryBuffer newData;
		if (ranges.size() == 2) {
			long offset = ranges.get(0) * rcdWidth;
			long size = ranges.get(1) * rcdWidth - offset;
			newData = data.slice(offset, size);
		} else {
			newData = HostMemoryBuffer.allocate(numRecord * rcdWidth);
			long dstOffset = 0;
			for (int i = 0; i < ranges.size(); i += 2) {
				long offset = ranges.get(i);
				long length = ranges.get(i + 1) - offset;
				newData.copyFromHostBuffer(dstOffset * rcdWidth,
						data, offset * rcdWidth, length * rcdWidth);
				dstOffset += length;
			}
		}
		data.close();
		data = newData;
	}

	private void gatherValidBuffer(List<Integer> ranges, int rangeLength) {
		if (ranges.isEmpty()) {
			valids = null;
			numNulls = 0;
			return;
		}
		if (ranges.size() == 2 && ranges.get(0) == 0 && ranges.get(1) == rowCnt) {
			return;
		}
		byte[] newValids = new byte[rangeLength];
		int dstOffset = 0;
		for (int i = 0; i < ranges.size(); i += 2) {
			int offset = ranges.get(i);
			int length = ranges.get(i + 1) - offset;
			System.arraycopy(valids, offset, newValids, dstOffset, length);
			dstOffset += length;
		}
		valids = newValids;
	}

	private HostMemoryBuffer buildOffsetBuffer(int numRecord) {
		if (arrayOffsets == null) return null;

		HostMemoryBuffer offBuf = HostMemoryBuffer.allocate((numRecord + 1) * 4L);
		offBuf.setInt(0L, 0);
		int lastIndex = 0;
		for (int i = 0; i < numRecord; ++i) {
			lastIndex += arrayLengths[i];
			offBuf.setInt((i + 1) * 4L, lastIndex);
		}

		return offBuf;
	}

	private HostMemoryBuffer buildNullMask(int numRecord) {
		long actualBytes = ((long) numRecord + 7) >> 3;
		long paddingBytes = ((actualBytes + 63) >> 6) << 6;
		HostMemoryBuffer nullMask = HostMemoryBuffer.allocate(paddingBytes);
		for (int i = 0; i < numRecord - 7; i += 8) {
			int mask = (valids[i] ^ 1)
					| ((valids[i + 1] ^ 1) << 1)
					| ((valids[i + 2] ^ 1) << 2)
					| ((valids[i + 3] ^ 1) << 3)
					| ((valids[i + 4] ^ 1) << 4)
					| ((valids[i + 5] ^ 1) << 5)
					| ((valids[i + 6] ^ 1) << 6)
					| ((valids[i + 7] ^ 1) << 7);
			nullMask.setByte(i >> 3, (byte) mask);
		}
		int lastByte = 0;
		int j = 0;
		for (int i = (numRecord >> 3) << 3; i < numRecord; i++) {
			lastByte |= ((valids[i] ^ 1) << j++);
		}
		if (j > 0) {
			nullMask.setByte(numRecord >> 3, (byte) lastByte);
		}

		return nullMask;
	}

	public void reAllocate(int newCapacity) {
		this.capacity = 0;
		this.rowCnt = 0;
		this.elementsAppended = 0;
		this.numNulls = 0;
		data = null;
		charOffset = null;
		valids = null;
		lastCharRowId = -1;
		arrayLengths = null;
		arrayOffsets = null;
		selectedLength = 0;
		childrenRanges = new ArrayList<>();
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
		if (isAllNull) return true;
		if (valids == null) return false;
		return valids[rowId] == 1;
	}

	@Override
	public void putNotNull(int rowId) {
		if (!hasNull() || valids == null) return;
		valids[rowId] = 0;
	}

	@Override
	public void putNull(int rowId) {
		if (valids == null) {
			initNullMask(elementsAppended > 0 ? elementsAppended : capacity);
		}
		valids[rowId] = 1;
		++numNulls;
	}

	@Override
	public void putNulls(int rowId, int count) {
		if (valids == null) {
			initNullMask(elementsAppended > 0 ? elementsAppended : capacity);
		}
		for (int i = 0; i < count; ++i) {
			valids[rowId + i] = 1;
		}
		numNulls += count;
	}

	@Override
	public void putNotNulls(int rowId, int count) {
		if (!hasNull() || valids == null) return;
		for (int i = 0; i < count; ++i) {
			valids[rowId + i] = 0;
		}
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
		arrayOffsets[rowId] = offset;
		arrayLengths[rowId] = length;

		if (length == 0) return;

		if (!childrenRanges.isEmpty() && childrenRanges.get(childrenRanges.size() - 1) == offset) {
			childrenRanges.set(childrenRanges.size() - 1, offset + length);
		} else {
			childrenRanges.add(offset);
			childrenRanges.add(offset + length);
		}
		selectedLength += length;
	}

	@Override
	public int putByteArray(int rowId, byte[] value, int offset, int length) {
		int result = arrayData().appendBytes(length, value, offset);
		for (int i = lastCharRowId + 1; i < rowId; ++i) {
			charOffset.setInt((i + 1) * 4L, result);
		}
		charOffset.setInt((rowId + 1) * 4L, result + length);
		lastCharRowId = rowId;
		return result;
	}

	@Override
	public void reserve(int requiredCapacity) {
		rowCnt = requiredCapacity;
		super.reserve(requiredCapacity);
	}

	@Override
	protected void reserveInternal(int newCap) {
		if (valids != null) {
			initNullMask(newCap);
		}
		if (type instanceof ArrayType || type instanceof MapType) {
			int[] newLengths = new int[newCap];
			int[] newOffsets = new int[newCap];
			if (this.arrayLengths != null) {
				System.arraycopy(this.arrayLengths, 0, newLengths, 0, capacity);
				System.arraycopy(this.arrayOffsets, 0, newOffsets, 0, capacity);
			}
			arrayLengths = newLengths;
			arrayOffsets = newOffsets;
		} else if (isArray()) {
			HostMemoryBuffer newOffsets = moveBuffer(
					HostMemoryBuffer.allocate((newCap + 1) * 4L), charOffset);
			if (charOffset == null) {
				newOffsets.setInt(0, 0);
			}
			charOffset = newOffsets;
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

	private void initNullMask(int rowCapacity) {
		if (valids == null) {
			valids = new byte[rowCapacity];
		} else {
			byte[] newValids = new byte[rowCapacity];
			System.arraycopy(valids, 0, newValids, 0, valids.length);
			valids = newValids;
		}
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

	@Override
	public void close() {
		super.close();
		arrayLengths = null;
		arrayOffsets = null;
		if (data != null) data.close();
		if (charOffset != null) charOffset.close();
	}

	private void dumpOffsetVector(HostMemoryBuffer offsetBuffer, int rdSeed) {
		StringBuffer buffer = new StringBuffer();
		buffer.append("\n[").append(rdSeed).append("]offsetVector: ");
		for (int i = 0; i < offsetBuffer.getLength(); i += 4) {
			buffer.append(offsetBuffer.getInt(i)).append(", ");
		}
		System.err.println(buffer);
	}

	private void dumpLongVector(HostMemoryBuffer buffer, int rdSeed) {
		StringBuffer sb = new StringBuffer();
		sb.append('[').append(rdSeed).append("] LongBuffer: ");
		for (int i = 0; i < buffer.getLength(); i += 8)
			sb.append(buffer.getLong(i)).append(", ");
		System.err.println(sb);
	}

	private void dumpStringVector(HostMemoryBuffer offsetBuffer, HostMemoryBuffer dataBuffer, int rdSeed) {
		StringBuffer buffer = new StringBuffer();
		buffer.append("\n[").append(rdSeed).append("]stringOffset: ");
		for (int i = 0; i < offsetBuffer.getLength(); i += 4) {
			buffer.append(offsetBuffer.getInt(i)).append(", ");
		}
		buffer.append("\nstringVector: ");
		for (int i = 0; i < offsetBuffer.getLength() - 4; i += 4) {
			for (int j = offsetBuffer.getInt(i); j < offsetBuffer.getInt(i + 4); ++j) {
				if (j >= dataBuffer.getLength()) {
					buffer.append("+++");
					System.err.println(buffer);
					return;
				}
				buffer.append((char) dataBuffer.getByte(j));
			}
			buffer.append(", ");
		}
		System.err.println(buffer);
	}
}
