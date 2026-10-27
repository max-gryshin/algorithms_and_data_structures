package weights

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
)

// safetensors layout (Hugging Face):
//
//	N        uint64 little-endian: length of the header
//	header   N bytes of JSON: {"name": {"dtype": "BF16", "shape": [4096, 4096],
//	         "data_offsets": [begin, end]}, ..., "__metadata__": {...}}
//	data     raw little-endian row-major tensors; offsets are relative to its start
//
// No code is executed on load (unlike pickle-based .bin/.pt), and every tensor can be
// read in place from an mmapped file.

// DType is a safetensors dtype name.
type DType string

const (
	F32  DType = "F32"
	F16  DType = "F16"
	BF16 DType = "BF16"
	F64  DType = "F64"
)

// Size is the number of bytes per element (0 for an unsupported dtype).
func (d DType) Size() int {
	switch d {
	case F32:
		return 4
	case F16, BF16:
		return 2
	case F64:
		return 8
	}
	return 0
}

// TensorInfo is one entry of the header.
type TensorInfo struct {
	DType       DType    `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// File is an opened safetensors file.
type File struct {
	Tensors map[string]TensorInfo
	Meta    map[string]string
	data    []byte // data section
	mapping []byte // whole mmapped file, nil if read into memory
}

// Open maps path into memory (falls back to reading it when mmap is unavailable).
func Open(path string) (*File, error) {
	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close() // the mapping stays valid after close
	st, err := fd.Stat()
	if err != nil {
		return nil, err
	}
	if m, err := mmap(fd, int(st.Size())); err == nil {
		f, err := Parse(m)
		if err != nil {
			munmap(m)
			return nil, fmt.Errorf("weights: %s: %w", path, err)
		}
		f.mapping = m
		return f, nil
	}
	b, err := io.ReadAll(fd)
	if err != nil {
		return nil, err
	}
	return Parse(b)
}

// Mapped reports whether the file is mmapped.
func (f *File) Mapped() bool { return f.mapping != nil }

// Close unmaps the file. Tensors returned by Float32 are copies and stay valid.
func (f *File) Close() error {
	if f.mapping == nil {
		return nil
	}
	m := f.mapping
	f.mapping, f.data = nil, nil
	return munmap(m)
}

// Parse reads a safetensors file held in memory (b is not copied).
func Parse(b []byte) (*File, error) {
	if len(b) < 8 {
		return nil, fmt.Errorf("safetensors: file of %d bytes", len(b))
	}
	n := binary.LittleEndian.Uint64(b)
	if n > uint64(len(b)-8) || n > 100<<20 {
		return nil, fmt.Errorf("safetensors: header length %d, file %d bytes", n, len(b))
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		return nil, fmt.Errorf("safetensors: header: %w", err)
	}
	f := &File{Tensors: map[string]TensorInfo{}, data: b[8+n:]}
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &f.Meta); err != nil {
				return nil, fmt.Errorf("safetensors: metadata: %w", err)
			}
			continue
		}
		var ti TensorInfo
		if err := json.Unmarshal(msg, &ti); err != nil {
			return nil, fmt.Errorf("safetensors: %s: %w", name, err)
		}
		numel := 1
		for _, s := range ti.Shape {
			numel *= s
		}
		lo, hi := ti.DataOffsets[0], ti.DataOffsets[1]
		switch {
		case ti.DType.Size() == 0:
			// unsupported dtypes (I64 position ids, F8…) are listed but cannot be converted
		case lo < 0 || hi < lo || hi > int64(len(f.data)):
			return nil, fmt.Errorf("safetensors: %s: offsets [%d, %d] outside data of %d bytes", name, lo, hi, len(f.data))
		case hi-lo != int64(numel*ti.DType.Size()):
			return nil, fmt.Errorf("safetensors: %s: %d bytes for %v %s", name, hi-lo, ti.Shape, ti.DType)
		}
		f.Tensors[name] = ti
	}
	return f, nil
}

// Names lists the tensors in sorted order.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.Tensors))
	for k := range f.Tensors {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

// Float32 converts a tensor to float32 (a fresh copy) and returns its shape.
func (f *File) Float32(name string) ([]float32, []int, error) {
	ti, ok := f.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	b := f.data[ti.DataOffsets[0]:ti.DataOffsets[1]]
	sz := ti.DType.Size()
	if sz == 0 {
		return nil, nil, fmt.Errorf("safetensors: %s has unsupported dtype %s", name, ti.DType)
	}
	out := make([]float32, len(b)/sz)
	le := binary.LittleEndian
	for i := range out {
		switch ti.DType {
		case F32:
			out[i] = math.Float32frombits(le.Uint32(b[4*i:]))
		case F16:
			out[i] = F16ToF32(le.Uint16(b[2*i:]))
		case BF16:
			out[i] = BF16ToF32(le.Uint16(b[2*i:]))
		case F64:
			out[i] = float32(math.Float64frombits(le.Uint64(b[8*i:])))
		}
	}
	return out, slices.Clone(ti.Shape), nil
}

// Tensor is an input of Write.
type Tensor struct {
	Name  string
	DType DType // stored dtype; Data is rounded to it
	Shape []int
	Data  []float32
}

// Write produces a safetensors file. The header is padded with spaces to a multiple
// of 8 bytes so that the data section stays aligned (as the reference writer does).
func Write(w io.Writer, ts []Tensor, meta map[string]string) error {
	header := map[string]any{}
	if meta != nil {
		header["__metadata__"] = meta
	}
	var off int64
	for _, t := range ts {
		if t.DType.Size() == 0 {
			return fmt.Errorf("safetensors: %s: unsupported dtype %s", t.Name, t.DType)
		}
		end := off + int64(len(t.Data)*t.DType.Size())
		header[t.Name] = TensorInfo{t.DType, t.Shape, [2]int64{off, end}}
		off = end
	}
	js, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(js)%8 != 0 {
		js = append(js, ' ')
	}
	bw := bufio.NewWriter(w)
	binary.Write(bw, binary.LittleEndian, uint64(len(js)))
	bw.Write(js)
	var buf [8]byte
	le := binary.LittleEndian
	for _, t := range ts {
		for _, x := range t.Data {
			switch t.DType {
			case F32:
				le.PutUint32(buf[:], math.Float32bits(x))
			case F16:
				le.PutUint16(buf[:], F32ToF16(x))
			case BF16:
				le.PutUint16(buf[:], F32ToBF16(x))
			case F64:
				le.PutUint64(buf[:], math.Float64bits(float64(x)))
			}
			bw.Write(buf[:t.DType.Size()])
		}
	}
	return bw.Flush()
}
