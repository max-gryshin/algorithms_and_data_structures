package gpt

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"sort"
)

// Checkpoint file format (lecture L9) — the same idea as safetensors (L12):
//
//	"GOGPTv01"                  8 bytes magic
//	H                           uint64 little-endian: length of the header
//	header                      H bytes of JSON: config, meta, tensor table
//	data                        raw little-endian tensors, at the offsets of the table
//
// The header is readable with `head -c 4000 model.ckpt`; the data can be mmapped and
// read without parsing (L12). Model weights are F32; Extra arrays (optimizer moments)
// are F64 so that a resumed run continues bit for bit.
const checkpointMagic = "GOGPTv01"

// Checkpoint is a model plus everything needed to resume or use it.
type Checkpoint struct {
	Model *Model
	Meta  map[string]string    // free-form: tokenizer, step, data path, git commit…
	Extra map[string][]float64 // optimizer state and other float arrays
}

type tensorEntry struct {
	Name   string `json:"name"`
	DType  string `json:"dtype"` // "F32" or "F64"
	Shape  []int  `json:"shape"`
	Offset int64  `json:"offset"` // bytes from the start of the data section
}

type checkpointHeader struct {
	Config  Config            `json:"config"`
	Meta    map[string]string `json:"meta,omitempty"`
	Tensors []tensorEntry     `json:"tensors"`
}

// Save writes the checkpoint to w.
func (c *Checkpoint) Save(w io.Writer) error {
	h := checkpointHeader{Config: c.Model.Cfg, Meta: c.Meta}
	var off int64
	params := c.Model.NamedParameters()
	for _, np := range params {
		h.Tensors = append(h.Tensors, tensorEntry{np.Name, "F32", np.P.Data.Shape(), off})
		off += int64(np.P.Data.Size()) * 4
	}
	extra := sortedKeys(c.Extra)
	for _, k := range extra {
		h.Tensors = append(h.Tensors, tensorEntry{k, "F64", []int{len(c.Extra[k])}, off})
		off += int64(len(c.Extra[k])) * 8
	}
	js, err := json.Marshal(h)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(w, 1<<20)
	bw.WriteString(checkpointMagic)
	binary.Write(bw, binary.LittleEndian, uint64(len(js)))
	bw.Write(js)
	var buf [8]byte
	for _, np := range params {
		for _, x := range np.P.Data.Data() {
			binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(x))
			bw.Write(buf[:4])
		}
	}
	for _, k := range extra {
		for _, x := range c.Extra[k] {
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(x))
			bw.Write(buf[:])
		}
	}
	return bw.Flush()
}

// SaveFile writes atomically: to a temporary file in the same directory, then rename.
// A crash in the middle of saving never leaves a truncated checkpoint behind.
func (c *Checkpoint) SaveFile(path string) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name()) // no-op after a successful rename
	if err := c.Save(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// LoadCheckpoint reads a checkpoint written by Save and rebuilds the model.
func LoadCheckpoint(r io.Reader) (*Checkpoint, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	magic := make([]byte, len(checkpointMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, err
	}
	if string(magic) != checkpointMagic {
		return nil, fmt.Errorf("gpt: not a checkpoint (magic %q)", magic)
	}
	var hl uint64
	if err := binary.Read(br, binary.LittleEndian, &hl); err != nil {
		return nil, err
	}
	if hl > 1<<30 {
		return nil, fmt.Errorf("gpt: header of %d bytes", hl)
	}
	js := make([]byte, hl)
	if _, err := io.ReadFull(br, js); err != nil {
		return nil, err
	}
	var h checkpointHeader
	if err := json.Unmarshal(js, &h); err != nil {
		return nil, fmt.Errorf("gpt: checkpoint header: %w", err)
	}
	if err := h.Config.Validate(); err != nil {
		return nil, err
	}
	m := New(rand.New(rand.NewPCG(0, 0)), h.Config)
	byName := map[string]NamedParam{}
	for _, np := range m.NamedParameters() {
		byName[np.Name] = np
	}
	c := &Checkpoint{Model: m, Meta: h.Meta, Extra: map[string][]float64{}}
	var off int64
	var buf [8]byte
	for _, e := range h.Tensors {
		if e.Offset != off {
			return nil, fmt.Errorf("gpt: tensor %s at offset %d, expected %d", e.Name, e.Offset, off)
		}
		switch e.DType {
		case "F32":
			np, ok := byName[e.Name]
			if !ok {
				return nil, fmt.Errorf("gpt: unknown tensor %s", e.Name)
			}
			if !slices.Equal(np.P.Data.Shape(), e.Shape) {
				return nil, fmt.Errorf("gpt: %s has shape %v, model wants %v", e.Name, e.Shape, np.P.Data.Shape())
			}
			d := np.P.Data.Data()
			for i := range d {
				if _, err := io.ReadFull(br, buf[:4]); err != nil {
					return nil, err
				}
				d[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[:4]))
			}
			off += int64(len(d)) * 4
			delete(byName, e.Name)
		case "F64":
			if len(e.Shape) != 1 {
				return nil, fmt.Errorf("gpt: extra %s has shape %v", e.Name, e.Shape)
			}
			x := make([]float64, e.Shape[0])
			for i := range x {
				if _, err := io.ReadFull(br, buf[:]); err != nil {
					return nil, err
				}
				x[i] = math.Float64frombits(binary.LittleEndian.Uint64(buf[:]))
			}
			c.Extra[e.Name] = x
			off += int64(len(x)) * 8
		default:
			return nil, fmt.Errorf("gpt: tensor %s has dtype %q", e.Name, e.DType)
		}
	}
	if len(byName) > 0 {
		return nil, fmt.Errorf("gpt: checkpoint misses %d tensors, e.g. %s", len(byName), sortedKeys(byName)[0])
	}
	return c, nil
}

// LoadCheckpointFile opens path and calls LoadCheckpoint.
func LoadCheckpointFile(path string) (*Checkpoint, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c, err := LoadCheckpoint(f)
	if err != nil {
		return nil, errors.Join(fmt.Errorf("gpt: %s", path), err)
	}
	return c, nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
