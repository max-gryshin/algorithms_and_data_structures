package serve

import "fmt"

// BlockAllocator manages a paged KV-cache (PagedAttention, Kwon et al., 2023): memory
// is split into NumBlocks blocks of BlockSize tokens; a sequence owns a block table — a
// list of block ids, like a page table. Blocks are reference counted so that sequences
// with a common prefix share its blocks (prefix caching). Not safe for concurrent use:
// the engine goroutine owns it.
type BlockAllocator struct {
	BlockSize int
	refs      []int32
	free      []int // stack of free block ids
}

func NewBlockAllocator(numBlocks, blockSize int) *BlockAllocator {
	a := &BlockAllocator{BlockSize: blockSize, refs: make([]int32, numBlocks)}
	for i := numBlocks - 1; i >= 0; i-- {
		a.free = append(a.free, i)
	}
	return a
}

func (a *BlockAllocator) NumBlocks() int { return len(a.refs) }
func (a *BlockAllocator) Free() int      { return len(a.free) }
func (a *BlockAllocator) Used() int      { return len(a.refs) - len(a.free) }
func (a *BlockAllocator) RefCount(id int) int {
	return int(a.refs[id])
}

// Alloc takes n blocks with refcount 1, all or nothing.
func (a *BlockAllocator) Alloc(n int) ([]int, bool) {
	if n > len(a.free) {
		return nil, false
	}
	ids := make([]int, n)
	for i := range ids {
		id := a.free[len(a.free)-1]
		a.free = a.free[:len(a.free)-1]
		a.refs[id] = 1
		ids[i] = id
	}
	return ids, true
}

// Retain adds a reference (another sequence or the prefix cache shares the block).
func (a *BlockAllocator) Retain(ids ...int) {
	for _, id := range ids {
		if a.refs[id] <= 0 {
			panic(fmt.Sprintf("serve: retain of free block %d", id))
		}
		a.refs[id]++
	}
}

// Release drops a reference; a block whose count reaches 0 returns to the free list.
func (a *BlockAllocator) Release(ids ...int) {
	for _, id := range ids {
		if a.refs[id] <= 0 {
			panic(fmt.Sprintf("serve: double free of block %d", id))
		}
		a.refs[id]--
		if a.refs[id] == 0 {
			a.free = append(a.free, id)
		}
	}
}

// BlocksFor is ⌈tokens / blockSize⌉.
func BlocksFor(tokens, blockSize int) int { return (tokens + blockSize - 1) / blockSize }

// FragStats compares the paged layout with contiguous preallocation of maxLen tokens.
type FragStats struct {
	Tokens        int     // tokens actually stored
	PagedSlots    int     // Σ ⌈len/B⌉·B
	PagedWaste    float64 // 1 − Tokens/PagedSlots: only the tail of the last block
	ContigSlots   int     // n · maxLen
	ContigWaste   float64 // 1 − Tokens/ContigSlots
	PagedPerToken float64 // PagedSlots / Tokens
}

// Fragmentation computes FragStats for sequences of the given lengths.
func Fragmentation(lens []int, blockSize, maxLen int) FragStats {
	var f FragStats
	for _, n := range lens {
		f.Tokens += n
		f.PagedSlots += BlocksFor(n, blockSize) * blockSize
		f.ContigSlots += maxLen
	}
	if f.PagedSlots > 0 {
		f.PagedWaste = 1 - float64(f.Tokens)/float64(f.PagedSlots)
		f.PagedPerToken = float64(f.PagedSlots) / float64(f.Tokens)
	}
	if f.ContigSlots > 0 {
		f.ContigWaste = 1 - float64(f.Tokens)/float64(f.ContigSlots)
	}
	return f
}
