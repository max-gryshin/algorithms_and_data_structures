package rag

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
)

// VectorIndex stores vectors and finds the most similar ones to a query.
// Vectors are normalized on Add, so similarity = cosine = dot product.
type VectorIndex interface {
	Add(v []float32) int
	Search(q []float32, k int) []Hit
	Len() int
}

// Flat is exact brute-force search: compare the query with every vector, O(N·d).
type Flat struct {
	mu   sync.RWMutex
	vecs [][]float32
}

func (f *Flat) Add(v []float32) int {
	v = normalized(v)
	f.mu.Lock()
	defer f.mu.Unlock()
	f.vecs = append(f.vecs, v)
	return len(f.vecs) - 1
}

func (f *Flat) Len() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.vecs)
}

// Search keeps the k best candidates in a min-heap of size k: O(N·d + N·log k).
func (f *Flat) Search(q []float32, k int) []Hit {
	q = normalized(q)
	f.mu.RLock()
	defer f.mu.RUnlock()
	h := &nodeHeap{} // min-heap by similarity: the root is the worst of the current top-k
	for i, v := range f.vecs {
		s := Dot(q, v)
		if h.Len() < k {
			heap.Push(h, node{int32(i), s})
		} else if s > (*h)[0].sim {
			(*h)[0] = node{int32(i), s}
			heap.Fix(h, 0)
		}
	}
	return h.sortedHits()
}

// HNSW is a Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).
// Each vector gets a random top level ℓ = ⌊−ln(U)·mL⌋; on every level ≤ ℓ it is linked to
// its M nearest neighbours (2M on level 0). Search descends greedily from the sparse top
// level to level 0, then runs a beam search of width ef.
type HNSW struct {
	M              int // links per node on levels ≥ 1
	EfConstruction int // beam width while inserting
	EfSearch       int // beam width while searching (≥ k)

	mu       sync.RWMutex
	rng      *rand.Rand
	mL       float64
	vecs     [][]float32
	links    [][][]int32 // links[node][level] = neighbour ids
	entry    int32
	maxLevel int
	visits   sync.Pool // *visitSet, reused between searches
}

// visitSet marks visited nodes without clearing memory between searches:
// a node is visited iff mark[id] == gen, and every search bumps gen.
type visitSet struct {
	mark []uint32
	gen  uint32
}

func (g *HNSW) getVisits() *visitSet {
	v, _ := g.visits.Get().(*visitSet)
	if v == nil {
		v = &visitSet{}
	}
	if len(v.mark) < len(g.vecs) {
		v.mark = make([]uint32, len(g.vecs)+len(g.vecs)/2)
		v.gen = 0
	}
	v.gen++
	if v.gen == 0 { // wrapped around: reset
		clear(v.mark)
		v.gen = 1
	}
	return v
}

// NewHNSW returns an empty graph. Typical values: m = 16, efConstruction = 200, efSearch = 64.
func NewHNSW(m, efConstruction, efSearch int, seed uint64) *HNSW {
	return &HNSW{
		M: m, EfConstruction: efConstruction, EfSearch: efSearch,
		rng: rand.New(rand.NewPCG(seed, seed)), mL: 1 / math.Log(float64(m)),
		entry: -1,
	}
}

func (g *HNSW) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.vecs)
}

// Levels returns how many nodes have each top level (level 0 = only the bottom layer).
func (g *HNSW) Levels() []int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	counts := make([]int, g.maxLevel+1)
	for _, l := range g.links {
		counts[len(l)-1]++
	}
	return counts
}

func (g *HNSW) Add(v []float32) int {
	v = normalized(v)
	g.mu.Lock()
	defer g.mu.Unlock()

	id := int32(len(g.vecs))
	level := int(-math.Log(1-g.rng.Float64()) * g.mL)
	g.vecs = append(g.vecs, v)
	g.links = append(g.links, make([][]int32, level+1))
	if g.entry < 0 {
		g.entry, g.maxLevel = id, level
		return int(id)
	}

	cur := g.entry
	for l := g.maxLevel; l > level; l-- {
		cur = g.greedy(v, cur, l)
	}
	eps := []int32{cur}
	for l := min(level, g.maxLevel); l >= 0; l-- {
		cands := g.searchLayer(v, eps, g.EfConstruction, l)
		for _, n := range g.selectNeighbors(cands, g.M) {
			g.links[id][l] = append(g.links[id][l], n.id)
			g.links[n.id][l] = append(g.links[n.id][l], id)
			g.prune(n.id, l)
		}
		eps = eps[:0]
		for _, c := range cands {
			eps = append(eps, c.id)
		}
	}
	if level > g.maxLevel {
		g.entry, g.maxLevel = id, level
	}
	return int(id)
}

// prune shrinks the neighbour list of node n on level l back to the limit
// (M, or 2M on level 0) using the same diversity heuristic as insertion.
func (g *HNSW) prune(n int32, l int) {
	maxLinks := g.M
	if l == 0 {
		maxLinks = 2 * g.M
	}
	ls := g.links[n][l]
	if len(ls) <= maxLinks {
		return
	}
	cands := make([]node, len(ls))
	for i, m := range ls {
		cands[i] = node{m, Dot(g.vecs[n], g.vecs[m])}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].sim > cands[j].sim })
	g.links[n][l] = g.links[n][l][:0]
	for _, c := range g.selectNeighbors(cands, maxLinks) {
		g.links[n][l] = append(g.links[n][l], c.id)
	}
}

// selectNeighbors picks up to m links from cands (sorted best first) with the HNSW
// heuristic: a candidate is kept only if it is closer to the base point than to any
// already kept neighbour. This favours links in different directions ("highways"
// between clusters) over many links into one cluster. Free slots are then filled
// with the best discarded candidates.
func (g *HNSW) selectNeighbors(cands []node, m int) []node {
	var kept, discarded []node
	for _, c := range cands {
		if len(kept) >= m {
			break
		}
		good := true
		for _, k := range kept {
			if Dot(g.vecs[c.id], g.vecs[k.id]) > c.sim {
				good = false
				break
			}
		}
		if good {
			kept = append(kept, c)
		} else {
			discarded = append(discarded, c)
		}
	}
	for _, d := range discarded {
		if len(kept) >= m {
			break
		}
		kept = append(kept, d)
	}
	return kept
}

// greedy walks to the most similar neighbour until no neighbour improves (beam width 1).
func (g *HNSW) greedy(q []float32, cur int32, l int) int32 {
	best := Dot(q, g.vecs[cur])
	for changed := true; changed; {
		changed = false
		for _, n := range g.links[cur][l] {
			if s := Dot(q, g.vecs[n]); s > best {
				best, cur, changed = s, n, true
			}
		}
	}
	return cur
}

// searchLayer is a beam search on one level. candidates: max-heap of nodes to expand
// (best first); results: min-heap of the best ef nodes seen (worst at the root).
// Stops when the best unexpanded candidate is worse than the worst result.
// Returns results sorted best first.
func (g *HNSW) searchLayer(q []float32, eps []int32, ef, l int) []node {
	vs := g.getVisits()
	defer g.visits.Put(vs)
	cands, results := &maxNodeHeap{}, &nodeHeap{}
	for _, e := range eps {
		vs.mark[e] = vs.gen
		n := node{e, Dot(q, g.vecs[e])}
		heap.Push(cands, n)
		heap.Push(results, n)
	}
	for cands.Len() > 0 {
		c := heap.Pop(cands).(node)
		if results.Len() >= ef && c.sim < (*results)[0].sim {
			break
		}
		for _, n := range g.links[c.id][l] {
			if vs.mark[n] == vs.gen {
				continue
			}
			vs.mark[n] = vs.gen
			s := Dot(q, g.vecs[n])
			if results.Len() < ef || s > (*results)[0].sim {
				heap.Push(cands, node{n, s})
				heap.Push(results, node{n, s})
				if results.Len() > ef {
					heap.Pop(results)
				}
			}
		}
	}
	out := make([]node, results.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(results).(node)
	}
	return out
}

func (g *HNSW) Search(q []float32, k int) []Hit {
	q = normalized(q)
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.entry < 0 {
		return nil
	}
	cur := g.entry
	for l := g.maxLevel; l > 0; l-- {
		cur = g.greedy(q, cur, l)
	}
	res := g.searchLayer(q, []int32{cur}, max(g.EfSearch, k), 0)
	if len(res) > k {
		res = res[:k]
	}
	hits := make([]Hit, len(res))
	for i, n := range res {
		hits[i] = Hit{Index: int(n.id), Score: float64(n.sim)}
	}
	return hits
}

type node struct {
	id  int32
	sim float32
}

// nodeHeap is a min-heap by similarity.
type nodeHeap []node

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].sim < h[j].sim }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x any)        { *h = append(*h, x.(node)) }
func (h *nodeHeap) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// sortedHits drains the heap into hits sorted best first.
func (h *nodeHeap) sortedHits() []Hit {
	out := make([]Hit, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		n := heap.Pop(h).(node)
		out[i] = Hit{Index: int(n.id), Score: float64(n.sim)}
	}
	return out
}

// maxNodeHeap is a max-heap by similarity.
type maxNodeHeap struct{ nodeHeap }

func (h maxNodeHeap) Less(i, j int) bool { return h.nodeHeap[i].sim > h.nodeHeap[j].sim }

func normalized(v []float32) []float32 {
	c := append([]float32(nil), v...)
	Normalize(c)
	return c
}
