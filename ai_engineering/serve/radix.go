package serve

// RadixCache is a prefix cache over token ids (RadixAttention, SGLang, Zheng et al., 2023):
// a compressed trie whose edges are token runs; every node holds the KV blocks of its
// edge. Edges are split only at block boundaries, so a node always owns whole blocks
// (a partially filled last block is never shared). The cache holds one reference to each
// block it stores; a node is evictable when nobody else references its blocks (refcount 1).
type RadixCache struct {
	alloc *BlockAllocator
	bs    int
	root  *radixNode
	clock int64
}

type radixNode struct {
	key      []int
	blocks   []int
	children []*radixNode
	parent   *radixNode
	lastUse  int64
}

func NewRadixCache(alloc *BlockAllocator) *RadixCache {
	return &RadixCache{alloc: alloc, bs: alloc.BlockSize, root: &radixNode{}}
}

// common returns the length of the common prefix of a and b rounded down to whole blocks.
func (c *RadixCache) common(a, b []int) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n / c.bs * c.bs
}

// Match returns the longest cached block-aligned prefix of tokens and its blocks. The
// caller receives its own reference to every returned block (Release them when done).
func (c *RadixCache) Match(tokens []int) (n int, blocks []int) {
	c.clock++
	node := c.root
	for {
		var next *radixNode
		cp := 0
		for _, ch := range node.children {
			if cp = c.common(ch.key, tokens[n:]); cp > 0 {
				next = ch
				break
			}
		}
		if next == nil {
			break
		}
		next.lastUse = c.clock
		blocks = append(blocks, next.blocks[:cp/c.bs]...)
		n += cp
		if cp < len(next.key) {
			break
		}
		node = next
	}
	c.alloc.Retain(blocks...)
	return n, blocks
}

// Insert stores the full blocks of tokens (len(blocks) ≥ len(tokens)/bs). Blocks of a
// prefix that is already cached are not inserted; new ones get a cache reference.
func (c *RadixCache) Insert(tokens, blocks []int) {
	c.clock++
	tokens = tokens[:len(tokens)/c.bs*c.bs]
	node, n := c.root, 0
	for n < len(tokens) {
		var next *radixNode
		cp := 0
		for _, ch := range node.children {
			if cp = c.common(ch.key, tokens[n:]); cp > 0 {
				next = ch
				break
			}
		}
		if next == nil {
			leaf := &radixNode{key: append([]int(nil), tokens[n:]...), parent: node, lastUse: c.clock,
				blocks: append([]int(nil), blocks[n/c.bs:len(tokens)/c.bs]...)}
			c.alloc.Retain(leaf.blocks...)
			node.children = append(node.children, leaf)
			return
		}
		if cp < len(next.key) { // split the edge at a block boundary
			mid := &radixNode{key: next.key[:cp], blocks: next.blocks[:cp/c.bs], parent: node,
				children: []*radixNode{next}}
			for i, ch := range node.children {
				if ch == next {
					node.children[i] = mid
				}
			}
			next.key, next.blocks, next.parent = next.key[cp:], next.blocks[cp/c.bs:], mid
			next = mid
		}
		next.lastUse = c.clock
		node, n = next, n+cp
	}
}

// Evict frees up to want blocks by removing least recently used leaves whose blocks are
// referenced only by the cache. It returns the number of blocks freed. O(nodes) per
// eviction: production code keeps the leaves in a heap.
func (c *RadixCache) Evict(want int) int {
	freed := 0
	for freed < want {
		var victim *radixNode
		var walk func(*radixNode)
		walk = func(n *radixNode) {
			for _, ch := range n.children {
				walk(ch)
			}
			if n == c.root || len(n.children) > 0 {
				return
			}
			for _, b := range n.blocks {
				if c.alloc.RefCount(b) != 1 {
					return
				}
			}
			if victim == nil || n.lastUse < victim.lastUse {
				victim = n
			}
		}
		walk(c.root)
		if victim == nil {
			break
		}
		c.alloc.Release(victim.blocks...)
		freed += len(victim.blocks)
		p := victim.parent
		for i, ch := range p.children {
			if ch == victim {
				p.children = append(p.children[:i], p.children[i+1:]...)
				break
			}
		}
	}
	return freed
}

// CachedBlocks is the number of blocks held by the tree.
func (c *RadixCache) CachedBlocks() int {
	var count func(*radixNode) int
	count = func(n *radixNode) int {
		s := len(n.blocks)
		for _, ch := range n.children {
			s += count(ch)
		}
		return s
	}
	return count(c.root)
}
