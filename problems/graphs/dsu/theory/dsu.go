package theory

// Disjoint Set Union
type DSU struct {
	parent []int
	rank   []int
}

func NewDSU(n int) *DSU {
	parent := make([]int, n)
	rank := make([]int, n)

	for i := range parent {
		parent[i] = i
	}

	return &DSU{
		parent: parent,
		rank:   rank,
	}
}

func (d *DSU) Find(x int) int {
	if d.parent[x] != x {
		// Сглаживаем дерево, перенаправляя узел к корню
		d.parent[x] = d.Find(d.parent[x]) // path compression
	}

	return d.parent[x]
}

func (d *DSU) Union(x, y int) bool {
	px := d.Find(x)
	py := d.Find(y)

	if px == py {
		return false
	}

	if d.rank[px] < d.rank[py] {
		d.parent[px] = py
	} else if d.rank[px] > d.rank[py] {
		d.parent[py] = px
	} else {
		d.parent[py] = px
		d.rank[px]++
	}
	return true
}

func (d *DSU) Connect(x, y int) bool {
	return d.Find(x) == d.Find(y)
}
