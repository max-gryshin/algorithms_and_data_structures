package theory

type Color int

const (
	White Color = iota
	Gray
	Black
)

// Query представляет запрос на поиск LCA между двумя вершинами u и v
type Query struct {
	Index int // Оригинальный индекс запроса для сохранения порядка ответов
	U, V  int
}

// Node представляет вершину дерева со списками смежности для ребер и запросов
type Node struct {
	ID       int
	Children []int
	Queries  []Query
}

// DisjointSetUnionLCA расширенная DSU
type DisjointSetUnionLCA struct {
	DSU
	Ancestor []int   // Дополнительный массив для алгоритма Тарьяна
	Visited  []Color // Отслеживание полностью обработанных вершин
}

func NewDisjointSetUnionLCA(n int) *DisjointSetUnionLCA {
	return &DisjointSetUnionLCA{
		DSU: DSU{
			Parent: make([]int, n),
			Rank:   make([]int, n),
		},
		Ancestor: make([]int, n),
		Visited:  make([]Color, n), // White by default
	}
}

func (dsu *DisjointSetUnionLCA) MakeSet(x int) {
	dsu.Parent[x] = x
	dsu.Rank[x] = 0
	dsu.Ancestor[x] = x
	dsu.Visited[x] = White
}

func (dsu *DisjointSetUnionLCA) LCA(u int, tree []Node, answers []int) {
	// MAKE-SET(u)
	dsu.MakeSet(u)
	// ANCESTOR[FIND-SET(u)] = u
	dsu.Ancestor[dsu.FindSet(u)] = u

	// for each child v of u in T
	for _, v := range tree[u].Children {
		dsu.LCA(v, tree, answers)

		// UNION(u, v)
		dsu.Union(u, v)

		// ANCESTOR[FIND-SET(u)] = u
		// После объединения корнем DSU мог стать v,
		// но фактическим предком текущей ветки всё еще остается u!
		dsu.Ancestor[dsu.FindSet(u)] = u
	}

	dsu.Visited[u] = Black

	// for each node v such that {u, v} ∈ P (проверяем запросы для вершины u)
	for _, query := range tree[u].Queries {
		v := query.V
		if u == query.V { // Если запрос вида (u, u)
			v = query.U
		}

		if dsu.Visited[v] == Black {
			// Результат: ANCESTOR[FIND-SET(v)]
			answers[query.Index] = dsu.Ancestor[dsu.FindSet(v)]
		}
	}
}
