package theory

const MaxLog = 20

type BinaryLiftingLCA struct {
	up    [][]int
	depth []int
	n     int
	log   int
}

func NewBinaryLiftingLCA(n int) *BinaryLiftingLCA {
	return &BinaryLiftingLCA{
		up:    make([][]int, n),
		depth: make([]int, n),
		n:     n,
		log:   MaxLog,
	}
}

// Build - предварительная обработка O(N log N):
//   - Заполняет up[v][i] для всех вершин
//   - up[v][i] = предок v на расстоянии 2^i
func (bl *BinaryLiftingLCA) Build(tree []Node, root int) {
	for i := 0; i < bl.n; i++ {
		bl.up[i] = make([]int, bl.log)
	}

	bl.dfs(root, -1, tree)
}

func (bl *BinaryLiftingLCA) dfs(u, parent int, tree []Node) {
	bl.up[u][0] = parent

	for i := 1; i < bl.log; i++ {
		if bl.up[u][i-1] != -1 {
			bl.up[u][i] = bl.up[bl.up[u][i-1]][i-1]
		} else {
			bl.up[u][i] = -1
		}
	}

	for _, v := range tree[u].Children {
		bl.depth[v] = bl.depth[u] + 1
		bl.dfs(v, u, tree)
	}
}

// LiftUpподнять вершину u на steps шагов вверх:
// - Разбирает steps на биты (двоичное представление)
// - Для каждого установленного бита делает прыжок на 2^i
// - Выравнивает по глубине (поднимает глубокого)
// - Двоичным поиском поднимает обоих, пока они не совпадут
// - Возвращает родителя последней позиции
func (bl *BinaryLiftingLCA) LiftUp(u, steps int) int {
	for i := 0; i < bl.log; i++ {
		if (steps>>uint(i))&1 > 0 {
			u = bl.up[u][i]
			if u == -1 {
				return -1
			}
		}
	}
	return u
}

func (bl *BinaryLiftingLCA) Query(u, v int) int {
	if bl.depth[u] < bl.depth[v] {
		u, v = v, u
	}

	u = bl.LiftUp(u, bl.depth[u]-bl.depth[v])

	if u == v {
		return u
	}

	for i := bl.log - 1; i >= 0; i-- {
		if bl.up[u][i] != bl.up[v][i] {
			u = bl.up[u][i]
			v = bl.up[v][i]
		}
	}

	return bl.up[u][0]
}
