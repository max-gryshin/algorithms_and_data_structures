package theory

import (
	"slices"
	"testing"
)

func TestLCA(t *testing.T) {
	// Создадим дерево из 8 вершин (0..7). Корень — 0.
	//         0
	//       /   \
	//      1     2
	//    / | \    \
	//   3  4  5    6
	//               \
	//                7
	table := []struct {
		numVertices     int
		queries         []Query
		tree            func(numVertices int, queries []Query) []Node
		answers         func(queriesNum int) []int
		answersExpected []int
		dsu             func(numVertices int) *DisjointSetUnionLCA
	}{
		{
			numVertices: 8,
			// Список offline запросов (пары вершин, у которых ищем LCA)
			queries: []Query{
				{Index: 0, U: 3, V: 5}, // Ожидаемый ответ: 1
				{Index: 1, U: 4, V: 7}, // Ожидаемый ответ: 0
				{Index: 2, U: 6, V: 7}, // Ожидаемый ответ: 2
				{Index: 3, U: 3, V: 4}, // Ожидаемый ответ: 1
			},
			tree: func(numVertices int, queries []Query) []Node {
				return buildTree(numVertices, queries)
			},
			answers: func(queriesNum int) []int {
				return make([]int, queriesNum)
			},
			answersExpected: []int{1, 0, 6, 1},
			dsu: func(numVertices int) *DisjointSetUnionLCA {
				return NewDisjointSetUnionLCA(numVertices)
			},
		},
	}
	for _, row := range table {
		dsu := row.dsu(row.numVertices)
		answers := row.answers(len(row.queries))
		dsu.LCA(0, row.tree(row.numVertices, row.queries), answers)
		if !slices.Equal(answers, row.answersExpected) {
			t.Errorf("expected answer %v, but got %v", row.answersExpected, answers)
		}
	}

	//fmt.Println("Результаты поиска наименьшего общего предка (LCA):")
	//for i, q := range queries {
	//	fmt.Printf("LCA для вершин (%d, %d) => %d\n", q.U, q.V, answers[i])
	//}
}

func buildTree(numVertices int, queries []Query) []Node {
	tree := make([]Node, numVertices)
	for i := 0; i < numVertices; i++ {
		tree[i] = Node{ID: i}
	}

	// Построение связей дерева (отец -> сын)
	tree[0].Children = []int{1, 2}
	tree[1].Children = []int{3, 4, 5}
	tree[2].Children = []int{6}
	tree[6].Children = []int{7}

	// Распределяем запросы по спискам смежности вершин (чтобы при обработке u проверить v)
	for _, query := range queries {
		tree[query.U].Queries = append(tree[query.U].Queries, query)
		tree[query.V].Queries = append(tree[query.V].Queries, query)
	}

	return tree
}
