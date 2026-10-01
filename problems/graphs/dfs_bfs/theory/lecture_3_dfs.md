# Лекция 3: DFS, Связность и Топологическая Сортировка

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Depth-First Search (DFS)](#depth-first-search-dfs)
2. [Связные компоненты](#связные-компоненты)
3. [Обход деревьев](#обход-деревьев)
4. [Топологическая сортировка](#топологическая-сортировка)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Depth-First Search (DFS)

### Концепция

DFS (поиск в глубину) — это алгоритм обхода графа, который идёт **максимально глубоко** перед тем как вернуться назад.

```
        0
       /|\
      1 2 3
     /     \
    4       5

DFS порядок: 0 → 1 → 4 → 2 → 3 → 5
(или другие варианты в зависимости от порядка соседей)
```

### Рекурсивная реализация

```go
// DFS рекурсивная версия
func DFSRecursive(graph map[int][]int, node int, visited map[int]bool, result *[]int) {
	visited[node] = true
	*result = append(*result, node)
	
	for _, neighbor := range graph[node] {
		if !visited[neighbor] {
			DFSRecursive(graph, neighbor, visited, result)
		}
	}
}

// Использование
func DFS(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	result := []int{}
	DFSRecursive(graph, start, visited, &result)
	return result
}
```

### Итеративная реализация

```go
// DFS итеративная версия с явным стеком
func DFSIterative(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	stack := []int{start}
	result := []int{}
	
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		
		if !visited[node] {
			visited[node] = true
			result = append(result, node)
			
			// Добавляем соседей в стек (в обратном порядке для сохранения порядка)
			for i := len(graph[node]) - 1; i >= 0; i-- {
				neighbor := graph[node][i]
				if !visited[neighbor] {
					stack = append(stack, neighbor)
				}
			}
		}
	}
	
	return result
}
```

### Сложность DFS

| Метрика | Значение |
|---------|----------|
| **Время** | O(V + E) |
| **Память** | O(V) — стек рекурсии или явный стек |
| **Пройденные рёбра** | Все рёбра один раз |

### DFS vs BFS

```
                BFS                          DFS
┌──────────────────────────┬──────────────────────────┐
│ Использует очередь       │ Использует стек          │
│ Уровневый обход          │ Глубокий обход           │
│ Кратчайший путь          │ Проверка цикла           │
│ Меньше памяти для        │ Лучше для деревьев       │
│ сбалансированного графа  │                          │
└──────────────────────────┴──────────────────────────┘
```

---

## Связные компоненты

### Определение

**Связная компонента** — это максимальное подмножество вершин, где каждая вершина достижима из любой другой.

```
     1 - 2        4 - 5
      \ /         |
       3          6

Компоненты: {1, 2, 3}, {4, 5, 6}
```

### Поиск компонент

```go
// Найти все связные компоненты
func FindComponents(graph map[int][]int, numVertices int) [][]int {
	visited := make(map[int]bool)
	components := [][]int{}
	
	for i := 0; i < numVertices; i++ {
		if !visited[i] {
			component := []int{}
			dfsComponent(graph, i, visited, &component)
			components = append(components, component)
		}
	}
	
	return components
}

func dfsComponent(graph map[int][]int, node int, visited map[int]bool, component *[]int) {
	visited[node] = true
	*component = append(*component, node)
	
	for _, neighbor := range graph[node] {
		if !visited[neighbor] {
			dfsComponent(graph, neighbor, visited, component)
		}
	}
}

// Number of Islands - найти количество компонент в матрице
func NumIslands(grid [][]byte) int {
	if len(grid) == 0 {
		return 0
	}
	
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	
	count := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == '1' && !visited[i][j] {
				dfsIsland(grid, i, j, visited, m, n)
				count++
			}
		}
	}
	
	return count
}

func dfsIsland(grid [][]byte, i, j int, visited [][]bool, m, n int) {
	if i < 0 || i >= m || j < 0 || j >= n || visited[i][j] || grid[i][j] == '0' {
		return
	}
	
	visited[i][j] = true
	
	dfsIsland(grid, i+1, j, visited, m, n)
	dfsIsland(grid, i-1, j, visited, m, n)
	dfsIsland(grid, i, j+1, visited, m, n)
	dfsIsland(grid, i, j-1, visited, m, n)
}
```

---

## Обход деревьев

### Pre-order, In-order, Post-order

```
      1
     / \
    2   3
   / \
  4   5

Pre-order (корень - левое - правое):     1, 2, 4, 5, 3
In-order (левое - корень - правое):      4, 2, 5, 1, 3
Post-order (левое - правое - корень):    4, 5, 2, 3, 1
```

### Реализация на Go

```go
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// Pre-order обход
func PreorderTraversal(root *TreeNode) []int {
	result := []int{}
	preorder(root, &result)
	return result
}

func preorder(node *TreeNode, result *[]int) {
	if node == nil {
		return
	}
	*result = append(*result, node.Val)
	preorder(node.Left, result)
	preorder(node.Right, result)
}

// In-order обход (для BST дает отсортированный результат)
func InorderTraversal(root *TreeNode) []int {
	result := []int{}
	inorder(root, &result)
	return result
}

func inorder(node *TreeNode, result *[]int) {
	if node == nil {
		return
	}
	inorder(node.Left, result)
	*result = append(*result, node.Val)
	inorder(node.Right, result)
}

// Post-order обход
func PostorderTraversal(root *TreeNode) []int {
	result := []int{}
	postorder(root, &result)
	return result
}

func postorder(node *TreeNode, result *[]int) {
	if node == nil {
		return
	}
	postorder(node.Left, result)
	postorder(node.Right, result)
	*result = append(*result, node.Val)
}

// Итеративный In-order обход
func InorderIterative(root *TreeNode) []int {
	result := []int{}
	stack := []*TreeNode{}
	current := root
	
	for current != nil || len(stack) > 0 {
		for current != nil {
			stack = append(stack, current)
			current = current.Left
		}
		
		current = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		result = append(result, current.Val)
		current = current.Right
	}
	
	return result
}
```

---

## Топологическая сортировка

### Определение

**Топологическая сортировка** — это линейное упорядочение вершин ориентированного ациклического графа (DAG) такое, что для каждого ребра (u, v), вершина u предшествует v.

```
Граф:
  1 → 2 → 3
  1 → 4

Топологическая сортировка: 1, 2, 4, 3
или: 1, 4, 2, 3
```

### Алгоритм (DFS-based)

```go
// Топологическая сортировка используя DFS
func TopologicalSort(graph map[int][]int, numVertices int) []int {
	visited := make(map[int]bool)
	stack := []int{}
	
	for i := 0; i < numVertices; i++ {
		if !visited[i] {
			topoDFS(graph, i, visited, &stack)
		}
	}
	
	// Обратить стек
	result := make([]int, len(stack))
	for i, v := range stack {
		result[len(stack)-1-i] = v
	}
	
	return result
}

func topoDFS(graph map[int][]int, node int, visited map[int]bool, stack *[]int) {
	visited[node] = true
	
	for _, neighbor := range graph[node] {
		if !visited[neighbor] {
			topoDFS(graph, neighbor, visited, stack)
		}
	}
	
	*stack = append(*stack, node)
}

// Kahn's Algorithm (BFS-based)
func TopologicalSortKahn(graph map[int][]int, numVertices int) []int {
	// Подсчитать in-degree
	inDegree := make(map[int]int)
	for i := 0; i < numVertices; i++ {
		inDegree[i] = 0
	}
	
	for i := 0; i < numVertices; i++ {
		for _, neighbor := range graph[i] {
			inDegree[neighbor]++
		}
	}
	
	// Найти вершины с in-degree = 0
	queue := []int{}
	for i := 0; i < numVertices; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	
	result := []int{}
	
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)
		
		for _, neighbor := range graph[node] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	
	if len(result) != numVertices {
		// Граф содержит цикл
		return nil
	}
	
	return result
}
```

### Пример: Course Schedule

```go
// Проверить возможность прохождения всех курсов с заданными зависимостями
func CanFinish(numCourses int, prerequisites [][]int) bool {
	// Построить граф
	graph := make(map[int][]int)
	inDegree := make(map[int]int)
	
	for i := 0; i < numCourses; i++ {
		graph[i] = []int{}
		inDegree[i] = 0
	}
	
	for _, prereq := range prerequisites {
		course := prereq[0]
		preq := prereq[1]
		graph[preq] = append(graph[preq], course)
		inDegree[course]++
	}
	
	// Kahn's algorithm
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	
	count := 0
	
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		count++
		
		for _, neighbor := range graph[node] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	
	return count == numCourses
}

// Найти порядок прохождения курсов
func FindOrder(numCourses int, prerequisites [][]int) []int {
	// Аналогично CanFinish, но возвращаем сам порядок
	graph := make(map[int][]int)
	inDegree := make(map[int]int)
	
	for i := 0; i < numCourses; i++ {
		graph[i] = []int{}
		inDegree[i] = 0
	}
	
	for _, prereq := range prerequisites {
		course := prereq[0]
		preq := prereq[1]
		graph[preq] = append(graph[preq], course)
		inDegree[course]++
	}
	
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if inDegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	
	result := []int{}
	
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)
		
		for _, neighbor := range graph[node] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}
	
	if len(result) != numCourses {
		return []int{} // Цикл существует
	}
	
	return result
}
```

---

## Практические примеры

### Пример 1: Max Area of Island

```go
// Найти максимальную площадь острова
func MaxAreaOfIsland(grid [][]int) int {
	if len(grid) == 0 {
		return 0
	}
	
	m, n := len(grid), len(grid[0])
	maxArea := 0
	
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 {
				area := dfsArea(grid, i, j, m, n)
				if area > maxArea {
					maxArea = area
				}
			}
		}
	}
	
	return maxArea
}

func dfsArea(grid [][]int, i, j, m, n int) int {
	if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0 {
		return 0
	}
	
	grid[i][j] = 0 // Помечаем как посещённое
	
	area := 1
	area += dfsArea(grid, i+1, j, m, n)
	area += dfsArea(grid, i-1, j, m, n)
	area += dfsArea(grid, i, j+1, m, n)
	area += dfsArea(grid, i, j-1, m, n)
	
	return area
}
```

### Пример 2: Detect Cycle in Undirected Graph

```go
// Проверить наличие цикла в неориентированном графе
func HasCycle(n int, edges [][]int) bool {
	graph := make(map[int][]int)
	for i := 0; i < n; i++ {
		graph[i] = []int{}
	}
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	
	visited := make(map[int]bool)
	
	for i := 0; i < n; i++ {
		if !visited[i] {
			if dfsCycle(graph, i, -1, visited) {
				return true
			}
		}
	}
	
	return false
}

func dfsCycle(graph map[int][]int, node, parent int, visited map[int]bool) bool {
	visited[node] = true
	
	for _, neighbor := range graph[node] {
		if !visited[neighbor] {
			if dfsCycle(graph, neighbor, node, visited) {
				return true
			}
		} else if neighbor != parent {
			return true // Найден цикл
		}
	}
	
	return false
}
```

---

## Анализ сложности

| Алгоритм | Время | Память |
|----------|-------|--------|
| **DFS** | O(V + E) | O(V) |
| **Найти компоненты** | O(V + E) | O(V) |
| **Топо-сортировка DFS** | O(V + E) | O(V) |
| **Топо-сортировка Kahn** | O(V + E) | O(V) |
| **Проверка цикла** | O(V + E) | O(V) |

---

## Рекомендуемые задачи на LeetCode

- [Number of Islands](https://leetcode.com/problems/number-of-islands/) — поиск компонент
- [Max Area of Island](https://leetcode.com/problems/max-area-of-island/) — DFS для вычисления
- [Course Schedule](https://leetcode.com/problems/course-schedule/) — топо-сортировка
- [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) — найти порядок
- [Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) — обход дерева
- [Detect Cycle in Directed Graph](https://leetcode.com/problems/detect-cycle-in-a-directed-graph/) — DFS
- [Surrounded Regions](https://leetcode.com/problems/surrounded-regions/) — DFS на матрице

---

## Заключение

DFS — это фундаментальный алгоритм с множеством приложений:

1. **Поиск компонент связности** — разделение графа на подграфы
2. **Обход деревьев** — обработка иерархических данных
3. **Топологическая сортировка** — расписания, зависимости
4. **Обнаружение циклов** — проверка ацикличности
5. **Поиск путей** — существование пути между вершинами

Ключевое отличие от BFS: DFS идёт глубоко, что полезно для структурированных данных, а не для поиска кратчайшего пути.

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
