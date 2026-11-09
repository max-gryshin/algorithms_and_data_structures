# Лекция 1: BFS, Дейкстра, и Основы Теории Графов

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в графы](#введение-в-графы)
2. [Представление графов](#представление-графов)
3. [Breadth-First Search (BFS)](#breadth-first-search-bfs)
4. [Поиск кратчайшего пути в невзвешенных графах](#поиск-кратчайшего-пути-в-невзвешенных-графах)
5. [Алгоритм Дейкстры](#алгоритм-дейкстры)
6. [Практические примеры](#практические-примеры)
7. [Анализ сложности](#анализ-сложности)

---

## Введение в графы

### Что такое граф?

Граф — это одна из самых универсальных и мощных структур данных в информатике. Граф G = (V, E) состоит из:

- **V** (Vertices/Вершины) — множество узлов
- **E** (Edges/Рёбра) — множество связей между вершинами

#### Классификация графов:

```
┌─────────────────┬──────────────────┐
│     Граф        │  Характеристики  │
├─────────────────┼──────────────────┤
│ Ориентированный │  (u→v) ≠ (v→u)  │
│ Неориентир.     │  (u,v) = (v,u)  │
│ Взвешенный      │  каждое ребро   │
│                 │  имеет вес       │
│ Невзвешенный    │  все веса = 1    │
│ Циклический     │  содержит циклы  │
│ Ациклический    │  нет циклов      │
└─────────────────┴──────────────────┘
```

### Примеры графов в реальной жизни:

- **Социальные сети**: вершины = люди, рёбра = дружба
- **Дорожные сети**: вершины = города, рёбра = дороги (с расстояниями)
- **Интернет**: вершины = серверы, рёбра = соединения
- **Молекулярные структуры**: вершины = атомы, рёбра = химические связи

---

## Представление графов

В этом курсе мы рассматриваем два основных способа представления:

### 1. Список смежности (Adjacency List) — рекомендуется

```go
// Неориентированный граф
graph := map[int][]int{
    0: {1, 2},
    1: {0, 3},
    2: {0, 4},
    3: {1},
    4: {2},
}

// Ориентированный граф
digraph := map[int][]int{
    0: {1, 2},
    1: {3},
    2: {4},
    3: {},
    4: {},
}
```

**Преимущества**:
- O(1) доступ к соседям вершины
- Эффективен для разреженных графов (sparse graphs)
-占 O(V + E) памяти

**Недостатки**:
- Проверка существования ребра занимает O(degree(v)) времени

### 2. Матрица смежности (Adjacency Matrix)

```go
// Неориентированный граф
matrix := [][]int{
    {0, 1, 1, 0, 0},
    {1, 0, 0, 1, 0},
    {1, 0, 0, 0, 1},
    {0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0},
}
```

**Преимущества**:
- O(1) проверка существования ребра
- Удобна для плотных графов (dense graphs)

**Недостатки**:
- O(V²) памяти даже для разреженных графов
- Неэффективна для больших графов

---

## Breadth-First Search (BFS)

### Концепция

BFS — это алгоритм обхода графа "в ширину". Начиная с исходной вершины, мы посещаем **все вершины на расстоянии k перед тем, как посетить вершины на расстоянии k+1**.

```
      0
     / \
    1   2
   /     \
  3       4

BFS порядок: 0 → 1 → 2 → 3 → 4
```

### Алгоритм

```
BFS(graph, start):
    queue = [start]
    visited = {start}
    
    while queue not empty:
        node = queue.pop_front()
        process(node)
        
        for neighbor in graph[node]:
            if neighbor not visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### Реализация на Go

```go
// BFS обход графа в ширину
func BFS(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	queue := []int{start}
	visited[start] = true
	result := []int{}
	
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)
		
		for _, neighbor := range graph[node] {
			if !visited[neighbor] {
				visited[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	
	return result
}
```

### Сложность

| Метрика | Значение |
|---------|----------|
| **Время** | O(V + E) |
| **Память** | O(V) |
| **Пройденные рёбра** | все рёбра один раз |

**Почему O(V + E)?**
- Каждая вершина добавляется в queue ровно один раз: O(V)
- Каждое ребро проверяется один раз: O(E)

---

## Поиск кратчайшего пути в невзвешенных графах

### Задача: Shortest Path in Binary Matrix

**Формулировка**: Дана матрица m×n, содержащая 0 (проходимо) и 1 (препятствие). Найти кратчайший путь из верхнего левого угла в нижний правый.

```
Input:
0 0 1
0 0 0
1 0 0

Output: 4
Path: (0,0) → (0,1) → (1,1) → (1,2) → (2,2)
```

### Почему BFS?

В невзвешенных графах **BFS всегда находит кратчайший путь**. Это потому что:
- BFS посещает вершины в порядке расстояния от источника
- Первый раз, когда мы достигаем целевую вершину — это кратчайший путь

```go
// Найти кратчайший путь в бинарной матрице
// Time: O(m*n), Space: O(m*n)
func ShortestPathBinaryMatrix(grid [][]int) int {
	if len(grid) == 0 || grid[0][0] == 1 {
		return -1
	}
	
	m, n := len(grid), len(grid[0])
	
	if m == 1 && n == 1 {
		return 0
	}
	
	// 8 направлений (вверх, вниз, влево, вправо + диагонали)
	directions := [][2]int{
		{-1, -1}, {-1, 0}, {-1, 1},
		{0, -1}, {0, 1},
		{1, -1}, {1, 0}, {1, 1},
	}
	
	type cell struct {
		row, col, dist int
	}
	
	queue := []cell{{0, 0, 1}}
	grid[0][0] = 1 // Помечаем как посещённое
	
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		
		for _, d := range directions {
			newRow, newCol := cur.row+d[0], cur.col+d[1]
			
			if newRow >= 0 && newRow < m && newCol >= 0 && newCol < n && grid[newRow][newCol] == 0 {
				if newRow == m-1 && newCol == n-1 {
					return cur.dist + 1
				}
				
				grid[newRow][newCol] = 1
				queue = append(queue, cell{newRow, newCol, cur.dist + 1})
			}
		}
	}
	
	return -1
}
```

---

## Алгоритм Дейкстры

### Проблема невзвешенного BFS

BFS работает отлично для невзвешенных графов, но что делать, если рёбра имеют разные весы?

```
     (1)       (5)
  A ────→ B ────→ C

BFS выберет: A → B → C (расстояние: 6)
Оптимально:  A → B → C (расстояние: 6) ✓

Но если добавить:
     (1)       (5)
  A ────→ B ────→ C
  └──────────────┘
      (7)

BFS: A → B → C (расстояние: 6)
Оптимально: A → C (расстояние: 7) или A → B → C (6) ✓
```

### Алгоритм Дейкстры

Дейкстра использует **жадный подход** с приоритетной очередью:

1. Начиная с источника, всегда выбираем непосещённую вершину с **минимальным расстоянием**
2. Обновляем расстояния до её соседей
3. Повторяем до тех пор, пока не посетим все вершины (или целевую вершину)

```
Инвариант: При выборе вершины u из приоритетной очереди,
dist[u] = кратчайшее расстояние от source до u
```

### Псевдокод

```
Dijkstra(graph, source):
    dist = {v: ∞ for v in graph}
    dist[source] = 0
    pq = [(0, source)]  # (расстояние, вершина)
    visited = set()
    
    while pq not empty:
        current_dist, u = pq.pop_min()
        
        if u in visited:
            continue
        
        visited.add(u)
        
        for v in neighbors(graph, u):
            if dist[u] + weight(u,v) < dist[v]:
                dist[v] = dist[u] + weight(u,v)
                pq.push((dist[v], v))
    
    return dist
```

### Реализация на Go

```go
import "container/heap"

type edge struct {
	vertex, weight int
}

// Найти кратчайшие пути от start до всех вершин
func Dijkstra(graph map[int][]edge, start int) map[int]int {
	distances := make(map[int]int)
	for vertex := range graph {
		distances[vertex] = math.MaxInt
	}
	distances[start] = 0
	
	pq := &PriorityQueue{}
	heap.Push(pq, &Item{value: start, priority: 0})
	
	for pq.Len() > 0 {
		item := heap.Pop(pq).(*Item)
		currentDist := item.priority
		currentVertex := item.value
		
		if currentDist > distances[currentVertex] {
			continue
		}
		
		for _, e := range graph[currentVertex] {
			distance := currentDist + e.weight
			
			if distance < distances[e.vertex] {
				distances[e.vertex] = distance
				heap.Push(pq, &Item{value: e.vertex, priority: distance})
			}
		}
	}
	
	return distances
}

// Структура для приоритетной очереди
type Item struct {
	value    int
	priority int
	index    int
}

type PriorityQueue []*Item

func (pq PriorityQueue) Len() int           { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].priority < pq[j].priority }
func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	item := x.(*Item)
	item.index = len(*pq)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1
	*pq = old[0 : n-1]
	return item
}
```

### Пример: Network Delay Time

**Задача**: В сети из N узлов есть задержки. Сигнал отправляется из узла K. Найти время, за которое все узлы получат сигнал.

```go
// Найти время доставки сигнала до всех узлов
func NetworkDelayTime(times [][]int, n, k int) int {
	// Построить граф
	graph := make(map[int][]edge)
	for i := 1; i <= n; i++ {
		graph[i] = []edge{}
	}
	
	for _, time := range times {
		u, v, w := time[0], time[1], time[2]
		graph[u] = append(graph[u], edge{v, w})
	}
	
	// Дейкстра
	distances := make(map[int]int)
	for i := 1; i <= n; i++ {
		distances[i] = math.MaxInt
	}
	distances[k] = 0
	
	pq := &PriorityQueue{}
	heap.Push(pq, &Item{value: k, priority: 0})
	
	for pq.Len() > 0 {
		item := heap.Pop(pq).(*Item)
		currentDist := item.priority
		u := item.value
		
		if currentDist > distances[u] {
			continue
		}
		
		for _, e := range graph[u] {
			if currentDist+e.weight < distances[e.vertex] {
				distances[e.vertex] = currentDist + e.weight
				heap.Push(pq, &Item{value: e.vertex, priority: distances[e.vertex]})
			}
		}
	}
	
	maxTime := 0
	for i := 1; i <= n; i++ {
		if distances[i] == math.MaxInt {
			return -1
		}
		if distances[i] > maxTime {
			maxTime = distances[i]
		}
	}
	return maxTime
}

// Пример использования:
// times := [][]int{{1,2,1},{2,3,2},{1,3,4}}
// n, k := 3, 1
// fmt.Println(NetworkDelayTime(times, n, k)) // Output: 4
```

---

## Практические примеры

### Пример 1: Поиск уровней вершин в графе

```go
// Найти уровень (расстояние) каждой вершины от start
func GetLevels(graph map[int][]int, start int) map[int]int {
	levels := make(map[int]int)
	levels[start] = 0
	queue := []int{start}
	
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		
		for _, neighbor := range graph[node] {
			if _, exists := levels[neighbor]; !exists {
				levels[neighbor] = levels[node] + 1
				queue = append(queue, neighbor)
			}
		}
	}
	
	return levels
}
```

### Пример 2: Все кратчайшие пути

```go
// Найти все кратчайшие пути от start до end
func AllShortestPaths(graph map[int][]edge, start, end int) [][]int {
	distances := make(map[int]int)
	parents := make(map[int][]int)
	
	for vertex := range graph {
		distances[vertex] = math.MaxInt
	}
	distances[start] = 0
	
	pq := &PriorityQueue{}
	heap.Push(pq, &Item{value: start, priority: 0})
	
	for pq.Len() > 0 {
		item := heap.Pop(pq).(*Item)
		currentDist := item.priority
		u := item.value
		
		if currentDist > distances[u] {
			continue
		}
		
		for _, e := range graph[u] {
			v := e.vertex
			newDist := currentDist + e.weight
			
			if newDist < distances[v] {
				distances[v] = newDist
				parents[v] = []int{u}
				heap.Push(pq, &Item{value: v, priority: newDist})
			} else if newDist == distances[v] {
				parents[v] = append(parents[v], u)
			}
		}
	}
	
	// Восстановить все пути
	var paths [][]int
	var backtrack func(int, []int)
	backtrack = func(node int, path []int) {
		if node == start {
			newPath := make([]int, 0, len(path)+1)
			newPath = append(newPath, start)
			newPath = append(newPath, path...)
			paths = append(paths, newPath)
			return
		}
		for _, parent := range parents[node] {
			newPath := make([]int, len(path)+1)
			newPath[0] = node
			copy(newPath[1:], path)
			backtrack(parent, newPath)
		}
	}
	
	backtrack(end, []int{})
	return paths
}
```

---

## Анализ сложности

### BFS

| Аспект | Сложность | Примечание |
|--------|-----------|-----------|
| **Время** | O(V + E) | Каждая вершина и ребро посещаются один раз |
| **Память** | O(V) | Очередь + множество посещённых |
| **Кратчайший путь** | O(V + E) | Гарантирует в невзвешенных графах |

### Дейкстра

| Аспект | Сложность | Примечание |
|--------|-----------|-----------|
| **Время (мин-куча)** | O((V+E) log V) | Куча размера V, операции O(log V) |
| **Время (Фибоначчи)** | O(E + V log V) | Теоретический оптимум |
| **Память** | O(V) | Куча + расстояния |
| **Предусловие** | Неотрицательные веса | Отказывает при отрицательных весах! |

### Когда использовать что?

```
Невзвешенный граф?
  ↓
  └→ BFS (O(V+E))
  
Взвешенный граф, неотрицательные веса?
  ↓
  └→ Дейкстра (O((V+E) log V))
  
Взвешенный граф, отрицательные веса?
  ↓
  └→ Беллман-Форд (лекция 11)
```

---

## Практические советы

### 1. Выбор структуры данных
- **Список смежности** (~99% случаев) — эффективен, гибкий
- **Матрица смежности** — когда граф очень плотный (E ≈ V²)

### 2. Оптимизации в реальных приложениях
- Используйте **bidirectional BFS** для поиска на больших графах
- Для Дейкстры рассмотрите **A* algorithm** если есть эвристика
- Кэшируйте результаты для повторяющихся запросов

### 3. Распространённые ошибки
- ❌ Забыть пометить вершину как посещённую (бесконечный цикл)
- ❌ Использовать Дейкстру с отрицательными весами
- ❌ Нарушить инвариант приоритетной очереди

---

## Заключение

BFS и алгоритм Дейкстры — это фундаментальные алгоритмы на графах, которые встречаются в огромном числе приложений:

- **Социальные сети**: рекомендации "друзья друзей"
- **GPS навигация**: поиск кратчайшего маршрута
- **Игры**: поиск кратчайшего пути для NPC
- **Телекоммуникации**: оптимальная маршрутизация пакетов

Овладение этими алгоритмами — критически важно для любого, кто работает в Computer Science.

---

## Рекомендуемые задачи для практики

На [LeetCode](https://leetcode.com):
- [Number of Islands](https://leetcode.com/problems/number-of-islands/) — BFS/DFS
- [Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/) — BFS
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/) — Дейкстра
- [Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) — Дейкстра
- [Word Ladder](https://leetcode.com/problems/word-ladder/) — BFS
- [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) — модифицированный Дейкстра

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
