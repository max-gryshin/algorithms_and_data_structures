# Лекция 9: Minimum Spanning Tree (Краскал и Прим)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в MST](#введение-в-mst)
2. [Алгоритм Краскала](#алгоритм-краскала)
3. [Алгоритм Прима](#алгоритм-прима)
4. [Сравнение методов](#сравнение-методов)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Введение в MST

### Определение

**Minimum Spanning Tree (MST)** — это подмножество рёбер графа, которое:

1. **Связывает все вершины** (spanning tree)
2. **Не содержит циклов** (tree)
3. **Имеет минимальный общий вес** (minimum)

```
Граф:          MST:
  1---5---2      1---5
  |   |   |      |
  8   1   2  =>  8  1---3
  |   |   |      |       |
  3---2---4      3---2---4

Общий вес MST: 1+5+1+2 = 9
```

### Свойства MST

```
1. Уникальность рёбер (если все веса разные)
2. Может быть несколько MST (при одинаковых весах)
3. MST всегда имеет V-1 рёбер
4. Не содержит циклов
5. Дерево имеет диаметр ≤ диаметра исходного графа
```

### Приложения MST

- **Проектирование сетей** — минимальная стоимость соединения узлов
- **Кластеризация** — групирование похожих объектов
- **Сжатие данных** — алгоритмы кодирования
- **Генетические алгоритмы** — построение генеалогических деревьев
- **Маршрутизация** — оптимальная сетевая структура

---

## Алгоритм Краскала

### Идея

**Жадный подход**: сортируем рёбра по весу и добавляем их в MST, если они не создают цикл.

```
Шаги:
1. Отсортировать все рёбра по весу (возрастание)
2. Инициализировать DSU (Disjoint Set Union)
3. Для каждого ребра в отсортированном порядке:
   - Если ребро соединяет две разные компоненты
   - Добавить ребро в MST и объединить компоненты
```

### Реализация на Go

```go
type Edge struct {
	u, v, weight int
}

// Алгоритм Краскала для нахождения MST
func KruskalMST(vertices int, edges []Edge) ([]Edge, int) {
	// Отсортировать рёбра по весу
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].weight < edges[j].weight
	})
	
	// DSU (Disjoint Set Union)
	dsu := NewDSU(vertices)
	mst := []Edge{}
	totalWeight := 0
	
	for _, edge := range edges {
		// Если вершины в разных компонентах
		if dsu.Union(edge.u, edge.v) {
			mst = append(mst, edge)
			totalWeight += edge.weight
			
			// Если найдено V-1 рёбер, MST готово
			if len(mst) == vertices-1 {
				break
			}
		}
	}
	
	return mst, totalWeight
}

// DSU структура
type DSU struct {
	parent []int
	rank   []int
}

func NewDSU(n int) *DSU {
	parent := make([]int, n)
	rank := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
		rank[i] = 0
	}
	return &DSU{parent, rank}
}

func (dsu *DSU) Find(x int) int {
	if dsu.parent[x] != x {
		dsu.parent[x] = dsu.Find(dsu.parent[x])
	}
	return dsu.parent[x]
}

func (dsu *DSU) Union(x, y int) bool {
	px := dsu.Find(x)
	py := dsu.Find(y)
	
	if px == py {
		return false
	}
	
	if dsu.rank[px] < dsu.rank[py] {
		dsu.parent[px] = py
	} else if dsu.rank[px] > dsu.rank[py] {
		dsu.parent[py] = px
	} else {
		dsu.parent[py] = px
		dsu.rank[px]++
	}
	
	return true
}
```

### Пример Краскала пошагово

```
Граф:
  1 --5-- 2      Рёбра (отсортированы):
  |   |   |      1-3: 1 ✓
  8   1   2      1-2: 5
  |   |   |      3-4: 2
  3 --2-- 4      4-2: 2
              2-3: 1 ✓
              1-4: 8
              ... (остальные)

Шаг 1: (1,3, вес=1) → добавить в MST
Шаг 2: (2,3, вес=1) → добавить в MST
Шаг 3: (3,4, вес=2) → добавить в MST
Шаг 4: (1,2, вес=5) → добавить в MST (V-1=4 рёбер)

MST вес: 1+1+2+5 = 9
```

---

## Алгоритм Прима

### Идея

**Пошаговое расширение**: начинаем с одной вершины и постепенно расширяем дерево, добавляя минимальное ребро, соединяющее дерево с оставшейся частью графа.

```
Шаги:
1. Выбрать стартовую вершину
2. Добавить все её соседей в приоритетную очередь
3. Пока MST не содержит все вершины:
   - Выбрать ребро с минимальным весом из приоритетной очереди
   - Если оно соединяет новую вершину, добавить её в MST
   - Добавить все соседей новой вершины в очередь
```

### Реализация на Go

```go
// Алгоритм Прима для нахождения MST
func PrimaMST(vertices int, edges [][]int) ([][]int, int) {
	// Построить список смежности с весами
	graph := make([][]Edge, vertices)
	for _, e := range edges {
		u, v, w := e[0], e[1], e[2]
		graph[u] = append(graph[u], Edge{u, v, w})
		graph[v] = append(graph[v], Edge{v, u, w})
	}
	
	visited := make([]bool, vertices)
	minHeap := &MinHeap{}
	mst := [][]int{}
	totalWeight := 0
	
	// Начать с вершины 0
	visited[0] = true
	for _, neighbor := range graph[0] {
		heap.Push(minHeap, neighbor)
	}
	
	for minHeap.Len() > 0 {
		edge := heap.Pop(minHeap).(Edge)
		
		// Пропустить если обе вершины уже в дереве
		if visited[edge.v] {
			continue
		}
		
		// Добавить ребро в MST
		mst = append(mst, []int{edge.u, edge.v, edge.weight})
		totalWeight += edge.weight
		visited[edge.v] = true
		
		// Добавить все соседей новой вершины
		for _, neighbor := range graph[edge.v] {
			if !visited[neighbor.v] {
				heap.Push(minHeap, neighbor)
			}
		}
	}
	
	return mst, totalWeight
}

// MinHeap для приоритетной очереди
type MinHeap []Edge

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].weight < h[j].weight }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(Edge))
}

func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}
```

### Пример Прима пошагово

```
Граф:
  1 --5-- 2      Стартуем с вершины 1:
  |   |   |
  8   1   2    Шаг 1: в дереве {1}, мин.соседи: (1,3):1, (1,2):5, (1,4):8
  |   |   |      → добавить (1,3):1, в дереве {1,3}
  3 --2-- 4
              Шаг 2: в дереве {1,3}, мин.соседи: (3,4):2, (1,2):5, (1,4):8
                    → добавить (3,4):2, в дереве {1,3,4}
              
              Шаг 3: в дереве {1,3,4}, мин.соседи: (4,2):2, (1,2):5, (1,4):8
                    → добавить (4,2):2, в дереве {1,3,4,2}
              
              MST готово: (1,3):1 + (3,4):2 + (4,2):2 = 5
```

---

## Сравнение методов

### Краскал vs Прим

| Аспект | Краскал | Прим |
|--------|---------|------|
| **Подход** | Жадный на рёбрах | Жадный на вершинах |
| **Структура данных** | DSU | Приоритетная очередь |
| **Время (мин-куча)** | O(E log E) | O(E log V) |
| **Память** | O(V + E) | O(V + E) |
| **Лучше для** | Разреженных графов | Плотных графов |
| **Реализация** | Проще | Сложнее |

### Когда какой использовать?

```
Если E << V²  (разреженный граф)
  └─ Краскал: O(E log E) = O(E log V)
  
Если E ≈ V²   (плотный граф)
  └─ Прим: O(V² + E) = O(V²)
  
Если нужно простое решение
  └─ Краскал
  
Если граф очень большой и памяти мало
  └─ Прим (нет списка всех рёбер)
```

---

## Практические примеры

### Пример 1: Min Cost to Connect All Points

```go
// Найти минимальную стоимость соединения всех точек
func MinCostConnectPoints(points [][]int) int {
	n := len(points)
	
	// Построить полный граф (вычислить все расстояния)
	edges := []Edge{}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dist := abs(points[i][0]-points[j][0]) + abs(points[i][1]-points[j][1])
			edges = append(edges, Edge{i, j, dist})
		}
	}
	
	// Применить Краскал
	_, totalCost := KruskalMST(n, edges)
	return totalCost
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
```

### Пример 2: Network Delay Time (MST вариант)

```go
// Найти максимальное время в MST (диаметр дерева)
func MaxNetworkDelayInMST(edges []Edge, vertices int) int {
	mst, _ := KruskalMST(vertices, edges)
	
	// Построить граф из MST
	graph := make([][]int, vertices)
	for _, e := range mst {
		graph[e.u] = append(graph[e.u], e.weight)
		graph[e.v] = append(graph[e.v], e.weight)
	}
	
	// BFS для нахождения диаметра
	maxDist := 0
	for i := 0; i < vertices; i++ {
		dist := bfsDist(graph, i, vertices)
		if dist > maxDist {
			maxDist = dist
		}
	}
	
	return maxDist
}

func bfsDist(graph [][]int, start, n int) int {
	dist := make([]int, n)
	queue := []int{start}
	maxD := 0
	
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		
		for i, d := range graph[u] {
			if dist[i] == 0 && i != start {
				dist[i] = dist[u] + d
				queue = append(queue, i)
				if dist[i] > maxD {
					maxD = dist[i]
				}
			}
		}
	}
	
	return maxD
}
```

### Пример 3: Redundant Connection in MST

```go
// Найти лишнее ребро, которое создаёт цикл
func FindRedundantConnection(edges [][]int) []int {
	n := len(edges)
	dsu := NewDSU(n + 1)
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		
		if !dsu.Union(u, v) {
			return edge // Это ребро создаёт цикл
		}
	}
	
	return []int{}
}
```

---

## Анализ сложности

### Краскал

| Операция | Время |
|----------|-------|
| **Сортировка рёбер** | O(E log E) |
| **DSU операции** | O(E · α(V)) ≈ O(E) |
| **Итого** | **O(E log E)** |

### Прим

| Операция | Время |
|----------|-------|
| **Вставка в очередь** | O(E log V) |
| **Удаление из очереди** | O(E log V) |
| **Итого** | **O(E log V)** |

### Память

| Метод | Память |
|-------|--------|
| **Краскал** | O(E) для списка рёбер + O(V) для DSU |
| **Прим** | O(V + E) для списка смежности |

---

## Доказательство корректности

### Лемма о разрезе (Cut Lemma)

```
Пусть S — подмножество вершин MST.
Пусть (u, v) — минимальное ребро пересекающее разрез (S, V-S).
Тогда (u, v) принадлежит некоторому MST.

Следствие:
- Краскал выбирает минимальное ребро (по сортировке)
- Прим выбирает минимальное ребро (из приоритетной очереди)
- Оба гарантируют MST
```

---

## Рекомендуемые задачи на LeetCode

- [Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/) — Краскал
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/) — обнаружение цикла
- [Connecting Cities With Minimum Cost](https://leetcode.com/problems/connecting-cities-with-minimum-cost/) — MST
- [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) — компоненты связности

---

## Заключение

MST — это фундаментальная задача графов:

1. **Краскал** — жадный на рёбрах, использует DSU
2. **Прим** — жадный на вершинах, использует приоритетную очередь
3. **Оба работают** благодаря лемме о разрезе
4. **Выбор зависит** от размера и плотности графа

Приложения:
- Дизайн сетей (телеком, электрические сети)
- Кластеризация данных
- Обработка изображений
- Гене́тические алгоритмы

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
