# Лекция 12: Maximum Flow и Min Cut

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Advanced

---

## Содержание
1. [Введение в потоки](#введение-в-потоки)
2. [Ford-Fulkerson алгоритм](#ford-fulkerson-алгоритм)
3. [Edmonds-Karp алгоритм](#edmonds-karp-алгоритм)
4. [Dinic алгоритм](#dinic-алгоритм)
5. [Приложения потоков](#приложения-потоков)
6. [Практические примеры](#практические-примеры)

---

## Введение в потоки

### Определение задачи

**Flow Network** — это ориентированный граф G = (V, E) где:
- Каждое ребро (u,v) имеет **capacity** c(u,v) ≥ 0
- Есть **source** (исток) s и **sink** (сток) t
- Граф может иметь несколько путей из s в t

**Maximum Flow** — найти максимальное количество "потока", которое можно прогнать из s в t, не превышая пропускные способности рёбер.

```
Граф:
    s --1--> a --2--> t
    |        |       ^
    +--3-----+-------+
    
    capacity(s,a) = 1
    capacity(s,b) = 3
    capacity(a,t) = 2
    capacity(b,t) = 2
    
    Maximum flow = 3 (все рёбра полностью используются)
```

### Ограничения потока

```
1. Capacity constraint: flow(u,v) ≤ capacity(u,v)
2. Flow conservation: Σ flow(u,v) = Σ flow(v,w) для всех v ≠ s,t
3. Antisymmetry: flow(u,v) = -flow(v,u)
```

---

## Ford-Fulkerson алгоритм

### Идея: Augmenting Path

**Идея**: Повторно найти путь из s в t, где есть "место" для потока, и добавить поток.

```go
// Ford-Fulkerson алгоритм
func FordFulkerson(graph [][]int, source, sink int) int {
	n := len(graph)
	capacity := make([][]int, n)
	for i := range capacity {
		capacity[i] = make([]int, n)
		copy(capacity[i], graph[i])
	}
	
	maxFlow := 0
	
	for {
		// Найти увеличивающий путь из source в sink
		parent := make([]int, n)
		for i := range parent {
			parent[i] = -1
		}
		
		// BFS для нахождения пути
		queue := []int{source}
		parent[source] = source
		
		for len(queue) > 0 && parent[sink] == -1 {
			u := queue[0]
			queue = queue[1:]
			
			for v := 0; v < n; v++ {
				if parent[v] == -1 && capacity[u][v] > 0 {
					parent[v] = u
					queue = append(queue, v)
					
					if v == sink {
						break
					}
				}
			}
		}
		
		if parent[sink] == -1 {
			break // Нет пути
		}
		
		// Найти минимальную пропускную способность на пути
		pathFlow := math.MaxInt
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			pathFlow = min(pathFlow, capacity[u][v])
		}
		
		// Обновить пропускные способности
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			capacity[u][v] -= pathFlow
			capacity[v][u] += pathFlow // Обратный поток
		}
		
		maxFlow += pathFlow
	}
	
	return maxFlow
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Сложность Ford-Fulkerson

```
Time: O(E * maxFlow)  — может быть экспоненциальным!
Худший случай: когда maxFlow очень большой или пути неудачные
```

---

## Edmonds-Karp алгоритм

### Улучшение Ford-Fulkerson

**Edmonds-Karp** — это вариант Ford-Fulkerson, который **всегда использует кратчайший путь** (BFS вместо DFS).

```go
// Edmonds-Karp алгоритм (улучшенный Ford-Fulkerson)
func EdmondsKarp(graph [][]int, source, sink int) int {
	n := len(graph)
	capacity := make([][]int, n)
	for i := range capacity {
		capacity[i] = make([]int, n)
		copy(capacity[i], graph[i])
	}
	
	maxFlow := 0
	
	for {
		// BFS для кратчайшего пути
		parent := make([]int, n)
		for i := range parent {
			parent[i] = -1
		}
		
		queue := []int{source}
		parent[source] = source
		found := false
		
		for len(queue) > 0 && !found {
			u := queue[0]
			queue = queue[1:]
			
			for v := 0; v < n; v++ {
				if parent[v] == -1 && capacity[u][v] > 0 {
					parent[v] = u
					queue = append(queue, v)
					
					if v == sink {
						found = true
						break
					}
				}
			}
		}
		
		if !found {
			break
		}
		
		// Найти минимум на пути
		pathFlow := math.MaxInt
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			pathFlow = min(pathFlow, capacity[u][v])
		}
		
		// Обновить граф
		for v := sink; v != source; v = parent[v] {
			u := parent[v]
			capacity[u][v] -= pathFlow
			capacity[v][u] += pathFlow
		}
		
		maxFlow += pathFlow
	}
	
	return maxFlow
}
```

### Сложность Edmonds-Karp

```
Time: O(V * E²)  — полиномиальна!
Всегда завершится за разумное время
```

---

## Dinic алгоритм

### Оптимизированный подход: Level Graph

**Dinic** строит **level graph** и находит **blocking flows**.

```go
type Edge struct {
	to, rev, flow, cap int
}

type Dinic struct {
	graph [][]Edge
	level []int
	iter  []int
}

func NewDinic(n int) *Dinic {
	return &Dinic{
		graph: make([][]Edge, n),
		level: make([]int, n),
		iter:  make([]int, n),
	}
}

// Добавить ребро u -> v с capacity cap
func (d *Dinic) AddEdge(u, v, cap int) {
	d.graph[u] = append(d.graph[u], Edge{v, len(d.graph[v]), 0, cap})
	d.graph[v] = append(d.graph[v], Edge{u, len(d.graph[u]) - 1, 0, 0})
}

// BFS для построения level graph
func (d *Dinic) bfs(source, sink int) bool {
	for i := range d.level {
		d.level[i] = -1
	}
	d.level[source] = 0
	
	queue := []int{source}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		
		for _, e := range d.graph[u] {
			if d.level[e.to] < 0 && e.flow < e.cap {
				d.level[e.to] = d.level[u] + 1
				queue = append(queue, e.to)
			}
		}
	}
	
	return d.level[sink] >= 0
}

// DFS для найденного blocking flow
func (d *Dinic) dfs(u, sink, f int) int {
	if u == sink {
		return f
	}
	
	for i := d.iter[u]; i < len(d.graph[u]); i++ {
		d.iter[u] = i
		e := &d.graph[u][i]
		
		if d.level[u] < d.level[e.to] && e.flow < e.cap {
			push := d.dfs(e.to, sink, min(f, e.cap-e.flow))
			if push > 0 {
				e.flow += push
				d.graph[e.to][e.rev].flow -= push
				return push
			}
		}
	}
	
	return 0
}

// Найти максимальный поток
func (d *Dinic) MaxFlow(source, sink int) int {
	flow := 0
	
	for d.bfs(source, sink) {
		for i := range d.iter {
			d.iter[i] = 0
		}
		
		for {
			f := d.dfs(source, sink, math.MaxInt)
			if f == 0 {
				break
			}
			flow += f
		}
	}
	
	return flow
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Сложность Dinic

```
Time: O(V² * E)  — лучше чем Edmonds-Karp в практике
Может быть O(E * sqrt(V)) для некоторых графов
```

---

## Приложения потоков

### 1. Паросочетание в двудольном графе

```go
// Найти максимальное паросочетание в двудольном графе
func MaxBipartiteMatching(leftSize, rightSize int, edges [][]int) int {
	// Создать flow network:
	// source -> все левые вершины (capacity 1)
	// все левые -> соответствующие правые (capacity 1)
	// все правые -> sink (capacity 1)
	
	n := 2 + leftSize + rightSize
	d := NewDinic(n)
	source, sink := 0, 1
	
	for i := 0; i < leftSize; i++ {
		d.AddEdge(source, 2+i, 1)
	}
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		d.AddEdge(2+u, 2+leftSize+v, 1)
	}
	
	for i := 0; i < rightSize; i++ {
		d.AddEdge(2+leftSize+i, sink, 1)
	}
	
	return d.MaxFlow(source, sink)
}
```

### 2. Min Cut

```go
// Min Cut = Max Flow (теорема Форда-Фалкерсона)
// Рёбра в min cut — это рёбра, которые полностью насыщены и
// где у вас может пройти путь из source в вершину, но не в sink
```

### 3. Максимальное биpartite matching с весами

```go
// Венгерский алгоритм через потоки минимальной стоимости
// Требует модификации для учёта весов (не рассматривается здесь)
```

---

## Практические примеры

### Пример 1: Network Flow

```go
// LeetCode 1584 - Min Cost to Connect All Points с потоками
func MinCostConnectPoints(points [][]int) int {
	// Это скорее для MST, но демонстрирует концепцию
	// Maximum flow здесь не оптимален
	// Используйте Kruskal/Prim вместо этого
}
```

### Пример 2: Valid Path in Graph

```go
// Проверить есть ли путь из source в sink (вариант max flow)
func ValidPath(n int, edges [][]int, source, sink int) bool {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	
	visited := make(map[int]bool)
	
	var dfs func(int) bool
	dfs = func(u int) bool {
		if u == sink {
			return true
		}
		visited[u] = true
		
		for _, v := range graph[u] {
			if !visited[v] && dfs(v) {
				return true
			}
		}
		
		return false
	}
	
	return dfs(source)
}
```

---

## Сравнение алгоритмов

| Алгоритм | Время | Место | Когда использовать |
|----------|-------|-------|-------------------|
| **Ford-Fulkerson** | O(E * maxFlow) | O(V²) | Маленькие графы |
| **Edmonds-Karp** | O(V * E²) | O(V²) | Стабильный выбор |
| **Dinic** | O(V² * E) | O(V + E) | Большие графы |
| **Push-Relabel** | O(V³) | O(V) | Очень большие графы |

---

## Рекомендуемые задачи на LeetCode

- [Maximum Network Rank](https://leetcode.com/problems/maximum-network-rank/)
- [Evaluate Division](https://leetcode.com/problems/evaluate-division/) (graph path)
- [Network Delay Time](https://leetcode.com/problems/network-delay-time/) (не flow, а путь)
- [Shortest Path in Weighted Graph](https://leetcode.com/problems/the-time-when-all-cities-become-reachable/)

*Примечание: Maximum Flow задачи встречаются реже в LeetCode, чем в ICPC/Codeforces*

---

## Заключение

1. **Ford-Fulkerson** — базовая идея, может быть неэффективной
2. **Edmonds-Karp** — гарантированно полиномиальна
3. **Dinic** — быстрый на практике, рекомендуется
4. **Приложения** — паросочетание, min cut, вырезание графа
5. **Min-Cost Max Flow** — для задач с весами (продвинуто)

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
