# Лекция 11: Кратчайшие пути (Продолжение)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Advanced

---

## Содержание
1. [Обзор алгоритмов кратчайшего пути](#обзор-алгоритмов-кратчайшего-пути)
2. [Алгоритм Беллмана-Форда](#алгоритм-беллмана-форда)
3. [Алгоритм Флойда-Уоршелла](#алгоритм-флойда-уоршелла)
4. [SPFA (Shortest Path Faster Algorithm)](#spfa-shortest-path-faster-algorithm)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Обзор алгоритмов кратчайшего пути

### Классификация

```
┌─────────────────────────────────────────┐
│   Алгоритмы кратчайшего пути           │
├─────────────────────────────────────────┤
│ Одноисточник (Single Source):           │
│  ├─ Невзвешенный: BFS (O(V+E))          │
│  ├─ Неотрицательные веса: Дейкстра      │
│  └─ Отрицательные веса: Беллман-Форд    │
│                                         │
│ Все пары (All Pairs):                   │
│  ├─ Небольшие графы: Флойд-Уоршелл     │
│  └─ Повторить Дейкстру для каждой вер. │
│                                         │
│ Специальные случаи:                     │
│  ├─ DAG: топо-сортировка + релаксация   │
│  └─ Все с неотрицательными: BFS/Дейкстра
└─────────────────────────────────────────┘
```

### Сравнительная таблица

| Алгоритм | Время | Веса | Отрицательные циклы |
|----------|-------|------|-------------------|
| **BFS** | O(V+E) | Нет | Нет |
| **Дейкстра** | O((V+E) log V) | ≥0 | Нет |
| **Беллман-Форд** | O(V·E) | Любые | Обнаруживает |
| **SPFA** | O(E) avg, O(V·E) worst | Любые | Обнаруживает |
| **Флойд-Уоршелл** | O(V³) | Любые | Обнаруживает |

---

## Алгоритм Беллмана-Форда

### Идея

**Релаксация рёбер**: повторно релаксируем все рёбра V-1 раз. Если на V-й итерации что-то изменится — есть отрицательный цикл.

```
Релаксация ребра (u,v) с весом w:
  if dist[u] + w < dist[v]:
      dist[v] = dist[u] + w
```

### Почему V-1 итераций?

```
Кратчайший путь без циклов имеет максимум V-1 рёбер.
Каждая итерация "распространяет" кратчайшие пути на один шаг дальше.
После V-1 итераций все кратчайшие пути найдены.
На V-й итерации проверяем наличие отрицательного цикла.
```

### Реализация на Go

```go
// Беллман-Форд для нахождения кратчайших путей
func BellmanFord(vertices int, edges []Edge, source int) ([]int, bool) {
	distances := make([]int, vertices)
	for i := range distances {
		distances[i] = math.MaxInt / 2 // "бесконечность"
	}
	distances[source] = 0
	
	// V-1 релаксаций
	for i := 0; i < vertices-1; i++ {
		for _, edge := range edges {
			if distances[edge.u] != math.MaxInt/2 &&
				distances[edge.u]+edge.weight < distances[edge.v] {
				distances[edge.v] = distances[edge.u] + edge.weight
			}
		}
	}
	
	// Проверка отрицательного цикла
	for _, edge := range edges {
		if distances[edge.u] != math.MaxInt/2 &&
			distances[edge.u]+edge.weight < distances[edge.v] {
			return nil, true // Отрицательный цикл обнаружен
		}
	}
	
	return distances, false
}

// Восстановить путь от source до target
func BellmanFordPath(vertices int, edges []Edge, source, target int) ([]int, bool) {
	distances := make([]int, vertices)
	parent := make([]int, vertices)
	
	for i := range distances {
		distances[i] = math.MaxInt / 2
		parent[i] = -1
	}
	distances[source] = 0
	
	// V-1 релаксаций
	for i := 0; i < vertices-1; i++ {
		for _, edge := range edges {
			if distances[edge.u] != math.MaxInt/2 &&
				distances[edge.u]+edge.weight < distances[edge.v] {
				distances[edge.v] = distances[edge.u] + edge.weight
				parent[edge.v] = edge.u
			}
		}
	}
	
	// Проверка отрицательного цикла
	for _, edge := range edges {
		if distances[edge.u] != math.MaxInt/2 &&
			distances[edge.u]+edge.weight < distances[edge.v] {
			return nil, true
		}
	}
	
	// Восстановить путь
	path := []int{}
	for v := target; v != -1; v = parent[v] {
		path = append([]int{v}, path...)
	}
	
	return path, false
}
```

### Пример Беллмана-Форда

```
Граф:           Итерация 1:        Итерация 2:      Итерация 3:
  1 --1-- 2       dist[2]=1         dist[3]=2        dist[4]=2
  |   2   |       dist[3]=INF       dist[4]=INF
  3   |   3
  |   v   |
  4 --1-- 3

Источник: 1
dist[1]=0, dist[2]=1, dist[3]=2, dist[4]=2
```

---

## Алгоритм Флойда-Уоршелла

### Идея

**Динамическое программирование**: построить кратчайшие пути используя промежуточные вершины.

```
dist[i][j][k] = кратчайший путь от i до j,
                используя только вершины 0..k как промежуточные

dist[i][j][k] = min(
    dist[i][j][k-1],           // не используем k
    dist[i][k][k-1] + dist[k][j][k-1]  // используем k
)
```

### Реализация на Go

```go
// Флойд-Уоршелл для всех пар кратчайших путей
func FloydWarshall(vertices int, edges [][]int) [][]int {
	// Инициализировать матрицу расстояний
	INF := math.MaxInt / 2
	dist := make([][]int, vertices)
	for i := range dist {
		dist[i] = make([]int, vertices)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
			} else {
				dist[i][j] = INF
			}
		}
	}
	
	// Добавить рёбра
	for _, edge := range edges {
		u, v, w := edge[0], edge[1], edge[2]
		dist[u][v] = w
	}
	
	// DP: для каждого промежуточного узла k
	for k := 0; k < vertices; k++ {
		for i := 0; i < vertices; i++ {
			for j := 0; j < vertices; j++ {
				if dist[i][k] != INF && dist[k][j] != INF {
					dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])
				}
			}
		}
	}
	
	return dist
}

// Обнаружить отрицательный цикл
func HasNegativeCycle(vertices int, dist [][]int) bool {
	for i := 0; i < vertices; i++ {
		if dist[i][i] < 0 {
			return true
		}
	}
	return false
}

// Восстановить путь от i к j
func ReconstructPath(dist [][]int, next [][]int, i, j int) []int {
	if dist[i][j] == math.MaxInt/2 {
		return nil // Нет пути
	}
	
	path := []int{i}
	for i != j {
		i = next[i][j]
		path = append(path, i)
	}
	return path
}

// Флойд-Уоршелл с восстановлением пути
func FloydWarshallWithPath(vertices int, edges [][]int) ([][]int, [][]int) {
	INF := math.MaxInt / 2
	dist := make([][]int, vertices)
	next := make([][]int, vertices)
	
	for i := range dist {
		dist[i] = make([]int, vertices)
		next[i] = make([]int, vertices)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
				next[i][j] = i
			} else {
				dist[i][j] = INF
				next[i][j] = -1
			}
		}
	}
	
	// Добавить рёбра
	for _, edge := range edges {
		u, v, w := edge[0], edge[1], edge[2]
		dist[u][v] = w
		next[u][v] = v
	}
	
	// DP
	for k := 0; k < vertices; k++ {
		for i := 0; i < vertices; i++ {
			for j := 0; j < vertices; j++ {
				if dist[i][k] != INF && dist[k][j] != INF {
					if dist[i][k]+dist[k][j] < dist[i][j] {
						dist[i][j] = dist[i][k] + dist[k][j]
						next[i][j] = next[i][k]
					}
				}
			}
		}
	}
	
	return dist, next
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Пример Флойда-Уоршелла

```
Граф:
  1 --4-- 2
  |       |
  2       1
  |       |
  3 --5-- 4

Начальная матрица (с рёбрами):
    0   1   2   3
0  [0   4  INF 2]
1  [INF 0  1  INF]
2  [INF INF 0  5]
3  [INF INF INF 0]

После k=0:
    0   1   2   3
0  [0   4  INF 2]
1  [INF 0  1  INF]
2  [INF INF 0  5]
3  [INF INF INF 0]

После k=1:
    0   1   2   3
0  [0   4  5   2]
1  [INF 0  1  INF]
2  [INF INF 0  5]
3  [INF INF INF 0]

... и так далее
```

---

## SPFA (Shortest Path Faster Algorithm)

### Идея

Вариант Беллмана-Форда, который релаксирует рёбра только если произошли изменения.

```go
// SPFA — улучшенная версия Беллмана-Форда
func SPFA(vertices int, edges []Edge, source int) ([]int, bool) {
	distances := make([]int, vertices)
	inQueue := make([]bool, vertices)
	count := make([]int, vertices)
	
	for i := range distances {
		distances[i] = math.MaxInt / 2
	}
	distances[source] = 0
	
	queue := []int{source}
	inQueue[source] = true
	
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		inQueue[u] = false
		
		for _, edge := range edges {
			if edge.u != u {
				continue
			}
			
			v := edge.v
			if distances[u] != math.MaxInt/2 &&
				distances[u]+edge.weight < distances[v] {
				distances[v] = distances[u] + edge.weight
				count[v]++
				
				// Обнаружить отрицательный цикл
				if count[v] >= vertices {
					return nil, true
				}
				
				if !inQueue[v] {
					queue = append(queue, v)
					inQueue[v] = true
				}
			}
		}
	}
	
	return distances, false
}
```

### Сложность

```
Среднее: O(E)       — намного быстрее на практике
Худшее: O(V·E)      — когда много отрицательных циклов
```

---

## Практические примеры

### Пример 1: Cheapest Flights Within K Stops

```go
// Найти самый дешёвый путь с максимум K промежуточных остановок
func FindCheapestPrice(n int, flights [][]int, src, dst, k int) int {
	// DP подход (похож на Беллман-Форд)
	// dp[i][j] = минимальная цена до города j с максимум i рейсами
	
	INF := math.MaxInt / 2
	dp := make([][]int, k+2)
	for i := range dp {
		dp[i] = make([]int, n)
		for j := range dp[i] {
			dp[i][j] = INF
		}
	}
	
	dp[0][src] = 0
	
	for i := 1; i <= k+1; i++ {
		dp[i][src] = 0
		
		for _, flight := range flights {
			u, v, price := flight[0], flight[1], flight[2]
			
			if dp[i-1][u] != INF {
				dp[i][v] = min(dp[i][v], dp[i-1][u]+price)
			}
		}
	}
	
	result := dp[k+1][dst]
	if result == INF {
		return -1
	}
	return result
}
```

### Пример 2: Network Delay Time (все пары)

```go
// Найти время, за которое все узлы получат сигнал (от любого источника)
func NetworkDelayTimeAllPairs(times [][]int, n int) int {
	dist, _ := FloydWarshall(n, times)
	
	maxDelay := 0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if dist[i][j] != math.MaxInt/2 && dist[i][j] > maxDelay {
				maxDelay = dist[i][j]
			}
		}
	}
	
	return maxDelay
}
```

### Пример 3: Floyd-Warshall для обнаружения цикла

```go
// Проверить есть ли отрицательный цикл и его размер
func DetectNegativeCycle(n int, edges [][]int) bool {
	dist, _ := FloydWarshallWithPath(n, edges)
	
	for i := 0; i < n; i++ {
		if dist[i][i] < 0 {
			return true
		}
	}
	
	return false
}
```

---

## Анализ сложности

### Время выполнения

| Алгоритм | Время | Когда использовать |
|----------|-------|-------------------|
| **Дейкстра** | O((V+E) log V) | Неотрицательные веса, один источник |
| **Беллман-Форд** | O(V·E) | Отрицательные веса, один источник |
| **SPFA** | O(E) avg, O(V·E) worst | Отрицательные веса на практике |
| **Флойд-Уоршелл** | O(V³) | Все пары, малые V |

### Память

| Алгоритм | Память |
|----------|--------|
| **Дейкстра** | O(V + E) |
| **Беллман-Форд** | O(V) |
| **SPFA** | O(V + E) |
| **Флойд-Уоршелл** | O(V²) |

---

## Таблица выбора алгоритма

```
┌─────────────────────────────────────────────────────┐
│        Как выбрать алгоритм кратчайшего пути?      │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Один источник?                                    │
│  ├─ ДА:                                            │
│  │  ├─ Неотрицательные веса?                       │
│  │  │  └─ ДА: Дейкстра O((V+E) log V)            │
│  │  └─ Отрицательные веса?                        │
│  │     ├─ ДА: Беллман-Форд O(V·E)                │
│  │     └─ (или SPFA на практике O(E))            │
│  └─ НЕТ (все пары):                               │
│     ├─ V ≤ 500: Флойд-Уоршелл O(V³)             │
│     └─ V > 500: Дейкстра V раз O(V(V+E) log V)  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Рекомендуемые задачи на LeetCode

- [Network Delay Time](https://leetcode.com/problems/network-delay-time/) — Дейкстра
- [Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/) — Беллман-Форд вариант
- [Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) — Дейкстра
- [Shortest Path in a Grid with Obstacles Elimination](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/) — BFS/Дейкстра
- [Minimum Cost to Make at Least One Valid Path in a Grid](https://leetcode.com/problems/minimum-cost-to-make-at-least-one-valid-path-in-a-grid/) — 0-1 BFS

---

## Заключение

Три главных алгоритма кратчайшего пути:

1. **Дейкстра** — для неотрицательных весов, один источник
2. **Беллман-Форд** — для отрицательных весов, обнаружение циклов
3. **Флойд-Уоршелл** — для всех пар, особенно на малых графах

Выбор зависит от:
- Размера графа
- Наличия отрицательных весов
- Нужны ли все пары кратчайших путей
- Допустимое время выполнения

Практический совет: сначала проверь отрицательные циклы, потом выбирай алгоритм.

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
