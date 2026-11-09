# Лекция 4: Мосты, Точки Артикуляции и Алгоритмы Тарджана/Косарайю

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Advanced

---

## Содержание
1. [Точки артикуляции](#точки-артикуляции)
2. [Мосты в графе](#мосты-в-графе)
3. [Алгоритм Тарджана](#алгоритм-тарджана)
4. [Алгоритм Косарайю](#алгоритм-косарайю)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Точки артикуляции

### Определение

**Точка артикуляции (Articulation Point)** — это вершина, удаление которой увеличивает количество связных компонент в графе.

```
Граф:
    1 --- 2 --- 3
    |           |
    4 --------- 5

Точка артикуляции: вершина 2
Если удалить 2: граф разбивается на {1,4,5} и {3}
```

### Алгоритм DFS для поиска точек артикуляции

Ключевая идея:
- Вершина `u` — точка артикуляции если:
  1. `u` — корень DFS дерева и имеет ≥2 детей, или
  2. `u` — не корень и существует ребёнок `v` такой, что ни `v`, ни его потомки не могут достичь предка `u` или выше

```go
// Найти все точки артикуляции
func FindArticulationPoints(n int, edges [][]int) []int {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	
	visited := make([]bool, n)
	disc := make([]int, n)        // discovery time
	low := make([]int, n)         // lowest discovery time reachable
	parent := make([]int, n)
	for i := range parent {
		parent[i] = -1
	}
	
	articulationPoints := []int{}
	time := 0
	
	var dfs func(int)
	dfs = func(u int) {
		visited[u] = true
		disc[u] = time
		low[u] = time
		time++
		
		children := 0
		
		for _, v := range graph[u] {
			if !visited[v] {
				children++
				parent[v] = u
				dfs(v)
				
				// Проверяем, может ли v достичь предка u
				low[u] = min(low[u], low[v])
				
				// u — точка артикуляции в двух случаях:
				// 1) u — корень и имеет >= 2 детей
				if parent[u] == -1 && children > 1 {
					articulationPoints = append(articulationPoints, u)
				}
				
				// 2) u — не корень и low[v] >= disc[u]
				if parent[u] != -1 && low[v] >= disc[u] {
					articulationPoints = append(articulationPoints, u)
				}
			} else if v != parent[u] {
				// Back edge к предку
				low[u] = min(low[u], disc[v])
			}
		}
	}
	
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i)
		}
	}
	
	// Удалить дубликаты
	m := make(map[int]bool)
	result := []int{}
	for _, ap := range articulationPoints {
		if !m[ap] {
			m[ap] = true
			result = append(result, ap)
		}
	}
	
	return result
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

---

## Мосты в графе

### Определение

**Мост (Bridge)** — это ребро, удаление которого увеличивает количество связных компонент.

```
Граф:
    1 --- 2 --- 3
          |
          4

Мост: ребро (2, 3)
Если удалить (2,3): вершина 3 отделяется от остального графа
```

### Алгоритм Тарджана для поиска мостов

```go
// Найти все мосты в графе
func FindBridges(n int, edges [][]int) [][]int {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
	}
	
	visited := make([]bool, n)
	disc := make([]int, n)        // discovery time
	low := make([]int, n)         // lowest discovery time reachable
	time := 0
	bridges := [][]int{}
	
	var dfs func(int, int)
	dfs = func(u, p int) {
		visited[u] = true
		disc[u] = time
		low[u] = time
		time++
		
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v, u)
				low[u] = min(low[u], low[v])
				
				// Если низшая точка v выше discovery time u,
				// то (u,v) — мост
				if low[v] > disc[u] {
					bridges = append(bridges, []int{u, v})
				}
			} else if v != p {
				// Back edge
				low[u] = min(low[u], disc[v])
			}
		}
	}
	
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	
	return bridges
}
```

---

## Алгоритм Тарджана

### Strongly Connected Components (SCC)

**Сильно связная компонента** — это максимальное подмножество вершин, где каждая вершина достижима из любой другой.

```
Граф:
  1 → 2 → 3
  ↑       ↓
  └───────┘

SCC: {1, 2, 3}
```

### Алгоритм на Go

```go
// Найти все SCC используя алгоритм Тарджана
func FindSCCTarjan(n int, edges [][]int) [][]int {
	graph := make([][]int, n)
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
	}
	
	disc := make([]int, n)
	low := make([]int, n)
	stackMember := make([]bool, n)
	stack := []int{}
	
	for i := range disc {
		disc[i] = -1
		low[i] = -1
	}
	
	sccs := [][]int{}
	time := 0
	
	var dfs func(int)
	dfs = func(u int) {
		disc[u] = time
		low[u] = time
		time++
		stack = append(stack, u)
		stackMember[u] = true
		
		for _, v := range graph[u] {
			if disc[v] == -1 {
				dfs(v)
				low[u] = min(low[u], low[v])
			} else if stackMember[v] {
				low[u] = min(low[u], disc[v])
			}
		}
		
		// Если u — root узла, pop SCC из stack
		if low[u] == disc[u] {
			scc := []int{}
			for {
				v := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				stackMember[v] = false
				scc = append(scc, v)
				if v == u {
					break
				}
			}
			sccs = append(sccs, scc)
		}
	}
	
	for i := 0; i < n; i++ {
		if disc[i] == -1 {
			dfs(i)
		}
	}
	
	return sccs
}
```

---

## Алгоритм Косарайю

### Альтернативный подход к поиску SCC

Косарайю использует два DFS прохода:
1. DFS на исходном графе, запомнить порядок finish time
2. DFS на транспонированном графе в обратном порядке finish time

```go
// Найти все SCC используя алгоритм Косарайю
func FindSCCKosaraju(n int, edges [][]int) [][]int {
	// Построить граф
	graph := make([][]int, n)
	graphT := make([][]int, n) // Транспонированный граф
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		graph[u] = append(graph[u], v)
		graphT[v] = append(graphT[v], u)
	}
	
	// Шаг 1: DFS на исходном графе и запомнить порядок
	visited := make([]bool, n)
	finishOrder := []int{}
	
	var dfs1 func(int)
	dfs1 = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs1(v)
			}
		}
		finishOrder = append(finishOrder, u)
	}
	
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs1(i)
		}
	}
	
	// Шаг 2: DFS на транспонированном графе в обратном порядке
	visited = make([]bool, n)
	sccs := [][]int{}
	
	var dfs2 func(int, *[]int)
	dfs2 = func(u int, scc *[]int) {
		visited[u] = true
		*scc = append(*scc, u)
		for _, v := range graphT[u] {
			if !visited[v] {
				dfs2(v, scc)
			}
		}
	}
	
	for i := len(finishOrder) - 1; i >= 0; i-- {
		u := finishOrder[i]
		if !visited[u] {
			scc := []int{}
			dfs2(u, &scc)
			sccs = append(sccs, scc)
		}
	}
	
	return sccs
}
```

### Косарайю vs Тарджан

| Аспект | Косарайю | Тарджан |
|--------|----------|---------|
| **DFS проходов** | 2 | 1 |
| **Память** | O(V+E) для транспозиции | O(V) |
| **Сложность** | O(V+E) | O(V+E) |
| **Реализация** | Проще | Сложнее |
| **Практика** | Часто использует | Иногда используется |

---

## Практические примеры

### Пример 1: Critical Connections in a Network

```go
// Найти все критичные соединения (мосты) в сети
func CriticalConnections(n int, connections [][]int) [][]int {
	graph := make([][]int, n)
	edgeIndex := make(map[[2]int]int)
	
	for i, conn := range connections {
		u, v := conn[0], conn[1]
		graph[u] = append(graph[u], v)
		graph[v] = append(graph[v], u)
		if u > v {
			u, v = v, u
		}
		edgeIndex[[2]int{u, v}] = i
	}
	
	visited := make([]bool, n)
	disc := make([]int, n)
	low := make([]int, n)
	time := 0
	bridges := [][]int{}
	
	var dfs func(int, int)
	dfs = func(u, p int) {
		visited[u] = true
		disc[u] = time
		low[u] = time
		time++
		
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v, u)
				low[u] = min(low[u], low[v])
				
				if low[v] > disc[u] {
					a, b := u, v
					if a > b {
						a, b = b, a
					}
					bridges = append(bridges, []int{a, b})
				}
			} else if v != p {
				low[u] = min(low[u], disc[v])
			}
		}
	}
	
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	
	return bridges
}
```

### Пример 2: Strongly Connected Components (LeetCode версия)

```go
// Вернуть все SCC в виде списка списков
func StronglyConnectedComponents(n int, edges [][]int) [][]int {
	return FindSCCKosaraju(n, edges)
}

// Пример использования:
// edges := [][]int{{1,0},{0,2},{2,1}}
// result := StronglyConnectedComponents(3, edges)
// result будет содержать [[0,1,2]] или похожее группирование
```

---

## Анализ сложности

| Алгоритм | Время | Память | Применение |
|----------|-------|--------|-----------|
| **Поиск артикуляций** | O(V+E) | O(V) | Критичные вершины |
| **Поиск мостов (Тарджан)** | O(V+E) | O(V) | Критичные рёбра |
| **SCC Тарджан** | O(V+E) | O(V) | Один проход |
| **SCC Косарайю** | O(V+E) | O(V+E) | Два прохода |

---

## Рекомендуемые задачи на LeetCode

- [Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/) — Hard
- [Find The Celebrity](https://leetcode.com/problems/find-the-celebrity/) — O(n) решение используя идеи графов
- [Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/) — компоненты связности
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/) — поиск цикла
- [Accounts Merge](https://leetcode.com/problems/accounts-merge/) — нахождение компонент

---

## Заключение

Эта лекция охватывает продвинутые методы анализа графов:

1. **Точки артикуляции** — критичные вершины сети
2. **Мосты** — критичные рёбра сети
3. **SCC** — группирование вершин в компоненты
4. **Тарджан и Косарайю** — два подхода к одной проблеме

Эти алгоритмы критичны для:
- Анализа сетевой надёжности
- Социальных сетей (нахождение групп)
- Компиляции и анализа зависимостей
- Маршрутизации в сетях

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
