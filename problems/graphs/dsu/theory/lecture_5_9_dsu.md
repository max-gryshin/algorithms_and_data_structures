# Лекция 5 & 9: Disjoint Set Union (Система непересекающихся множеств)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в DSU](#введение-в-dsu)
2. [Базовая реализация](#базовая-реализация)
3. [Оптимизации](#оптимизации)
4. [Применение в задачах](#применение-в-задачах)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Введение в DSU

### Что такое DSU?

**Disjoint Set Union (DSU)** — это структура данных, которая поддерживает разбиение множества элементов на непересекающиеся подмножества и позволяет эффективно:

1. **Find** — найти представителя (корень) множества, содержащего элемент
2. **Union** — объединить два множества

```
Начально: {1}, {2}, {3}, {4}, {5}

Union(1, 2): {1,2}, {3}, {4}, {5}
Union(3, 4): {1,2}, {3,4}, {5}
Union(1, 3): {1,2,3,4}, {5}

Find(2) = 1 (или представитель множества {1,2,3,4})
```

### Применения

- **Проверка связности** — достижимость между двумя вершинами
- **Кластеризация** — группирование связанных элементов
- **Поиск циклов** — в неориентированных графах
- **Минимальное остовное дерево** — алгоритм Краскала
- **Эквивалентность классов** — социальные сети, аккаунты

---

## Базовая реализация

### Наивный подход (неэффективно)

```go
// Базовая реализация (без оптимизаций)
type BasicDSU struct {
	parent []int
}

func NewBasicDSU(n int) *BasicDSU {
	parent := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
	}
	return &BasicDSU{parent}
}

// O(n) в худшем случае
func (dsu *BasicDSU) Find(x int) int {
	if dsu.parent[x] != x {
		dsu.parent[x] = dsu.Find(dsu.parent[x]) // Path compression
		return dsu.parent[x]
	}
	return x
}

// O(n) в худшем случае
func (dsu *BasicDSU) Union(x, y int) {
	px := dsu.Find(x)
	py := dsu.Find(y)
	if px != py {
		dsu.parent[px] = py
	}
}

// O(1)
func (dsu *BasicDSU) Connected(x, y int) bool {
	return dsu.Find(x) == dsu.Find(y)
}
```

---

## Оптимизации

### 1. Path Compression

**Path Compression** — при поиске представителя направляем все узлы прямо к корню.

```
До:         После:
  x           root
  |          / | \
  a         x  a  b
  |
  b
  |
 root
```

```go
// Find с path compression
func (dsu *DSU) Find(x int) int {
	if dsu.parent[x] != x {
		dsu.parent[x] = dsu.Find(dsu.parent[x]) // Рекурсивное сжатие
	}
	return dsu.parent[x]
}
```

### 2. Union by Rank

**Union by Rank** — при объединении подвешиваем меньшее дерево к большему.

```go
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

// Union by rank
func (dsu *DSU) Union(x, y int) bool {
	px := dsu.Find(x)
	py := dsu.Find(y)
	
	if px == py {
		return false // Уже в одном множестве
	}
	
	// Подвесить меньшее дерево к большему
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

func (dsu *DSU) Connected(x, y int) bool {
	return dsu.Find(x) == dsu.Find(y)
}
```

### 3. Union by Size

Альтернатива union by rank — использовать размер множества:

```go
type DSUSize struct {
	parent []int
	size   []int
}

func NewDSUSize(n int) *DSUSize {
	parent := make([]int, n)
	size := make([]int, n)
	for i := 0; i < n; i++ {
		parent[i] = i
		size[i] = 1
	}
	return &DSUSize{parent, size}
}

func (dsu *DSUSize) Find(x int) int {
	if dsu.parent[x] != x {
		dsu.parent[x] = dsu.Find(dsu.parent[x])
	}
	return dsu.parent[x]
}

// Union by size
func (dsu *DSUSize) Union(x, y int) bool {
	px := dsu.Find(x)
	py := dsu.Find(y)
	
	if px == py {
		return false
	}
	
	// Подвесить меньшее множество к большему
	if dsu.size[px] < dsu.size[py] {
		dsu.parent[px] = py
		dsu.size[py] += dsu.size[px]
	} else {
		dsu.parent[py] = px
		dsu.size[px] += dsu.size[py]
	}
	
	return true
}

func (dsu *DSUSize) GetSize(x int) int {
	px := dsu.Find(x)
	return dsu.size[px]
}
```

---

## Применение в задачах

### 1. Number of Provinces (LeetCode 547)

```go
// Найти количество провинций (связных компонент)
func FindCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	dsu := NewDSU(n)
	
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if isConnected[i][j] == 1 {
				dsu.Union(i, j)
			}
		}
	}
	
	// Подсчитать уникальные представители
	count := 0
	for i := 0; i < n; i++ {
		if dsu.Find(i) == i {
			count++
		}
	}
	
	return count
}
```

### 2. Redundant Connection (LeetCode 684)

```go
// Найти ребро, которое создаёт цикл
func FindRedundantConnection(edges [][]int) []int {
	n := len(edges)
	dsu := NewDSU(n + 1)
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		
		// Если они уже в одном множестве, это ребро создаёт цикл
		if dsu.Find(u) == dsu.Find(v) {
			return edge
		}
		
		dsu.Union(u, v)
	}
	
	return []int{}
}
```

### 3. Accounts Merge (LeetCode 721)

```go
// Объединить аккаунты одного человека
func AccountsMerge(accounts [][]string) [][]string {
	emailToOwner := make(map[string]int)
	dsu := NewDSU(len(accounts))
	
	// Первый проход: связать аккаунты содержащие одинаковые emails
	for i, account := range accounts {
		for j := 1; j < len(account); j++ {
			email := account[j]
			
			if owner, exists := emailToOwner[email]; exists {
				dsu.Union(i, owner)
			} else {
				emailToOwner[email] = i
			}
		}
	}
	
	// Второй проход: сгруппировать emails по аккаунту
	emailsByOwner := make(map[int][]string)
	for email, owner := range emailToOwner {
		root := dsu.Find(owner)
		emailsByOwner[root] = append(emailsByOwner[root], email)
	}
	
	// Построить результат
	result := [][]string{}
	for i, emails := range emailsByOwner {
		account := []string{accounts[i][0]}
		sort.Strings(emails)
		account = append(account, emails...)
		result = append(result, account)
	}
	
	return result
}
```

---

## Практические примеры

### Пример 1: Detect Cycle in Undirected Graph

```go
// Проверить наличие цикла в неориентированном графе
func HasCycleDSU(n int, edges [][]int) bool {
	dsu := NewDSU(n)
	
	for _, edge := range edges {
		u, v := edge[0], edge[1]
		
		if !dsu.Union(u, v) {
			return true // Цикл найден
		}
	}
	
	return false
}
```

### Пример 2: Employees Under the Same Manager

```go
// Найти всех подчинённых менеджера (по цепочке управления)
func FindEmployeesUnderManager(managers map[int]int, managerId int) []int {
	dsu := NewDSU(len(managers))
	
	for employee, manager := range managers {
		dsu.Union(employee, manager)
	}
	
	result := []int{}
	for employee := range managers {
		if dsu.Find(employee) == dsu.Find(managerId) && employee != managerId {
			result = append(result, employee)
		}
	}
	
	return result
}
```

### Пример 3: Kruskal's Algorithm для MST

```go
type Edge struct {
	u, v, weight int
}

// Найти минимальное остовное дерево используя Kruskal
func KruskalMST(n int, edges []Edge) []Edge {
	// Сортировать рёбра по весу
	sort.Slice(edges, func(i, j int) bool {
		return edges[i].weight < edges[j].weight
	})
	
	dsu := NewDSU(n)
	mst := []Edge{}
	
	for _, edge := range edges {
		if dsu.Union(edge.u, edge.v) {
			mst = append(mst, edge)
			if len(mst) == n-1 {
				break
			}
		}
	}
	
	return mst
}
```

---

## Анализ сложности

### DSU без оптимизаций

| Операция | Время |
|----------|-------|
| Find | O(n) |
| Union | O(n) |

### DSU с Path Compression

| Операция | Время |
|----------|-------|
| Find | O(α(n))* |
| Union | O(α(n))* |

*α(n) — обратная функция Аккермана (практически константа ≤ 4)

### DSU с Union by Rank + Path Compression

| Операция | Время |
|----------|-------|
| Find | O(α(n)) |
| Union | O(α(n)) |
| **m операций** | **O(m·α(n))** |

### Ацкерманова функция

```
α(n) — обратная функция Аккермана

α(1) = 1
α(2) = 1
α(3) = 2
α(4) = 2
α(2^16) = 3
α(2^65536) = 4

Для практических n: α(n) ≤ 4
```

---

## Сравнение методов

| Метод | Find | Union | Память | Применение |
|-------|------|-------|--------|-----------|
| **Базовый** | O(n) | O(n) | O(n) | Образование |
| **Path Compression** | O(α(n)) | O(α(n)) | O(n) | Практика |
| **Union by Rank** | O(log n) | O(log n) | O(n) | Гарантия |
| **Оба** | O(α(n)) | O(α(n)) | O(n) | Оптимум |

---

## Рекомендуемые задачи на LeetCode

- [Number of Provinces](https://leetcode.com/problems/number-of-provinces/) — подсчёт компонент
- [Redundant Connection](https://leetcode.com/problems/redundant-connection/) — поиск лишнего ребра
- [Accounts Merge](https://leetcode.com/problems/accounts-merge/) — объединение аккаунтов
- [Friend Circles](https://leetcode.com/problems/friend-circles/) — связные компоненты
- [Couples Holding Hands](https://leetcode.com/problems/couples-holding-hands/) — оптимальные пары
- [Smallest String With Swaps](https://leetcode.com/problems/smallest-string-with-swaps/) — перестановки в компонентах

---

## Важные свойства

1. **Equi-valence Relation**: DSU моделирует отношение эквивалентности
   - Рефлексивность: x ~ x
   - Симметричность: x ~ y ⟹ y ~ x
   - Транзитивность: x ~ y, y ~ z ⟹ x ~ z

2. **Лемма об уровне**: С union by rank высота дерева не превышает O(log n)

3. **Инвариант Path Compression**: Все узлы на пути Find указывают прямо на корень

---

## Заключение

DSU — это одна из самых элегантных структур данных в Computer Science:

- **Простота** — несколько строк кода
- **Эффективность** — O(α(n)) ≈ O(1) на практике
- **Универсальность** — применяется во многих алгоритмах
- **Теория** — красивая математика

Используется в:
- Алгоритме Краскала (MST)
- Проверке связности
- Обнаружении циклов
- Эквивалентности и кластеризации

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
