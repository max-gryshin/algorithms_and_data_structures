# Лекция 6b: Segment Tree и Fenwick Tree (Range Queries)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Advanced

---

## Содержание
1. [Range Queries Problem](#range-queries-problem)
2. [Segment Tree](#segment-tree)
3. [Fenwick Tree (Binary Indexed Tree)](#fenwick-tree-binary-indexed-tree)
4. [Сравнение методов](#сравнение-методов)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Range Queries Problem

### Определение

**Range Query** — это задача ответить на запрос о некоторой функции на диапазоне [L, R]:
- Сумма элементов с индексом L по R
- Минимум/максимум на диапазоне
- XOR всех элементов
- И т.д.

```
Массив: [1, 3, 5, 7, 9, 11]
        [0, 1, 2, 3, 4, 5]

Query(2, 4) = сумма элементов с индекса 2 по 4 = 5+7+9 = 21
Update(3, 10) = изменить элемент с индекса 3 на 10

После обновления:
Массив: [1, 3, 5, 10, 9, 11]
Query(2, 4) = 5+10+9 = 24
```

### Наивное решение

```go
// Без оптимизаций
func RangeSum(arr []int, l, r int) int {
	sum := 0
	for i := l; i <= r; i++ {
		sum += arr[i]
	}
	return sum // O(n)
}

func Update(arr []int, idx, val int) {
	arr[idx] = val // O(1)
}
```

**Проблема**: Query O(n), плохо при m запросов

---

## Segment Tree

### Идея

**Segment Tree** — это дерево, где каждый узел представляет диапазон массива и хранит результат функции (сумму, минимум, максимум) на этом диапазоне.

```
Массив: [1, 3, 5, 7]

         [16]        ← сумма [0,3]
        /    \
      [4]     [12]   ← [0,1], [2,3]
     /  \    /   \
   [1] [3] [5]  [7]  ← листья [0], [1], [2], [3]
```

### Реализация Segment Tree

```go
type SegmentTree struct {
	tree []int
	n    int
}

func NewSegmentTree(arr []int) *SegmentTree {
	n := len(arr)
	st := &SegmentTree{
		tree: make([]int, 4*n),
		n:    n,
	}
	st.build(arr, 0, 0, n-1)
	return st
}

// Построить дерево рекурсивно
func (st *SegmentTree) build(arr []int, node, start, end int) {
	if start == end {
		st.tree[node] = arr[start]
	} else {
		mid := (start + end) / 2
		leftChild := 2*node + 1
		rightChild := 2*node + 2
		
		st.build(arr, leftChild, start, mid)
		st.build(arr, rightChild, mid+1, end)
		
		st.tree[node] = st.tree[leftChild] + st.tree[rightChild]
	}
}

// Запрос суммы на диапазоне [l, r]
func (st *SegmentTree) RangeSum(l, r int) int {
	return st.rangeSumQuery(0, 0, st.n-1, l, r)
}

func (st *SegmentTree) rangeSumQuery(node, start, end, l, r int) int {
	if r < start || end < l {
		return 0 // Нет пересечения
	}
	
	if l <= start && end <= r {
		return st.tree[node] // Полностью в диапазоне
	}
	
	mid := (start + end) / 2
	leftChild := 2*node + 1
	rightChild := 2*node + 2
	
	leftSum := st.rangeSumQuery(leftChild, start, mid, l, r)
	rightSum := st.rangeSumQuery(rightChild, mid+1, end, l, r)
	
	return leftSum + rightSum
}

// Обновить элемент с индексом idx на значение val
func (st *SegmentTree) Update(idx, val int) {
	st.updateHelper(0, 0, st.n-1, idx, val)
}

func (st *SegmentTree) updateHelper(node, start, end, idx, val int) {
	if start == end {
		st.tree[node] = val
	} else {
		mid := (start + end) / 2
		leftChild := 2*node + 1
		rightChild := 2*node + 2
		
		if idx <= mid {
			st.updateHelper(leftChild, start, mid, idx, val)
		} else {
			st.updateHelper(rightChild, mid+1, end, idx, val)
		}
		
		st.tree[node] = st.tree[leftChild] + st.tree[rightChild]
	}
}

// Поиск минимума на диапазоне
func (st *SegmentTree) RangeMin(l, r int) int {
	return st.rangeMinQuery(0, 0, st.n-1, l, r)
}

func (st *SegmentTree) rangeMinQuery(node, start, end, l, r int) int {
	if r < start || end < l {
		return math.MaxInt
	}
	
	if l <= start && end <= r {
		return st.tree[node]
	}
	
	mid := (start + end) / 2
	leftChild := 2*node + 1
	rightChild := 2*node + 2
	
	leftMin := st.rangeMinQuery(leftChild, start, mid, l, r)
	rightMin := st.rangeMinQuery(rightChild, mid+1, end, l, r)
	
	return min(leftMin, rightMin)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Lazy Propagation (Оптимизация)

Для обновления диапазонов (не одного элемента):

```go
// Segment Tree с Lazy Propagation
type LazySegmentTree struct {
	tree []int
	lazy []int
	n    int
}

func NewLazySegmentTree(n int) *LazySegmentTree {
	return &LazySegmentTree{
		tree: make([]int, 4*n),
		lazy: make([]int, 4*n),
		n:    n,
	}
}

func (lst *LazySegmentTree) updateRange(node, start, end, l, r, val int) {
	if lst.lazy[node] != 0 {
		lst.tree[node] += (end - start + 1) * lst.lazy[node]
		
		if start != end {
			lst.lazy[2*node+1] += lst.lazy[node]
			lst.lazy[2*node+2] += lst.lazy[node]
		}
		lst.lazy[node] = 0
	}
	
	if start > end || start > r || end < l {
		return
	}
	
	if l <= start && end <= r {
		lst.tree[node] += (end - start + 1) * val
		if start != end {
			lst.lazy[2*node+1] += val
			lst.lazy[2*node+2] += val
		}
		return
	}
	
	mid := (start + end) / 2
	lst.updateRange(2*node+1, start, mid, l, r, val)
	lst.updateRange(2*node+2, mid+1, end, l, r, val)
	lst.tree[node] = lst.tree[2*node+1] + lst.tree[2*node+2]
}

func (lst *LazySegmentTree) queryRange(node, start, end, l, r int) int {
	if start > end || start > r || end < l {
		return 0
	}
	
	if lst.lazy[node] != 0 {
		lst.tree[node] += (end - start + 1) * lst.lazy[node]
		
		if start != end {
			lst.lazy[2*node+1] += lst.lazy[node]
			lst.lazy[2*node+2] += lst.lazy[node]
		}
		lst.lazy[node] = 0
	}
	
	if l <= start && end <= r {
		return lst.tree[node]
	}
	
	mid := (start + end) / 2
	leftSum := lst.queryRange(2*node+1, start, mid, l, r)
	rightSum := lst.queryRange(2*node+2, mid+1, end, l, r)
	
	return leftSum + rightSum
}
```

---

## Fenwick Tree (Binary Indexed Tree)

### Идея

**Fenwick Tree** (Binary Indexed Tree) — это более компактная структура для range queries, использующая свойства бинарной нумерации.

```
Индексы (1-based): 1 2 3 4 5 6 7 8
Бинарные:          001 010 011 100 101 110 111 1000

BIT[i] хранит сумму элементов от индекса (i - (i & -i) + 1) до i
```

### Реализация Fenwick Tree

```go
type FenwickTree struct {
	tree []int
	n    int
}

func NewFenwickTree(n int) *FenwickTree {
	return &FenwickTree{
		tree: make([]int, n+1),
		n:    n,
	}
}

// Инициализировать из массива
func NewFenwickTreeFromArray(arr []int) *FenwickTree {
	ft := NewFenwickTree(len(arr))
	for i, val := range arr {
		ft.Update(i, val)
	}
	return ft
}

// Получить самый последний установленный бит
func lowbit(x int) int {
	return x & (-x)
}

// Обновить элемент с индексом idx на delta
func (ft *FenwickTree) Update(idx, delta int) {
	idx++ // 1-based indexing
	for idx <= ft.n {
		ft.tree[idx] += delta
		idx += lowbit(idx)
	}
}

// Получить префиксную сумму до индекса idx (inclusive, 0-based)
func (ft *FenwickTree) PrefixSum(idx int) int {
	idx++ // 1-based indexing
	sum := 0
	for idx > 0 {
		sum += ft.tree[idx]
		idx -= lowbit(idx)
	}
	return sum
}

// Получить сумму на диапазоне [l, r]
func (ft *FenwickTree) RangeSum(l, r int) int {
	if l == 0 {
		return ft.PrefixSum(r)
	}
	return ft.PrefixSum(r) - ft.PrefixSum(l-1)
}

// Найти k-й нулевой элемент (0-based)
func (ft *FenwickTree) FindKth(k int) int {
	// Бинарный поиск + Fenwick Tree
	left, right := 0, ft.n-1
	
	for left < right {
		mid := (left + right) / 2
		if ft.PrefixSum(mid) < k {
			left = mid + 1
		} else {
			right = mid
		}
	}
	
	return left
}
```

### Визуализация Fenwick Tree

```
Для массива [1, 2, 3, 4, 5]:

Индекс:    1  2  3  4  5
Массив:   [1, 2, 3, 4, 5]
BIT:      [1, 3, 3, 10, 5]

BIT[1] = 1           (сумма от 1 до 1)
BIT[2] = 1+2 = 3     (сумма от 1 до 2)
BIT[3] = 3           (сумма от 3 до 3)
BIT[4] = 1+2+3+4 = 10 (сумма от 1 до 4)
BIT[5] = 5           (сумма от 5 до 5)
```

---

## Сравнение методов

### По операциям

| Метод | Build | Query | Update |
|-------|-------|-------|--------|
| **Prefix Sum Array** | O(n) | O(1) | O(n) |
| **Segment Tree** | O(n) | O(log n) | O(log n) |
| **Fenwick Tree** | O(n log n) | O(log n) | O(log n) |
| **Segment Tree + Lazy** | O(n) | O(log n) | O(log n) range |

### По памяти

| Метод | Память |
|-------|---------|
| **Array** | O(n) |
| **Segment Tree** | O(4n) = O(n) |
| **Fenwick Tree** | O(n) |

### Когда использовать

```
Fenwick Tree:
  ✓ Проще реализовать
  ✓ Меньше кода
  ✓ Хорошо для суммы/XOR
  ✗ Сложнее для минимума

Segment Tree:
  ✓ Поддерживает минимум/максимум
  ✓ С Lazy Propagation для range updates
  ✓ Более гибкий
  ✗ Больше кода
```

---

## Практические примеры

### Пример 1: Range Sum Query - Mutable

```go
// LeetCode 307
type NumArray struct {
	ft *FenwickTree
}

func NewNumArray(nums []int) *NumArray {
	ft := NewFenwickTree(len(nums))
	for i, num := range nums {
		ft.Update(i, num)
	}
	return &NumArray{ft: ft}
}

func (na *NumArray) Update(index int, val int) {
	// Получить текущее значение
	current := na.ft.RangeSum(index, index)
	delta := val - current
	na.ft.Update(index, delta)
}

func (na *NumArray) SumRange(left int, right int) int {
	return na.ft.RangeSum(left, right)
}
```

### Пример 2: Count of Smaller Numbers After Self

```go
// Найти количество меньших элементов справа от каждого элемента
func CountSmaller(nums []int) []int {
	n := len(nums)
	result := make([]int, n)
	
	// Координатная компрессия
	sorted := make([]int, n)
	copy(sorted, nums)
	sort.Ints(sorted)
	
	rank := make(map[int]int)
	idx := 1
	for i := 0; i < n; i++ {
		if i == 0 || sorted[i] != sorted[i-1] {
			rank[sorted[i]] = idx
			idx++
		}
	}
	
	ft := NewFenwickTree(n + 1)
	
	// Обходим справа налево
	for i := n - 1; i >= 0; i-- {
		// Количество элементов меньше nums[i]
		if rank[nums[i]] > 1 {
			result[i] = ft.PrefixSum(rank[nums[i]] - 2)
		}
		ft.Update(rank[nums[i]]-1, 1)
	}
	
	return result
}
```

### Пример 3: 2D Range Sum Query

```go
// Range Sum в 2D массиве
type NumMatrix struct {
	ft    []*FenwickTree
	m, n  int
}

func NewNumMatrix(matrix [][]int) *NumMatrix {
	m, n := len(matrix), len(matrix[0])
	nm := &NumMatrix{
		ft: make([]*FenwickTree, m),
		m:  m,
		n:  n,
	}
	
	for i := 0; i < m; i++ {
		nm.ft[i] = NewFenwickTree(n)
		for j := 0; j < n; j++ {
			nm.ft[i].Update(j, matrix[i][j])
		}
	}
	
	return nm
}

func (nm *NumMatrix) Update(row, col, val int) {
	current := nm.ft[row].RangeSum(col, col)
	delta := val - current
	nm.ft[row].Update(col, delta)
}

func (nm *NumMatrix) SumRegion(row1, col1, row2, col2 int) int {
	sum := 0
	for i := row1; i <= row2; i++ {
		sum += nm.ft[i].RangeSum(col1, col2)
	}
	return sum
}
```

---

## Анализ сложности

### Построение

| Структура | Время |
|-----------|-------|
| **Segment Tree** | O(n) |
| **Fenwick Tree** | O(n log n) |

### Запрос и обновление

| Структура | Query | Update |
|-----------|-------|--------|
| **Array** | O(1) | O(1) |
| **Prefix Sum** | O(1) | O(n) |
| **Segment Tree** | O(log n) | O(log n) |
| **Fenwick Tree** | O(log n) | O(log n) |
| **Seg Tree + Lazy** | O(log n) | O(log n) диапазон |

---

## Рекомендуемые задачи на LeetCode

- [Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)
- [Range Sum Query 2D - Mutable](https://leetcode.com/problems/range-sum-query-2d-mutable/)
- [Counting Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [Reverse Pairs](https://leetcode.com/problems/reverse-pairs/)
- [Segment Tree Modification](https://leetcode.com/problems/segment-tree-modification/)
- [The Skyline Problem](https://leetcode.com/problems/the-skyline-problem/)

---

## Заключение

1. **Fenwick Tree** — проще, компактнее, для суммы/XOR
2. **Segment Tree** — универсальнее, поддерживает минимум/максимум
3. **Lazy Propagation** — для range updates
4. **Координатная компрессия** — важна для больших значений

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
