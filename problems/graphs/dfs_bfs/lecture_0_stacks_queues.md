# Лекция 0: Стеки, Очереди и Основные Структуры Данных

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Fundamental

---

## Содержание
1. [Стеки (Stacks)](#стеки-stacks)
2. [Очереди (Queues)](#очереди-queues)
3. [Приложения в алгоритмах](#приложения-в-алгоритмах)
4. [Практические примеры](#практические-примеры)
5. [Анализ сложности](#анализ-сложности)

---

## Стеки (Stacks)

### Определение

**Стек** — это структура данных, которая работает по принципу **LIFO** (Last In, First Out):
- Последний добавленный элемент — первый удаляемый

```
Push(3) → [3]
Push(1) → [3, 1]
Push(4) → [3, 1, 4]
Pop()   → [3, 1]  (возвращает 4)
Peek()  → 1 (смотрит на верх, не удаляет)
```

### Реализация на Go

```go
// Стек с использованием срезов
type Stack struct {
	items []int
}

func (s *Stack) Push(value int) {
	s.items = append(s.items, value)
}

func (s *Stack) Pop() (int, bool) {
	if len(s.items) == 0 {
		return 0, false
	}
	value := s.items[len(s.items)-1]
	s.items = s.items[:len(s.items)-1]
	return value, true
}

func (s *Stack) Peek() (int, bool) {
	if len(s.items) == 0 {
		return 0, false
	}
	return s.items[len(s.items)-1], true
}

func (s *Stack) IsEmpty() bool {
	return len(s.items) == 0
}

func (s *Stack) Size() int {
	return len(s.items)
}
```

### Приложения стеков

```
1. Обратная польская нотация (RPN)
2. Проверка сбалансированности скобок
3. Обход графов (DFS)
4. Вычисление выражений
5. История отмены/повтора (Undo/Redo)
6. Компилятор (синтаксический анализ)
```

### Пример 1: Valid Parentheses

```go
// Проверить сбалансированность скобок
func IsValid(s string) bool {
	stack := &Stack{}
	pairs := map[rune]rune{
		')': '(',
		'}': '{',
		']': '[',
	}
	
	for _, char := range s {
		if char == '(' || char == '{' || char == '[' {
			stack.Push(int(char))
		} else if closer, ok := pairs[char]; ok {
			top, found := stack.Pop()
			if !found || rune(top) != closer {
				return false
			}
		}
	}
	
	return stack.IsEmpty()
}
```

### Пример 2: Evaluate Reverse Polish Notation

```go
// Вычислить выражение в обратной польской нотации
func EvalRPN(tokens []string) int {
	stack := []int{}
	
	for _, token := range tokens {
		switch token {
		case "+", "-", "*", "/":
			if len(stack) < 2 {
				return 0
			}
			
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			
			var result int
			switch token {
			case "+":
				result = a + b
			case "-":
				result = a - b
			case "*":
				result = a * b
			case "/":
				result = a / b
			}
			
			stack = append(stack, result)
		default:
			num := 0
			for _, digit := range token {
				num = num*10 + int(digit-'0')
			}
			stack = append(stack, num)
		}
	}
	
	if len(stack) != 1 {
		return 0
	}
	return stack[0]
}
```

---

## Очереди (Queues)

### Определение

**Очередь** — это структура данных, которая работает по принципу **FIFO** (First In, First Out):
- Первый добавленный элемент — первый удаляемый

```
Enqueue(3) → [3]
Enqueue(1) → [3, 1]
Enqueue(4) → [3, 1, 4]
Dequeue()  → [1, 4]  (возвращает 3)
Front()    → 1 (смотрит на начало)
```

### Реализация на Go

```go
// Очередь с использованием контейнера
import "container/list"

type Queue struct {
	items *list.List
}

func NewQueue() *Queue {
	return &Queue{items: list.New()}
}

func (q *Queue) Enqueue(value int) {
	q.items.PushBack(value)
}

func (q *Queue) Dequeue() (int, bool) {
	if q.items.Len() == 0 {
		return 0, false
	}
	element := q.items.Front()
	q.items.Remove(element)
	return element.Value.(int), true
}

func (q *Queue) Front() (int, bool) {
	if q.items.Len() == 0 {
		return 0, false
	}
	return q.items.Front().Value.(int), true
}

func (q *Queue) IsEmpty() bool {
	return q.items.Len() == 0
}

func (q *Queue) Size() int {
	return q.items.Len()
}

// Оптимизированная версия со срезом (circular buffer)
type CircularQueue struct {
	items []int
	front int
	rear  int
	size  int
	cap   int
}

func NewCircularQueue(capacity int) *CircularQueue {
	return &CircularQueue{
		items: make([]int, capacity),
		front: 0,
		rear:  -1,
		size:  0,
		cap:   capacity,
	}
}

func (q *CircularQueue) Enqueue(value int) bool {
	if q.size == q.cap {
		return false // Очередь полна
	}
	q.rear = (q.rear + 1) % q.cap
	q.items[q.rear] = value
	q.size++
	return true
}

func (q *CircularQueue) Dequeue() (int, bool) {
	if q.size == 0 {
		return 0, false
	}
	value := q.items[q.front]
	q.front = (q.front + 1) % q.cap
	q.size--
	return value, true
}

func (q *CircularQueue) Front() (int, bool) {
	if q.size == 0 {
		return 0, false
	}
	return q.items[q.front], true
}

func (q *CircularQueue) IsEmpty() bool {
	return q.size == 0
}

func (q *CircularQueue) IsFull() bool {
	return q.size == q.cap
}
```

### Приложения очередей

```
1. BFS (поиск в ширину)
2. Системы обработки задач (job queues)
3. Буферизация данных
4. Коммуникация между процессами
5. Очередь задач на сервере
6. Симуляция (моделирование процессов)
```

### Пример 1: Number of Islands (BFS вариант)

```go
// Найти количество островов используя очередь (BFS)
func NumIslandsBFS(grid [][]byte) int {
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
				bfsIsland(grid, i, j, visited, m, n)
				count++
			}
		}
	}
	
	return count
}

func bfsIsland(grid [][]byte, startI, startJ int, visited [][]bool, m, n int) {
	queue := NewQueue()
	queue.Enqueue(startI)
	queue.Enqueue(startJ)
	visited[startI][startJ] = true
	
	directions := []int{-1, 0, 1, 0, -1} // чтобы циклить удобно
	
	for !queue.IsEmpty() {
		i, _ := queue.Dequeue()
		j, _ := queue.Dequeue()
		
		for d := 0; d < 4; d++ {
			ni := i + directions[d]
			nj := j + directions[d+1]
			
			if ni >= 0 && ni < m && nj >= 0 && nj < n &&
				grid[ni][nj] == '1' && !visited[ni][nj] {
				visited[ni][nj] = true
				queue.Enqueue(ni)
				queue.Enqueue(nj)
			}
		}
	}
}
```

### Пример 2: Moving Average from Data Stream

```go
// Вычислять скользящее среднее для потока данных
type MovingAverage struct {
	queue *list.List
	sum   int
	size  int
	cap   int
}

func NewMovingAverage(size int) *MovingAverage {
	return &MovingAverage{
		queue: list.New(),
		sum:   0,
		size:  0,
		cap:   size,
	}
}

func (m *MovingAverage) Next(val int) float64 {
	if m.size == m.cap {
		front := m.queue.Front()
		m.sum -= front.Value.(int)
		m.queue.Remove(front)
		m.size--
	}
	
	m.queue.PushBack(val)
	m.sum += val
	m.size++
	
	return float64(m.sum) / float64(m.size)
}
```

---

## Приложения в алгоритмах

### Стеки vs Очереди

```
            Стек (LIFO)              Очередь (FIFO)
┌──────────────────────────┬──────────────────────────┐
│ Операция  O(1)           │ Операция  O(1)           │
│ Push      добавить сверху│ Enqueue   добавить в конец
│ Pop       удалить сверху │ Dequeue   удалить с начала
│ Peek      смотреть сверху│ Front     смотреть начало│
│                          │                          │
│ Применение:              │ Применение:              │
│ - DFS                    │ - BFS                    │
│ - Скобки                 │ - Job scheduling         │
│ - Undo/Redo              │ - Буферизация            │
│ - Вычисления             │ - Симуляция              │
└──────────────────────────┴──────────────────────────┘
```

### Deque (Double-Ended Queue)

```go
// Двусторонняя очередь - поддерживает операции с обоих концов
type Deque struct {
	items *list.List
}

func (d *Deque) PushFront(value int) {
	d.items.PushFront(value)
}

func (d *Deque) PushBack(value int) {
	d.items.PushBack(value)
}

func (d *Deque) PopFront() (int, bool) {
	if d.items.Len() == 0 {
		return 0, false
	}
	elem := d.items.Front()
	d.items.Remove(elem)
	return elem.Value.(int), true
}

func (d *Deque) PopBack() (int, bool) {
	if d.items.Len() == 0 {
		return 0, false
	}
	elem := d.items.Back()
	d.items.Remove(elem)
	return elem.Value.(int), true
}

// Пример: Sliding Window Maximum
func MaxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 || k == 0 {
		return []int{}
	}
	
	deque := list.New()
	result := []int{}
	
	for i, num := range nums {
		// Удалить элементы вне окна
		if deque.Len() > 0 && deque.Front().Value.(int) < i-k+1 {
			deque.Remove(deque.Front())
		}
		
		// Удалить меньшие элементы с конца
		for deque.Len() > 0 {
			back := deque.Back()
			if nums[back.Value.(int)] <= num {
				deque.Remove(back)
			} else {
				break
			}
		}
		
		deque.PushBack(i)
		
		// Первое окно готово
		if i >= k-1 {
			result = append(result, nums[deque.Front().Value.(int)])
		}
	}
	
	return result
}
```

---

## Практические примеры

### Пример 1: Daily Temperatures

```go
// Найти дни до первого дня с более высокой температурой (используя стек)
func DailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	result := make([]int, n)
	stack := []int{} // Индексы в убывающем порядке температур
	
	for i := 0; i < n; i++ {
		for len(stack) > 0 && temperatures[stack[len(stack)-1]] < temperatures[i] {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			result[idx] = i - idx
		}
		stack = append(stack, i)
	}
	
	return result
}
```

### Пример 2: Trapping Rain Water

```go
// Вычислить объём воды, которая может быть заперта между стенами
func Trap(height []int) int {
	if len(height) == 0 {
		return 0
	}
	
	stack := []int{} // Индексы в возрастающем порядке высот
	water := 0
	
	for i := 0; i < len(height); i++ {
		for len(stack) > 0 && height[stack[len(stack)-1]] < height[i] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			
			if len(stack) == 0 {
				break
			}
			
			left := stack[len(stack)-1]
			width := i - left - 1
			h := min(height[left], height[i]) - height[top]
			water += width * h
		}
		stack = append(stack, i)
	}
	
	return water
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

### Пример 3: LRU Cache (используя Deque и Hash Map)

```go
// Least Recently Used Cache
type LRUCache struct {
	cache map[int]*list.Element
	list  *list.List
	cap   int
}

type pair struct {
	key   int
	value int
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		cache: make(map[int]*list.Element),
		list:  list.New(),
		cap:   capacity,
	}
}

func (lru *LRUCache) Get(key int) int {
	if elem, ok := lru.cache[key]; ok {
		lru.list.MoveToBack(elem)
		return elem.Value.(pair).value
	}
	return -1
}

func (lru *LRUCache) Put(key int, value int) {
	if elem, ok := lru.cache[key]; ok {
		elem.Value = pair{key, value}
		lru.list.MoveToBack(elem)
		return
	}
	
	elem := lru.list.PushBack(pair{key, value})
	lru.cache[key] = elem
	
	if lru.list.Len() > lru.cap {
		front := lru.list.Front()
		lru.list.Remove(front)
		delete(lru.cache, front.Value.(pair).key)
	}
}
```

---

## Анализ сложности

| Операция | Стек | Очередь | Deque |
|----------|------|---------|-------|
| Push/Enqueue/PushFront | O(1) | O(1) | O(1) |
| Pop/Dequeue/PopFront | O(1) | O(1) | O(1) |
| Peek/Front/Back | O(1) | O(1) | O(1) |
| **Память** | O(n) | O(n) | O(n) |

---

## Рекомендуемые задачи на LeetCode

- [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/)
- [Evaluate Reverse Polish Notation](https://leetcode.com/problems/evaluate-reverse-polish-notation/)
- [Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
- [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/)
- [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
- [LRU Cache](https://leetcode.com/problems/lru-cache/)
- [Number of Islands](https://leetcode.com/problems/number-of-islands/) (BFS вариант)

---

## Заключение

Стеки и очереди — это **фундаментальные структуры данных**:

1. **Стеки** — LIFO, для DFS и обработки
2. **Очереди** — FIFO, для BFS и буферизации
3. **Deque** — оба конца, для скользящего окна
4. **LRU Cache** — комбинация очереди и hash map

Понимание этих структур критично для:
- Решения задач на графах
- Обработки потоков данных
- Оптимизации алгоритмов
- Проектирования систем

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
