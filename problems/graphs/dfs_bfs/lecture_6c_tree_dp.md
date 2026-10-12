# Лекция 6c: Tree DP (Dynamic Programming на деревьях)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Advanced

---

## Содержание
1. [Введение в Tree DP](#введение-в-tree-dp)
2. [Переоценка и переиспользование](#переоценка-и-переиспользование)
3. [Перерасчёт и рекалкуляция](#перерасчёт-и-рекалкуляция)
4. [Практические примеры](#практические-примеры)
5. [Анализ сложности](#анализ-сложности)

---

## Введение в Tree DP

### Отличие от обычного DP

```
Обычное DP:           Tree DP:
Линейный массив       Структура дерева
dp[i] зависит         dp[node] зависит
от dp[i-1]            от dp[child1], dp[child2], ...
```

### Базовый шаблон

```go
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func DFS(node *TreeNode) int {
	if node == nil {
		return 0
	}
	
	left := DFS(node.Left)
	right := DFS(node.Right)
	
	// Вычислить результат для текущего узла
	result := combine(node, left, right)
	
	return result
}
```

---

## Переоценка и переиспользование

### Пример 1: Maximum Path Sum

```go
// Максимальная сумма пути в дереве
func MaxPathSum(root *TreeNode) int {
	maxSum := math.MinInt
	
	// DFS возвращает максимальную сумму пути от node к листу
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		
		// Максимальный путь вниз от левого поддерева
		leftMax := max(0, dfs(node.Left))
		// Максимальный путь вниз от правого поддерева
		rightMax := max(0, dfs(node.Right))
		
		// Обновить глобальный максимум
		maxSum = max(maxSum, node.Val+leftMax+rightMax)
		
		// Вернуть максимальный путь, идущий вниз
		return node.Val + max(leftMax, rightMax)
	}
	
	dfs(root)
	return maxSum
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

### Пример 2: Diameter of Binary Tree

```go
// Диаметр дерева (самый длинный путь между двумя узлами)
func DiameterOfBinaryTree(root *TreeNode) int {
	diameter := 0
	
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		
		left := dfs(node.Left)
		right := dfs(node.Right)
		
		// Обновить диаметр
		diameter = max(diameter, left+right)
		
		// Вернуть высоту
		return 1 + max(left, right)
	}
	
	dfs(root)
	return diameter
}
```

### Пример 3: Balanced Binary Tree

```go
// Проверить сбалансированность дерева
func IsBalanced(root *TreeNode) bool {
	_, ok := dfsBalance(root)
	return ok
}

func dfsBalance(node *TreeNode) (int, bool) {
	if node == nil {
		return 0, true
	}
	
	leftHeight, leftOk := dfsBalance(node.Left)
	if !leftOk {
		return 0, false
	}
	
	rightHeight, rightOk := dfsBalance(node.Right)
	if !rightOk {
		return 0, false
	}
	
	// Проверить баланс текущего узла
	if abs(leftHeight-rightHeight) > 1 {
		return 0, false
	}
	
	return 1 + max(leftHeight, rightHeight), true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
```

---

## Перерасчёт и рекалкуляция

### Пример: House Robber III

```go
// Ограбить дом в виде дерева (не смежные узлы)
func Rob(root *TreeNode) int {
	left, _ := dfsRob(root)
	return left
}

// Возвращает (maxWithRoot, maxWithoutRoot)
func dfsRob(node *TreeNode) (int, int) {
	if node == nil {
		return 0, 0
	}
	
	leftWith, leftWithout := dfsRob(node.Left)
	rightWith, rightWithout := dfsRob(node.Right)
	
	// Если берём текущий узел, не можем брать соседей
	withRoot := node.Val + leftWithout + rightWithout
	
	// Если не берём, можем взять или не взять соседей (берём максимум)
	withoutRoot := max(leftWith, leftWithout) + max(rightWith, rightWithout)
	
	return withRoot, withoutRoot
}
```

### Пример: Binary Tree Maximum Path Sum (улучшенный)

```go
// Максимум для поддерева + глобальный максимум
func MaxPathSumV2(root *TreeNode) int {
	result := math.MinInt
	
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		
		// Важно: если путь отрицательный, не берём его
		leftSum := max(0, dfs(node.Left))
		rightSum := max(0, dfs(node.Right))
		
		// Максимум через текущий узел
		result = max(result, node.Val+leftSum+rightSum)
		
		// Максимум вниз (один из потомков или сам узел)
		return node.Val + max(leftSum, rightSum)
	}
	
	dfs(root)
	return result
}
```

---

## Практические примеры

### Пример 1: Lowest Common Ancestor

```go
// Найти наименьший общий предок двух узлов
func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	
	if root == p || root == q {
		return root
	}
	
	left := LowestCommonAncestor(root.Left, p, q)
	right := LowestCommonAncestor(root.Right, p, q)
	
	// Оба в разных поддеревьях
	if left != nil && right != nil {
		return root
	}
	
	// Оба в левом или оба в правом
	if left != nil {
		return left
	}
	return right
}
```

### Пример 2: Binary Tree Cameras

```go
// Минимальное количество камер для наблюдения всего дерева
func MinCameraCover(root *TreeNode) int {
	// 0: не наблюдается, 1: наблюдается, 2: имеет камеру
	state, count := dfsCamera(root)
	if state == 0 {
		count++
	}
	return count
}

// Возвращает (состояние, минимум камер)
func dfsCamera(node *TreeNode) (int, int) {
	if node == nil {
		return 1, 0 // листья считаются наблюдаемыми
	}
	
	leftState, leftCam := dfsCamera(node.Left)
	rightState, rightCam := dfsCamera(node.Right)
	
	cams := leftCam + rightCam
	
	// Если оба дитя имеют камеры, это узел наблюдается
	if leftState == 2 || rightState == 2 {
		return 1, cams
	}
	
	// Если оба ребенка не наблюдаются, нужна камера здесь
	if leftState == 0 || rightState == 0 {
		return 2, cams + 1
	}
	
	// Оба ребенка наблюдаются, но без камер
	return 0, cams
}
```

### Пример 3: Distribute Coins in Binary Tree

```go
// Распределить монеты так, чтобы каждый узел имел 1 монету
func DistributeCoins(root *TreeNode) int {
	moves := 0
	
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		
		left := dfs(node.Left)
		right := dfs(node.Right)
		
		// Записываем ходы
		moves += abs(left) + abs(right)
		
		// Возвращаем избыток/дефицит после сбалансирования этого узла
		// node имеет node.Val монет, нужна 1
		// left и right монеты передаются + дефицит/избыток
		return node.Val + left + right - 1
	}
	
	dfs(root)
	return moves
}
```

---

## Шаблоны Tree DP

### Паттерн 1: Top-Down DP

```go
// Мемоизация сверху вниз
type TreeDPTopDown struct {
	memo map[*TreeNode]int
}

func NewTreeDPTopDown() *TreeDPTopDown {
	return &TreeDPTopDown{
		memo: make(map[*TreeNode]int),
	}
}

func (t *TreeDPTopDown) DFS(node *TreeNode) int {
	if node == nil {
		return 0
	}
	
	if val, ok := t.memo[node]; ok {
		return val
	}
	
	result := node.Val + t.DFS(node.Left) + t.DFS(node.Right)
	t.memo[node] = result
	return result
}
```

### Паттерн 2: Bottom-Up DP

```go
// Постройка вверх от листьев
func TreeDPBottomUp(root *TreeNode) int {
	result := 0
	
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		
		// Сначала обработать детей (Bottom-Up)
		left := dfs(node.Left)
		right := dfs(node.Right)
		
		// Потом текущий узел
		current := combine(node, left, right)
		
		// Глобальное обновление
		result = max(result, current)
		
		return current
	}
	
	dfs(root)
	return result
}

func combine(node *TreeNode, left, right int) int {
	return node.Val + left + right
}
```

---

## Анализ сложности

### Все Tree DP задачи

| Аспект | Сложность |
|--------|-----------|
| **Время** | O(n) |
| **Память** | O(h) где h = высота дерева |
| **Мемоизация** | O(n) дополнительно |

Где n = количество узлов в дереве.

---

## Рекомендуемые задачи на LeetCode

- [Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
- [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- [House Robber III](https://leetcode.com/problems/house-robber-iii/)
- [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/)
- [Distribute Coins in Binary Tree](https://leetcode.com/problems/distribute-coins-in-binary-tree/)
- [Maximum Product of Splitted Binary Tree](https://leetcode.com/problems/maximum-product-of-splitted-binary-tree/)

---

## Заключение

1. **Tree DP** — применяется к иерархическим структурам
2. **Bottom-Up** — обрабатываем детей перед родителем
3. **Мемоизация** — если решение пересчитывается много раз
4. **Two-State DP** — часто возвращаем пару значений (с/без)

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
