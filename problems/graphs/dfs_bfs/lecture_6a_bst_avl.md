# Лекция 6a: Binary Search Tree и AVL Tree (Поиск и Балансировка)

**Курс**: CS 161 — Design of Algorithms  
**Преподаватель**: Кафедра Computer Science, Stanford University  
**Уровень**: Intermediate

---

## Содержание
1. [Введение в двоичные деревья поиска](#введение-в-двоичные-деревья-поиска)
2. [Binary Search Tree (BST)](#binary-search-tree-bst)
3. [Балансировка дерева](#балансировка-дерева)
4. [AVL Tree](#avl-tree)
5. [Практические примеры](#практические-примеры)
6. [Анализ сложности](#анализ-сложности)

---

## Введение в двоичные деревья поиска

### Структура дерева

```
Дерево = набор узлов с иерархией:

       4
      / \
     2   6
    / \ / \
   1  3 5  7

Свойства:
- Корень (root): узел без родителя
- Листья (leaves): узлы без детей
- Высота: максимальное расстояние от корня до листа
```

### Реализация базовой структуры на Go

```go
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// Создать новый узел
func NewNode(val int) *TreeNode {
	return &TreeNode{Val: val}
}
```

---

## Binary Search Tree (BST)

### Определение и свойства

**BST** — это дерево, где для каждого узла:
- Все значения в **левом поддереве** < значение узла
- Все значения в **правом поддереве** > значение узла

```
Правильный BST:        Неправильный (не BST):
      4                      4
     / \                     / \
    2   6                    2   6
   / \ / \                  / \ / \
  1  3 5  7               1  3 5  8  (8 > 6, но в левом!)
```

### Реализация BST

```go
// Вставить значение в BST
func (root *TreeNode) Insert(val int) {
	if val < root.Val {
		if root.Left == nil {
			root.Left = NewNode(val)
		} else {
			root.Left.Insert(val)
		}
	} else if val > root.Val {
		if root.Right == nil {
			root.Right = NewNode(val)
		} else {
			root.Right.Insert(val)
		}
	}
	// val == root.Val: не вставляем дубликаты
}

// Поиск значения в BST
func (root *TreeNode) Search(val int) bool {
	if root == nil {
		return false
	}
	
	if val < root.Val {
		return root.Left.Search(val)
	} else if val > root.Val {
		return root.Right.Search(val)
	}
	
	return true // val == root.Val
}

// Найти минимальное значение
func (root *TreeNode) FindMin() int {
	for root.Left != nil {
		root = root.Left
	}
	return root.Val
}

// Найти максимальное значение
func (root *TreeNode) FindMax() int {
	for root.Right != nil {
		root = root.Right
	}
	return root.Val
}

// Удалить значение из BST
func (root *TreeNode) Delete(val int) *TreeNode {
	if root == nil {
		return nil
	}
	
	if val < root.Val {
		root.Left = root.Left.Delete(val)
	} else if val > root.Val {
		root.Right = root.Right.Delete(val)
	} else {
		// Найдено значение для удаления
		
		// Случай 1: нет левого поддерева
		if root.Left == nil {
			return root.Right
		}
		
		// Случай 2: нет правого поддерева
		if root.Right == nil {
			return root.Left
		}
		
		// Случай 3: оба поддерева существуют
		// Найти минимум в правом поддереве (inorder successor)
		minRight := root.Right.FindMin()
		root.Val = minRight
		root.Right = root.Right.Delete(minRight)
	}
	
	return root
}

// Inorder обход (возвращает отсортированный список)
func (root *TreeNode) Inorder() []int {
	if root == nil {
		return []int{}
	}
	
	result := []int{}
	result = append(result, root.Left.Inorder()...)
	result = append(result, root.Val)
	result = append(result, root.Right.Inorder()...)
	
	return result
}
```

### Проблема дисбаланса

```
Вставить: 1, 2, 3, 4, 5

Хорошее дерево (BST):
    3
   / \
  2   4
 /     \
1       5

Плохое дерево (Skewed):
1
 \
  2
   \
    3
     \
      4
       \
        5

Это как связный список! O(n) поиск вместо O(log n)
```

---

## Балансировка дерева

### Концепция баланса

```
Балансирующий фактор (Balance Factor) = высота левого - высота правого

      4        BF = 1 (высота_левого=2, высота_правого=1)
     / \
    2   6
   /     \
  1       7

Идеально сбалансированное: BF в диапазоне [-1, 1] для всех узлов
```

### Ротации для восстановления баланса

```
Левая ротация:        Правая ротация:
    x                      y
     \                    /
      y          →       x
     / \                / \
    a   b              a   b

Код:
```

```go
// Левая ротация
func rotateLeft(x *TreeNode) *TreeNode {
	y := x.Right
	x.Right = y.Left
	y.Left = x
	return y
}

// Правая ротация
func rotateRight(y *TreeNode) *TreeNode {
	x := y.Left
	y.Left = x.Right
	x.Right = y
	return x
}
```

### Типы дисбаланса

```
LL (Left-Left):      LR (Left-Right):     RR (Right-Right):   RL (Right-Left):
    z                   z                       x                  x
   /                   /                         \                  \
  y                   x                           y                   y
 /                     \                           \                 /
x                       y                           z               z
Fix: Right Rot      Fix: L+R Rot             Fix: Left Rot      Fix: R+L Rot
```

---

## AVL Tree

### Определение

**AVL дерево** — это самобалансирующееся двоичное дерево поиска, где для каждого узла разница высот левого и правого поддеревьев не превышает 1.

### Реализация AVL дерева

```go
type AVLNode struct {
	Val    int
	Left   *AVLNode
	Right  *AVLNode
	Height int
}

func NewAVLNode(val int) *AVLNode {
	return &AVLNode{Val: val, Height: 1}
}

// Получить высоту узла
func getHeight(node *AVLNode) int {
	if node == nil {
		return 0
	}
	return node.Height
}

// Получить балансирующий фактор
func getBalance(node *AVLNode) int {
	if node == nil {
		return 0
	}
	return getHeight(node.Left) - getHeight(node.Right)
}

// Обновить высоту узла
func updateHeight(node *AVLNode) {
	if node == nil {
		return
	}
	node.Height = 1 + max(getHeight(node.Left), getHeight(node.Right))
}

// Левая ротация
func rotateLeftAVL(x *AVLNode) *AVLNode {
	y := x.Right
	x.Right = y.Left
	y.Left = x
	updateHeight(x)
	updateHeight(y)
	return y
}

// Правая ротация
func rotateRightAVL(y *AVLNode) *AVLNode {
	x := y.Left
	y.Left = x.Right
	x.Right = y
	updateHeight(y)
	updateHeight(x)
	return x
}

// Вставить значение в AVL дерево
func (root *AVLNode) InsertAVL(val int) *AVLNode {
	if root == nil {
		return NewAVLNode(val)
	}
	
	if val < root.Val {
		root.Left = root.Left.InsertAVL(val)
	} else if val > root.Val {
		root.Right = root.Right.InsertAVL(val)
	} else {
		return root // дубликат
	}
	
	updateHeight(root)
	balance := getBalance(root)
	
	// LL случай
	if balance > 1 && val < root.Left.Val {
		return rotateRightAVL(root)
	}
	
	// RR случай
	if balance < -1 && val > root.Right.Val {
		return rotateLeftAVL(root)
	}
	
	// LR случай
	if balance > 1 && val > root.Left.Val {
		root.Left = rotateLeftAVL(root.Left)
		return rotateRightAVL(root)
	}
	
	// RL случай
	if balance < -1 && val < root.Right.Val {
		root.Right = rotateRightAVL(root.Right)
		return rotateLeftAVL(root)
	}
	
	return root
}

// Удалить значение из AVL дерева
func (root *AVLNode) DeleteAVL(val int) *AVLNode {
	if root == nil {
		return nil
	}
	
	if val < root.Val {
		root.Left = root.Left.DeleteAVL(val)
	} else if val > root.Val {
		root.Right = root.Right.DeleteAVL(val)
	} else {
		if root.Left == nil {
			return root.Right
		}
		if root.Right == nil {
			return root.Left
		}
		
		// Найти inorder successor
		minRight := root.Right
		for minRight.Left != nil {
			minRight = minRight.Left
		}
		root.Val = minRight.Val
		root.Right = root.Right.DeleteAVL(minRight.Val)
	}
	
	if root == nil {
		return nil
	}
	
	updateHeight(root)
	balance := getBalance(root)
	
	// Балансировка после удаления
	if balance > 1 {
		if getBalance(root.Left) < 0 {
			root.Left = rotateLeftAVL(root.Left)
		}
		return rotateRightAVL(root)
	}
	
	if balance < -1 {
		if getBalance(root.Right) > 0 {
			root.Right = rotateRightAVL(root.Right)
		}
		return rotateLeftAVL(root)
	}
	
	return root
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```

---

## Практические примеры

### Пример 1: Validate Binary Search Tree

```go
// Проверить является ли дерево правильным BST
func IsValidBST(root *TreeNode) bool {
	return isValidBSTHelper(root, math.MinInt64, math.MaxInt64)
}

func isValidBSTHelper(root *TreeNode, min, max int64) bool {
	if root == nil {
		return true
	}
	
	if int64(root.Val) <= min || int64(root.Val) >= max {
		return false
	}
	
	return isValidBSTHelper(root.Left, min, int64(root.Val)) &&
		isValidBSTHelper(root.Right, int64(root.Val), max)
}
```

### Пример 2: Kth Smallest Element in BST

```go
// Найти k-й наименьший элемент в BST
func KthSmallest(root *TreeNode, k int) int {
	count := 0
	result := 0
	
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		
		inorder(node.Left)
		count++
		if count == k {
			result = node.Val
			return
		}
		inorder(node.Right)
	}
	
	inorder(root)
	return result
}
```

### Пример 3: Closest Binary Search Tree Value

```go
// Найти значение в BST ближайшее к target
func ClosestValue(root *TreeNode, target float64) int {
	closest := float64(root.Val)
	
	var dfs func(*TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		
		diff := math.Abs(float64(node.Val) - target)
		if diff < math.Abs(closest-target) {
			closest = float64(node.Val)
		}
		
		if target < float64(node.Val) {
			dfs(node.Left)
		} else {
			dfs(node.Right)
		}
	}
	
	dfs(root)
	return int(closest)
}
```

---

## Анализ сложности

### BST (несбалансированное)

| Операция | Худший | Средний |
|----------|--------|---------|
| **Поиск** | O(n) | O(log n) |
| **Вставка** | O(n) | O(log n) |
| **Удаление** | O(n) | O(log n) |
| **Память** | O(n) | O(n) |

### AVL Tree (сбалансированное)

| Операция | Время | Примечание |
|----------|-------|-----------|
| **Поиск** | O(log n) | Гарантировано |
| **Вставка** | O(log n) | С ротациями |
| **Удаление** | O(log n) | С ротациями |
| **Память** | O(n) | Требует хранения высоты |

### Сравнение структур

| Структура | Поиск | Вставка | Удаление | Балансировка |
|-----------|-------|---------|----------|--------------|
| **Array** | O(log n) бин. | O(n) | O(n) | - |
| **BST** | O(log n) avg | O(log n) avg | O(log n) avg | нет |
| **AVL** | O(log n) | O(log n) | O(log n) | да |
| **Hash Map** | O(1) avg | O(1) avg | O(1) avg | - |

---

## Рекомендуемые задачи на LeetCode

- [Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [Kth Smallest Element in a BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)
- [Lowest Common Ancestor of a BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
- [Binary Search Tree Iterator](https://leetcode.com/problems/binary-search-tree-iterator/)
- [Closest Binary Search Tree Value](https://leetcode.com/problems/closest-binary-search-tree-value/)
- [Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)

---

## Заключение

1. **BST** — эффективен в среднем, но может деградировать
2. **AVL** — гарантирует логарифмический поиск через ротации
3. **Выбор** зависит от требований к гарантиям сложности
4. **Приложения** — базы данных, системы индексирования

---

**CS 161 — Design of Algorithms**  
*Stanford University, Department of Computer Science*
