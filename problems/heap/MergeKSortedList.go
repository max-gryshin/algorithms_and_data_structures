package heappattern

import "container/heap"

type MinHeap []*ListNode

func (m MinHeap) Len() int {
	return len(m)
}

func (m MinHeap) Less(i, j int) bool {
	return m[i].Val < m[j].Val
}

func (m MinHeap) Swap(i, j int) {
	m[i], m[j] = m[j], m[i]
}

func (m *MinHeap) Push(n any) {
	item := n.(*ListNode)
	*m = append(*m, item)
}

func (m *MinHeap) Pop() any {
	old := *m
	length := len(*m)
	item := old[length-1]
	old[length-1] = nil
	*m = old[:length-1]
	return item
}

func NewMinHeap() *MinHeap {
	m := &MinHeap{}
	heap.Init(m)

	return m
}

// MergeKLists В min-heap мы храним указатели на текущие минимальные элементы каждого списка.
// Пока есть кандидаты:
// 1. Взять минимальный текущий узел из heap.
// 2. Прицепить его к результату.
// 3. Если у этого узла есть следующий элемент:
// добавить следующий элемент в heap.
func MergeKLists(lists []*ListNode) *ListNode {
	h := &MinHeap{}
	heap.Init(h)
	// кладем только heads
	for _, list := range lists {
		if list != nil {
			heap.Push(h, list)
		}
	}

	// Этот прием называется паттерном фиктивной головы (Dummy Head или Sentinel Node).
	// Две переменные нужны для решения двух конкретных задач:
	// сохранить ссылку на начало списка и избавиться от лишних проверок на nil.
	dummy := &ListNode{}
	current := dummy
	for h.Len() > 0 {
		node := heap.Pop(h).(*ListNode) // вернуть список с наименьшим значением в head
		if node.Next != nil {
			heap.Push(h, node.Next)
		}
		current.Next = node
		current = current.Next // current теперь указывает на новый конец списка
	}
	return dummy.Next
}
