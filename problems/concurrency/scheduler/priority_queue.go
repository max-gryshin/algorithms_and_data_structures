package scheduler

type PrioritizedTask struct {
	Priority int    // Чем выше число, тем раньше запустится
	Index    int    // Индекс внутри heap (нужен для container/heap)
	TaskID   string // ID задачи
}

// Реализация интерфейса heap.Interface для max-heap
type PriorityQueue []*PrioritizedTask

func (pq PriorityQueue) Len() int           { return len(pq) }
func (pq PriorityQueue) Less(i, j int) bool { return pq[i].Priority > pq[j].Priority } // Max-heap
func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].Index = i
	pq[j].Index = j
}

func (pq *PriorityQueue) Push(x any) {
	n := len(*pq)
	item := x.(*PrioritizedTask)
	item.Index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.Index = -1
	*pq = old[0 : n-1]
	return item
}
