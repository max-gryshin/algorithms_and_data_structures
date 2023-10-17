package data_structure

type Queue struct {
	Tail *ListNode
	Head *ListNode
}

func NewQueue() *Queue {
	return &Queue{}
}

func (q *Queue) Enqueue(v int) {
	node := &ListNode{Key: v}
	if q.Head == nil {
		q.Head = node
	} else {
		q.Tail.Next = node
	}
	q.Tail = node
}

func (q *Queue) DeQueue() int {
	if q.Head == nil {
		return -1
	}
	temp := q.Head
	q.Head = q.Head.Next
	if q.Head == nil {
		q.Tail = nil
	}
	return temp.Key
}
