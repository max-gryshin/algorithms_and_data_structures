package data_structure

import "testing"

var qDeQueue = []struct {
	val []int
	deq []int
}{
	{
		[]int{1, 2, 3, 4, 5},
		[]int{1, 2},
	},
}

func TestQueue_DeQueue(t *testing.T) {
	q := NewQueue()
	for _, v := range qDeQueue {
		for _, val := range v.val {
			q.Enqueue(val)
		}
		for _, deqVal := range v.deq {
			d := q.DeQueue()
			if d != deqVal {
				t.Errorf("Not valid dequeued value in queue: expected %d, actual %d", deqVal, d)
			}
		}
	}
}
