package data_structure

import "testing"

var doubleListTests = []struct {
	n    []int
	list *DoubleList
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		NewDoubleList(),
	},
}

func TestDoubleList_Add(t *testing.T) {
	list := NewDoubleList()
	var nextVal int
	for _, l := range doubleListTests {
		for _, v := range l.n {
			list.Add(v)
			link := list.Search(v)
			if link.next != nil && link.next.key != nextVal {
				t.Errorf("Not valid next value in linked list: expected %d, actual %d", nextVal, link.prev.key)
			}
			nextVal = v
		}
	}
}
