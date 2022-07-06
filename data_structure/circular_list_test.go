package data_structure

import (
	"testing"
)

var circularListTests = []struct {
	n []int
	v []int
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 10},
	},
}

var circularListTestsDelete = []struct {
	n []int
	d []int
	e []int
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{10, 9},
		[]int{8, 7, 6, 5, 4, 3, 2, 1, 0, 8},
	},
}

func TestCircularList_Add(t *testing.T) {
	list := NewCircularList()
	var nextVal int
	for _, l := range circularListTests {
		for _, v := range l.n {
			list.Add(v)
			link := list.Search(v)
			if link.next != nil && link.next.key != nextVal {
				t.Errorf("Not valid Next value in circular list: expected %d, actual %d", nextVal, link.prev.key)
			}
			nextVal = v
		}
		for _, v := range l.v {
			if list.Current().key != v {
				t.Errorf("Not valid Next value in circular list: expected %d, actual %d", v, list.Current().key)
			}
			list = list.Next()
		}
	}
}

func TestCircularList_Delete(t *testing.T) {
	list := NewCircularList()
	for _, l := range circularListTestsDelete {
		for _, v := range l.n {
			list.Add(v)
		}
		for _, v := range l.d {
			list.Delete(v)
			list = list.Next()
		}
		for _, e := range l.e {
			if list.Current().key != e {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", e, list.Current().key)
			}
			list = list.Next()
		}
	}
}
