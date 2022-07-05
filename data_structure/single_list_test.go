package data_structure

import (
	"testing"
)

var singleListTests = []struct {
	n []int
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	},
}

var singleListTestsDelete = []struct {
	n []int
	v []int
	d []int
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{10, 9, 8, 7, 6, 5, 4, 3, 2},
		[]int{1},
	},
	{
		[]int{},
		[]int{},
		[]int{1},
	},
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{9, 8, 7, 6, 5, 4, 3, 2, 1},
		[]int{10},
	},
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{10, 9, 8, 7, 6, 4, 3, 2, 1},
		[]int{5},
	},
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{},
		[]int{10, 9, 8, 7, 6, 4, 3, 2, 1},
	},
}

func TestSingleList_Add(t *testing.T) {
	var lastVal int
	var ls *ListNode
	for _, l := range singleListTests {
		for _, v := range l.n {
			ls = AddNode(ls, v)
			if ls != nil {
				lastVal = v
			}
		}
		if ls.Key != lastVal {
			t.Errorf("Not valid current value in linked list: expected %d, actual %d", ls.Key, lastVal)
		}
	}
}

func TestSingleList_Delete(t *testing.T) {
	var ls *ListNode
	for _, l := range singleListTestsDelete {
		for _, v := range l.n {
			ls = AddNode(ls, v)
		}
		for _, v := range l.d {
			DeleteNode(ls, v)
		}
		for _, v := range l.v {
			if ls.Key != v {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", v, ls.Key)
			}
			if ls.Next == nil {
				break
			}
			NextNode(ls)
		}
	}
}
