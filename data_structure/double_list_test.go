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

var doubleListTestsCurrent = []struct {
	n []int
	v []int
}{
	{
		[]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		[]int{10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
	},
}

func TestDoubleList_Add(t *testing.T) {
	list := NewDoubleList()
	var nextVal int
	for _, l := range doubleListTests {
		for _, v := range l.n {
			list.Add(v)
			link := list.Search(v)
			if link.next != nil && link.next.Key != nextVal {
				t.Errorf("Not valid Next value in linked list: expected %d, actual %d", nextVal, link.prev.Key)
			}
			nextVal = v
		}
	}
}

func TestDoubleList_Delete(t *testing.T) {
	list := NewDoubleList()
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.n {
			list.Add(v)
		}
	}
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.v {
			if list.Current().Key != v {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", v, list.Current().Key)
			}
			list.Delete(v)
		}
	}
}

func TestDoubleList_Current(t *testing.T) {
	list := NewDoubleList()
	for _, l := range doubleListTests {
		for _, v := range l.n {
			list.Add(v)
			if list.Current().Key != v {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", v, list.Current().Key)
			}
		}
	}
}

func TestDoubleList_Next(t *testing.T) {
	list := NewDoubleList()
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.n {
			list.Add(v)
		}
	}
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.v {
			if list.Current().Key != v {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", v, list.Current().Key)
			}
			list.Next()
		}
	}
}

func TestDoubleList_Prev(t *testing.T) {
	list := NewDoubleList()
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.n {
			list.Add(v)
		}
	}
	for _, l := range doubleListTestsCurrent {
		for range l.v {
			list.Next()
		}
	}
	for _, l := range doubleListTestsCurrent {
		for _, v := range l.n {
			if list.Current().Key != v {
				t.Errorf("Not valid current value in linked list: expected %d, actual %d", v, list.Current().Key)
			}
			list.Prev()
		}
	}
}
