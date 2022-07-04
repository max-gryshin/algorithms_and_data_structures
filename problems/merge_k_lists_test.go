package problems

import (
	"algorithms_and_data_structures/data_structure"
	"testing"
)

var kList1 = []int{5, 4, 1}
var kList2 = []int{4, 3, 1}
var kList3 = []int{6, 2}
var expected = []int{1, 1, 2, 3, 4, 4, 5, 6}

func TestMergeKLists(t *testing.T) {
	var ls1 *data_structure.ListNode
	var ls2 *data_structure.ListNode
	var ls3 *data_structure.ListNode
	for _, v := range kList1 {
		ls1 = data_structure.AddNode(ls1, v)
	}
	for _, v := range kList2 {
		ls2 = data_structure.AddNode(ls2, v)
	}
	for _, v := range kList3 {
		ls3 = data_structure.AddNode(ls3, v)
	}
	result := mergeKLists([]*data_structure.ListNode{ls1, ls2, ls3})
	for _, v := range expected {
		if v != result.Key {
			t.Errorf("Not valid current value of merged linked list: expected %d, actual %d", v, result.Key)
			break
		}
		if result.Next == nil {
			break
		}
		data_structure.NextNode(result)
	}
}

func TestMergeKListsEmpty(t *testing.T) {
	result := mergeKLists([]*data_structure.ListNode{})
	var emptyExpected *data_structure.ListNode
	if result != emptyExpected {
		t.Errorf("Not valid current value of merged linked list: expected %v, actual %v", emptyExpected, result)
	}
}
func TestMergeKListsEmptyList(t *testing.T) {
	result := mergeKLists([]*data_structure.ListNode{&data_structure.ListNode{}})
	if result.Key != 0 && result.Next != nil {
		t.Errorf("Not valid merged linked list: expected %v, actual %v", &data_structure.ListNode{}, result)
	}
}
