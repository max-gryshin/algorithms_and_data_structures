package problems

import "testing"

func TestAddTwoNumbers(t *testing.T) {
	// 4342
	var list1 = &ListNode{2, &ListNode{4, &ListNode{3, &ListNode{4, nil}}}}
	// 665
	var list2 = &ListNode{5, &ListNode{6, &ListNode{6, nil}}}
	// 5007
	var listExpected = &ListNode{7, &ListNode{0, &ListNode{0, &ListNode{5, nil}}}}
	listActual := addTwoNumbers(list1, list2)
	for listActual != nil {
		if listActual.Val != listExpected.Val {
			t.Errorf(
				"the result of mergeTwoLists(%+v, %+v) not correct: \n expected %+v,\n actual %+v\n",
				list1,
				list2,
				listExpected.Val,
				listActual.Val,
			)
		}
		if listExpected == nil || listActual == nil {
			break
		}
		listActual = listActual.Next
		listExpected = listExpected.Next
	}
}
