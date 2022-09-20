package problems

import "testing"

func TestReverseList(t *testing.T) {
	var list = &ListNode{
		1,
		&ListNode{2,
			&ListNode{3, &ListNode{4, &ListNode{5, &ListNode{6, &ListNode{7,
				&ListNode{8, &ListNode{9, &ListNode{10, nil}}},
			}}}}},
		},
	}
	var reversedList = &ListNode{
		10,
		&ListNode{9,
			&ListNode{8, &ListNode{7, &ListNode{6, &ListNode{5, &ListNode{4,
				&ListNode{3, &ListNode{2, &ListNode{1, nil}}},
			}}}}},
		},
	}
	result := reverseListGeeks(list)
	for result != nil || reversedList != nil {
		if result.Val != reversedList.Val {
			t.Errorf(
				" the result of ReverseList(%+v) not correct:\n expected %+v,\n actual %+v\n",
				list,
				reversedList,
				result,
			)
		}
		if result == nil || reversedList == nil {
			break
		}
		result = result.Next
		reversedList = reversedList.Next
	}

}
