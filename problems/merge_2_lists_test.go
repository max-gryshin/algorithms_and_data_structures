package problems

import "testing"

func TestMergeTwoLists(t *testing.T) {
	var list1 = ListNode{
		1,
		&ListNode{3,
			&ListNode{5, &ListNode{7, &ListNode{9, &ListNode{11, nil}}}},
		},
	}
	var list2 = ListNode{
		2,
		&ListNode{4,
			&ListNode{6, &ListNode{8, &ListNode{10, &ListNode{12, nil}}}},
		},
	}
	var expectedResult = ListNode{
		1,
		&ListNode{2,
			&ListNode{3, &ListNode{4, &ListNode{5, &ListNode{6, &ListNode{7,
				&ListNode{8, &ListNode{9, &ListNode{10, &ListNode{11, &ListNode{12, nil}}}}},
			}}}}},
		},
	}
	data := []struct {
		n1             *ListNode
		n2             *ListNode
		expectedResult *ListNode
	}{
		{
			&list1,
			&list2,
			&expectedResult,
		},
	}
	for _, v := range data {
		res := mergeTwoLists(v.n1, v.n2)
		for res != nil {
			if res.Val != v.expectedResult.Val {
				t.Errorf(
					" the result of mergeTwoLists(%+v, %+v) not correct:\n expected %+v,\n actual %+v\n",
					v.n1,
					v.n2,
					v.expectedResult,
					res,
				)
			}
			if res.Next == nil || v.expectedResult.Next == nil {
				break
			}
			res = res.Next
			*v.expectedResult = *v.expectedResult.Next
		}
	}
}
