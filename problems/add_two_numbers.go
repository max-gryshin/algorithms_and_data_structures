package problems

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	l3 := &ListNode{}
	curr := l3
	rem := 0
	for l1 != nil || l2 != nil {
		sum := rem
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}
		rem = sum / 10
		curr.Next = &ListNode{Val: sum % 10}
		curr = curr.Next
	}
	if rem > 0 {
		curr.Next = &ListNode{Val: rem}
	}
	return l3.Next
}
