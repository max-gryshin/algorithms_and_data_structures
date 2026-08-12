package fast_slow_pointer

func fastAndSlowPointers(head *ListNode) bool {
	if head == nil {
		return false
	}
	fast := head.Next
	slow := head
	for head != nil && head.Next != nil {
		if slow == fast {
			return true
		}
		if head.Next.Next == nil {
			return false
		}
		fast = head.Next.Next
		slow = head.Next
		head = head.Next
	}
	return false
}

func detectCycle(head *ListNode) *ListNode {
	slow := head
	fast := head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next

		if slow == fast {
			start := head

			for start != slow {
				start = start.Next
				slow = slow.Next
			}

			return start
		}
	}

	return nil
}
