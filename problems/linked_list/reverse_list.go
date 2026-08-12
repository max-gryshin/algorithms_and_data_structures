package linked_list

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseListSwapMemory(head *ListNode) *ListNode {
	var reversedList *ListNode
	for head != nil {
		reversedList = &ListNode{head.Val, reversedList}
		head = head.Next
	}
	return reversedList
}

func reverseList(head *ListNode) *ListNode {
	var front *ListNode
	mid, end := head, head
	for mid != nil {
		end = mid.Next
		mid.Next = front
		front, mid = mid, end
	}
	return front
}

func reverseListGeeks(head *ListNode) *ListNode {
	var prev, next *ListNode
	for head != nil {
		next = head.Next
		head.Next = prev
		prev = head
		head = next
	}

	return prev
}

func reverseListV2(head *ListNode) *ListNode {
	var rev, next *ListNode
	for head != nil {
		next = head.Next
		head.Next = rev
		rev = head
		head = next
	}
	return rev
}

//1 -> 2 -> 3 -> 4 -> nil
//
//должно превратиться в
//
//4 -> 3 -> 2 -> 1 -> nil

func reverseListV3(head *ListNode) *ListNode {
	var prev, next *ListNode
	for head != nil {
		next = head.Next
		head.Next = prev
		prev = head
		head = next
	}

	return prev
}
