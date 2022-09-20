package problems

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
	var prev *ListNode
	var curr = head
	var next *ListNode
	for curr != nil {
		next = curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}

	return prev
}
