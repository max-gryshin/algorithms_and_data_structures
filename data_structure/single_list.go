package data_structure

type ListNode struct {
	Key  int
	Next *ListNode
}

func AddNode(headRef *ListNode, v int) *ListNode {
	n := &ListNode{
		Key: v,
	}
	n.Next = headRef
	headRef = n

	return headRef
}

func DeleteNode(headRef *ListNode, v int) {
	if headRef == nil {
		return
	}
	for headRef != nil {
		if headRef.Key == v {
			*headRef = *headRef.Next
			break
		}
		if headRef.Next != nil && headRef.Next.Key == v {
			headRef.Next = headRef.Next.Next
			break
		}
		headRef = headRef.Next
	}
}

func NextNode(headRef *ListNode) {
	if headRef.Next != nil {
		*headRef = *headRef.Next
	}
}
