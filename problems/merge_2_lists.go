package problems

type ListNode struct {
	Val  int
	Next *ListNode
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	list := &ListNode{}
	tail := list
	for list1 != nil || list2 != nil {
		if list2 == nil {
			tail.Next = list1
			break
		}
		if list1 == nil {
			tail.Next = list2
			break
		}
		if list1.Val < list2.Val {
			tail.Next = list1
			list1 = list1.Next
		} else {
			tail.Next = list2
			list2 = list2.Next
		}
		tail = tail.Next
	}

	return list.Next
}

func mergeTwoListsV2(list1 *ListNode, list2 *ListNode) *ListNode {
	// Этот прием называется паттерном фиктивной головы (Dummy Head или Sentinel Node).
	// Две переменные нужны для решения двух конкретных задач:
	// сохранить ссылку на начало списка и избавиться от лишних проверок на nil.
	dummy := &ListNode{}
	current := dummy

	for list1 != nil && list2 != nil {
		if list1.Val <= list2.Val {
			// тут кладем наименьшее значение в current
			current.Next = list1
			// смещаем current
			list1.Next = list1
		} else {
			// тут кладем наименьшее значение в current
			current.Next = list2
			list2 = list2.Next
		}
		current = current.Next
	}
	if list1 == nil { // если list1 закончился, то кладем в конец current list2
		current.Next = list2
	}
	if list2 == nil { // если list2 закончился, то кладем в конец current list1
		current.Next = list1
	}
	return dummy.Next
}
