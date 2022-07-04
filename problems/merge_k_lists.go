package problems

import "algorithms_and_data_structures/data_structure"

func mergeKLists(lists []*data_structure.ListNode) *data_structure.ListNode {
	var temp, result *data_structure.ListNode
	if len(lists) == 0 {
		return result
	}
	lLen := len(lists) - 1
	count := 0
	for {
		if lists[count] == nil {
			if count == lLen {
				break
			}
			count++
			continue
		}
		temp = appendNode(temp, lists[count].Key)
		if lists[count].Next != nil {
			data_structure.NextNode(lists[count])
		} else {
			lists[count] = nil
		}
		if count < lLen {
			count++
		} else {
			count = 0
		}
	}
	for {
		result = data_structure.AddNode(result, temp.Key)
		if temp.Next == nil {
			break
		}
		data_structure.NextNode(temp)
	}

	return result
}

func appendNode(ls *data_structure.ListNode, v int) *data_structure.ListNode {
	if ls == nil {
		return data_structure.AddNode(ls, v)
	}
	var prev *data_structure.ListNode
	for {
		if ls == nil || v >= ls.Key {
			ls = data_structure.AddNode(ls, v)
			break
		} else {
			prev = data_structure.AddNode(prev, ls.Key)
		}
		if ls.Next == nil {
			break
		}
		data_structure.NextNode(ls)
	}
	if prev != nil {
		for {
			ls = &data_structure.ListNode{
				Key:  prev.Key,
				Next: ls,
			}
			if prev.Next == nil {
				break
			}
			data_structure.NextNode(prev)
		}
	}
	return ls
}
