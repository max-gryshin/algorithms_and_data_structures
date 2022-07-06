package data_structure

type DoubleLink struct {
	Key  int
	prev *DoubleLink
	next *DoubleLink
}

type LinkedList struct {
	head   *DoubleLink
	bucket []DoubleLink
}

func (ll *LinkedList) Insert(x int) {
	newLink := DoubleLink{
		Key:  x,
		next: ll.head,
	}
	if ll.head != nil {
		ll.head.prev = &newLink
	}
	ll.head = &newLink
	newLink.prev = nil
	ll.bucket = append(ll.bucket, newLink)
	return
}

func Delete(ll *LinkedList) (x int) {
	ll.bucket = append(ll.bucket[:x], ll.bucket[x+1:]...)
	return
}
