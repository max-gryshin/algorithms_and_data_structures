package data_structure

type Link struct {
	key  int
	prev *Link
	next *Link
}

type LinkedList struct {
	head   *Link
	bucket []Link
}

func (ll *LinkedList) ListInsert(x int) {
	newLink := Link{
		key:  x,
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

func listDelete(ll *LinkedList) (x int) {
	ll.bucket = append(ll.bucket[:x], ll.bucket[x+1:]...)
	return
}
