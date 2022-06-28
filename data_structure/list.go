package data_structure

type DoubleList struct {
	head *Link
}

func NewDoubleList() *DoubleList {
	return &DoubleList{
		head: nil,
	}
}

func (l *DoubleList) Add(v int) {
	x := Link{
		key: v,
	}
	x.next = l.head
	if l.head != nil {
		l.head.prev = &x
	}
	l.head = &x
	x.prev = nil
}

func (l *DoubleList) Delete(v int) {
	x := Link{
		key: v,
	}
	if x.prev != nil {
		x.prev.next = x.next
	} else {
		l.head = x.next
	}
	if x.next != nil {
		x.next.prev = x.prev
	}
}

func (l *DoubleList) Search(v int) Link {
	x := l.head
	for x != nil && x.key != v {
		x = x.next
	}

	return *x
}
