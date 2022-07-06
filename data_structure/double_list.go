package data_structure

type DoubleList struct {
	head *DoubleLink
}

func NewDoubleList() *DoubleList {
	return &DoubleList{
		head: nil,
	}
}

func (l *DoubleList) Add(v int) {
	x := &DoubleLink{
		Key:  v,
		next: l.head,
	}
	if l.head != nil {
		l.head.prev = x
	}
	l.head = x
	x.prev = nil
}

func (l *DoubleList) Delete(v int) {
	x := l.Search(v)
	if x.prev != nil {
		x.prev.next = x.next
	} else {
		l.head = x.next
	}
	if x.next != nil {
		x.next.prev = x.prev
	}
}

func (l *DoubleList) Search(v int) DoubleLink {
	x := l.head
	for x != nil && x.Key != v {
		x = x.next
	}

	return *x
}

func (l *DoubleList) Next() {
	if l.head.next != nil {
		l.head = l.head.next
	}
}

func (l *DoubleList) Prev() {
	if l.head.prev != nil {
		l.head = l.head.prev
	}
}

func (l *DoubleList) Current() DoubleLink {
	return *l.head
}
