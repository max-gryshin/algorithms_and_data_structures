package data_structure

type CircularList struct {
	prev, next *CircularList
	key        int
}

func NewCircularList() *CircularList {
	cl := new(CircularList)
	cl.next = cl
	cl.prev = cl
	return cl
}

func (l *CircularList) Add(v int) {
	l.next = &CircularList{
		key:  v,
		next: l.next,
		prev: l,
	}
}

func (l *CircularList) Delete(v int) {
	x := l.Search(v)
	x.prev.next = x.next
	x.next.prev = x.prev
}

func (l *CircularList) Search(v int) CircularList {
	x := l
	for x.key != v {
		x = x.next
	}

	return *x
}

func (l *CircularList) Next() *CircularList {
	return l.next
}

func (l *CircularList) Current() *CircularList {
	return l
}

func (l *CircularList) Prev() *CircularList {
	return l.prev
}
