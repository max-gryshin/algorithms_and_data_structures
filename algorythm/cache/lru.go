package cache

import (
	"sync"
	"time"
)

type item struct {
	value interface{}
	exp   time.Time
	next  *item
	prev  *item
	key   string
}

type cache struct {
	data map[string]*item
	ttl  time.Duration
	size int

	mu    sync.RWMutex
	first *item
	last  *item
}

type Cache interface {
	// Set sets value to cache with given key
	Set(key string, value interface{})
	// Get returns value from cache by given key
	Get(key string) (interface{}, bool)
}

// New creates new cache with given ttl and size
func New(ttl time.Duration, size int) Cache {
	return &cache{
		data: make(map[string]*item, size),
		ttl:  ttl,
		size: size,
	}
}

// Set sets value to cache with given key
func (c *cache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// check if item exists
	if item, ok := c.data[key]; ok {
		item.value = value
		item.exp = time.Now().Add(c.ttl)
		// move item to the end of the list to prevent it from being removed
		c.moveBack(item)
		return
	}

	// remove first item if cache is full. first item is the oldest one
	if len(c.data) == c.size {
		c.remove(c.first)
	}

	item := &item{
		value: value,
		exp:   time.Now().Add(c.ttl),
		key:   key,
	}

	c.data[key] = item

	// if cache is empty, set first and last items to the new one
	if c.first == nil {
		c.first = item
		c.last = item
		return
	}

	// add new item to the end of the list
	item.prev = c.last
	c.last.next = item
	c.last = item
}

// moveBack moves item to the end of the list
func (c *cache) moveBack(item *item) {
	// if item is already the last one, do nothing
	if c.last == item {
		return
	}

	// if item is the first one, set first item to the next one
	if c.first == item {
		c.first = item.next
		c.first.prev = nil
	} else {
		// remove item from the list
		item.prev.next = item.next
		item.next.prev = item.prev
	}

	// add item to the end of the list
	item.next = nil
	c.last.next = item
	item.prev = c.last
	c.last = item
}

// Get returns value from cache by given key
func (c *cache) Get(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// check if item exists
	item, ok := c.data[key]
	if !ok {
		return nil, false
	}

	// remove item if it's expired
	if item.exp.Before(time.Now()) {
		c.remove(item)
		return nil, false
	}

	// move item to the end of the list to prevent it from being removed
	c.moveBack(item)
	return item.value, true
}

// remove removes item from the list and map
func (c *cache) remove(item *item) {
	// if item is the first one, set first item to the next one
	if c.first == item {
		// if item is the only one, set first and last items to nil else set first item to the next one
		c.first = item.next
		if c.first != nil {
			c.first.prev = nil
		} else {
			c.last = nil
		}
	} else {
		// remove item from the list
		item.prev.next = item.next
		if item.next != nil {
			item.next.prev = item.prev
		}
	}

	// remove item from the map
	item.next = nil
	item.prev = nil
	delete(c.data, item.key)
}
