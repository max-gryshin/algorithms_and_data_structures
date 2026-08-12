package concurrency

import "sync"

type Cache struct {
	m  map[string]string
	mu sync.RWMutex
}

func (c *Cache) Get(key string) (string, bool) {
	c.mu.RLock()
	val, ok := c.m[key]
	c.mu.RUnlock()
	return val, ok
}

func (c *Cache) Set(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[key] = value
}
func (c *Cache) Delete(key string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.m, key)
}
