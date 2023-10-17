package cache

import (
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func Test_cache_Set(t *testing.T) {
	c := New(time.Second, 5)
	result, found := c.Get("key1")
	assert.Equal(t, false, found)
	assert.Equal(t, nil, result)

	c.Set("key1", "value1")
	result, found = c.Get("key1")
	assert.Equal(t, true, found)
	assert.Equal(t, "value1", result)
	time.Sleep(time.Second)
	result, found = c.Get("key1")
	assert.Equal(t, false, found)
	assert.Equal(t, nil, result)

	c.Set("key1", "value1")
	c.Set("key2", "value2")
	c.Set("key3", "value3")
	c.Set("key4", "value4")
	c.Set("key5", "value5")
	result, found = c.Get("key1")
	assert.Equal(t, true, found)
	assert.Equal(t, "value1", result)
	c.Set("key6", "value6")
	result, found = c.Get("key2")
	assert.Equal(t, false, found)
	assert.Equal(t, nil, result)
}

func Test_cache_Get(t *testing.T) {
	c := New(time.Second, 5)
	result, found := c.Get("key1")
	assert.Equal(t, false, found)
	assert.Equal(t, nil, result)

	c.Set("key1", "value1")

	result, found = c.Get("key1")
	assert.Equal(t, true, found)
	assert.Equal(t, "value1", result)

	time.Sleep(time.Second)
	result, found = c.Get("key1")
	assert.Equal(t, false, found)
	assert.Equal(t, nil, result)
}

func Benchmark_cache_Set(b *testing.B) {
	c := New(time.Second, 500)

	for i := 0; i < b.N; i++ {
		c.Set(strconv.Itoa(i), "value")
	}
}
