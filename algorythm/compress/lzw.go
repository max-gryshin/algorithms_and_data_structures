// Package compress
// Implement by https://habr.com/ru/company/otus/blog/581728

package compress

import (
	"fmt"
	"runtime"
	"sync"
)

func CompressLZW(s string) []int {
	code := 256
	d := make(map[string]int)
	for i := 0; i < 256; i++ {
		d[string(rune(i))] = i
	}
	var currChar string
	var result []int
	for _, c := range []byte(s) {
		if _, isTrue := d[currChar+string(c)]; isTrue {
			currChar += string(c)
		} else {
			result = append(result, d[currChar])
			d[currChar+string(c)] = code
			code++
			currChar = string(c)
		}
	}
	if currChar != "" {
		result = append(result, d[currChar])
	}
	return result
}

func DecompressLZW(compressed []int) string {
	var code = 256
	var entry string
	dictionary := make(map[int]string)
	for i := 0; i < 256; i++ {
		dictionary[i] = string(rune(i))
	}
	char := string(rune(compressed[0]))
	res := char
	for _, v := range compressed[1:] {
		if x, ok := dictionary[v]; ok {
			entry = x
		} else if v == code {
			entry = char + char[:1]
		} else {
			panic(fmt.Sprintf("Bad compressed element: %dictionary", v))
		}
		res += entry
		dictionary[code] = char + entry[:1]
		code++
		char = entry
	}
	return res
}

func DecompressConcurrentLZW(compressed []int) string {
	var code = 256
	dictionary := make(map[int]string)
	for i := 0; i < 256; i++ {
		dictionary[i] = string(rune(i))
	}
	var result string
	resultMap := make(map[int]string)
	rangeCompressed := countRange(len(compressed))
	numGor := len(rangeCompressed) - 1
	m := sync.Mutex{}
	wt := sync.WaitGroup{}
	for i := 0; i < numGor; i++ {
		compressedCollection := compressed[rangeCompressed[i]:rangeCompressed[i+1]]
		wt.Add(1)
		go func(mut *sync.Mutex, it int, coll []int) {
			defer func() {
				wt.Done()
			}()
			if len(coll) == 0 {
				fmt.Printf(" goroutine with empty compressed collection %d", it)
			}
			var entry string
			char := string(rune(coll[0]))
			res := char
			for _, v := range coll[1:] {
				m.Lock()
				if x, ok := dictionary[v]; ok {
					entry = x
				} else if v == code {
					entry = char + char[:1]
				} else {
					panic(fmt.Sprintf("Bad compressed element: %d dictionary", v))
				}
				res += entry
				m.Unlock()
				m.Lock()
				dictionary[code] = char + entry[:1]
				m.Unlock()
				code++
				char = entry
			}

			m.Lock()
			resultMap[it] = res
			m.Unlock()
		}(&m, i, compressedCollection)
	}
	wt.Wait()
	for i := 0; i < numGor; i++ {
		result += resultMap[i]
	}

	return result
}

func countRange(cLen int) []int {
	var res []int
	numCPU := runtime.NumCPU()
	if cLen >= 20000 {
		numCPU *= cLen / 40000
	}
	res = append(res, 0)
	count := 0
	for {
		if count >= cLen {
			res = append(res, cLen)
			break
		}
		count += cLen / numCPU
		res = append(res, count)
	}
	return res
}
