package compress

import (
	"fmt"
	"os"
	"testing"
	"time"
)

var dataToCompress = `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.`

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func TestCompressLZWest(t *testing.T) {
	dat, err := os.ReadFile("/home/anduser/projsets/algorithms_and_data_structures/algorythm/compress/lotr_short.txt")
	check(err)
	fmt.Printf("size of file before compression: %d kilo bytes", len(dat)/1000)
	resCompress := CompressLZW(string(dat))
	fmt.Printf("size of compressed text is %d kilo bytes \n", len(resCompress)/1000)
	start := time.Now()
	resDecompress := DecompressConcurrentLZW(resCompress)
	elapsed := time.Since(start)
	fmt.Printf("decompresson took  %f seconds \n", elapsed.Seconds())
	f, err := os.Create("/home/anduser/projsets/algorithms_and_data_structures/algorythm/compress/decompress_lotr.txt")
	check(err)
	defer func(f *os.File) {
		err = f.Close()
		if err != nil {
			panic(err)
		}
	}(f)
	resCode, err := f.WriteString(resDecompress)
	check(err)
	fmt.Printf("result code: %d", resCode)
	err = f.Sync()
	if err != nil {
		panic(err)
	}
}
