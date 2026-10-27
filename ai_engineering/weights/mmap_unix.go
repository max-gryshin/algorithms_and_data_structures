//go:build darwin || linux || freebsd

package weights

import (
	"os"
	"syscall"
)

// mmap maps the whole file read-only. Pages are loaded by the OS on first access and
// shared with the page cache: opening a 5 GB file costs no reads and no copy.
func mmap(f *os.File, size int) ([]byte, error) {
	if size == 0 {
		return nil, syscall.EINVAL
	}
	return syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
}

func munmap(b []byte) error { return syscall.Munmap(b) }
