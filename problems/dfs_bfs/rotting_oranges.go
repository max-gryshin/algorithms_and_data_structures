package dfs_bfs

const (
	empty  = 0
	fresh  = 1
	rotten = 2
)

type rottenPos struct {
	i int
	j int
}

// [
//  [2,1,1],
//  [1,1,0],
//  [0,1,1]
// ]

// [
//  [2,1,1],
//  [0,1,1],
//  [1,0,1]
// ]

// [[0,2]]

func orangesRotting(grid [][]int) int {
	queue := []rottenPos{}
	freshCount := 0
	hasFresh := false
	iLen := len(grid)
	jLen := 0
	for i := 0; i < len(grid); i++ {
		jLen = max(jLen, len(grid[i]))
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == rotten {
				queue = append(queue, rottenPos{i: i, j: j})
			}
			if grid[i][j] == fresh {
				freshCount++
				hasFresh = true
			}
		}
	}
	minutes := 0
	for len(queue) > 0 {
		currentQueueLen := len(queue)
		var goNext bool
		for i := 0; i < currentQueueLen; i++ {
			rottenElement := queue[0]
			queue = queue[1:]
			if rottenElement.i+1 < iLen {
				if grid[rottenElement.i+1][rottenElement.j] == fresh {
					grid[rottenElement.i+1][rottenElement.j] = rotten
					queue = append(queue, rottenPos{i: rottenElement.i + 1, j: rottenElement.j})
					freshCount--
					goNext = true
				}
			}
			if rottenElement.j+1 < jLen {
				if grid[rottenElement.i][rottenElement.j+1] == fresh {
					grid[rottenElement.i][rottenElement.j+1] = rotten
					queue = append(queue, rottenPos{i: rottenElement.i, j: rottenElement.j + 1})
					freshCount--
					goNext = true
				}
			}
			if rottenElement.i > 0 {
				if grid[rottenElement.i-1][rottenElement.j] == fresh {
					grid[rottenElement.i-1][rottenElement.j] = rotten
					queue = append(queue, rottenPos{i: rottenElement.i - 1, j: rottenElement.j})
					freshCount--
					goNext = true
				}
			}
			if rottenElement.j > 0 {
				if grid[rottenElement.i][rottenElement.j-1] == fresh {
					grid[rottenElement.i][rottenElement.j-1] = rotten
					queue = append(queue, rottenPos{i: rottenElement.i, j: rottenElement.j - 1})
					freshCount--
					goNext = true
				}
			}
		}
		if goNext {
			minutes++
		}
	}
	if freshCount > 0 {
		return -1
	}
	if minutes > 0 {
		return minutes
	}
	if !hasFresh {
		return 0
	}
	return -1
}
