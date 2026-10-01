package dfs_bfs

//Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
//An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
//You may assume all four edges of the grid are all surrounded by water.
//
//Example 1:
//
//Input: grid = [
//  ["1","1","1","1","0"],
//  ["1","1","0","1","0"],
//  ["1","1","0","0","0"],
//  ["0","0","0","0","0"]
//]
//Output: 1
//
//Example 2:
//
//Input: grid = [
//  ["1","1","0","0","0"],
//  ["1","1","0","0","0"],
//  ["0","0","1","0","0"],
//  ["0","0","0","1","1"]
//]
//Output: 3

// проходимся по матрице BFSем - кладем первый участок земли в очередь и мапу уже посещенных островов в подцикле
// BFSа идем left, up, right, down и если это земля то добавляем в мапу посещеных участков земли
// когда из под цикла вышли увеличиваем количество островов +1 ну
// и разумеется каждый участок земли проверяем не посещали ли его уже - если да то скип главный цикл итерируется по всем элементам

// for every cell:
//
//	if cell == '1':
//	    islands++
//	    BFS(cell)
//
//	    BFS:
//	        mark current as visited
//	        check 4 directions
//	        if neighbor == '1':
//	            mark visited
//	            add to queue

type islandPosition struct {
	i int
	j int
}

func numIslands(grid [][]byte) int {
	var (
		isAnyIsland bool
		islands     int
	)
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '0' {
				continue
			}
			islands++
			queue := []islandPosition{islandPosition{i: i, j: j}}
			isAnyIsland = true
			grid[i][j] = '0'
			for len(queue) > 0 {
				currLand := queue[0]
				queue = queue[1:]
				if currLand.i > 0 && grid[currLand.i-1][j] == '1' {
					grid[currLand.i-1][j] = '0'
					queue = append(queue, islandPosition{i: i - 1, j: j})
				}
				if currLand.j > 0 && grid[currLand.i][j-1] == '1' {
					grid[currLand.i][j-1] = '0'
					queue = append(queue, islandPosition{i: i, j: j - 1})
				}
				if currLand.i+1 < len(grid) && grid[currLand.i+1][j] == '1' {
					grid[currLand.i+1][j] = '0'
					queue = append(queue, islandPosition{i: i + 1, j: j})
				}
				if currLand.j+1 < len(grid[i]) && grid[currLand.i][j+1] == '1' {
					grid[currLand.i][j+1] = '0'
					queue = append(queue, islandPosition{i: i, j: j + 1})
				}
			}
		}
	}
	if !isAnyIsland {
		return 0
	}

	return islands
}
