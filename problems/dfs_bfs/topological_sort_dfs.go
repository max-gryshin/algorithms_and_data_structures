package dfs_bfs

import (
	"fmt"
	"slices"
)

type Color int

const (
	White Color = iota
	Gray
	Black
)

func topologicalSort(graph map[int][]int) ([]int, error) {
	visited := make(map[int]Color)
	result := make([]int, 0)
	var dfs func(id int) error
	dfs = func(id int) error {
		// mark as visited
		visited[id] = Gray
		vertex, ok := graph[id]
		if !ok {
			return fmt.Errorf("id %d not found", id)
		}
		for _, node := range vertex {
			state := visited[node]
			// cycle detected
			if state == Gray {
				return fmt.Errorf("cycle detected; cycle id is %d", node)
			}
			// call dfs if not visited yet
			if state == White {
				if err := dfs(node); err != nil {
					return err
				}
			}
		}
		visited[id] = Black
		result = append(result, id)

		return nil
	}
	for id := range graph {
		// call dfs only for white nodes
		if visited[id] == White {
			if err := dfs(id); err != nil {
				return nil, err
			}
		}
	}
	// since it is dfs - the order is reversed
	slices.Reverse(result)

	return result, nil
}
