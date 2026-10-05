package dsu

import "algorithms_and_data_structures/problems/graphs/dsu/theory"

func FindEmployeesUnderManager(managers map[int]int, managerId int) []int {
	dsu := theory.NewDSU(len(managers))

	for employee, manager := range managers {
		dsu.Union(employee, manager)
	}

	result := []int{}
	for employee := range managers {
		if dsu.FindSet(employee) == dsu.FindSet(managerId) && employee != managerId {
			result = append(result, employee)
		}
	}

	return result
}
