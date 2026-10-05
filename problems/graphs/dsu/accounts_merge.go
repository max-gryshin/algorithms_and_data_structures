package dsu

import "algorithms_and_data_structures/problems/graphs/dsu/theory"

// Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
// Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]

func accountsMerge(accounts [][]string) [][]string {
	dsu := theory.NewDSU(len(accounts))

	emailsToOwner := make(map[string]int)
	// union by same email
	for accountID, account := range accounts { // index of accounts's slice is a account id
		for i := 1; i < len(account); i++ {
			if accountIDExisting, ok := emailsToOwner[account[i]]; ok {
				dsu.Union(accountIDExisting, accountID)
			} else {
				emailsToOwner[account[i]] = accountID
			}
		}
	}
	emailsByOwner := make(map[int][]string)
	// group emails by root (account id)
	for email, owner := range emailsToOwner {
		root := dsu.FindSet(owner)
		emailsByOwner[root] = append(emailsByOwner[root], email)
	}

	result := [][]string{}
	for root, emails := range emailsByOwner {
		row := []string{}
		row = append(row, accounts[root][0])
		row = append(row, emails...)
		result = append(result, row)
	}

	return result
}
