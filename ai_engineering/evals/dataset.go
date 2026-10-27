// Package evals is a small evaluation harness for LLM systems (lecture L19):
// datasets of cases, graders (exact, contains, regex, JSON shape, LLM-as-judge),
// a concurrent runner, reports with bootstrap confidence intervals, run-to-run diffs
// with McNemar's test, latency percentiles and minimal tracing.
package evals

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

// Case is one test example.
type Case struct {
	ID       string         `json:"id"`
	Input    string         `json:"input"`
	Expected string         `json:"expected,omitempty"`
	Tags     []string       `json:"tags,omitempty"`
	Meta     map[string]any `json:"meta,omitempty"`
}

// Dataset is a named list of cases.
type Dataset struct {
	Name  string `json:"name"`
	Cases []Case `json:"cases"`
}

// LoadDataset reads either a JSON object {"name": ..., "cases": [...]} or JSON Lines
// (one Case per line). Case IDs must be unique and non-empty.
func LoadDataset(r io.Reader) (*Dataset, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	ds := &Dataset{}
	var probe map[string]json.RawMessage
	if json.Unmarshal(data, &probe) == nil && probe["cases"] != nil {
		if err := json.Unmarshal(data, ds); err != nil {
			return nil, fmt.Errorf("evals: dataset: %w", err)
		}
	} else {
		sc := bufio.NewScanner(strings.NewReader(string(data)))
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for line := 1; sc.Scan(); line++ {
			if strings.TrimSpace(sc.Text()) == "" {
				continue
			}
			var c Case
			if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
				return nil, fmt.Errorf("evals: line %d: %w", line, err)
			}
			ds.Cases = append(ds.Cases, c)
		}
	}
	seen := map[string]bool{}
	for _, c := range ds.Cases {
		if c.ID == "" || seen[c.ID] {
			return nil, fmt.Errorf("evals: empty or duplicate case id %q", c.ID)
		}
		seen[c.ID] = true
	}
	return ds, nil
}

// LoadDatasetFile loads a dataset from disk; the file name becomes the default name.
func LoadDatasetFile(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	ds, err := LoadDataset(f)
	if err == nil && ds.Name == "" {
		ds.Name = path
	}
	return ds, err
}
