package rag

import (
	"fmt"
	"strings"
)

// Chunk is a retrievable piece of a document.
type Chunk struct {
	ID      string // "<source>#<n>"
	Source  string // file name or URL
	Heading string // heading path, e.g. "L1 > 3. Скалярное произведение"
	Text    string
}

// ChunkMarkdown splits a Markdown document into chunks along its headings.
// A section longer than maxRunes is cut into windows of maxRunes runes that
// overlap by overlap runes, preferring to cut at a blank line or a newline.
// Lines inside ``` code fences are never treated as headings.
func ChunkMarkdown(source, text string, maxRunes, overlap int) []Chunk {
	type section struct {
		heading string
		body    strings.Builder
	}
	var (
		sections []*section
		path     []string // path[level-1] = heading text
		inFence  bool
	)
	cur := &section{}
	sections = append(sections, cur)
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inFence = !inFence
		}
		if level := headingLevel(line); level > 0 && !inFence {
			title := strings.TrimSpace(line[level:])
			if len(path) >= level {
				path = path[:level-1]
			}
			for len(path) < level-1 {
				path = append(path, "")
			}
			path = append(path, title)
			cur = &section{heading: joinPath(path)}
			sections = append(sections, cur)
		}
		cur.body.WriteString(line)
		cur.body.WriteByte('\n')
	}

	var out []Chunk
	for _, s := range sections {
		body := strings.TrimSpace(s.body.String())
		if body == "" {
			continue
		}
		for _, piece := range window([]rune(body), maxRunes, overlap) {
			out = append(out, Chunk{
				ID:      fmt.Sprintf("%s#%d", source, len(out)),
				Source:  source,
				Heading: s.heading,
				Text:    piece,
			})
		}
	}
	return out
}

func headingLevel(line string) int {
	n := 0
	for n < len(line) && line[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(line) || line[n] != ' ' {
		return 0
	}
	return n
}

func joinPath(p []string) string {
	var parts []string
	for _, s := range p {
		if s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, " > ")
}

// window cuts r into pieces of at most size runes with the given overlap.
func window(r []rune, size, overlap int) []string {
	if len(r) <= size {
		return []string{string(r)}
	}
	var out []string
	for start := 0; start < len(r); {
		end := min(start+size, len(r))
		if end < len(r) {
			end = cutPoint(r, start, end)
		}
		out = append(out, strings.TrimSpace(string(r[start:end])))
		if end == len(r) {
			break
		}
		start = max(end-overlap, start+1)
	}
	return out
}

// cutPoint moves end back to just after a blank line or newline in the second half
// of [start, end), so that chunks end on paragraph boundaries when possible.
func cutPoint(r []rune, start, end int) int {
	mid := start + (end-start)/2
	for i := end - 1; i > mid; i-- {
		if r[i] == '\n' && r[i-1] == '\n' {
			return i + 1
		}
	}
	for i := end - 1; i > mid; i-- {
		if r[i] == '\n' {
			return i + 1
		}
	}
	return end
}
