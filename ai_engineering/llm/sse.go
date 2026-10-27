package llm

import (
	"bufio"
	"io"
	"iter"
	"strings"
)

// SSEEvent is one Server-Sent Event: lines "event: ..." and "data: ..." terminated by a blank line.
type SSEEvent struct {
	Event string
	Data  string
	ID    string
}

// ReadSSE parses a text/event-stream body (WHATWG HTML spec, "Server-sent events"):
//   - a line "field: value" sets a field (one optional space after the colon is dropped);
//   - several "data:" lines are joined with "\n";
//   - a line starting with ":" is a comment (servers send them as keep-alives);
//   - a blank line dispatches the accumulated event.
func ReadSSE(r io.Reader) iter.Seq2[SSEEvent, error] {
	return func(yield func(SSEEvent, error) bool) {
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 64*1024), 8*1024*1024) // one event can be a large JSON document
		var ev SSEEvent
		var data []string
		dispatch := func() bool {
			if len(data) == 0 && ev.Event == "" {
				return true
			}
			ev.Data = strings.Join(data, "\n")
			ok := yield(ev, nil)
			ev, data = SSEEvent{}, nil
			return ok
		}
		for sc.Scan() {
			line := strings.TrimSuffix(sc.Text(), "\r")
			if line == "" {
				if !dispatch() {
					return
				}
				continue
			}
			if strings.HasPrefix(line, ":") {
				continue
			}
			field, value, _ := strings.Cut(line, ":")
			value = strings.TrimPrefix(value, " ")
			switch field {
			case "event":
				ev.Event = value
			case "data":
				data = append(data, value)
			case "id":
				ev.ID = value
			}
		}
		if err := sc.Err(); err != nil {
			yield(SSEEvent{}, err)
			return
		}
		dispatch() // a stream may end without the final blank line
	}
}
