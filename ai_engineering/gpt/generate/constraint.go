package generate

import "fmt"

// Constraint restricts generation to the strings of a formal language (constrained
// decoding, L10). Before every step the generator asks Allowed for each token of the
// vocabulary and sets the logits of forbidden tokens to −∞ — the model can only
// choose among continuations that keep the output a valid prefix. This is how
// "structured outputs" / JSON mode of LLM APIs work (Willard & Louf 2023).
type Constraint interface {
	// Allowed reports whether the bytes of a token may be appended now.
	Allowed(token []byte) bool
	// Advance commits a token's bytes (they must be Allowed).
	Advance(token []byte)
	// Done reports that the output is complete; generation stops.
	Done() bool
}

// ---------- generic JSON: a pushdown automaton over bytes ----------

// JSON accepts one JSON object or array (like the "JSON mode" of LLM APIs). It is a
// pushdown automaton: a state for the current lexical position plus a stack of open
// containers ('{' or '['). Strings are checked to be valid UTF-8.
//
// MaxDepth and MaxString bound nesting and string length so that even an untrained
// model must finish: past the limit, only the closing token is allowed.
type JSON struct {
	MaxDepth  int // 0 = 8
	MaxString int // bytes per string, 0 = 64
	st        jsonState
}

type jsonState struct {
	stack  []byte // open containers
	mode   jsonMode
	key    bool // the current string is an object key
	strLen int  // bytes in the current string
	utf    utf8State
	hex    int // hex digits still expected after \u
	lit    string
	wsRun  int // consecutive whitespace bytes (bounded, so that the output must progress)
}

type jsonMode int

const (
	mValue    jsonMode = iota // expect a value
	mObjFirst                 // after '{': key or '}'
	mObjKey                   // after ',' in an object: key
	mColon                    // after a key: ':'
	mObjNext                  // after a value in an object: ',' or '}'
	mArrFirst                 // after '[': value or ']'
	mArrNext                  // after a value in an array: ',' or ']'
	mString                   // inside "…"
	mEscape                   // after '\' inside a string
	mNumMinus                 // after '-'
	mNumZero                  // after a leading 0
	mNumInt                   // in integer digits
	mNumDot                   // after '.'
	mNumFrac                  // in fraction digits
	mNumE                     // after 'e'/'E'
	mNumESign                 // after the exponent sign
	mNumExp                   // in exponent digits
	mLiteral                  // inside true/false/null
	mDone                     // top-level value complete
)

func (j *JSON) limits() (int, int) {
	d, s := j.MaxDepth, j.MaxString
	if d == 0 {
		d = 8
	}
	if s == 0 {
		s = 64
	}
	return d, s
}

func (j *JSON) Allowed(tok []byte) bool {
	s := j.st
	s.stack = append([]byte(nil), j.st.stack...)
	for _, c := range tok {
		if !j.feed(&s, c) {
			return false
		}
	}
	return true
}

func (j *JSON) Advance(tok []byte) {
	for _, c := range tok {
		if !j.feed(&j.st, c) {
			panic(fmt.Sprintf("generate: JSON constraint rejects %q", tok))
		}
	}
}

func (j *JSON) Done() bool { return j.st.mode == mDone }

func isWS(c byte) bool    { return c == ' ' || c == '\n' || c == '\t' || c == '\r' }
func isDigit(c byte) bool { return c >= '0' && c <= '9' }
func isHex(c byte) bool   { return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F' }

// afterValue moves to the state that follows a complete value.
func afterValue(s *jsonState) {
	switch {
	case len(s.stack) == 0:
		s.mode = mDone
	case s.stack[len(s.stack)-1] == '{':
		s.mode = mObjNext
	default:
		s.mode = mArrNext
	}
}

// feed advances s by one byte; false means the byte is not allowed.
func (j *JSON) feed(s *jsonState, c byte) bool {
	maxDepth, maxStr := j.limits()
	if !isWS(c) {
		s.wsRun = 0
	}
	// A number ends at the first byte that cannot continue it; that byte is then
	// processed in the state after the value.
	switch s.mode {
	case mNumZero, mNumInt, mNumFrac, mNumExp:
		switch {
		case isDigit(c) && s.mode != mNumZero:
			return s.strLen < maxStr && inc(&s.strLen)
		case c == '.' && (s.mode == mNumZero || s.mode == mNumInt):
			s.mode = mNumDot
			return true
		case (c == 'e' || c == 'E') && s.mode != mNumExp:
			s.mode = mNumE
			return true
		}
		afterValue(s)
		if s.mode == mDone {
			return false // top level is an object or array, never a bare number
		}
	}
	switch s.mode {
	case mDone:
		return false
	case mValue, mArrFirst:
		if isWS(c) {
			return len(s.stack) > 0 && ws(s) // no whitespace before the top-level value
		}
		if c == ']' && s.mode == mArrFirst {
			s.stack = s.stack[:len(s.stack)-1]
			afterValue(s)
			return true
		}
		if len(s.stack) == 0 && c != '{' && c != '[' {
			return false // top level: object or array only
		}
		switch {
		case c == '{' || c == '[':
			if len(s.stack) >= maxDepth {
				return false
			}
			s.stack = append(s.stack, c)
			if c == '{' {
				s.mode = mObjFirst
			} else {
				s.mode = mArrFirst
			}
		case c == '"':
			s.mode, s.key, s.strLen = mString, false, 0
		case c == '-':
			s.mode, s.strLen = mNumMinus, 0
		case c == '0':
			s.mode, s.strLen = mNumZero, 1
		case c >= '1' && c <= '9':
			s.mode, s.strLen = mNumInt, 1
		case c == 't':
			s.mode, s.lit = mLiteral, "rue"
		case c == 'f':
			s.mode, s.lit = mLiteral, "alse"
		case c == 'n':
			s.mode, s.lit = mLiteral, "ull"
		default:
			return false
		}
		return true
	case mObjFirst, mObjKey:
		switch {
		case isWS(c):
			return ws(s)
		case c == '"':
			s.mode, s.key, s.strLen = mString, true, 0
			return true
		case c == '}' && s.mode == mObjFirst:
			s.stack = s.stack[:len(s.stack)-1]
			afterValue(s)
			return true
		}
		return false
	case mColon:
		if isWS(c) {
			return ws(s)
		}
		if c == ':' {
			s.mode = mValue
			return true
		}
		return false
	case mObjNext, mArrNext:
		top := s.stack[len(s.stack)-1]
		switch {
		case isWS(c):
			return ws(s)
		case c == ',':
			if top == '{' {
				s.mode = mObjKey
			} else {
				s.mode = mValue
			}
			return true
		case c == '}' && top == '{', c == ']' && top == '[':
			s.stack = s.stack[:len(s.stack)-1]
			afterValue(s)
			return true
		}
		return false
	case mString:
		if s.utf.need > 0 {
			if !s.utf.cont(c) {
				return false
			}
			s.strLen++
			return true
		}
		switch {
		case c == '"':
			if s.key {
				s.mode = mColon
			} else {
				afterValue(s)
			}
			return true
		case s.strLen >= maxStr: // budget spent: only the closing quote
			return false
		case c == '\\':
			s.mode = mEscape
			return true
		case c < 0x20:
			return false // control characters must be escaped
		case c < 0x80:
			s.strLen++
			return true
		case !s.utf.lead(c):
			return false // invalid lead byte
		}
		// a multi-byte character must fit into the remaining budget
		if s.strLen+s.utf.need+1 > maxStr {
			return false
		}
		s.strLen++
		return true
	case mEscape:
		if s.hex > 0 {
			if !isHex(c) {
				return false
			}
			s.hex--
			if s.hex == 0 {
				s.mode = mString
			}
			return true
		}
		switch c {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			s.mode = mString
			s.strLen++
			return true
		case 'u':
			s.hex = 4
			s.strLen++
			return true
		}
		return false
	case mNumMinus:
		switch {
		case c == '0':
			s.mode = mNumZero
		case c >= '1' && c <= '9':
			s.mode = mNumInt
		default:
			return false
		}
		s.strLen = 1
		return true
	case mNumDot, mNumE, mNumESign:
		if isDigit(c) {
			if s.mode == mNumDot {
				s.mode = mNumFrac
			} else {
				s.mode = mNumExp
			}
			return true
		}
		if (c == '+' || c == '-') && s.mode == mNumE {
			s.mode = mNumESign
			return true
		}
		return false
	case mLiteral:
		if c != s.lit[0] {
			return false
		}
		s.lit = s.lit[1:]
		if s.lit == "" {
			afterValue(s)
		}
		return true
	}
	return false
}

func inc(n *int) bool { *n++; return true }

// ws allows at most 4 whitespace bytes in a row.
func ws(s *jsonState) bool {
	s.wsRun++
	return s.wsRun <= 4
}

// ---------- a fixed schema: a template automaton ----------

// FieldKind is the type of a schema field.
type FieldKind int

const (
	String FieldKind = iota // "…" of 1…MaxString bytes, no escapes
	Int                     // -?[0-9]{1,MaxDigits} without leading zeros
	Bool                    // true | false
)

// Field is one key of a flat JSON object.
type Field struct {
	Name string
	Kind FieldKind
}

// Schema accepts exactly one object with the given fields in the given order, e.g.
// {"name":"Анна","age":42} — the automaton is a sequence of fixed literals ({"name":")
// and typed slots. Unlike generic JSON it can only produce the shape you want, and the
// limits make it finish in a bounded number of bytes.
type Schema struct {
	Fields    []Field
	MaxString int // bytes per string value, 0 = 32
	MaxDigits int // digits per integer, 0 = 6
	st        schemaState
}

type schemaState struct {
	part   int    // index into parts()
	pos    int    // bytes consumed in this part
	lit    string // remaining bytes of a bool literal
	utf    utf8State
	neg    bool // integer slot: saw '-'
	digits int  // integer slot: digits so far
	zero   bool // integer slot: the first digit was 0
}

type schemaPart struct {
	lit  string // fixed text, or
	kind FieldKind
}

func (s *Schema) parts() []schemaPart {
	var ps []schemaPart
	for i, f := range s.Fields {
		sep := ","
		if i == 0 {
			sep = "{"
		}
		open := `"` + f.Name + `":`
		if f.Kind == String {
			open += `"`
		}
		ps = append(ps, schemaPart{lit: sep + open}, schemaPart{kind: f.Kind})
		if f.Kind == String {
			ps = append(ps, schemaPart{lit: `"`})
		}
	}
	return append(ps, schemaPart{lit: "}"})
}

func (s *Schema) Allowed(tok []byte) bool {
	st := s.st
	ps := s.parts()
	for _, c := range tok {
		if !s.feed(&st, ps, c) {
			return false
		}
	}
	return true
}

func (s *Schema) Advance(tok []byte) {
	ps := s.parts()
	for _, c := range tok {
		if !s.feed(&s.st, ps, c) {
			panic(fmt.Sprintf("generate: schema rejects %q", tok))
		}
	}
}

func (s *Schema) Done() bool { return s.st.part == len(s.parts()) }

func (s *Schema) feed(st *schemaState, ps []schemaPart, c byte) bool {
	maxStr, maxDig := s.MaxString, s.MaxDigits
	if maxStr == 0 {
		maxStr = 32
	}
	if maxDig == 0 {
		maxDig = 6
	}
	for st.part < len(ps) {
		p := ps[st.part]
		if p.lit != "" {
			if c != p.lit[st.pos] {
				return false
			}
			st.pos++
			if st.pos == len(p.lit) {
				st.part, st.pos = st.part+1, 0
			}
			return true
		}
		switch p.kind {
		case String:
			if st.utf.need > 0 {
				if !st.utf.cont(c) {
					return false
				}
				st.pos++
				return true
			}
			if c == '"' && st.pos > 0 { // the slot ends; '"' belongs to the next literal
				st.part, st.pos = st.part+1, 0
				continue
			}
			if c == '"' || c == '\\' || c < 0x20 || st.pos >= maxStr {
				return false
			}
			u := st.utf
			if c >= 0x80 && !u.lead(c) {
				return false
			}
			if st.pos+u.need+1 > maxStr {
				return false
			}
			st.utf = u
			st.pos++
			return true
		case Int:
			switch {
			case c == '-' && !st.neg && st.digits == 0:
				st.neg = true
				return true
			case isDigit(c) && st.digits < maxDig && !st.zero: // no digits after a leading 0
				st.zero = st.digits == 0 && c == '0'
				st.digits++
				return true
			case st.digits > 0: // the integer ends; c must start the next literal
				st.part, st.pos, st.neg, st.digits, st.zero = st.part+1, 0, false, 0, false
				continue
			}
			return false
		case Bool:
			if st.pos == 0 {
				switch c {
				case 't':
					st.lit = "rue"
				case 'f':
					st.lit = "alse"
				default:
					return false
				}
				st.pos = 1
				return true
			}
			if c != st.lit[0] {
				return false
			}
			st.lit = st.lit[1:]
			if st.lit == "" {
				st.part, st.pos = st.part+1, 0
			}
			return true
		}
	}
	return false // already complete
}

// utf8State validates UTF-8 byte by byte (RFC 3629): after a lead byte it knows how
// many continuation bytes must follow and the allowed range of the first one, which
// excludes overlong forms (E0 80…), UTF-16 surrogates (ED A0…) and code points
// above U+10FFFF (F4 90…).
type utf8State struct {
	need   int
	lo, hi byte // range of the next continuation byte
}

// lead starts a multi-byte character; false for a byte that cannot start one.
func (u *utf8State) lead(c byte) bool {
	u.lo, u.hi = 0x80, 0xBF
	switch {
	case c >= 0xC2 && c <= 0xDF:
		u.need = 1
	case c == 0xE0:
		u.need, u.lo = 2, 0xA0
	case c == 0xED:
		u.need, u.hi = 2, 0x9F
	case c >= 0xE1 && c <= 0xEF:
		u.need = 2
	case c == 0xF0:
		u.need, u.lo = 3, 0x90
	case c == 0xF4:
		u.need, u.hi = 3, 0x8F
	case c >= 0xF1 && c <= 0xF3:
		u.need = 3
	default:
		return false
	}
	return true
}

// cont consumes a continuation byte.
func (u *utf8State) cont(c byte) bool {
	if c < u.lo || c > u.hi {
		return false
	}
	u.need--
	u.lo, u.hi = 0x80, 0xBF
	return true
}
