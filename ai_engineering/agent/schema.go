package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// SchemaOf builds a JSON Schema for a Go type with reflect. It is how a tool's input
// struct becomes the "input_schema" the model sees, so the struct is the single source
// of truth for both the prompt and the decoder.
//
// Struct tags:
//
//	json:"name,omitempty"  property name; omitempty (or a pointer field) makes it optional
//	desc:"..."             property description — this is prompt text, write it for the model
//	enum:"a,b,c"           allowed string values
//
// Objects get "additionalProperties": false, so unknown keys are a validation error
// rather than silently ignored.
func SchemaOf(t reflect.Type) map[string]any {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		return map[string]any{"type": "array", "items": SchemaOf(t.Elem())}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": SchemaOf(t.Elem())}
	case reflect.Struct:
		props := map[string]any{}
		required := []string{}
		for f := range t.Fields() {
			if !f.IsExported() {
				continue
			}
			name, omitempty := jsonName(f)
			if name == "-" {
				continue
			}
			p := SchemaOf(f.Type)
			if d := f.Tag.Get("desc"); d != "" {
				p["description"] = d
			}
			if e := f.Tag.Get("enum"); e != "" {
				p["enum"] = strings.Split(e, ",")
			}
			props[name] = p
			if !omitempty && f.Type.Kind() != reflect.Pointer {
				required = append(required, name)
			}
		}
		return map[string]any{"type": "object", "properties": props, "required": required, "additionalProperties": false}
	}
	return map[string]any{} // interface{} and friends: any JSON value
}

func jsonName(f reflect.StructField) (name string, omitempty bool) {
	tag := f.Tag.Get("json")
	name, opts, _ := strings.Cut(tag, ",")
	if name == "" {
		name = f.Name
	}
	return name, strings.Contains(opts, "omitempty") || strings.Contains(opts, "omitzero")
}

// decodeInput validates raw tool input against the schema of In and decodes it.
// The model's JSON is untrusted input like any HTTP body: it must be an object, contain
// every required key, have no unknown keys, have the right types and respect enums.
// The error text goes back to the model as an is_error tool result, so it says exactly
// what to fix.
func decodeInput[In any](raw json.RawMessage, schema map[string]any) (In, error) {
	var in In
	if len(bytes.TrimSpace(raw)) == 0 {
		raw = json.RawMessage("{}")
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return in, fmt.Errorf("input must be a JSON object: %v", err)
	}
	if req, ok := schema["required"].([]string); ok {
		for _, k := range req {
			if _, ok := obj[k]; !ok {
				return in, fmt.Errorf("missing required field %q", k)
			}
		}
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		return in, fmt.Errorf("invalid input: %v", err)
	}
	props, _ := schema["properties"].(map[string]any)
	for k, p := range props {
		enum, ok := p.(map[string]any)["enum"].([]string)
		v, present := obj[k]
		if !ok || !present {
			continue
		}
		var s string
		if json.Unmarshal(v, &s) != nil || !slices.Contains(enum, s) {
			return in, fmt.Errorf("field %q must be one of %v, got %s", k, enum, v)
		}
	}
	return in, nil
}
