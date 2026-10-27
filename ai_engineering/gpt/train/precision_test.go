package train

import (
	"math"
	"testing"
)

func TestFP16KnownValues(t *testing.T) {
	for _, c := range []struct {
		x    float32
		bits uint16
	}{
		{1, 0x3C00}, {-2, 0xC000}, {0.1, 0x2E66}, {65504, 0x7BFF}, {65520, 0x7C00}, {1e6, 0x7C00},
		{float32(math.Ldexp(1, -14)), 0x0400}, {float32(math.Ldexp(1, -24)), 0x0001}, {1e-8, 0x0000},
	} {
		if got := ToFP16(c.x); got != c.bits {
			t.Errorf("ToFP16(%g) = %#04x, want %#04x", c.x, got, c.bits)
		}
	}
	// Every finite fp16 value survives the round trip fp16 → fp32 → fp16.
	for h := range 1 << 16 {
		if h&0x7C00 == 0x7C00 {
			continue // Inf/NaN
		}
		if got := ToFP16(FromFP16(uint16(h))); got != uint16(h) {
			t.Fatalf("round trip %#04x → %g → %#04x", h, FromFP16(uint16(h)), got)
		}
	}
}

func TestBF16KnownValues(t *testing.T) {
	for _, c := range []struct {
		x    float32
		bits uint16
	}{{1, 0x3F80}, {0.1, 0x3DCD}, {math.Pi, 0x4049}, {-3e38, 0xFF62}} {
		if got := ToBF16(c.x); got != c.bits {
			t.Errorf("ToBF16(%g) = %#04x, want %#04x", c.x, got, c.bits)
		}
	}
	// The stagnation problem: in bf16, 1 + 1e-3 is still 1 (the step to the next
	// representable number after 1 is 2⁻⁷ = 0.0078) — updates vanish without fp32 master weights.
	if RoundBF16(1+1e-3) != 1 || RoundBF16(1+0.004) != 1.0078125 {
		t.Fatalf("bf16(1.001) = %g, bf16(1.004) = %g", RoundBF16(1+1e-3), RoundBF16(1+0.004))
	}
}
