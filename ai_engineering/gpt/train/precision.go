package train

import "math"

// Mixed precision, concept only (L9): our training runs in float32, but these
// conversions show what bf16 and fp16 do to numbers — and why GPUs keep fp32
// "master weights" and scale the loss.
//
//	fp32: 1 sign | 8 exponent | 23 mantissa bits   ≈ 7 decimal digits, up to 3.4e38
//	bf16: 1 sign | 8 exponent |  7 mantissa bits   ≈ 2–3 digits, same range as fp32
//	fp16: 1 sign | 5 exponent | 10 mantissa bits   ≈ 3 digits, max 65 504, min normal 6.1e-5

// ToBF16 rounds x to the nearest bfloat16 (ties to even) and returns its 16 bits:
// bf16 is simply the upper half of the fp32 bit pattern.
func ToBF16(x float32) uint16 {
	b := math.Float32bits(x)
	if x != x { // NaN stays NaN
		return uint16(b>>16) | 0x40
	}
	// Add 0x7FFF plus the lowest kept bit: rounds to nearest, ties to even.
	b += 0x7FFF + (b>>16)&1
	return uint16(b >> 16)
}

// FromBF16 expands bf16 bits back to float32 (exact: append 16 zero bits).
func FromBF16(h uint16) float32 { return math.Float32frombits(uint32(h) << 16) }

// RoundBF16 is x rounded to bf16 precision, as a float32.
func RoundBF16(x float32) float32 { return FromBF16(ToBF16(x)) }

// ToFP16 rounds x to the nearest IEEE half precision value (ties to even), with
// overflow to ±Inf and gradual underflow to subnormals and zero.
func ToFP16(x float32) uint16 {
	b := math.Float32bits(x)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23&0xFF) - 127 // unbiased exponent
	man := b & 0x7FFFFF
	switch {
	case exp == 128: // Inf or NaN
		if man != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	case exp > 15: // too large: overflow
		return sign | 0x7C00
	case exp >= -14: // normal fp16: keep 10 of 23 mantissa bits, round the other 13
		h := uint32(exp+15)<<10 | man>>13
		rest := man & 0x1FFF
		if rest > 0x1000 || (rest == 0x1000 && h&1 == 1) {
			h++ // may carry into the exponent — that is correct rounding (up to Inf)
		}
		return sign | uint16(h)
	case exp >= -25: // subnormal fp16: value = m·2⁻²⁴
		m := man | 0x800000     // implicit leading 1
		shift := uint(-exp - 1) // x = m·2^(exp−23) = h·2⁻²⁴ → h = m >> (−exp−1)
		h := m >> shift
		rest := m & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rest > half || (rest == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	return sign // underflow to ±0
}

// FromFP16 converts half precision bits to float32 (exact).
func FromFP16(h uint16) float32 {
	sign := float32(1)
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int(h>>10) & 0x1F
	man := float64(h & 0x3FF)
	switch exp {
	case 0:
		return sign * float32(math.Ldexp(man, -24)) // subnormal: m·2⁻²⁴
	case 31:
		if man != 0 {
			return float32(math.NaN())
		}
		return sign * float32(math.Inf(1))
	}
	return sign * float32(math.Ldexp(1+man/1024, exp-15))
}

// RoundFP16 is x rounded to fp16 precision, as a float32.
func RoundFP16(x float32) float32 { return FromFP16(ToFP16(x)) }
