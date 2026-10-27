// Package weights loads real pretrained weights into gpt.Model and quantizes them
// (lecture L12): the safetensors format (reader with mmap, writer for tests), dtype
// conversions fp16/bf16 ↔ fp32, name mapping of Hugging Face GPT-2 and Llama
// checkpoints, int8 per-channel and int4 group-wise quantization with mat-vec kernels.
package weights

import "math"

// BF16ToF32 widens bfloat16: bf16 is the upper half of an fp32, so shift by 16.
func BF16ToF32(b uint16) float32 { return math.Float32frombits(uint32(b) << 16) }

// F32ToBF16 rounds to the nearest bfloat16 (ties to even), keeping NaN a NaN.
func F32ToBF16(f float32) uint16 {
	u := math.Float32bits(f)
	if u&0x7FFFFFFF > 0x7F800000 { // NaN: rounding could turn it into ±Inf
		return uint16(u>>16) | 0x40
	}
	u += 0x7FFF + (u>>16)&1 // add just under half an ulp, plus 1 if the kept part is odd
	return uint16(u >> 16)
}

// F16ToF32 widens IEEE-754 half precision (1 sign, 5 exponent, 10 mantissa bits).
func F16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	man := uint32(h & 0x3FF)
	switch exp {
	case 0: // zero or subnormal: man · 2⁻²⁴
		v := float32(man) / (1 << 24)
		if sign != 0 {
			v = -v
		}
		return v
	case 0x1F: // Inf or NaN
		return math.Float32frombits(sign | 0x7F800000 | man<<13)
	}
	return math.Float32frombits(sign | (exp-15+127)<<23 | man<<13)
}

// F32ToF16 rounds to the nearest half (ties to even); too large → ±Inf, too small → ±0.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int(b>>23) & 0xFF
	man := b & 0x7FFFFF
	if exp == 0xFF {
		if man != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	e := exp - 127 + 15 // exponent re-biased for half
	if e >= 0x1F {
		return sign | 0x7C00
	}
	if e <= 0 { // subnormal half: value = q · 2⁻²⁴, q = M >> (126 − exp)
		shift := uint(14 - e)
		if shift > 24 {
			return sign
		}
		m := man | 0x800000
		q := m >> shift
		rem, half := m&(1<<shift-1), uint32(1)<<(shift-1)
		if rem > half || rem == half && q&1 == 1 {
			q++
		}
		return sign | uint16(q)
	}
	h := uint32(e)<<10 | man>>13
	rem := man & 0x1FFF
	if rem > 0x1000 || rem == 0x1000 && h&1 == 1 {
		h++ // a carry into the exponent is correct, up to Inf
	}
	return sign | uint16(h)
}
