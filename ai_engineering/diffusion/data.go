package diffusion

import (
	"math"
	"math/rand/v2"
)

// Dataset is a 2-D point cloud with a class label per point.
type Dataset struct {
	X      []float32 // N×2, row-major
	Labels []int     // N labels in [0, NumClasses)
}

// Len returns the number of points.
func (d *Dataset) Len() int { return len(d.Labels) }

// Point returns the i-th point.
func (d *Dataset) Point(i int) (float32, float32) { return d.X[2*i], d.X[2*i+1] }

// TwoMoons generates the classic "two moons": class 0 is the upper half-circle
// (cos θ, sin θ), class 1 the lower one (1 − cos θ, ½ − sin θ), θ ∈ [0, π], plus
// Gaussian noise of the given std. Points are generated in pairs, so classes are balanced.
// The result is standardized to zero mean and unit variance per axis.
func TwoMoons(rng *rand.Rand, n int, noise float64) *Dataset {
	d := &Dataset{X: make([]float32, 0, 2*n), Labels: make([]int, 0, n)}
	for i := range n {
		th := rng.Float64() * math.Pi
		var x, y float64
		if i%2 == 0 {
			x, y = math.Cos(th), math.Sin(th)
		} else {
			x, y = 1-math.Cos(th), 0.5-math.Sin(th)
		}
		x += noise * rng.NormFloat64()
		y += noise * rng.NormFloat64()
		d.X = append(d.X, float32(x), float32(y))
		d.Labels = append(d.Labels, i%2)
	}
	Standardize(d.X)
	return d
}

// Spiral generates a two-armed Archimedean spiral r = θ/(3π), θ ∈ [π/2, 3π];
// arm 1 is arm 0 rotated by 180°. Standardized like TwoMoons.
func Spiral(rng *rand.Rand, n int, noise float64) *Dataset {
	d := &Dataset{X: make([]float32, 0, 2*n), Labels: make([]int, 0, n)}
	for i := range n {
		th := math.Pi/2 + rng.Float64()*2.5*math.Pi
		r := th / (3 * math.Pi)
		x, y := r*math.Cos(th), r*math.Sin(th)
		if i%2 == 1 {
			x, y = -x, -y
		}
		x += noise * rng.NormFloat64()
		y += noise * rng.NormFloat64()
		d.X = append(d.X, float32(x), float32(y))
		d.Labels = append(d.Labels, i%2)
	}
	Standardize(d.X)
	return d
}

// Standardize shifts and scales each of the two coordinates of pts (N×2) in place
// to zero mean and unit variance. Diffusion assumes data on the same scale as the
// N(0, I) noise it ends with.
func Standardize(pts []float32) {
	for c := range 2 {
		var s, s2 float64
		n := float64(len(pts) / 2)
		for i := c; i < len(pts); i += 2 {
			s += float64(pts[i])
		}
		mean := s / n
		for i := c; i < len(pts); i += 2 {
			dv := float64(pts[i]) - mean
			s2 += dv * dv
		}
		std := math.Sqrt(s2 / n)
		for i := c; i < len(pts); i += 2 {
			pts[i] = float32((float64(pts[i]) - mean) / std)
		}
	}
}

// Gaussian returns n points from N(0, I) in 2-D — the starting point of sampling
// and the "know-nothing" baseline for metrics.
func Gaussian(rng *rand.Rand, n int) []float32 {
	out := make([]float32, 2*n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

// Subset returns the points of class label (label < 0 returns a copy of all points).
func (d *Dataset) Subset(label int) []float32 {
	var out []float32
	for i, l := range d.Labels {
		if label < 0 || l == label {
			out = append(out, d.X[2*i], d.X[2*i+1])
		}
	}
	return out
}
