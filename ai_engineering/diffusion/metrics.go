package diffusion

import "math"

// MeanNNDist is the average, over points a, of the Euclidean distance to the nearest
// point of ref (both N×2). Small values mean "every generated point lies on the data";
// it does not notice missing modes — for that look at Chamfer.
func MeanNNDist(a, ref []float32) float64 {
	var s float64
	n := len(a) / 2
	for i := range n {
		ax, ay := a[2*i], a[2*i+1]
		best := math.Inf(1)
		for j := 0; j < len(ref); j += 2 {
			dx, dy := float64(ax-ref[j]), float64(ay-ref[j+1])
			best = min(best, dx*dx+dy*dy)
		}
		s += math.Sqrt(best)
	}
	return s / float64(n)
}

// Chamfer is the symmetric version ½·(MeanNNDist(a, b) + MeanNNDist(b, a)):
// it also grows when part of the data is not covered by the samples (mode dropping).
func Chamfer(a, b []float32) float64 { return (MeanNNDist(a, b) + MeanNNDist(b, a)) / 2 }

// Moments returns the per-axis mean and variance of points (N×2).
func Moments(pts []float32) (mean, variance [2]float64) {
	n := float64(len(pts) / 2)
	for i, v := range pts {
		mean[i%2] += float64(v)
	}
	mean[0] /= n
	mean[1] /= n
	for i, v := range pts {
		d := float64(v) - mean[i%2]
		variance[i%2] += d * d
	}
	variance[0] /= n
	variance[1] /= n
	return mean, variance
}
