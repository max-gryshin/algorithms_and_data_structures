// Command toy trains DDPM and flow-matching models on a 2-D toy dataset (lecture L15)
// and writes CSVs for plotting:
//
//	go run ./ai_engineering/diffusion/cmd/toy -data moons -out /tmp/l15
//
// Files: data.csv, ddpm.csv (trajectory snapshots + final samples), flow.csv
// (samples for different numbers of Euler steps), cfg.csv (class-conditional samples
// for several guidance weights). All in long format: series,step,label,x,y.
//
// Plot, e.g., with pandas:
//
//	df = pd.read_csv("flow.csv"); df[df.step == 8].plot.scatter("x", "y", s=2)
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"time"

	"algorithms_and_data_structures/ai_engineering/diffusion"
	"algorithms_and_data_structures/ai_engineering/nn"
)

func main() {
	dataName := flag.String("data", "moons", "dataset: moons or spiral")
	n := flag.Int("n", 4000, "dataset size")
	samples := flag.Int("samples", 1000, "points to generate per run")
	steps := flag.Int("steps", 3000, "training steps per model")
	hidden := flag.Int("hidden", 128, "hidden width of the denoiser MLP")
	T := flag.Int("T", 100, "DDPM steps")
	out := flag.String("out", ".", "output directory for CSV files")
	seed := flag.Uint64("seed", 1, "random seed")
	flag.Parse()

	rng := rand.New(rand.NewPCG(*seed, 15))
	gen := map[string]func(*rand.Rand, int) *diffusion.Dataset{
		"moons":  func(r *rand.Rand, n int) *diffusion.Dataset { return diffusion.TwoMoons(r, n, 0.05) },
		"spiral": func(r *rand.Rand, n int) *diffusion.Dataset { return diffusion.Spiral(r, n, 0.03) },
	}[*dataName]
	if gen == nil {
		log.Fatalf("unknown dataset %q", *dataName)
	}
	data := gen(rng, *n)
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	ref := data.X
	noise := diffusion.Gaussian(rng, *samples)
	fresh := gen(rand.New(rand.NewPCG(*seed+1, 16)), *samples).X // the ideal: new points from the true distribution
	fmt.Printf("dataset %s: %d points\n", *dataName, data.Len())
	fmt.Printf("baseline N(0,I):      nn-dist %.3f chamfer %.3f\n", diffusion.MeanNNDist(noise, ref), diffusion.Chamfer(noise, ref))
	fmt.Printf("ideal (fresh data):   nn-dist %.3f chamfer %.3f\n", diffusion.MeanNNDist(fresh, ref), diffusion.Chamfer(fresh, ref))
	write(*out, "data.csv", func(c *diffusion.CSV) error { return c.Add("data", 0, data.X, data.Labels) })

	cfg := diffusion.TrainConfig{Steps: *steps, Batch: 256, LR: 2e-3, CondDrop: 0.15}

	// DDPM, unconditional.
	sched := diffusion.LinearSchedule(*T, 1e-4, 0.1)
	fmt.Printf("schedule: T=%d ᾱ_T=%.4f √ᾱ_T=%.3f\n", *T, sched.AlphaBar[*T], math.Sqrt(sched.AlphaBar[*T]))
	ddpm := diffusion.NewDenoiser(rng, *hidden, 3, 16, 0)
	fmt.Printf("denoiser parameters: %d\n", nn.CountParams(ddpm.Parameters()))
	start := time.Now()
	losses := diffusion.TrainDDPM(ddpm, sched, data, cfg, rng)
	fmt.Printf("DDPM: trained %d steps in %v, loss %.3f → %.3f\n", *steps, time.Since(start).Round(time.Millisecond),
		avg(losses[:100]), avg(losses[len(losses)-100:]))
	write(*out, "ddpm.csv", func(c *diffusion.CSV) error {
		var err error
		x := diffusion.SampleDDPM(ddpm, sched, *samples, *T, diffusion.NoClass, 1, rng, func(t int, x []float32) {
			if err == nil && t%(max(1, *T/5)) == 0 {
				err = c.Add("ddpm", t, x, nil)
			}
		})
		if err != nil {
			return err
		}
		fmt.Printf("DDPM %3d steps: nn-dist %.3f chamfer %.3f\n", *T, diffusion.MeanNNDist(x, ref), diffusion.Chamfer(x, ref))
		for _, k := range []int{2, 4, 8, 16} {
			a := diffusion.SampleDDPM(ddpm, sched, *samples, k, diffusion.NoClass, 1, rng, nil)
			b := diffusion.SampleDDIM(ddpm, sched, *samples, k, diffusion.NoClass, 1, rng, nil)
			fmt.Printf("%2d steps: DDPM nn-dist %.3f chamfer %.3f | DDIM nn-dist %.3f chamfer %.3f\n", k,
				diffusion.MeanNNDist(a, ref), diffusion.Chamfer(a, ref), diffusion.MeanNNDist(b, ref), diffusion.Chamfer(b, ref))
			if err := c.Add(fmt.Sprintf("ddim%d", k), 0, b, nil); err != nil {
				return err
			}
		}
		return nil
	})

	// Flow matching, unconditional.
	flow := diffusion.NewDenoiser(rng, *hidden, 3, 16, 0)
	start = time.Now()
	losses = diffusion.TrainFlow(flow, data, cfg, rng)
	fmt.Printf("Flow: trained %d steps in %v, loss %.3f → %.3f\n", *steps, time.Since(start).Round(time.Millisecond),
		avg(losses[:100]), avg(losses[len(losses)-100:]))
	write(*out, "flow.csv", func(c *diffusion.CSV) error {
		for _, k := range []int{1, 2, 4, 8, 16, 64} {
			x := diffusion.SampleFlow(flow, *samples, k, diffusion.NoClass, 1, rng, nil)
			fmt.Printf("Flow %2d Euler steps: nn-dist %.3f chamfer %.3f\n", k, diffusion.MeanNNDist(x, ref), diffusion.Chamfer(x, ref))
			if err := c.Add("flow", k, x, nil); err != nil {
				return err
			}
		}
		return nil
	})

	// Classifier-free guidance with flow matching (2 classes).
	cond := diffusion.NewDenoiser(rng, *hidden, 3, 16, 2)
	diffusion.TrainFlow(cond, data, cfg, rng)
	write(*out, "cfg.csv", func(c *diffusion.CSV) error {
		for _, w := range []float64{0, 1, 3} {
			for cls := range 2 {
				x := diffusion.SampleFlow(cond, *samples/2, 32, cls, w, rng, nil)
				own, other := diffusion.MeanNNDist(x, data.Subset(cls)), diffusion.MeanNNDist(x, data.Subset(1-cls))
				fmt.Printf("CFG w=%.0f class %d: nn-dist to own class %.3f, to other %.3f\n", w, cls, own, other)
				labels := make([]int, len(x)/2)
				for i := range labels {
					labels[i] = cls
				}
				if err := c.Add(fmt.Sprintf("w=%g", w), 32, x, labels); err != nil {
					return err
				}
			}
		}
		return nil
	})
	fmt.Printf("CSV files written to %s\n", *out)
}

func write(dir, name string, fill func(*diffusion.CSV) error) {
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	c, err := diffusion.NewCSV(f)
	if err == nil {
		err = fill(c)
	}
	if err == nil {
		err = c.Flush()
	}
	if err != nil {
		log.Fatal(err)
	}
}

func avg(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
