package diffusion

import (
	"encoding/csv"
	"io"
	"strconv"
)

// CSV writes point clouds in long format, one point per row:
//
//	series,step,label,x,y
//	data,0,1,0.53,-1.20
//	ddpm,100,0,1.91,0.07
//
// Long format plots directly: filter by series/step, colour by label
// (pandas + matplotlib, gnuplot, any spreadsheet).
type CSV struct{ w *csv.Writer }

// NewCSV writes the header and returns the writer.
func NewCSV(w io.Writer) (*CSV, error) {
	c := &CSV{w: csv.NewWriter(w)}
	return c, c.w.Write([]string{"series", "step", "label", "x", "y"})
}

// Add appends the points pts (N×2). labels may be nil (written as -1) or hold N labels.
func (c *CSV) Add(series string, step int, pts []float32, labels []int) error {
	st := strconv.Itoa(step)
	for i := 0; i < len(pts)/2; i++ {
		l := -1
		if labels != nil {
			l = labels[i]
		}
		err := c.w.Write([]string{series, st, strconv.Itoa(l),
			strconv.FormatFloat(float64(pts[2*i]), 'f', 4, 32),
			strconv.FormatFloat(float64(pts[2*i+1]), 'f', 4, 32)})
		if err != nil {
			return err
		}
	}
	return nil
}

// Flush writes buffered rows and reports any write error.
func (c *CSV) Flush() error {
	c.w.Flush()
	return c.w.Error()
}
