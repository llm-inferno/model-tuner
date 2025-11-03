package utils

import (
	"bytes"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// print vector
func VecString(name string, v *mat.VecDense) string {
	x := v.RawVector().Data
	var b bytes.Buffer
	fmt.Fprint(&b, name+"=[")
	for i := range x {
		fmtString := "%10.3f "
		if math.Abs(x[i]) < 0.01 {
			fmtString = "%10.3E "
		}
		fmt.Fprintf(&b, fmtString, x[i])
	}
	fmt.Fprint(&b, "]")
	return b.String()
}

// print matrix
func MatString(name string, m *mat.Dense) string {
	x := m.RawMatrix().Data
	nRows, nCols := m.Dims()
	var b bytes.Buffer
	fmt.Fprint(&b, name+"=[")
	k := 0
	for range nRows {
		fmt.Fprint(&b, "[")
		for j := 0; j < nCols; j++ {
			fmtString := "%10.3f "
			if math.Abs(x[k]) < 0.01 {
				fmtString = "%10.3E "
			}
			fmt.Fprintf(&b, fmtString, x[k])
			k++
		}
		fmt.Fprint(&b, "]")
	}
	fmt.Fprint(&b, "]")
	return b.String()
}
