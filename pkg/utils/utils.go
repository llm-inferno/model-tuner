package utils

import (
	"bytes"
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// print vector
func VecString(name string, v *mat.VecDense) string {
	x := v.RawVector().Data
	var b bytes.Buffer
	fmt.Fprint(&b, name+"=[")
	for i := 0; i < len(x); i++ {
		fmt.Fprintf(&b, "%10.3f ", x[i])
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
	for i := 0; i < nRows; i++ {
		fmt.Fprint(&b, "[")
		for j := 0; j < nCols; j++ {
			fmt.Fprintf(&b, "%10.3f ", x[k])
			k++
		}
		fmt.Fprint(&b, "]")
	}
	fmt.Fprint(&b, "]")
	return b.String()
}
