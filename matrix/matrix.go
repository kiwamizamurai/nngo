// Package matrix provides minimal [][]float64-based matrix operations used by
// the nn package. The implementation favors readability over performance so it
// can serve as a teaching reference for matrix calculus and backprop.
package matrix

import (
	"math"
	"math/rand"
)

// Matrix is a row-major dense matrix represented as a slice of rows.
// Rows = len(m), Cols = len(m[0]).
type Matrix [][]float64

// New allocates a zero-initialized rows×cols matrix.
func New(rows, cols int) Matrix {
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

// Shape returns (rows, cols). An empty matrix returns (0, 0).
func (m Matrix) Shape() (int, int) {
	if len(m) == 0 {
		return 0, 0
	}
	return len(m), len(m[0])
}

// Randn fills a rows×cols matrix with samples from N(0, scale²).
func Randn(rows, cols int, scale float64, r *rand.Rand) Matrix {
	m := New(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = r.NormFloat64() * scale
		}
	}
	return m
}

// Dot returns the matrix product A(r×k) × B(k×c). Panics on shape mismatch.
func Dot(a, b Matrix) Matrix {
	ar, ak := a.Shape()
	bk, bc := b.Shape()
	if ak != bk {
		panic("matrix.Dot: inner dimension mismatch")
	}
	out := New(ar, bc)
	for i := 0; i < ar; i++ {
		for k := 0; k < ak; k++ {
			aik := a[i][k]
			for j := 0; j < bc; j++ {
				out[i][j] += aik * b[k][j]
			}
		}
	}
	return out
}

// AddRow returns A with row broadcast-added to every row (bias addition).
func AddRow(a Matrix, row []float64) Matrix {
	r, c := a.Shape()
	out := New(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[i][j] = a[i][j] + row[j]
		}
	}
	return out
}

// Transpose returns Aᵀ.
func Transpose(a Matrix) Matrix {
	r, c := a.Shape()
	out := New(c, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j][i] = a[i][j]
		}
	}
	return out
}

// Apply returns a new matrix with f applied element-wise.
func Apply(a Matrix, f func(float64) float64) Matrix {
	r, c := a.Shape()
	out := New(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[i][j] = f(a[i][j])
		}
	}
	return out
}

// ColSum returns the per-column sum (used for bias gradient: dB = sum over batch).
func ColSum(a Matrix) []float64 {
	r, c := a.Shape()
	out := make([]float64, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j] += a[i][j]
		}
	}
	return out
}

// SoftmaxRow applies softmax to each row. The standard max-subtraction trick
// is used to keep the exponentiation numerically stable.
func SoftmaxRow(a Matrix) Matrix {
	r, c := a.Shape()
	out := New(r, c)
	for i := 0; i < r; i++ {
		maxv := a[i][0]
		for j := 1; j < c; j++ {
			if a[i][j] > maxv {
				maxv = a[i][j]
			}
		}
		var sum float64
		for j := 0; j < c; j++ {
			out[i][j] = math.Exp(a[i][j] - maxv)
			sum += out[i][j]
		}
		for j := 0; j < c; j++ {
			out[i][j] /= sum
		}
	}
	return out
}
