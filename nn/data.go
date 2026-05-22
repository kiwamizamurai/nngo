package nn

import (
	"math"
	"math/rand"

	"github.com/kiwamizamurai/nngo/matrix"
)

// Spiral generates the classic CS231n-style spiral dataset: K interleaved arms
// of N points each centered on the origin. Returns the N*K × 2 input matrix
// and the corresponding integer labels in [0, K).
func Spiral(n, k int, seed int64) (matrix.Matrix, []int) {
	r := rand.New(rand.NewSource(seed))
	x := matrix.New(n*k, 2)
	y := make([]int, n*k)
	for j := 0; j < k; j++ {
		for i := 0; i < n; i++ {
			idx := j*n + i
			radius := float64(i) / float64(n)
			theta := float64(j)*4 + float64(i)/float64(n)*4 + r.NormFloat64()*0.2
			x[idx][0] = radius * math.Sin(theta)
			x[idx][1] = radius * math.Cos(theta)
			y[idx] = j
		}
	}
	return x, y
}
