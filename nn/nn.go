// Package nn implements a small multi-layer perceptron with hand-derived
// backpropagation. It is intentionally written to be read top-to-bottom as a
// teaching reference; for production work use an established framework.
package nn

import (
	"math"
	"math/rand"

	"github.com/kiwamizamurai/nngo/matrix"
)

// Activation is the elementwise non-linearity applied to hidden layers.
type Activation int

const (
	Sigmoid Activation = iota
	ReLU
	Tanh
)

func (a Activation) apply(z float64) float64 {
	switch a {
	case Sigmoid:
		return 1.0 / (1.0 + math.Exp(-z))
	case ReLU:
		if z > 0 {
			return z
		}
		return 0
	case Tanh:
		return math.Tanh(z)
	}
	return z
}

// deriv takes the *post-activation* value x = f(z) and returns f'(z). This
// shortcut works because each supported activation has a closed-form derivative
// expressible in terms of its output.
func (a Activation) deriv(x float64) float64 {
	switch a {
	case Sigmoid:
		return x * (1 - x)
	case ReLU:
		if x > 0 {
			return 1
		}
		return 0
	case Tanh:
		return 1 - x*x
	}
	return 1
}

// ActivationFromString maps "sigmoid"/"relu"/"tanh" to the enum, defaulting to Sigmoid.
func ActivationFromString(s string) Activation {
	switch s {
	case "relu":
		return ReLU
	case "tanh":
		return Tanh
	default:
		return Sigmoid
	}
}

// Loss selects the loss function and, implicitly, the output-layer activation.
//
//	MSE:          E = (1/2)‖X_L − T‖²   output = sigmoid
//	              δ_L = (X_L − T) ⊙ X_L ⊙ (1 − X_L)
//
//	CrossEntropy: E = −Σ T log P        output = softmax (P)
//	              δ_L = P − T           (softmax+CE simplification)
type Loss int

const (
	MSE Loss = iota
	CrossEntropy
)

// LossFromString maps "mse"/"ce" (and aliases) to the enum, defaulting to MSE.
func LossFromString(s string) Loss {
	switch s {
	case "ce", "crossentropy", "cross-entropy":
		return CrossEntropy
	default:
		return MSE
	}
}

// MLP is a fully-connected feed-forward network with L = len(sizes)-1 weight
// layers. Hidden layers share Activation; the output layer's activation is
// determined by LossFn.
//
//	X_0 ─W_1─► X_1 ─W_2─► X_2 ─...─► X_L
//	X_l = f_l(X_{l-1} W_l + b_l)
//
// Backprop (chain rule, δ notation):
//
//	δ_l-1 = (δ_l W_lᵀ) ⊙ f_{l-1}'(X_{l-1})
//	∂E/∂W_l = X_{l-1}ᵀ · δ_l
//	∂E/∂b_l = Σ_batch δ_l
type MLP struct {
	Ws     []matrix.Matrix // weights, length L
	Bs     [][]float64     // biases, length L
	Xs     []matrix.Matrix // cached activations, length L+1 (X_0 ... X_L)
	Act    Activation
	LossFn Loss
}

// NewMLP builds an MLP with the given layer sizes. sizes must have at least 2
// elements: sizes[0] is the input dimension, sizes[len-1] is the output
// dimension, and everything in between are hidden layer widths. Weights are
// initialized with He (ReLU) or Xavier (others) scaled Gaussians.
func NewMLP(sizes []int, seed int64, act Activation, lossFn Loss) *MLP {
	r := rand.New(rand.NewSource(seed))
	L := len(sizes) - 1
	m := &MLP{
		Ws:     make([]matrix.Matrix, L),
		Bs:     make([][]float64, L),
		Xs:     make([]matrix.Matrix, L+1),
		Act:    act,
		LossFn: lossFn,
	}
	for l := 0; l < L; l++ {
		scale := math.Sqrt(1.0 / float64(sizes[l]))
		if act == ReLU {
			scale = math.Sqrt(2.0 / float64(sizes[l]))
		}
		m.Ws[l] = matrix.Randn(sizes[l], sizes[l+1], scale, r)
		m.Bs[l] = make([]float64, sizes[l+1])
	}
	return m
}

func sigmoidScalar(z float64) float64 { return 1.0 / (1.0 + math.Exp(-z)) }

// Forward runs the network on x and returns the final activations X_L.
// All intermediate X_l values are cached for the subsequent Backward call.
func (m *MLP) Forward(x matrix.Matrix) matrix.Matrix {
	m.Xs[0] = x
	L := len(m.Ws)
	for l := 0; l < L; l++ {
		z := matrix.AddRow(matrix.Dot(m.Xs[l], m.Ws[l]), m.Bs[l])
		switch {
		case l == L-1 && m.LossFn == CrossEntropy:
			m.Xs[l+1] = matrix.SoftmaxRow(z)
		case l == L-1:
			m.Xs[l+1] = matrix.Apply(z, sigmoidScalar)
		default:
			m.Xs[l+1] = matrix.Apply(z, m.Act.apply)
		}
	}
	return m.Xs[L]
}

// Loss computes the configured loss against one-hot targets t (shape N×K).
func (m *MLP) Loss(t matrix.Matrix) float64 {
	xL := m.Xs[len(m.Ws)]
	n, c := xL.Shape()

	if m.LossFn == CrossEntropy {
		var sum float64
		for i := 0; i < n; i++ {
			for j := 0; j < c; j++ {
				if t[i][j] > 0 {
					p := xL[i][j]
					if p < 1e-12 {
						p = 1e-12
					}
					sum -= t[i][j] * math.Log(p)
				}
			}
		}
		return sum / float64(n)
	}

	var sum float64
	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			d := xL[i][j] - t[i][j]
			sum += 0.5 * d * d
		}
	}
	return sum / float64(n)
}

// Backward computes per-layer gradients and performs a single SGD step.
// Forward must have been called first to populate the activation cache.
func (m *MLP) Backward(t matrix.Matrix, lr float64) {
	L := len(m.Ws)
	xL := m.Xs[L]
	invN := 1.0 / float64(len(xL))

	deltas := make([]matrix.Matrix, L)
	if m.LossFn == CrossEntropy {
		deltas[L-1] = elemwise(xL, func(i, j int, x float64) float64 {
			return (x - t[i][j]) * invN
		})
	} else {
		deltas[L-1] = elemwise(xL, func(i, j int, x float64) float64 {
			return (x - t[i][j]) * x * (1 - x) * invN
		})
	}

	for l := L - 1; l >= 1; l-- {
		d := matrix.Dot(deltas[l], matrix.Transpose(m.Ws[l]))
		x := m.Xs[l]
		r, c := d.Shape()
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				d[i][j] *= m.Act.deriv(x[i][j])
			}
		}
		deltas[l-1] = d
	}

	for l := 0; l < L; l++ {
		dW := matrix.Dot(matrix.Transpose(m.Xs[l]), deltas[l])
		dB := matrix.ColSum(deltas[l])
		applyGrad(m.Ws[l], dW, lr)
		for i, g := range dB {
			m.Bs[l][i] -= lr * g
		}
	}
}

// Predict returns the argmax class for each input row.
func (m *MLP) Predict(x matrix.Matrix) []int {
	out := m.Forward(x)
	n, c := out.Shape()
	res := make([]int, n)
	for i := 0; i < n; i++ {
		best := 0
		for j := 1; j < c; j++ {
			if out[i][j] > out[i][best] {
				best = j
			}
		}
		res[i] = best
	}
	return res
}

// OneHot converts integer labels y ∈ [0, k) to a one-hot N×K matrix.
func OneHot(y []int, k int) matrix.Matrix {
	t := matrix.New(len(y), k)
	for i, v := range y {
		t[i][v] = 1
	}
	return t
}

func elemwise(a matrix.Matrix, f func(i, j int, v float64) float64) matrix.Matrix {
	r, c := a.Shape()
	out := matrix.New(r, c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[i][j] = f(i, j, a[i][j])
		}
	}
	return out
}

func applyGrad(w, dw matrix.Matrix, lr float64) {
	r, c := w.Shape()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			w[i][j] -= lr * dw[i][j]
		}
	}
}
