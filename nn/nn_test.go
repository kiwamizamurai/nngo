package nn

import (
	"math"
	"math/rand"
	"testing"

	"github.com/kiwamizamurai/nngo/matrix"
)

func TestForwardShape(t *testing.T) {
	m := NewMLP([]int{2, 4, 4, 3}, 42, Sigmoid, MSE)
	x := matrix.New(10, 2)
	out := m.Forward(x)
	r, c := out.Shape()
	if r != 10 || c != 3 {
		t.Errorf("output shape = (%d, %d), want (10, 3)", r, c)
	}
}

func TestForwardSigmoidInRange(t *testing.T) {
	m := NewMLP([]int{2, 3, 3}, 1, Sigmoid, MSE)
	r := rand.New(rand.NewSource(7))
	x := matrix.Randn(5, 2, 1.0, r)
	out := m.Forward(x)
	for i := range out {
		for j, v := range out[i] {
			if v <= 0 || v >= 1 {
				t.Errorf("sigmoid output[%d][%d] = %v, want in (0, 1)", i, j, v)
			}
		}
	}
}

func TestForwardSoftmaxSumsToOne(t *testing.T) {
	m := NewMLP([]int{2, 4, 3}, 1, ReLU, CrossEntropy)
	r := rand.New(rand.NewSource(7))
	x := matrix.Randn(8, 2, 1.0, r)
	out := m.Forward(x)
	for i := range out {
		var sum float64
		for _, v := range out[i] {
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("softmax row %d sum = %v, want 1", i, sum)
		}
	}
}

func TestOneHot(t *testing.T) {
	got := OneHot([]int{0, 2, 1}, 3)
	want := matrix.Matrix{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}
	for i := range want {
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Errorf("OneHot[%d][%d] = %v, want %v", i, j, got[i][j], want[i][j])
			}
		}
	}
}

func TestTrainingReducesLoss(t *testing.T) {
	cases := []struct {
		name   string
		act    Activation
		lossFn Loss
		lr     float64
	}{
		{"sigmoid_mse", Sigmoid, MSE, 1.0},
		{"relu_ce", ReLU, CrossEntropy, 0.3},
		{"tanh_mse", Tanh, MSE, 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, y := Spiral(60, 3, 1)
			target := OneHot(y, 3)
			m := NewMLP([]int{2, 8, 8, 3}, 2, tc.act, tc.lossFn)

			m.Forward(x)
			initial := m.Loss(target)

			for i := 0; i < 300; i++ {
				m.Forward(x)
				m.Backward(target, tc.lr)
			}

			m.Forward(x)
			final := m.Loss(target)

			if final >= initial {
				t.Errorf("loss did not decrease: initial=%v final=%v", initial, final)
			}
		})
	}
}

func TestPredictRange(t *testing.T) {
	m := NewMLP([]int{2, 4, 3}, 1, Sigmoid, MSE)
	x := matrix.Matrix{{0, 0}, {1, 1}, {-1, 0.5}}
	preds := m.Predict(x)
	if len(preds) != 3 {
		t.Fatalf("len(preds) = %d, want 3", len(preds))
	}
	for i, p := range preds {
		if p < 0 || p >= 3 {
			t.Errorf("pred[%d] = %d, want in [0, 3)", i, p)
		}
	}
}

func TestActivationFromString(t *testing.T) {
	cases := map[string]Activation{
		"sigmoid": Sigmoid,
		"relu":    ReLU,
		"tanh":    Tanh,
		"":        Sigmoid, // default
		"unknown": Sigmoid, // default
	}
	for in, want := range cases {
		if got := ActivationFromString(in); got != want {
			t.Errorf("ActivationFromString(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestLossFromString(t *testing.T) {
	cases := map[string]Loss{
		"mse":           MSE,
		"ce":            CrossEntropy,
		"crossentropy":  CrossEntropy,
		"cross-entropy": CrossEntropy,
		"":              MSE,
	}
	for in, want := range cases {
		if got := LossFromString(in); got != want {
			t.Errorf("LossFromString(%q) = %v, want %v", in, got, want)
		}
	}
}
