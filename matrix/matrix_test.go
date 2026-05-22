package matrix

import (
	"math"
	"math/rand"
	"testing"
)

const eps = 1e-9

func approxEqual(a, b Matrix) bool {
	ar, ac := a.Shape()
	br, bc := b.Shape()
	if ar != br || ac != bc {
		return false
	}
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			if math.Abs(a[i][j]-b[i][j]) > eps {
				return false
			}
		}
	}
	return true
}

func TestNewShape(t *testing.T) {
	m := New(3, 4)
	r, c := m.Shape()
	if r != 3 || c != 4 {
		t.Errorf("Shape = (%d, %d), want (3, 4)", r, c)
	}
	for i := range m {
		for j := range m[i] {
			if m[i][j] != 0 {
				t.Errorf("New[%d][%d] = %v, want 0", i, j, m[i][j])
			}
		}
	}
}

func TestEmptyShape(t *testing.T) {
	var m Matrix
	r, c := m.Shape()
	if r != 0 || c != 0 {
		t.Errorf("empty Shape = (%d, %d), want (0, 0)", r, c)
	}
}

func TestDotShape(t *testing.T) {
	a := New(2, 3)
	b := New(3, 4)
	r, c := Dot(a, b).Shape()
	if r != 2 || c != 4 {
		t.Errorf("Dot shape = (%d, %d), want (2, 4)", r, c)
	}
}

func TestDotIdentity(t *testing.T) {
	a := Matrix{{1, 2, 3}, {4, 5, 6}}
	id := Matrix{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if !approxEqual(Dot(a, id), a) {
		t.Errorf("A · I ≠ A")
	}
}

func TestDotMismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on shape mismatch")
		}
	}()
	Dot(New(2, 3), New(4, 5))
}

func TestAddRowBroadcast(t *testing.T) {
	a := Matrix{{1, 2}, {3, 4}, {5, 6}}
	got := AddRow(a, []float64{10, 20})
	want := Matrix{{11, 22}, {13, 24}, {15, 26}}
	if !approxEqual(got, want) {
		t.Errorf("AddRow = %v, want %v", got, want)
	}
}

func TestTransposeInvolution(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	a := Randn(3, 5, 1.0, r)
	if !approxEqual(Transpose(Transpose(a)), a) {
		t.Errorf("(Aᵀ)ᵀ ≠ A")
	}
}

func TestTransposeShape(t *testing.T) {
	a := New(4, 7)
	r, c := Transpose(a).Shape()
	if r != 7 || c != 4 {
		t.Errorf("Transpose shape = (%d, %d), want (7, 4)", r, c)
	}
}

func TestApply(t *testing.T) {
	a := Matrix{{1, -2}, {-3, 4}}
	got := Apply(a, func(x float64) float64 { return x * x })
	want := Matrix{{1, 4}, {9, 16}}
	if !approxEqual(got, want) {
		t.Errorf("Apply = %v, want %v", got, want)
	}
}

func TestColSum(t *testing.T) {
	a := Matrix{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	got := ColSum(a)
	want := []float64{12, 15, 18}
	for i, v := range want {
		if math.Abs(got[i]-v) > eps {
			t.Errorf("ColSum[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestSoftmaxRowSumsToOne(t *testing.T) {
	a := Matrix{{1, 2, 3}, {-1, 0, 1}, {100, -50, 50}}
	s := SoftmaxRow(a)
	for i := range s {
		var sum float64
		for _, v := range s[i] {
			if v < 0 || v > 1 {
				t.Errorf("softmax[%d] = %v, expected in [0,1]", i, v)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > eps {
			t.Errorf("row %d sum = %v, want 1", i, sum)
		}
	}
}

func TestSoftmaxRowStability(t *testing.T) {
	// Inputs of ~1000 would overflow Exp without the max-subtraction trick.
	a := Matrix{{1000, 1001, 1002}}
	s := SoftmaxRow(a)
	for _, v := range s[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("softmax produced %v (numerical instability)", v)
		}
	}
}
