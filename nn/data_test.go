package nn

import "testing"

func TestSpiralShape(t *testing.T) {
	x, y := Spiral(50, 3, 1)
	if len(x) != 150 {
		t.Errorf("len(x) = %d, want 150", len(x))
	}
	if len(y) != 150 {
		t.Errorf("len(y) = %d, want 150", len(y))
	}
	_, c := x.Shape()
	if c != 2 {
		t.Errorf("cols = %d, want 2", c)
	}
}

func TestSpiralLabelDistribution(t *testing.T) {
	_, y := Spiral(20, 4, 1)
	counts := make(map[int]int)
	for _, v := range y {
		counts[v]++
	}
	for c := 0; c < 4; c++ {
		if counts[c] != 20 {
			t.Errorf("class %d count = %d, want 20", c, counts[c])
		}
	}
}

func TestSpiralDeterministic(t *testing.T) {
	x1, y1 := Spiral(30, 3, 42)
	x2, y2 := Spiral(30, 3, 42)
	for i := range x1 {
		for j := range x1[i] {
			if x1[i][j] != x2[i][j] {
				t.Errorf("same seed produced different X at [%d][%d]", i, j)
			}
		}
	}
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Errorf("same seed produced different y at [%d]", i)
		}
	}
}
