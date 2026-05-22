//go:build js && wasm

// Command wasm exposes the nngo MLP to the browser via syscall/js. It is
// compiled to WebAssembly and consumed by docs/index.html.
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"syscall/js"

	"github.com/kiwamizamurai/nngo/matrix"
	"github.com/kiwamizamurai/nngo/nn"
)

// JS-facing surface (mounted on window.nngo):
//
//	nngo.init({n, k, layers, activation, loss, lr, seed}) -> {x, y, classes}
//	nngo.train(steps)               -> loss (number)
//	nngo.predictGrid(bound, res)    -> Int32Array(res*res)
//	nngo.getParams()                -> {W1, b1, ..., shapes, numLayers}
type state struct {
	net     *nn.MLP
	x       matrix.Matrix
	y       []int
	tOneHot matrix.Matrix
	k       int
	lr      float64
}

var s state

func jsInit(_ js.Value, args []js.Value) any {
	cfg := args[0]
	n := cfg.Get("n").Int()
	k := cfg.Get("k").Int()
	lr := cfg.Get("lr").Float()
	seed := int64(cfg.Get("seed").Int())

	layersJS := cfg.Get("layers")
	layers := make([]int, layersJS.Length())
	for i := 0; i < layersJS.Length(); i++ {
		layers[i] = layersJS.Index(i).Int()
	}

	actStr := "sigmoid"
	if v := cfg.Get("activation"); !v.IsUndefined() && !v.IsNull() {
		actStr = v.String()
	}
	lossStr := "mse"
	if v := cfg.Get("loss"); !v.IsUndefined() && !v.IsNull() {
		lossStr = v.String()
	}

	s.x, s.y = nn.Spiral(n, k, seed)
	s.tOneHot = nn.OneHot(s.y, k)
	s.net = nn.NewMLP(layers, seed+1, nn.ActivationFromString(actStr), nn.LossFromString(lossStr))
	s.k, s.lr = k, lr

	rows, _ := s.x.Shape()
	xFlat := make([]float64, rows*2)
	for i := 0; i < rows; i++ {
		xFlat[i*2] = s.x[i][0]
		xFlat[i*2+1] = s.x[i][1]
	}

	obj := js.Global().Get("Object").New()
	obj.Set("x", toFloat64Array(xFlat))
	obj.Set("y", toInt32Array(s.y))
	obj.Set("classes", k)
	return obj
}

func jsTrain(_ js.Value, args []js.Value) any {
	steps := 1
	if len(args) > 0 {
		steps = args[0].Int()
	}
	var loss float64
	for i := 0; i < steps; i++ {
		s.net.Forward(s.x)
		loss = s.net.Loss(s.tOneHot)
		s.net.Backward(s.tOneHot, s.lr)
	}
	return loss
}

func jsPredictGrid(_ js.Value, args []js.Value) any {
	bound := args[0].Float()
	res := args[1].Int()
	pts := matrix.New(res*res, 2)
	step := 2 * bound / float64(res-1)
	for i := 0; i < res; i++ {
		for j := 0; j < res; j++ {
			pts[i*res+j][0] = -bound + float64(j)*step
			pts[i*res+j][1] = -bound + float64(i)*step
		}
	}
	return toInt32Array(s.net.Predict(pts))
}

func jsGetParams(_ js.Value, _ []js.Value) any {
	obj := js.Global().Get("Object").New()
	shapes := js.Global().Get("Object").New()

	for l := 0; l < len(s.net.Ws); l++ {
		W := s.net.Ws[l]
		B := s.net.Bs[l]
		r, c := W.Shape()

		flat := make([]float64, r*c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				flat[i*c+j] = W[i][j]
			}
		}

		wName := fmt.Sprintf("W%d", l+1)
		bName := fmt.Sprintf("B%d", l+1)
		obj.Set(wName, toFloat64Array(flat))
		obj.Set(bName, toFloat64Array(B))
		shapes.Set(wName, makeShape(r, c))
		shapes.Set(bName, makeShape(1, len(B)))
	}
	obj.Set("shapes", shapes)
	obj.Set("numLayers", len(s.net.Ws))
	return obj
}

func makeShape(rows, cols int) js.Value {
	o := js.Global().Get("Object").New()
	o.Set("rows", rows)
	o.Set("cols", cols)
	return o
}

func toFloat64Array(f []float64) js.Value {
	bytes := make([]byte, len(f)*8)
	for i, v := range f {
		binary.LittleEndian.PutUint64(bytes[i*8:], math.Float64bits(v))
	}
	buf := js.Global().Get("ArrayBuffer").New(len(bytes))
	view := js.Global().Get("Uint8Array").New(buf)
	js.CopyBytesToJS(view, bytes)
	return js.Global().Get("Float64Array").New(buf)
}

func toInt32Array(v []int) js.Value {
	bytes := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(bytes[i*4:], uint32(int32(x)))
	}
	buf := js.Global().Get("ArrayBuffer").New(len(bytes))
	view := js.Global().Get("Uint8Array").New(buf)
	js.CopyBytesToJS(view, bytes)
	return js.Global().Get("Int32Array").New(buf)
}

func main() {
	nngo := js.Global().Get("Object").New()
	nngo.Set("init", js.FuncOf(jsInit))
	nngo.Set("train", js.FuncOf(jsTrain))
	nngo.Set("predictGrid", js.FuncOf(jsPredictGrid))
	nngo.Set("getParams", js.FuncOf(jsGetParams))
	js.Global().Set("nngo", nngo)
	select {}
}
