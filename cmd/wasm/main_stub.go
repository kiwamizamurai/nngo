//go:build !js || !wasm

// This stub exists so that `go vet`, `go test`, and `go build ./...` succeed on
// the host platform even though the real entry point is WebAssembly-only.
package main

import "fmt"

func main() {
	fmt.Println("nngo wasm: build with `GOOS=js GOARCH=wasm go build` (see Makefile)")
}
