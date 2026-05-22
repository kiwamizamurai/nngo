# nngo

[![CI](https://github.com/kiwamizamurai/nngo/actions/workflows/ci.yml/badge.svg)](https://github.com/kiwamizamurai/nngo/actions/workflows/ci.yml)
[![Pages](https://github.com/kiwamizamurai/nngo/actions/workflows/pages.yml/badge.svg)](https://github.com/kiwamizamurai/nngo/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Reference](https://pkg.go.dev/badge/github.com/kiwamizamurai/nngo.svg)](https://pkg.go.dev/github.com/kiwamizamurai/nngo)

A tiny, hand-derived multi-layer perceptron written in Go, compiled to
WebAssembly, and served as an interactive teaching playground on GitHub Pages.
Build a network layer-by-layer in the browser, watch the decision boundary
form, and see the underlying weight matrices update in real time.

**Live demo:** <https://kiwamizamurai.github.io/nngo/>

## What it shows

- **2D classification on the CS231n spiral dataset**, training entirely inside
  the browser.
- **Configurable architecture**: drag-and-drop layer chips to add, remove, and
  reorder hidden layers; edit per-layer sizes inline.
- **Activation choice**: sigmoid, ReLU, tanh.
- **Loss choice**: customized MSE (with sigmoid output) or Cross Entropy (with
  softmax output) — math derivations rendered in MathJax beside the controls.
- **Live weight matrices** rendered as bracketed matrices with both color and
  numeric values, plus a paper-style network diagram whose edge color/width
  reflect the current weights.

## What nngo gives you that micrograd and tinygrad don't

**Live, numeric weight matrices in the browser during training.**

Verified against the upstream implementations:

- [micrograd](https://github.com/karpathy/micrograd) ships
  `trace_graph.ipynb`, which uses graphviz to draw a static computation graph
  of *scalar* `Value` nodes (each labeled `data %.4f | grad %.4f`). It
  visualizes a single backward pass, not training, and shows individual
  scalars rather than weight matrices.
- [tinygrad](https://github.com/tinygrad/tinygrad) has no built-in network
  visualization — the repo is framework and accelerator code. Visualization,
  if any, is left to the user.

nngo renders each $W_l$ and $b_l$ as an actual matrix in $[\,\cdot\,]$
bracket notation, with **per-cell color and the numeric value** updating
every training step. You can watch `0.07 → 0.43 → -1.2` and read which
hidden unit took on which role.

## Architecture

```
nngo/
├── matrix/        small dense linear algebra ([][]float64-based)
├── nn/            MLP, activations, losses, spiral dataset generator
├── cmd/wasm/      WebAssembly entry point (window.nngo bindings)
└── docs/          static site (index.html + nngo.wasm + wasm_exec.js)
```

The `matrix` package has no external dependencies; `nn` depends only on
`matrix`; `cmd/wasm` is the only place that touches `syscall/js`. This makes
the core testable on any platform without WASM.

## Math reference (what the code computes)

For layers indexed $l = 1, \ldots, L$:

- **Forward**: $X_l = f_l(X_{l-1} W_l + b_l)$
- **MSE loss**: $E = \tfrac{1}{2}\lVert X_L - T\rVert^2$, output = sigmoid,
  $\delta_L = (X_L - T) \odot X_L (1 - X_L)$
- **CE loss**: $E = -\sum T \log P$, output = softmax, $\delta_L = P - T$
- **Recursion**: $\delta_{l-1} = (\delta_l W_l^\top) \odot f'_{l-1}(X_{l-1})$
- **Gradients**: $\nabla W_l = X_{l-1}^\top \delta_l$, $\nabla b_l = \mathbf{1}^\top \delta_l$
- **SGD update**: $W_l \leftarrow W_l - \eta \nabla W_l$

All of this is rendered live in the browser playground and implemented in
[`nn/nn.go`](nn/nn.go).

## Getting started

Prerequisites: Go 1.23+. Tests run on the host platform; the WASM build
targets `js/wasm`.

```sh
make test        # run unit tests for matrix and nn packages
make build       # compile WASM into docs/nngo.wasm
make serve       # build + serve docs/ at http://localhost:8080
make lint        # golangci-lint (if installed)
make fmt         # gofmt
```

## Deployment to GitHub Pages

The `.github/workflows/pages.yml` workflow builds the WASM artifact and
publishes `docs/` to GitHub Pages on every push to `main`. To set it up on a
fresh fork:

1. Fork the repository.
2. Repository settings → **Pages** → **Source: GitHub Actions**.
3. Push to `main` — the demo is live at `https://<you>.github.io/nngo/`.

## Contributing

Bug reports and small improvements are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). The scope of nngo is intentionally narrow
(stay tiny, stay readable); please open an issue before sending large features.

## License

[MIT](LICENSE)
