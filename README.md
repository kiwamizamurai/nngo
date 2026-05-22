# nngo

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

## Math reference (what the code computes)

https://kiwamizamurai.github.io/posts/2022-04-25/

