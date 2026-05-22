# Contributing to nngo

Thanks for considering a contribution. nngo is intentionally small; the
priority is readability over features.

## Before you start

For anything beyond a typo fix or obvious bug, **open an issue first** so we
can agree the change fits the project's scope. Things in scope:

- Bug fixes in the math, the WASM bridge, or the UI.
- Documentation improvements (especially clearer math explanations).
- Additional activations / losses *if* their derivative derivations are
  documented alongside the code.

Things generally **not** in scope:

- Convolutional layers, attention, or other modern architectures (use a real
  framework).
- Performance optimizations that obscure the math.
- New dependencies — the project deliberately has zero non-stdlib imports.

## Development workflow

```sh
make test        # unit tests for matrix and nn packages
make lint        # golangci-lint (install separately)
make fmt         # gofmt
make build       # compile WASM into docs/nngo.wasm
make serve       # build + serve docs/ at http://localhost:8080
```

Before opening a PR, please make sure `make test` and `make lint` both pass.

## Code style

- Standard `gofmt`.
- Doc comments on every exported identifier.
- Keep `matrix` dependency-free and `nn` depending only on `matrix`.
- For new math, add a comment with the derivation (one or two lines is fine).
