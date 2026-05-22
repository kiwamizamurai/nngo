.PHONY: all build serve test lint fmt vet clean

all: test build

# Build WASM into docs/ (consumed by GitHub Pages)
build:
	@if [ -f "$$(go env GOROOT)/lib/wasm/wasm_exec.js" ]; then \
		cp "$$(go env GOROOT)/lib/wasm/wasm_exec.js" docs/wasm_exec.js; \
	else \
		cp "$$(go env GOROOT)/misc/wasm/wasm_exec.js" docs/wasm_exec.js; \
	fi
	GOOS=js GOARCH=wasm go build -o docs/nngo.wasm ./cmd/wasm/
	@ls -lh docs/nngo.wasm

# Serve docs/ locally (requires python3)
serve: build
	cd docs && python3 -m http.server 8080

# Unit tests for matrix and nn packages (host arch, not WASM)
test:
	go test -race -count=1 ./matrix/... ./nn/...

lint:
	@command -v golangci-lint >/dev/null 2>&1 || { \
		echo "golangci-lint not found. Install: https://golangci-lint.run/usage/install/"; \
		exit 1; }
	golangci-lint run ./...

fmt:
	gofmt -s -w .

vet:
	go vet ./...

clean:
	rm -f docs/nngo.wasm docs/wasm_exec.js
