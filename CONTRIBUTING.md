# Contributing to ising-rs

## Before submitting a PR

1. **Format** your code:
   ```bash
   cargo fmt
   ```

2. **Lint** with clippy:
   ```bash
   cargo clippy -- -D warnings
   ```

3. **Run tests**:
   ```bash
   cargo test
   ```

All three checks run in CI and must pass before merging.

## Rust version

This project pins Rust **1.94** via `rust-toolchain.toml`. Install it with:
```bash
rustup install 1.94
```

## Project layout

- `src/` — library and binaries (Metropolis, Wolff, FSS, Kibble-Zurek)
- `analysis/` — Jupyter notebooks for data analysis and plotting
- `paper/` — LaTeX draft and figures
- `scripts/` — helper scripts for data generation

## Commit messages

- Use imperative mood ("Add feature", not "Added feature")
- Keep the first line under 72 characters
