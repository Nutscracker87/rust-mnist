# rust-mnist

A small **feed-forward neural network** in Rust for MNIST digit classification. It implements backpropagation and mini-batch stochastic gradient descent (SGD) from scratch using the [ndarray](https://crates.io/crates/ndarray) crate, with optional terminal visualisation of activations and average weight maps.

## Features

- **Architecture:** 784 → 36 → 10 (one hidden layer, sigmoid activations)
- **Training:** Mini-batch SGD; data shuffled each epoch
- **CLI:** Optional `--digit N` to visualise weight maps for a specific digit (0–9) after the last epoch

## Requirements

- **Rust** (e.g. via [rustup](https://rustup.rs/))
- **MNIST data** in a `data/` directory at the project root (see below)

## MNIST data setup

The [mnist](https://crates.io/crates/mnist) crate expects the standard MNIST files in the `data/` folder. Download if not presented, the four gzip files from [yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/):

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Place them in `data/` (no need to gunzip; the crate reads them as gzip). Your layout should look like:

```
rust-mnist/
  data/
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
  src/
  Cargo.toml
  ...
```

## Build and run

```bash
cargo build
cargo run
```

Training runs for 30 epochs and prints test accuracy after each. On the final epoch, weight visualisations are printed for a **random** digit unless you specify one:

```bash
# Visualise weight maps for digit 7
cargo run -- --digit 7

# Short form
cargo run -- -d 0
```

The `--` is required so that `--digit` and `-d` are passed to your program, not to Cargo.

## Licence

MIT (or your choice).
