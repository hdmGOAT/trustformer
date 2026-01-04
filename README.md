# Trustformer

Trustformer is a Transformer-based language model implementation in Rust. This project was built to learn how transformers work at a low level.

## Project Structure

- `src/`: Core implementation.
    - `embeddings/`: Token and positional embeddings.
    - `model/`: High-level model architecture (Decoder).
    - `sampling/`: Text generation and sampling strategies.
    - `tensor/`: Custom tensor operations and math backend.
    - `tokenizer/`: BPE tokenizer training and inference.
    - `transformer/`: Transformer blocks (Attention, FeedForward, LayerNorm).
    - `utils/`: Helper functions.
- `data/`: Directory for training data and other resources.
- `tests/`: Integration tests.

## Getting Started

1.  Ensure you have Rust installed.
2.  Place your training data in `data/training_data.txt`.
3.  Run the project:

```bash
cargo run
```

## Features

-   **Tokenizer**: Byte Pair Encoding (BPE) tokenizer.
-   **Model**: Decoder-only Transformer architecture.
-   **Sampling**: Text generation with temperature sampling.

## Usage

The `main.rs` file demonstrates how to:
1.  Load training data.
2.  Train the tokenizer.
3.  Initialize the model.
4.  Generate text based on a prompt.
