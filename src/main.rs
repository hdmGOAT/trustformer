use std::fs;
use trustformer::model::decoder::Decoder;
use trustformer::sampling::generate::generate;
use trustformer::tokenizer::core::Tokenizer;
use trustformer::tokenizer::training::byte_pair_encode;

fn main() {
    let text = fs::read_to_string("data/training_data.txt").expect("Failed to read training data");
    let bpe_vocab_size = 300;
    let vocab = byte_pair_encode(&text, bpe_vocab_size);

    let tokenizer = Tokenizer::new(vocab);

    let vocab_size = bpe_vocab_size + 3; 
    let max_seq_len = 64;
    let d_model = 32;
    let n_layers = 2;
    let n_heads = 4;
    let ffn_hidden = 64;

    let model = Decoder::new(
        vocab_size,
        max_seq_len,
        d_model,
        n_layers,
        n_heads,
        ffn_hidden,
    );

    let prompt = "hans";

    let rng = || rand::random::<f32>();

    let output =generate(
        &model,
        &tokenizer,
        prompt,
        16,
        1.0,      
        rng,
    );

    println!("Prompt: {}", prompt);
    println!("Generated: {}", output);
}
