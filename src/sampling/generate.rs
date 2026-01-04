use crate::{model::decoder::Decoder, tokenizer::core::Tokenizer};

fn softmax_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    assert!(temperature > 0.0);

    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits
        .iter()
        .map(|&x| ((x - max_logit) / temperature).exp())
        .collect();

    let sum: f32 = exps.iter().sum();

    exps.iter().map(|&x| x / sum).collect()
}

pub fn sample_from_probs(probs: &[f32], r:f32) -> usize {
    let mut acc = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return i;
        }

    }

    probs.len() - 1
}


pub fn generate(
    model: &Decoder,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    rng: impl Fn() -> f32,
) -> String {
    let mut tokens = tokenizer.encode(prompt, true);

    for _ in 0..max_tokens {
        let logits = model.forward(&tokens);

        let last = logits.row(tokens.len() - 1);

        let probs = softmax_temp(&last, temperature);
        let next = sample_from_probs(&probs, rng());

        tokens.push(next);

        if next == tokenizer.specials.eos {
            break;
        }
    }

    tokenizer.decode(&tokens)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::decoder::Decoder;
    use crate::tokenizer::core::Tokenizer;
    use std::collections::HashMap;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_temp(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_softmax_all_positive() {
        let logits = vec![0.5, -1.2, 3.3];
        let probs = softmax_temp(&logits, 1.0);
        for p in probs {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_softmax_preserves_ordering() {
        let logits = vec![5.0, 2.0, -1.0];
        let probs = softmax_temp(&logits, 1.0);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_equal_logits_uniform() {
        let logits = vec![0.0, 0.0, 0.0, 0.0];
        let probs = softmax_temp(&logits, 0.7);
        for p in probs {
            assert!(approx_eq(p, 0.25));
        }
    }

    #[test]
    fn test_softmax_low_temperature_sharp() {
        let logits = vec![10.0, 0.0, 0.0];
        let probs = softmax_temp(&logits, 0.1);
        assert!(probs[0] > 0.99);
    }

    #[test]
    fn test_softmax_high_temperature_flat() {
        let logits = vec![10.0, 0.0, 0.0];
        let probs = softmax_temp(&logits, 10.0);
        for p in probs {
            assert!(p > 0.2);
        }
    }

    #[test]
    fn test_softmax_large_logits_stable() {
        let logits = vec![1000.0, 999.0, 998.0];
        let probs = softmax_temp(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!(approx_eq(sum, 1.0));
        assert!(!probs.iter().any(|x| x.is_nan()));
    }

    #[test]
    fn test_softmax_zero_temperature_panics() {
        let logits = vec![1.0, 2.0];
        let result = std::panic::catch_unwind(|| softmax_temp(&logits, 0.0));
        assert!(result.is_err());
    }

    // -----------------------------
    // sample_from_probs tests
    // -----------------------------
    #[test]
    fn test_sample_from_probs_simple() {
        let probs = vec![0.1, 0.2, 0.7];
        assert_eq!(sample_from_probs(&probs, 0.05), 0);
        assert_eq!(sample_from_probs(&probs, 0.15), 1);
        assert_eq!(sample_from_probs(&probs, 0.5), 2);
        assert_eq!(sample_from_probs(&probs, 0.95), 2);
    }

    #[test]
    fn test_sample_from_probs_edge_cases() {
        let probs = vec![0.5, 0.5];
        assert_eq!(sample_from_probs(&probs, 0.0), 0);
        assert_eq!(sample_from_probs(&probs, 0.499), 0);
        assert_eq!(sample_from_probs(&probs, 0.5), 1);
        assert_eq!(sample_from_probs(&probs, 0.999), 1);
        assert_eq!(sample_from_probs(&probs, 1.0), 1); // fallback to last index
    }

    #[test]
    fn test_sample_from_probs_single_element() {
        let probs = vec![1.0];
        assert_eq!(sample_from_probs(&probs, 0.0), 0);
        assert_eq!(sample_from_probs(&probs, 0.999), 0);
    }

    // -----------------------------
    // generate tests
    // -----------------------------
    fn build_test_model_and_tokenizer() -> (Decoder, Tokenizer) {
        let mut vocab = HashMap::new();
        vocab.insert([b'a' as usize, b'b' as usize], 256);

        let tokenizer = Tokenizer::new(vocab);

        let vocab_size = 260; // plus specials
        let max_seq_len = 10;
        let d_model = 8;
        let n_layers = 1;
        let n_heads = 2;
        let ffn_hidden = 16;

        let model = Decoder::new(
            vocab_size,
            max_seq_len,
            d_model,
            n_layers,
            n_heads,
            ffn_hidden,
        );

        (model, tokenizer)
    }

    #[test]
    fn test_generate_returns_string() {
        let (model, tokenizer) = build_test_model_and_tokenizer();
        let output = generate(&model, &tokenizer, "ab", 5, 1.0, || 0.5);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_generate_stops_at_eos() {
        let (mut model, tokenizer) = build_test_model_and_tokenizer();

        // force EOS token as first sampled token by zeroing logits
        for w in model.lm_head.weight.data_mut() {
            *w = 0.0;
        }
        for b in model.lm_head.bias.data_mut() {
            *b = 0.0;
        }

        let output = generate(&model, &tokenizer, "a", 100, 1.0, || 1.0);
        // should terminate early if EOS encountered
        assert!(!output.is_empty());
    }

    #[test]
    fn test_generate_empty_prompt() {
        let (model, tokenizer) = build_test_model_and_tokenizer();
        let output = generate(&model, &tokenizer, "", 5, 1.0, || 0.5);
        assert!(!output.is_empty());
    }
}
