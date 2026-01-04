use std::collections::HashMap;

pub struct SpecialTokens {
    pub unk: usize,
    pub bos: usize,
    pub eos: usize,
}

pub struct Tokenizer {
    vocab: HashMap<[usize; 2], usize>,
    rev_vocab: HashMap<usize, [usize; 2]>,
    pub specials: SpecialTokens,
}

impl Tokenizer {
    pub fn new(vocab: HashMap<[usize; 2], usize>) -> Tokenizer {
        let rev_vocab = vocab.iter()
            .map(|(pair, &id)| (id, *pair))
            .collect::<HashMap<usize, [usize; 2]>>();

        let next_id = if rev_vocab.is_empty() {
            256
        } else {
            *rev_vocab.keys().max().unwrap() + 1
        };

        let specials = SpecialTokens {
            unk: next_id,
            bos: next_id + 1,
            eos: next_id + 2,
        };

        Self { 
            vocab, 
            rev_vocab, 
            specials,
        }
    }

    pub fn encode(&self, text: &str, add_bos_eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = text.as_bytes().iter().map(|&x| x as usize).collect();

        let mut i = 0;
        while i + 1 < tokens.len() {
            let pair = [tokens[i], tokens[i + 1]];

            if let Some(&merged) = self.vocab.get(&pair) {
                tokens[i] = merged;
                tokens.remove(i + 1);

                if i > 0 { i = i.saturating_sub(1); }
            } else {
                i += 1;
            }
        }

        if add_bos_eos {
            tokens.insert(0, self.specials.bos);
            tokens.push(self.specials.eos);
        }

        tokens
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut output: Vec<u8> = Vec::new();

        for &token in tokens {
            if token == self.specials.bos || token == self.specials.eos {
                continue;
            } else if token == self.specials.unk {
                output.push(b'?');
            } else {
                self.expand(token, &mut output);
            }
        }

        String::from_utf8_lossy(&output).to_string()
    }

    fn expand(&self, token: usize, acc: &mut Vec<u8>) {
        let mut stack = vec![token];

        while let Some(t) = stack.pop() {
            if t <= 255 {
                acc.push(t as u8);
            } else {
                let pair = self.rev_vocab.get(&t)
                    .expect("token not found in rev_vocab");
                stack.push(pair[1]);
                stack.push(pair[0]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::training::byte_pair_encode;

    #[test]
    fn test_encode_decode_roundtrip() {
        let text = "hello world";
        let mut vocab = HashMap::new();
        vocab.insert([b'h' as usize, b'e' as usize], 256);
        vocab.insert([b'l' as usize, b'l' as usize], 257);
        
        let tokenizer = Tokenizer::new(vocab);

        let encoded = tokenizer.encode(text, false);
        assert_eq!(encoded[0], 256);
        assert_eq!(encoded[1], 257);

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_with_bos_eos() {
        let text = "hello world";
        let mut vocab = HashMap::new();
        vocab.insert([b'h' as usize, b'e' as usize], 256);
        vocab.insert([b'l' as usize, b'l' as usize], 257);

        let tokenizer = Tokenizer::new(vocab);

        let encoded = tokenizer.encode(text, true);
        assert_eq!(encoded.first(), Some(&tokenizer.specials.bos));
        assert_eq!(encoded.last(), Some(&tokenizer.specials.eos));

        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_long_text_roundtrip_compression() {
        let text = "\
            Once upon a midnight dreary, while I pondered, weak and weary, \
            Over many a quaint and curious volume of forgotten lore— \
            While I nodded, nearly napping, suddenly there came a tapping, \
            As of some one gently rapping, rapping at my chamber door. \
            “’Tis some visitor,” I muttered, “tapping at my chamber door— \
            Only this and nothing more.”";

        let vocab_size = 350;
        let vocab = byte_pair_encode(text, vocab_size);

        let tokenizer = Tokenizer::new(vocab);

        let encoded = tokenizer.encode(text, true);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, text, "Decoded text does not match original");

        println!("Original length (bytes): {}", text.len());
        println!("Encoded length (tokens): {}", encoded.len());
        assert!(encoded.len() < text.len(), "BPE should compress the text");
    }

    #[test]
    fn test_decode_with_unk_token() {
        let vocab = HashMap::new();
        let tokenizer = Tokenizer::new(vocab);

        let tokens = vec![
            b'a' as usize,
            tokenizer.specials.unk,
            b'c' as usize
        ];

        let decoded = tokenizer.decode(&tokens);
        assert_eq!(decoded, "a?c");
    }

    #[test]
    fn test_decode_ignores_bos_eos_with_unk() {
        let vocab = HashMap::new();
        let tokenizer = Tokenizer::new(vocab);

        let tokens = vec![
            tokenizer.specials.bos,
            tokenizer.specials.unk,
            tokenizer.specials.eos,
        ];

        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, "?");
    }
}
