use std::collections::HashMap;

pub struct Tokenizer {
    vocab: HashMap<[usize; 2], usize>,
    rev_vocab: HashMap<usize, [usize; 2]>
}

impl Tokenizer {
    pub fn new(vocab: HashMap<[usize; 2], usize>) -> Tokenizer {
        let rev_vocab = vocab.iter()
            .map(|(pair, &id)| (id, *pair))
            .collect();
        Self { vocab, rev_vocab } 
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        todo!()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        todo!()
    }

    fn expand(&self, token: usize, acc: &mut Vec<u8>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::training::{byte_pair_encode, to_ids};

    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let text = "hello world";
        let mut vocab = HashMap::new();
        vocab.insert([b'h' as usize, b'e' as usize], 256);
        vocab.insert([b'l' as usize, b'l' as usize], 257);
        
        let tokenizer = Tokenizer::new(vocab);

        let encoded = tokenizer.encode(text);
        assert_eq!(encoded[0], 256);
        assert_eq!(encoded[1], 257);

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

        let initial_tokens = to_ids(text);
        
        let vocab_size = 350;
        let vocab = byte_pair_encode(&initial_tokens, vocab_size);

        let tokenizer = Tokenizer::new(vocab);

        let encoded = tokenizer.encode(text);

        let decoded = tokenizer.decode(&encoded);

        assert_eq!(decoded, text, "Decoded text does not match original");

        println!("Original length (bytes): {}", text.len());
        println!("Encoded length (tokens): {}", encoded.len());
        assert!(encoded.len() < text.len(), "BPE should compress the text");
    }
}
