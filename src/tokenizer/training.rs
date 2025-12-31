use std::collections::HashMap;

pub fn byte_pair_encode(
    tokens: &[usize], 
    vocab_size: usize
) -> HashMap<[usize; 2], usize> {
    assert!(vocab_size >= 256, "Vocabulary must at least accommodate the 256 UTF-8 bytes");

    let num_merges = vocab_size - 256;
    let mut curr: Vec<usize> = tokens.to_vec();
    let mut vocab: HashMap<[usize; 2], usize> = HashMap::new();
    let mut next_id = 256;

    for _ in 0..num_merges {
        let Some((pair, _count)) = get_max_pair(&curr) else { break };
        let new_token = next_id;
        next_id += 1;
        vocab.insert(pair, new_token);

        let mut new_curr = Vec::with_capacity(curr.len());
        let mut i = 0;

        while i < curr.len() {
            if i + 1 < curr.len() && [curr[i], curr[i+1]] == pair {
                new_curr.push(new_token);
                i += 2;
            } else {
                new_curr.push(curr[i]);
                i += 1;
            }
        }

        curr = new_curr;
    }

    vocab
}

pub fn get_max_pair(tokens: &[usize]) -> Option<([usize; 2], usize)> {
    let mut pairs: HashMap<[usize; 2], usize> = HashMap::new();

    for i in 1..tokens.len() {
        let key = [tokens[i - 1], tokens[i]];
        *pairs.entry(key).or_insert(0) += 1;
    }

    pairs.into_iter().max_by(|a, b| {
        let count_a = a.1;
        let count_b = b.1;
        
        match count_a.cmp(&count_b) {
            std::cmp::Ordering::Equal => {
                b.0.cmp(&a.0)
            }
            other => other,
        }
    })
}
pub fn to_ids(s: &str) -> Vec<usize> {
    s.as_bytes().iter().map(|&b| b as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pair_ascii() {
        let tokens = to_ids("banana");
        let result = get_max_pair(&tokens);
        assert_eq!(result, Some(([b'a' as usize, b'n' as usize], 2)));
    }

    #[test]
    fn max_pair_utf8() {
        let tokens = to_ids("éé"); 
        let result = get_max_pair(&tokens);
        assert_eq!(result, Some(([0xC3, 0xA9], 2)));
    }

    #[test]
    fn empty_input_has_no_max() {
        let tokens: Vec<usize> = vec![];
        let result = get_max_pair(&tokens);
        assert_eq!(result, None);
    }

    #[test]
    fn single_byte_has_no_max() {
        let tokens: Vec<usize> = vec![42];
        let result = get_max_pair(&tokens);
        assert_eq!(result, None);
    }

    #[test]
    fn deterministic_choice_when_unique_max() {
        let tokens = to_ids("abab");
        let result = get_max_pair(&tokens);
        assert_eq!(result, Some(([b'a' as usize, b'b' as usize], 2)));
    }

    #[test]
    fn encode_no_merges_if_vocab_full() {
        let tokens = to_ids("abab"); 
        let vocab = byte_pair_encode(&tokens, 256);
        
        assert!(vocab.is_empty());
    }

    #[test]
    fn encode_single_merge_replaces_all_instances() {
        let tokens = to_ids("abab"); 
        let vocab = byte_pair_encode(&tokens, 257);
        
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.get(&[b'a' as usize, b'b' as usize]), Some(&256));
    }

    #[test]
    fn encode_recursive_merge() {
        let tokens = to_ids("aaaa");
        let vocab = byte_pair_encode(&tokens, 258);

        assert_eq!(vocab.len(), 2);
        assert_eq!(vocab.get(&[256, 256]), Some(&257));
    }

    #[test]
    fn encode_stops_when_no_pairs_exist() {
        let tokens = to_ids("a"); 
        let vocab = byte_pair_encode(&tokens, 300);

        assert!(vocab.is_empty());
    }

    #[test]
    fn encode_utf8_multibyte_merge() {
        let input = "😊";
        let tokens = to_ids(input);
        
        let vocab = byte_pair_encode(&tokens, 256 + 3);

        assert_eq!(vocab.len(), 3);
    }
    
    #[test]
    #[should_panic(expected = "Vocabulary must at least accommodate")]
    fn panic_on_small_vocab_size() {
        let tokens = to_ids("abc");
        byte_pair_encode(&tokens, 255);
    }
}
