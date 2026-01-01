use fancy_regex::Regex;
use std::collections::HashMap;

const GPT4_SPLIT_PATTERN: &str = r#"(?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"#;

pub fn byte_pair_encode(
    text: &str, 
    vocab_size: usize
) -> HashMap<[usize; 2], usize> {
    assert!(vocab_size >= 256, "Vocabulary must at least accommodate the 256 UTF-8 bytes");
    let mut chunks = split_with_regex(text);

    let num_merges = vocab_size - 256;
    let mut vocab: HashMap<[usize; 2], usize> = HashMap::new();
    let mut next_id = 256;

    for _ in 0..num_merges {
        let Some((pair, _count)) = get_max_pair_global(&chunks) else { break };
        
        let new_token = next_id;
        next_id += 1;
        vocab.insert(pair, new_token);

        for chunk in chunks.iter_mut() {
            merge_pair_in_chunk(chunk, pair, new_token);
        }
    }

    vocab
}

fn split_with_regex(text: &str) -> Vec<Vec<usize>> {
    let re = Regex::new(GPT4_SPLIT_PATTERN).expect("Invalid Regex Pattern");
    
    re.find_iter(text)
        .map(|m| {
            m.unwrap().as_str().as_bytes().iter().map(|&b| b as usize).collect()
        })
        .collect()
}

pub fn get_max_pair_global(chunks: &[Vec<usize>]) -> Option<([usize; 2], usize)> {
    let mut stats: HashMap<[usize; 2], usize> = HashMap::new();

    for chunk in chunks {
        for i in 0..chunk.len().saturating_sub(1) {
            let pair = [chunk[i], chunk[i+1]];
            *stats.entry(pair).or_insert(0) += 1;
        }
    }

    stats.into_iter().max_by(|a, b| {
        a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0))
    })
}

fn merge_pair_in_chunk(chunk: &mut Vec<usize>, pair: [usize; 2], new_token: usize) {
    let mut new_chunk = Vec::with_capacity(chunk.len());
    let mut i = 0;

    while i < chunk.len() {
        if i + 1 < chunk.len() && chunk[i] == pair[0] && chunk[i+1] == pair[1] {
            new_chunk.push(new_token);
            i += 2; 
        } else {
            new_chunk.push(chunk[i]);
            i += 1;
        }
    }
    *chunk = new_chunk;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_ids(s: &str) -> Vec<usize> {
        s.as_bytes().iter().map(|&b| b as usize).collect()
    }

    #[test]
    fn max_pair_ascii() {
        let tokens = to_ids("banana");
        let chunks = vec![tokens]; 
        let result = get_max_pair_global(&chunks);
        assert_eq!(result, Some(([b'a' as usize, b'n' as usize], 2)));
    }

    #[test]
    fn max_pair_utf8() {
        let tokens = to_ids("éé"); 
        let chunks = vec![tokens];
        let result = get_max_pair_global(&chunks);
        assert_eq!(result, Some(([0xC3, 0xA9], 2)));
    }

    #[test]
    fn encode_basic_merge() {
        let vocab = byte_pair_encode("abab", 257);
        assert_eq!(vocab.len(), 1);
        assert_eq!(vocab.get(&[97, 98]), Some(&256));
    }

    #[test]
    fn regex_prevents_cross_boundary_merge() {
        let text = "A\nA\nA\nA\nA";
        let vocab = byte_pair_encode(text, 257);
        
        assert!(vocab.is_empty(), "Vocab should be empty because \\n forces strict splits");
    }

    #[test]
    fn heavy_frequency_overcomes_tie_breaker() {
        let text = "loop loop loop oo oo"; 
        let vocab = byte_pair_encode(text, 257);
        
        assert!(vocab.contains_key(&[b'o' as usize, b'o' as usize]));
    }
}
