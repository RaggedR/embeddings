use rayon::prelude::*;

/// Compute cosine similarity matrix between Q×D queries and N×D corpus.
/// Both inputs must be L2-normalized rows (flat arrays in row-major order).
/// Returns Q×N similarity matrix (flat, row-major).
pub fn cosine_similarity_matrix(
    queries: &[f32],
    corpus: &[f32],
    q: usize,
    n: usize,
    d: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; q * n];

    result
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(qi, row)| {
            let query = &queries[qi * d..(qi + 1) * d];
            for ni in 0..n {
                let doc = &corpus[ni * d..(ni + 1) * d];
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += query[k] * doc[k];
                }
                row[ni] = dot;
            }
        });

    result
}

/// Compute full N×N pairwise cosine distance matrix (1 - cosine_sim).
/// Input: N×D matrix of L2-normalized rows (flat, row-major).
/// Returns N×N distance matrix (flat, row-major). Diagonal is 0.
pub fn pairwise_distances(embeddings: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n * n];

    // Compute upper triangle in parallel, then fill symmetrically
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let distances: Vec<(usize, usize, f32)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = &embeddings[i * d..(i + 1) * d];
            let b = &embeddings[j * d..(j + 1) * d];
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += a[k] * b[k];
            }
            (i, j, 1.0 - dot)
        })
        .collect();

    for (i, j, dist) in distances {
        result[i * n + j] = dist;
        result[j * n + i] = dist;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2_normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    #[test]
    fn test_cosine_sim_identical() {
        let mut v = vec![1.0f32, 2.0, 3.0];
        l2_normalize(&mut v);
        // Query = corpus = same vector
        let result = cosine_similarity_matrix(&v, &v, 1, 1, 3);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let mut a = vec![1.0f32, 0.0];
        let mut b = vec![0.0f32, 1.0];
        l2_normalize(&mut a);
        l2_normalize(&mut b);
        let result = cosine_similarity_matrix(&a, &b, 1, 1, 2);
        assert!(result[0].abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_matrix_shape() {
        // 2 queries × 3 corpus, dim=4
        let mut queries = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut corpus = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        for chunk in queries.chunks_mut(4) {
            l2_normalize(chunk);
        }
        for chunk in corpus.chunks_mut(4) {
            l2_normalize(chunk);
        }
        let result = cosine_similarity_matrix(&queries, &corpus, 2, 3, 4);
        assert_eq!(result.len(), 6); // 2×3
        // q0 matches corpus[0], q1 matches corpus[1]
        assert!((result[0] - 1.0).abs() < 1e-5); // q0·c0
        assert!(result[1].abs() < 1e-5); // q0·c1
        assert!(result[3].abs() < 1e-5); // q1·c0
        assert!((result[4] - 1.0).abs() < 1e-5); // q1·c1
    }

    #[test]
    fn test_pairwise_distances_diagonal_zero() {
        let mut vecs = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for chunk in vecs.chunks_mut(3) {
            l2_normalize(chunk);
        }
        let result = pairwise_distances(&vecs, 3, 3);
        assert_eq!(result.len(), 9);
        // Diagonal should be 0
        assert!(result[0].abs() < 1e-5);
        assert!(result[4].abs() < 1e-5);
        assert!(result[8].abs() < 1e-5);
    }

    #[test]
    fn test_pairwise_distances_symmetric() {
        let mut vecs = vec![1.0, 2.0, 0.0, 3.0, 0.5, 1.0, 0.0, 0.5];
        for chunk in vecs.chunks_mut(2) {
            l2_normalize(chunk);
        }
        let result = pairwise_distances(&vecs, 4, 2);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (result[i * 4 + j] - result[j * 4 + i]).abs() < 1e-5,
                    "Not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
