use rayon::prelude::*;

/// Brute-force batch kNN search.
/// queries: Q×D flat array, corpus: N×D flat array (both L2-normalized).
/// Returns (Q×K indices, Q×K similarities) sorted by descending similarity per query.
pub fn batch_knn(
    queries: &[f32],
    corpus: &[f32],
    q: usize,
    n: usize,
    d: usize,
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(n);
    let mut all_indices = vec![0usize; q * k];
    let mut all_sims = vec![0.0f32; q * k];

    // Process each query in parallel
    let results: Vec<(Vec<usize>, Vec<f32>)> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let query = &queries[qi * d..(qi + 1) * d];

            // Compute similarities to all corpus vectors
            let mut scored: Vec<(usize, f32)> = (0..n)
                .map(|ni| {
                    let doc = &corpus[ni * d..(ni + 1) * d];
                    let dot: f32 = query.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                    (ni, dot)
                })
                .collect();

            // Partial sort: move top-k to front
            let nth = k.min(scored.len()) - 1;
            scored.select_nth_unstable_by(nth, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Sort the top-k by descending similarity
            scored.truncate(k);
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let indices: Vec<usize> = scored.iter().map(|(i, _)| *i).collect();
            let sims: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
            (indices, sims)
        })
        .collect();

    for (qi, (indices, sims)) in results.into_iter().enumerate() {
        all_indices[qi * k..qi * k + k].copy_from_slice(&indices);
        all_sims[qi * k..qi * k + k].copy_from_slice(&sims);
    }

    (all_indices, all_sims)
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
    fn test_knn_exact_match() {
        // 3 corpus vectors (identity-like in 3D)
        let mut corpus = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for chunk in corpus.chunks_mut(3) {
            l2_normalize(chunk);
        }

        // Query matches corpus[1]
        let mut query = vec![0.0, 1.0, 0.0];
        l2_normalize(&mut query);

        let (indices, sims) = batch_knn(&query, &corpus, 1, 3, 3, 1);
        assert_eq!(indices[0], 1);
        assert!((sims[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_knn_top_k_ordering() {
        // 4 corpus vectors in 2D
        let mut corpus = vec![1.0, 0.0, 0.7, 0.7, 0.0, 1.0, -1.0, 0.0];
        for chunk in corpus.chunks_mut(2) {
            l2_normalize(chunk);
        }

        // Query close to (1, 0)
        let mut query = vec![0.9, 0.1];
        l2_normalize(&mut query);

        let (indices, sims) = batch_knn(&query, &corpus, 1, 4, 2, 3);
        // Most similar should be corpus[0]=(1,0), then corpus[1]=(0.7,0.7)
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        // Similarities should be descending
        assert!(sims[0] >= sims[1]);
        assert!(sims[1] >= sims[2]);
    }

    #[test]
    fn test_knn_multiple_queries() {
        let mut corpus = vec![1.0, 0.0, 0.0, 1.0];
        for chunk in corpus.chunks_mut(2) {
            l2_normalize(chunk);
        }

        let mut queries = vec![1.0, 0.0, 0.0, 1.0];
        for chunk in queries.chunks_mut(2) {
            l2_normalize(chunk);
        }

        let (indices, _) = batch_knn(&queries, &corpus, 2, 2, 2, 1);
        assert_eq!(indices[0], 0); // q0 matches c0
        assert_eq!(indices[1], 1); // q1 matches c1
    }

    #[test]
    fn test_knn_k_larger_than_n() {
        let mut corpus = vec![1.0, 0.0, 0.0, 1.0];
        for chunk in corpus.chunks_mut(2) {
            l2_normalize(chunk);
        }
        let mut query = vec![1.0, 0.0];
        l2_normalize(&mut query);

        // k=5 but only 2 corpus vectors
        let (indices, sims) = batch_knn(&query, &corpus, 1, 2, 2, 5);
        assert_eq!(indices.len(), 2);
        assert_eq!(sims.len(), 2);
    }
}
