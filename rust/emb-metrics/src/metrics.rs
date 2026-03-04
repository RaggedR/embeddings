use std::collections::HashMap;

/// Compute retrieval metrics over all queries.
///
/// - retrieved: Vec of retrieved index lists per query (ordered by rank)
/// - relevant: Vec of relevant index sets per query (ground truth)
/// - ks: K values for Recall@K and NDCG@K
///
/// Returns metrics: recall@K, mrr, ndcg@K for each K
pub fn batch_evaluate(
    retrieved: &[Vec<usize>],
    relevant: &[Vec<usize>],
    ks: &[usize],
) -> HashMap<String, f64> {
    let num_queries = retrieved.len();
    assert_eq!(num_queries, relevant.len());

    if num_queries == 0 {
        let mut m = HashMap::new();
        for &k in ks {
            m.insert(format!("recall@{}", k), 0.0);
            m.insert(format!("ndcg@{}", k), 0.0);
        }
        m.insert("mrr".to_string(), 0.0);
        return m;
    }

    let mut total_mrr = 0.0f64;
    let mut recall_sums: HashMap<usize, f64> = ks.iter().map(|&k| (k, 0.0)).collect();
    let mut ndcg_sums: HashMap<usize, f64> = ks.iter().map(|&k| (k, 0.0)).collect();

    for (ret, rel) in retrieved.iter().zip(relevant.iter()) {
        let rel_set: std::collections::HashSet<usize> = rel.iter().cloned().collect();

        if rel_set.is_empty() {
            continue;
        }

        // MRR: reciprocal rank of first relevant result
        for (rank, &idx) in ret.iter().enumerate() {
            if rel_set.contains(&idx) {
                total_mrr += 1.0 / (rank as f64 + 1.0);
                break;
            }
        }

        for &k in ks {
            let top_k = &ret[..k.min(ret.len())];

            // Recall@K
            let hits: usize = top_k.iter().filter(|idx| rel_set.contains(idx)).count();
            let recall = hits as f64 / rel_set.len() as f64;
            *recall_sums.get_mut(&k).unwrap() += recall;

            // NDCG@K
            let dcg = compute_dcg(top_k, &rel_set, k);
            let ideal_dcg = compute_ideal_dcg(rel_set.len(), k);
            if ideal_dcg > 0.0 {
                *ndcg_sums.get_mut(&k).unwrap() += dcg / ideal_dcg;
            }
        }
    }

    let mut metrics = HashMap::new();
    let n = num_queries as f64;

    metrics.insert("mrr".to_string(), total_mrr / n);

    for &k in ks {
        metrics.insert(format!("recall@{}", k), recall_sums[&k] / n);
        metrics.insert(format!("ndcg@{}", k), ndcg_sums[&k] / n);
    }

    metrics
}

/// DCG = sum of 1/log2(rank+1) for each relevant result in the top-k positions.
fn compute_dcg(
    retrieved: &[usize],
    relevant: &std::collections::HashSet<usize>,
    k: usize,
) -> f64 {
    let mut dcg = 0.0f64;
    for (i, &idx) in retrieved.iter().take(k).enumerate() {
        if relevant.contains(&idx) {
            dcg += 1.0 / (i as f64 + 2.0).log2();
        }
    }
    dcg
}

/// Ideal DCG: all relevant docs in top positions.
fn compute_ideal_dcg(num_relevant: usize, k: usize) -> f64 {
    let mut idcg = 0.0f64;
    for i in 0..k.min(num_relevant) {
        idcg += 1.0 / (i as f64 + 2.0).log2();
    }
    idcg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_recall() {
        // All relevant docs are retrieved in top positions
        let retrieved = vec![vec![0, 1, 2, 3, 4]];
        let relevant = vec![vec![0, 1, 2]];
        let ks = vec![5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["recall@5"] - 1.0).abs() < 1e-10);
        assert!((metrics["mrr"] - 1.0).abs() < 1e-10); // First result is relevant
    }

    #[test]
    fn test_zero_recall() {
        // No relevant docs in retrieved
        let retrieved = vec![vec![10, 11, 12, 13, 14]];
        let relevant = vec![vec![0, 1, 2]];
        let ks = vec![5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["recall@5"]).abs() < 1e-10);
        assert!((metrics["mrr"]).abs() < 1e-10);
    }

    #[test]
    fn test_partial_recall() {
        // 2 out of 4 relevant docs retrieved in top-5
        let retrieved = vec![vec![10, 0, 11, 1, 12]];
        let relevant = vec![vec![0, 1, 2, 3]];
        let ks = vec![5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["recall@5"] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_second_position() {
        // First relevant doc at position 2 (0-indexed: rank 1)
        let retrieved = vec![vec![10, 0, 1, 2, 3]];
        let relevant = vec![vec![0]];
        let ks = vec![5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["mrr"] - 0.5).abs() < 1e-10); // 1/(1+1)
    }

    #[test]
    fn test_multiple_queries_averaged() {
        // Query 0: perfect recall, Query 1: zero recall
        let retrieved = vec![vec![0, 1, 2], vec![10, 11, 12]];
        let relevant = vec![vec![0, 1], vec![0, 1]];
        let ks = vec![3];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["recall@3"] - 0.5).abs() < 1e-10); // (1.0 + 0.0) / 2
    }

    #[test]
    fn test_ndcg_perfect() {
        // All relevant in top positions → NDCG = 1.0
        let retrieved = vec![vec![0, 1, 2, 3, 4]];
        let relevant = vec![vec![0, 1, 2]];
        let ks = vec![5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["ndcg@5"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_k_values() {
        let retrieved = vec![vec![10, 0, 1, 2, 3]];
        let relevant = vec![vec![0, 1, 2, 3]];
        let ks = vec![1, 3, 5];

        let metrics = batch_evaluate(&retrieved, &relevant, &ks);
        assert!((metrics["recall@1"]).abs() < 1e-10); // idx 10 not relevant
        assert!((metrics["recall@3"] - 0.5).abs() < 1e-10); // 2 out of 4 (indices 0, 1)
        assert!((metrics["recall@5"] - 1.0).abs() < 1e-10); // all 4 found
    }

    #[test]
    fn test_empty_queries() {
        let metrics = batch_evaluate(&[], &[], &[5, 10]);
        assert!((metrics["recall@5"]).abs() < 1e-10);
        assert!((metrics["mrr"]).abs() < 1e-10);
    }
}
