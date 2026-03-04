mod cosine;
mod knn;
mod metrics;

use std::collections::HashMap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Compute cosine similarity matrix (Q×D queries vs N×D corpus → Q×N).
#[pyfunction]
fn cosine_similarity_matrix<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, f32>,
    corpus: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let q_shape = queries.shape();
    let c_shape = corpus.shape();
    let (q, d_q) = (q_shape[0], q_shape[1]);
    let (n, d_c) = (c_shape[0], c_shape[1]);

    if d_q != d_c {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: queries have dim {}, corpus has dim {}",
            d_q, d_c
        )));
    }

    let queries_slice = queries.as_slice()?;
    let corpus_slice = corpus.as_slice()?;

    let result = cosine::cosine_similarity_matrix(queries_slice, corpus_slice, q, n, d_q);
    let arr = Array2::from_shape_vec((q, n), result).unwrap();
    Ok(arr.into_pyarray(py))
}

/// Compute N×N pairwise cosine distance matrix.
#[pyfunction]
fn pairwise_distances<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = embeddings.shape();
    let (n, d) = (shape[0], shape[1]);

    let emb_slice = embeddings.as_slice()?;
    let result = cosine::pairwise_distances(emb_slice, n, d);
    let arr = Array2::from_shape_vec((n, n), result).unwrap();
    Ok(arr.into_pyarray(py))
}

/// Brute-force batch kNN. Returns (Q×K indices, Q×K similarities).
#[pyfunction]
fn batch_knn<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<'py, f32>,
    corpus: PyReadonlyArray2<'py, f32>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray2<usize>>, Bound<'py, PyArray2<f32>>)> {
    let q_shape = queries.shape();
    let c_shape = corpus.shape();
    let (q, d_q) = (q_shape[0], q_shape[1]);
    let (n, d_c) = (c_shape[0], c_shape[1]);

    if d_q != d_c {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: queries have dim {}, corpus has dim {}",
            d_q, d_c
        )));
    }

    let queries_slice = queries.as_slice()?;
    let corpus_slice = corpus.as_slice()?;

    let (indices, sims) = knn::batch_knn(queries_slice, corpus_slice, q, n, d_q, k);
    let actual_k = k.min(n);

    let idx_arr = Array2::from_shape_vec((q, actual_k), indices).unwrap();
    let sim_arr = Array2::from_shape_vec((q, actual_k), sims).unwrap();

    Ok((idx_arr.into_pyarray(py), sim_arr.into_pyarray(py)))
}

/// Compute retrieval metrics (Recall@K, MRR, NDCG@K).
#[pyfunction]
fn batch_evaluate(
    retrieved: Vec<Vec<usize>>,
    relevant: Vec<Vec<usize>>,
    ks: Vec<usize>,
) -> PyResult<HashMap<String, f64>> {
    Ok(metrics::batch_evaluate(&retrieved, &relevant, &ks))
}

/// Python module registration.
#[pymodule]
fn emb_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distances, m)?)?;
    m.add_function(wrap_pyfunction!(batch_knn, m)?)?;
    m.add_function(wrap_pyfunction!(batch_evaluate, m)?)?;
    Ok(())
}
