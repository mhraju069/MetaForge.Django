use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::Value;

// ── Data structures ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct PostImage {
    url: Option<String>,
    #[serde(default)]
    hash: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Post {
    post_id: String,
    caption: String,
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    images: Vec<PostImage>,
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na  = 0.0_f32;
    let mut nb  = 0.0_f32;
    for i in 0..v1.len() {
        dot += v1[i] * v2[i];
        na  += v1[i] * v1[i];
        nb  += v2[i] * v2[i];
    }
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na.sqrt() * nb.sqrt())
}

// ── search_context ────────────────────────────────────────────────────────────
// Returns ranked product context string including image URLs.
// Called from Python webhook for every text/audio query.

#[pyfunction]
fn search_context(
    query_vector_json: String,
    posts_json: String,
    limit: usize,
) -> PyResult<String> {
    let query_vec: Vec<f32> = serde_json::from_str(&query_vector_json).unwrap_or_default();
    let posts: Vec<Post>    = serde_json::from_str(&posts_json).unwrap_or_default();

    if query_vec.is_empty() {
        return Ok(String::new());
    }

    // Score every post that has a vector
    let mut scored: Vec<(f32, &Post)> = posts.iter()
        .filter_map(|p| {
            p.vector.as_ref().map(|v| (cosine_similarity(&query_vec, v), p))
        })
        .collect();

    // Sort descending by similarity
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut context = String::new();
    let mut count   = 0;

    for (score, post) in scored.iter().take(limit * 2) {   // over-fetch, filter below
        if *score < 0.45 { break; }   // below threshold — stop (already sorted)

        // Collect valid image URLs for this post
        let img_urls: Vec<&str> = post.images.iter()
            .filter_map(|i| i.url.as_deref())
            .filter(|u| !u.is_empty())
            .collect();

        // Build context line
        context.push_str(&format!(
            "---\nProduct (Match Confidence: {:.2}):\n{}\n",
            score, post.caption
        ));
        if !img_urls.is_empty() {
            context.push_str(&format!("Image URLs: {}\n", img_urls.join(", ")));
        }

        count += 1;
        if count >= limit { break; }
    }

    Ok(context)
}

// ── get_best_match ────────────────────────────────────────────────────────────
// Returns the single best matching post as JSON (score + caption + images).
// Useful for direct product lookups.

#[pyfunction]
fn get_best_match(query_vector_json: String, posts_json: String) -> PyResult<String> {
    let query_vec: Vec<f32> = serde_json::from_str(&query_vector_json).unwrap_or_default();
    let posts: Vec<Post>    = serde_json::from_str(&posts_json).unwrap_or_default();

    if query_vec.is_empty() || posts.is_empty() {
        return Ok(String::new());
    }

    let best = posts.iter()
        .filter_map(|p| {
            p.vector.as_ref().map(|v| (cosine_similarity(&query_vec, v), p))
        })
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((score, post)) = best {
        let img_urls: Vec<Value> = post.images.iter()
            .filter_map(|i| i.url.as_ref())
            .map(|u| Value::String(u.clone()))
            .collect();

        let result = serde_json::json!({
            "score":   score,
            "caption": post.caption,
            "post_id": post.post_id,
            "images":  img_urls,
        });
        Ok(result.to_string())
    } else {
        Ok(String::new())
    }
}

#[pyfunction]
fn fetch_meta_data(url: String, access_token: String) -> PyResult<String> {
    let client = reqwest::blocking::Client::new();
    let resp = client.get(url)
        .query(&[("access_token", access_token), ("limit", "50".to_string())])
        .send();

    match resp {
        Ok(res) => {
            if res.status().is_success() {
                let body = res.text().unwrap_or_default();
                Ok(body)
            } else {
                let err_msg = res.text().unwrap_or_default();
                Ok(format!(r#"{{"error": "API status {}", "details": {}}}"#, res.status(), err_msg))
            }
        }
        Err(e) => Ok(format!(r#"{{"error": "Request failed: {}"}}"#, e)),
    }
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn rust_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_context, m)?)?;
    m.add_function(wrap_pyfunction!(get_best_match, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_meta_data, m)?)?;
    Ok(())
}
