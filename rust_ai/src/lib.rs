use pyo3::prelude::*;
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::time::Duration;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Post {
    post_id: String,
    caption: String,
    vector: Option<Vec<f32>>,
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() { return 0.0; }
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        norm_a += v1[i] * v1[i];
        norm_b += v2[i] * v2[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[pyfunction]
fn search_context(query_vector_json: String, posts_json: String, limit: usize) -> PyResult<String> {
    let query_vec: Vec<f32> = serde_json::from_str(&query_vector_json).unwrap_or(vec![]);
    let posts: Vec<Post> = serde_json::from_str(&posts_json).unwrap_or(vec![]);

    let mut scored_posts: Vec<(f32, String)> = posts.into_iter()
        .filter_map(|p| {
            if let Some(v) = p.vector {
                let score = cosine_similarity(&query_vec, &v);
                Some((score, p.caption))
            } else { None }
        }).collect();

    scored_posts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut context = String::new();
    for (score, caption) in scored_posts.iter().take(limit) {
        if *score > 0.35 {
            context.push_str(&format!("- Product Info: {}\n", caption));
        }
    }
    Ok(context)
}

#[pyfunction]
fn generate_reply(api_key: String, user_msg_json: String, posts: String) -> PyResult<String> {
    let client = Client::builder().timeout(Duration::from_secs(30)).build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    let input: Value = serde_json::from_str(&user_msg_json).unwrap_or(json!({}));
    let query_vector: Vec<f32> = input["vector"].as_array()
        .map(|v| v.iter().filter_map(|val| val.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_else(|| vec![]);
    
    let user_message = input["text"].as_str().unwrap_or("");
    let context = search_context(serde_json::to_string(&query_vector).unwrap(), posts, 5)?;

    let system_prompt = format!(
        "You are a human shop assistant. ONLY use this context:\n{}\nRules: No JSON, no fake prices, say 'I don't know' if not found.",
        context
    );

    let payload = json!({
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    });

    let response = client.post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&payload).send()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    let res: Value = response.json().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;
    let content = res["choices"][0]["message"]["content"].as_str().unwrap_or("Error").to_string();
    Ok(content)
}

#[pymodule]
fn rust_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search_context, m)?)?;
    m.add_function(wrap_pyfunction!(generate_reply, m)?)?;
    Ok(())
}
