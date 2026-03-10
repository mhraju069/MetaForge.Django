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

/// Calculate cosine similarity between two vectors.
/// Much more efficient in Rust than Python.
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        norm_a += v1[i] * v1[i];
        norm_b += v2[i] * v2[i];
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

/// A Professional-grade Vector search in Rust.
/// Finds the most semantically relevant posts based on vector similarity.
fn find_best_semantic_context(query_vector: Vec<f32>, posts: Vec<Post>, limit: usize) -> String {
    let mut scored_posts: Vec<(f32, String)> = posts.into_iter()
        .filter_map(|p| {
            if let Some(v) = p.vector {
                let score = cosine_similarity(&query_vector, &v);
                Some((score, p.caption))
            } else {
                None
            }
        }).collect();

    // Sort by similarity score descending
    scored_posts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build context from top semantically similar posts
    let mut context = String::new();
    for (score, caption) in scored_posts.iter().take(limit) {
        if *score > 0.4 { // Threshold to avoid irrelevant garbage
            context.push_str(&format!("- Related Post/Product: {}\n", caption));
        }
    }
    context
}

#[pyfunction]
fn generate_reply(api_key: String, user_message_vector_json: String, posts: String) -> PyResult<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to build client: {}", e)))?;

    // 1. Parse Input Data (Professional Vectorized RAG)
    let user_msg_json: Value = serde_json::from_str(&user_message_vector_json).unwrap_or(json!({}));
    
    let query_vector: Vec<f32> = user_msg_json["vector"]
        .as_array()
        .map(|v| v.iter().filter_map(|val| val.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_else(|| vec![]);

    let user_message = user_msg_json["text"].as_str().unwrap_or("");

    let posts_data: Vec<Post> = serde_json::from_str(&posts)
        .unwrap_or_else(|_| vec![]);

    // 2. High-Performance Semantic Search
    let context = find_best_semantic_context(query_vector, posts_data, 5);

    // 3. Construct Intelligent Prompt
    let system_prompt = if context.is_empty() {
        "You are a professional assistant for MetaForge. Provide helpful responses based on store data.".to_string()
    } else {
        format!(
            "You are a professional MetaForge Shop Assistant. Use this SEMANTICALLY RELEVANT product context: \n{}\n\nUse this to accurately answer the user's question. Be succinct and helpful.",
            context
        )
    };

    let payload = json!({
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    });

    // 4. Call LLM
    let response = client.post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Request failed: {}", e)))?;

    let res_json: serde_json::Value = response.json()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to parse JSON: {}", e)))?;

    let content = res_json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("I'm sorry, I'm having trouble retrieving details from the shop right now.")
        .to_string();

    Ok(content)
}

#[pymodule]
fn rust_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_reply, m)?)?;
    Ok(())
}
