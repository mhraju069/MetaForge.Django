use pyo3::prelude::*;
use reqwest::blocking::Client;
use serde_json::json;
use std::time::Duration;

#[pyfunction]
fn generate_reply(api_key: String, user_message: String) -> PyResult<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to build client: {}", e)))?;

    let payload = json!({
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for MetaForge."},
            {"role": "user", "content": user_message}
        ]
    });

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
        .unwrap_or("No content returned from AI")
        .to_string();

    Ok(content)
}

#[pymodule]
fn rust_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_reply, m)?)?;
    Ok(())
}
