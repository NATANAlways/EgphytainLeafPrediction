const API_BASE = "http://localhost:8000";

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Could not reach the backend — is it running on :8000?");
  return res.json();
}

export async function predict(modelKey, file) {
  const form = new FormData();
  form.append("model_key", modelKey);
  form.append("file", file);

  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Prediction failed.");
  return data;
}

export async function explain(modelKey, file) {
  const form = new FormData();
  form.append("model_key", modelKey);
  form.append("file", file);

  const res = await fetch(`${API_BASE}/explain`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "AI validation failed.");
  return data;
}
