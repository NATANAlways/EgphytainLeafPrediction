import { useEffect, useRef, useState } from "react";
import { explain, fetchModels, predict } from "./api";
import ProbabilityChart from "./ProbabilityChart";
import "./App.css";

export default function App() {
  const [models, setModels] = useState([]);
  const [modelsError, setModelsError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [explainOpen, setExplainOpen] = useState(false);
  const [explainData, setExplainData] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState(null);

  const fileInputRef = useRef(null);

  function loadModels() {
    setModelsError(null);
    fetchModels()
      .then((list) => {
        setModels(list);
        const firstAvailable = list.find((m) => m.available);
        setSelectedModel(firstAvailable ? firstAvailable.key : list[0]?.key ?? null);
      })
      .catch((err) => setModelsError(err.message));
  }

  useEffect(loadModels, []);

  function resetExplain() {
    setExplainOpen(false);
    setExplainData(null);
    setExplainError(null);
  }

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreviewUrl(URL.createObjectURL(f));
    resetExplain();
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!file || !selectedModel) return;
    setLoading(true);
    setError(null);
    setResult(null);
    resetExplain();
    try {
      const data = await predict(selectedModel, file);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleToggleExplain() {
    const next = !explainOpen;
    setExplainOpen(next);
    if (next && !explainData && !explainLoading) {
      setExplainLoading(true);
      setExplainError(null);
      try {
        const data = await explain(selectedModel, file);
        setExplainData(data);
      } catch (err) {
        setExplainError(err.message);
      } finally {
        setExplainLoading(false);
      }
    }
  }

  const selectedModelInfo = models.find((m) => m.key === selectedModel);

  return (
    <div className="page">
      <div className="card">
        <h1>Leaf Species Classifier</h1>
        <p className="subtitle">Upload a leaf photo and pick a model to identify the species.</p>

        {modelsError && (
          <div className="banner banner--error">
            {modelsError} — start it with:
            <code>uvicorn webapp.backend.app.main:app --reload --port 8000</code>
            <button type="button" className="retry-btn" onClick={loadModels}>
              Retry
            </button>
          </div>
        )}

        <form onSubmit={handleSubmit} className="form">
          <fieldset className="field">
            <legend>Model</legend>
            <div className="model-list">
              {models.map((m) => (
                <label
                  key={m.key}
                  className={`model-option${!m.available ? " model-option--disabled" : ""}`}
                >
                  <input
                    type="radio"
                    name="model"
                    value={m.key}
                    checked={selectedModel === m.key}
                    disabled={!m.available}
                    onChange={() => setSelectedModel(m.key)}
                  />
                  <span>{m.label}</span>
                  {!m.available && <span className="badge">not trained yet</span>}
                </label>
              ))}
            </div>
          </fieldset>

          <div className="field">
            <label htmlFor="file-input" className="dropzone" onClick={() => fileInputRef.current?.click()}>
              {previewUrl ? (
                <img src={previewUrl} alt="Selected leaf preview" className="preview" />
              ) : (
                <span className="dropzone-hint">Click to choose a leaf image (.jpg/.png)</span>
              )}
            </label>
            <input
              id="file-input"
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />
          </div>

          <button
            type="submit"
            className="submit-btn"
            disabled={!file || !selectedModel || !selectedModelInfo?.available || loading}
          >
            {loading ? "Predicting…" : "Predict"}
          </button>
        </form>

        {error && <div className="banner banner--error">{error}</div>}

        {result && (
          <div className="result">
            <div className="result-headline">
              <span className="result-class">{result.predicted_class}</span>
              <span className="result-confidence">{(result.confidence * 100).toFixed(1)}% confidence</span>
            </div>
            <ProbabilityChart probabilities={result.probabilities} predictedClass={result.predicted_class} />

            <button type="button" className="explain-btn" onClick={handleToggleExplain}>
              {explainOpen ? "Hide AI Validation" : "AI Validation"}
            </button>
            <p className="explain-hint">
              Shows which pixels this model actually used for the prediction — useful for
              spotting when it's relying on background instead of the leaf itself.
            </p>

            {explainOpen && (
              <div className="explain-panel">
                {explainLoading && <p className="explain-status">Computing Grad-CAM…</p>}
                {explainError && <div className="banner banner--error">{explainError}</div>}
                {explainData && (
                  <div className="explain-images">
                    <figure>
                      <img src={explainData.gradcam} alt="Grad-CAM heatmap" />
                      <figcaption>Grad-CAM — where the model looked</figcaption>
                    </figure>
                    {explainData.cbam_attention && (
                      <figure>
                        <img src={explainData.cbam_attention} alt="CBAM spatial attention" />
                        <figcaption>CBAM attention (learned, not gradient-based)</figcaption>
                      </figure>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
