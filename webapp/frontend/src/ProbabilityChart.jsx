// Fixed categorical color per leaf class — same order/hues used in the training
// notebooks' ROC/PR curves, so a class reads as the same color everywhere.
const CLASS_COLORS = {
  Apple:     { light: "#2a78d6", dark: "#3987e5" },
  Berry:     { light: "#008300", dark: "#008300" },
  Fig:       { light: "#e87ba4", dark: "#d55181" },
  Guava:     { light: "#eda100", dark: "#c98500" },
  Orange:    { light: "#1baf7a", dark: "#199e70" },
  Palm:      { light: "#eb6834", dark: "#d95926" },
  Persimmon: { light: "#4a3aa7", dark: "#9085e9" },
  Tomato:    { light: "#e34948", dark: "#e66767" },
};

export default function ProbabilityChart({ probabilities, predictedClass }) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="prob-chart" role="table" aria-label="Per-class prediction probabilities">
      {entries.map(([cls, prob]) => {
        const colors = CLASS_COLORS[cls] ?? { light: "#898781", dark: "#898781" };
        const isPredicted = cls === predictedClass;
        return (
          <div className="prob-row" key={cls} role="row">
            <span className={`prob-label${isPredicted ? " prob-label--predicted" : ""}`} role="cell">
              {cls}
            </span>
            <div className="prob-track" role="cell">
              <div
                className="prob-fill"
                style={{
                  width: `${Math.max(prob * 100, 1.5)}%`,
                  "--fill-light": colors.light,
                  "--fill-dark": colors.dark,
                }}
              />
            </div>
            <span className="prob-value" role="cell">{(prob * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
