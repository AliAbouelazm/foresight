const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function getHealth() {
  return request("/health");
}

export function getRandomTrajectory() {
  return request("/trajectories/random");
}

export function getSampleTrajectory(id) {
  return request(`/trajectories/samples/${id}`);
}

export function predict(inputSequence, modelName = "transformer", mcPasses = 10) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({
      input_sequence: inputSequence,
      model_name: modelName,
      mc_passes: mcPasses,
    }),
  });
}

export function getTrainingHistory(modelName = "cnn") {
  return request(`/model/training-history?model_name=${modelName}`);
}

export function getMetrics(modelName = "transformer") {
  return request(`/model/metrics?model_name=${modelName}`);
}
