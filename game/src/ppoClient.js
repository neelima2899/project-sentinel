const API_URL = "http://127.0.0.1:8000/predict";

export async function predictAdversaryAction(observation, signal) {
  if (
    !Array.isArray(observation) ||
    observation.length !== 15 ||
    !observation.every(Number.isFinite)
  ) {
    throw new Error("Invalid PPO observation");
  }

  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ observation }),
    signal
  });

  if (!response.ok) {
    throw new Error(`PPO API error: ${response.status}`);
  }

  const result = await response.json();

  if (
    !Number.isInteger(result.action) ||
    result.action < 0 ||
    result.action > 4
  ) {
    throw new Error("PPO returned an invalid action");
  }

  return result;
}
