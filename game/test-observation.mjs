import { buildObservation } from "./src/ppoObservation.js";

const state = {
  p: { x: 0, z: 3 },
  g: { x: -4, z: -3 },
  e: { x: 4, z: -3 },
  player: 100,
  guardian: 120,
  enemy: 140,
  aiSteps: 0,
  nextAttackStep: 1,
  powerStrikeActive: false,
  powerStrikeCountdown: 0,
  powerStrikeCooldown: 0
};

const obs = buildObservation(state);

console.log("Observation count:", obs.length);
console.log("Observations:", obs);

if (obs.length !== 15 || obs.some(v => !Number.isFinite(v))) {
  throw new Error("Invalid PPO observation");
}

const expected = [
  -0.25, 0.375, -0.5, 0,
  1, 1, 1,
  Math.sqrt(52) / 16, 0.5,
  0, 0, 1 / 3,
  0, 0, 0
];

if (obs.some((value, i) => Math.abs(value - expected[i]) > 1e-6)) {
  throw new Error("Observation values do not match expected values");
}

console.log("PASS: All 15 PPO observations match expected values");
