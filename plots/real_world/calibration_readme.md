# Calibration Analysis

The `calibration_curve.png` shows the reliability diagrams for the traditional ML baselines. A Brier score closer to 0 indicates better calibration.

## Why isn't the LLM on this graph?
The LLM generates discrete text classifications (`[AKI_STAGE_1+]` or `[NORMAL]`) rather than continuous probabilities. Because the API we use for evaluation does not return log-probabilities for the generated tokens, we cannot mathematically construct a continuous calibration curve or compute a Brier score for the LLM without modifying the evaluation server to expose logprobs.
