# Community Leaderboard

Results submitted by the community via pull request.

## How to submit

1. Run the stress test against your model
2. Note your average score and worst-performing position
3. Open a PR adding a row to the table below
4. Include your `heatmap_*.png` in `results/community/` if you have one

```bash
# Run and get your stats
llm-stress-test run --model YOUR_MODEL --trials 3
# Check results/run_*.json for avg_score
```

---

## Results table

| Model | Provider | Max Tested | Avg Score | Worst Position | Worst Context | Trials | Date | Submitted by |
|-------|----------|-----------|-----------|---------------|---------------|--------|------|-------------|
| *Your model here* | | | | | | | | |

---

## Score interpretation

- **Avg Score** — mean retrieval score across all (context length × depth) cells tested
- **Worst Position** — the needle depth (0–100%) where the model performed worst
- **Worst Context** — the context length (in k tokens) where performance degraded most

---

## Notes

- Scores are not comparable across needle implementations — this repo uses "FLAMINGO-7429"
- Local (Ollama) results depend heavily on hardware and quantization level
- Higher trial counts → more reliable averages (we recommend ≥ 3)
