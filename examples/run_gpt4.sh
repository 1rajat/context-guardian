#!/usr/bin/env bash
# Example: run the full stress test against gpt-4o up to 32k context
# Requires: OPENAI_API_KEY exported in your environment

set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:?Please set OPENAI_API_KEY}"

echo "=== Cost estimate (no actual API calls) ==="
llm-stress-test run --model gpt-4o --max-context 32k --trials 3 --cost-estimate

echo ""
echo "=== Running quick mode first (sanity check) ==="
llm-stress-test run --model gpt-4o --quick --trials 1 --output results/

echo ""
echo "=== Full run up to 32k ==="
llm-stress-test run \
  --model gpt-4o \
  --max-context 32k \
  --trials 3 \
  --output results/

echo ""
echo "Done! Check results/ for JSON + heatmaps."
