## What changed

<!-- One or two sentences. Which module owns this change? -->

## Definition of done

- [ ] `ruff check .` passes
- [ ] `python scripts/check_quality.py` passes
- [ ] `python -m pytest` passes (no GPU needed)
- [ ] No new code in `server/main.py` (it is at its cap — extract a module)
- [ ] New transport method added to **both** `server/gpu.py` and `tests/fakes.py`
- [ ] `server/jarvis/` untouched (it is frozen by checksum)
- [ ] If the HTTP contract changed, `tests/test_contracts.py` was updated
- [ ] If a size limit was raised in `quality/budgets.json`, say why below

## Why this is the right module

<!-- If you added a file: which layer is it in, and what may it import? -->

## Cost impact

- [ ] No GPU work required for this change
- [ ] If a render was run: instance destroyed (`jl list` is empty)
