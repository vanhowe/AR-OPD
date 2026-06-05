Code evaluation datasets bundled directly in the repo for offline or deterministic evaluation.

Included files:
- `humanevalplus_eval.jsonl`
- `mbppplus_eval.jsonl`

Sources:
- `humanevalplus_eval.jsonl` is mirrored from `evalplus/humanevalplus:test.jsonl`
- `mbppplus_eval.jsonl` is converted from `evalplus/mbppplus:data/test-00000-of-00001-d5781c9c51e02795.parquet`

`eval_code.py` now checks these local files first and only falls back to `evalplus` package downloads if they are missing.
