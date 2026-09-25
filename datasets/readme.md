# Benchmark data

The files below were already present in the original repository. The retained snapshots are unchanged. `manifest.json` records SHA-256 hashes and file sizes so future runs can identify these exact snapshots. Original download revisions and preprocessing scripts were not included.

| Benchmark | Local files | Source |
| --- | --- | --- |
| MMLU | `mmlu.csv` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |
| MGSM | `mgsm/mgsm_*.tsv` (11 languages) | [juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm) |
| IFEval | `ifeval/ifeval_input_data_{train,test}.jsonl` | [Google Research IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) |
| GPQA | `gpqa_diamond.csv` (legacy, not registered) | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) |

Dataset contents retain upstream terms; the repository's MIT license does not relicense benchmark data. Consult the upstream source for access conditions and attribution. The links identify benchmark sources, not verified provenance for every local split or transformation.

See [reproducibility notes](../docs/reproducibility.md) for the actual sampling rules and IFEval scoring limitations.
