# Cleanup validation

Validated locally on 2026-09-25 with Python 3.12.13 on macOS arm64.

- 15 offline regression/integration tests passed (`python -m unittest discover -s tests -v`).
- All Python source files compiled successfully.
- All shell scripts passed `bash -n`.
- `git diff --check` passed.
- All three registered task loaders successfully read the bundled datasets.
- Dataset hashes matched `datasets/manifest.json`.
- Common credential-pattern scans of the original Git history and release files found no matches. The removed `src/key.env` file was empty. This is a limited pattern scan, not proof that every historical artifact is free of sensitive information.

Tests cover configuration merging, environment credentials, redacted logging, dependency-free CLI help, local client initialization, the MGSM runner/evaluator interface and result writes, credential-free source snapshots, dataset integrity, and serving-script refusal/default behavior. Model calls and GPU commands are replaced or avoided in these checks.

The installed client versions included OpenAI 2.54.0, NumPy 2.5.3, pandas 2.3.3, PyYAML 6.0.3, and tqdm 4.70.1. These describe this cleanup check, not the paper's original environment.

GPU serving, model downloads, live evolution, and GitHub Actions have not been run. The new CI workflow is configured to run CPU checks on Python 3.10 and 3.12 after publication. See [reproducibility notes](reproducibility.md) for substantive release gaps.
