# Changelog

## 0.2.0

- Release the engineering examples and experiment analytics developed since 0.1.5.
- Use benchfun 0.2.0 for benchmark definitions, including its shared ZDT1 objective.
  Corrected objective formulas mean historical benchmark results are not directly
  comparable; record both crate versions and the RNG seed in reported experiments.
- Replace async batch barriers with a bounded rolling queue, preserving candidate
  order while avoiding idle slots. Cancel and drain remaining tasks after a failure.
- Update the cantilever example to trussx 0.3, explicitly use the specified tip load,
  and validate candidate areas before structural analysis.
- Test both structural response and async refill/cancellation behavior.
- Update CI tooling and lint/test all targets and features.

benchfun 0.2.0 must be published before jeans 0.2.0 can be verified and published
against crates.io. For coordinated local validation, use Cargo's `--config
'patch.crates-io.benchfun.path="/path/to/benchfun"'` override; no filesystem path is
embedded in the published manifest.
