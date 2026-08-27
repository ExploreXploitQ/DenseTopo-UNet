# Contributing

DenseTopo-UNet welcomes focused improvements to correctness, reproducibility, documentation, and scientific evaluation. The project is alpha research software, so changes must preserve explicit information and evidence boundaries.

## Development setup

```bash
python -m pip install -e '.[dev]'
make check
```

Use Python 3.10 or newer. Tests must run on CPU unless explicitly marked `gpu`. Do not commit local raw volumes, checkpoints, credentials, environment directories, or generated run outputs.

## Change process

1. Open an issue for large behavior, schema, or scientific-protocol changes.
2. Add or update a test that states the intended observable behavior.
3. Implement the smallest coherent change.
4. Update public documentation and example configuration when an interface changes.
5. Run `make check` and the synthetic smoke workflow when relevant.
6. Describe limitations, failed checks, and evidence status honestly in the pull request.

## Code requirements

- Preserve the one-decompressed-channel inference boundary.
- Reject malformed data instead of silently guessing metadata.
- Keep file writes atomic where interruption would corrupt a run.
- Keep public names, comments, messages, documentation, and figure labels in English.
- Add type annotations to public and internal functions.
- Avoid dataset-specific paths, variable names, and undocumented defaults.
- Maintain compatibility checks when checkpoint semantics change.

## Scientific claims

A new performance statement must identify data provenance, splits, upstream compressor and version, error bound, topology extractor and version, persistence threshold, matching rule, seeds, baselines, and per-sample metrics. Synthetic proxies and software tests cannot support compressor or restoration-quality claims.

Do not add benchmark tables without the underlying machine-readable results and a reproducible command. Report negative results and samples where FC counts increase.

## Documentation and commits

Write commit messages in the imperative mood and keep unrelated changes separate. Markdown links must resolve locally. Native SVG figures need `<title>` and `<desc>` metadata.

## Conduct and legal status

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Contributors must have the right to submit their work. The repository currently contains no license grant; accepting a contribution does not itself create a general license for the repository.
