# Security policy

## Supported versions

DenseTopo-UNet is alpha research software. Security fixes are applied to the latest development version only; no long-term support branch currently exists.

| Version | Supported |
| --- | --- |
| latest `main` | yes |
| older commits or local forks | no |

## Reporting a vulnerability

Do not open a public issue for an unpatched vulnerability. Use GitHub's private vulnerability reporting or security-advisory feature for this repository. Include:

- affected version or commit;
- operating system and Python/PyTorch versions;
- minimal reproduction steps;
- expected and observed impact;
- whether untrusted manifests, checkpoints, paths, or raw files are involved;
- any proposed mitigation.

Allow maintainers reasonable time to reproduce and address the issue before public disclosure.

## Trust boundary

Manifests, YAML files, CSV labels, raw fields, and checkpoints should be treated as trusted local research inputs. PyTorch checkpoint deserialization is not a safe sandbox for untrusted files. Run the package with the minimum filesystem permissions required and inspect data provenance before use.

The project does not transmit telemetry, upload data, or manage credentials. Users are responsible for securing training data, output directories, cluster jobs, and upstream compressor or topology tools.
