# CICE-JAX

A JAX implementation of the CICE sea-ice model, developed to run CICE physics on GPUs while preserving differentiability for gradient-based applications.

Under the project author's direction, **Codex (GPT-6 Astra)** has assisted with software design, migration of the Fortran code to JAX, test development, and numerical comparison tooling. The original sea-ice model and its physical formulations are the work of the CICE and Icepack contributors.

## Upstream baseline

This independent repository contains the JAX implementation, ported from official **CICE6.6.3**, commit [`c734c12bfb19632cd700cb06b22c3ea62b4adeae`](https://github.com/CICE-Consortium/CICE/commit/c734c12bfb19632cd700cb06b22c3ea62b4adeae). Its Git history starts with the JAX implementation; the upstream commit identifies the source baseline rather than a parent in this repository's history.

The Fortran reference with local diagnostic extensions is maintained in the [`CICE-Fortran-debug`](https://github.com/TwitchZhu/CICE/tree/CICE-Fortran-debug) branch of the separate [CICE fork](https://github.com/TwitchZhu/CICE). Its Icepack baseline is [`daa41638c6cef298583f35ee4dec5ab1bb077aea`](https://github.com/CICE-Consortium/Icepack/commit/daa41638c6cef298583f35ee4dec5ab1bb077aea). The projects are maintained separately; later upstream changes are ported as needed. The fork's former [`CICE-JAX` branch](https://github.com/TwitchZhu/CICE/tree/CICE-JAX) is retained as a migration reference.

## Implementation status

The standalone JAX path includes initialization, thermodynamics, horizontal dynamics, transport, ridging, and NetCDF output. A GPU forward run has completed the five-day gx3 experiment described below, including thermodynamics, standard B-grid EVP dynamics, remapping, and ridging.

GPU execution and differentiable time integration are the project's goals. Completing this forward experiment does not establish end-to-end gradient correctness, validation of every optional physics branch, or long-term stability. These remain ongoing work.

## Five-day gx3 comparison

The local `gx3_5day` experiment follows the official `smoke gx3 8x2 diag1,run5day` configuration: a **100 × 116** grid, five ice-thickness categories, seven ice layers, one snow layer, and **120 time steps of 3,600 seconds** from 2005-01-01 to 2005-01-06. It uses mushy-layer thermodynamics, dEdd shortwave radiation, level-ice melt ponds, standard EVP with 240 subcycles, remap transport, and ridging.

The Fortran runs used GNU Fortran 11.4.0 and Intel Fortran 2021.10.0, each with 8 MPI ranks and 2 OpenMP threads. Their archived source and input checksum manifests match. They used CICE with local diagnostic extensions, rather than an untouched upstream build. JAX ran the full grid on one CUDA device with JAX/jaxlib 0.6.2 and 64-bit arithmetic enabled.

The following results summarize the archived comparison report **2026-11-05 14:27:05** for the final restart, `iced.2005-01-06-00000.nc`, at step 120. They describe that recorded experiment, not an automatic validation of subsequent source changes.

### Main state differences

Relative L2 differences are `100 × ||right − left||₂ / ||left||₂`, evaluated over jointly unmasked finite values without area weighting. In each item below, the three percentages are ordered **GNU–Intel / GNU–JAX / Intel–JAX**:

- **Ice area by thickness category (`aicen`):** 0.008568% / 0.008928% / 0.011502%.
- **Ice volume per unit grid-cell area by category (`vicen`):** 0.007198% / 0.006971% / 0.009797%.
- **Snow volume per unit grid-cell area by category (`vsnon`):** 0.011411% / 0.019872% / 0.022577%.
- **Ice velocity, u component (`uvel`):** 3.181685% / 3.187455% / 0.123387%.
- **Ice velocity, v component (`vvel`):** 2.852447% / 2.887091% / 0.207696%.

For these category area and volume fields, JAX differences are of the same order as the GNU–Intel comparison, although snow-volume differences are larger. For velocity, JAX is closer to Intel: the GNU–JAX differences are close to the GNU–Intel differences. Maximum absolute velocity differences, in the same comparison order, are **0.140721 / 0.140839 / 0.003363 m/s** for `uvel` and **0.117035 / 0.117121 / 0.007354 m/s** for `vvel`.

All five fields have matching missing-value masks and no unmasked NaN or Inf values in the report. The results are not bit-for-bit identical. GNU–Intel differences provide an empirical reference for these builds, not an acceptance threshold or proof that the remaining differences are caused only by roundoff.

## Create and run a JAX case on a server

The case creator bundles the `gx3_5day` preset and generates an independent `ice_in`, `run.py`, and comparison script. See [the server command sequence](docs/SERVER_RUN.md) for environment setup, input data, direct GPU execution, and optional Fortran comparison.

From `cice-dirs/cases/`, with `CICE-JAX/` and `CICE/` beside `cice-dirs/`:

```bash
python ../../CICE-JAX/cice.setup --case gx3_5day_jax --preset gx3_5day --inputdata ../../CICE/input
python gx3_5day_jax/run.py
```

Activate the CUDA Python environment first. The JAX case runs one full grid on one GPU; it does not need a Fortran build or MPI launch. GPU profiling is optional via `run.py --profile`.

## Repository contents

The repository keeps source code, configuration, and validation/comparison utilities:

- `cicecore/`: JAX model state, drivers, dynamics, infrastructure, and diagnostics.
- `icepack/`: JAX column-physics implementation and original Icepack license files.
- `configuration/`: model configuration, bundled case presets, and the generated-case runner template.
- `test/`: validation and comparison utilities.
- `environment.yml`: the development environment, including JAX with CUDA 12 support on Linux.

Local experiment directories (`cases/`) and development notes are excluded from Git. Generated history fields, restart files, logs, profiling records, and large snapshots are also kept outside version control. The `gx3_5day` configuration snapshots and comparison utility are distributed under `configuration/cases/`; input datasets and archived experiment outputs are not included.

## License and attribution

Original CICE license, copyright, and distribution-policy documents are retained at the repository root: [LICENSE.pdf](LICENSE.pdf), [COPYRIGHT.pdf](COPYRIGHT.pdf), and [DistributionPolicy.pdf](DistributionPolicy.pdf). The corresponding Icepack documents are retained under `icepack/`: [LICENSE.pdf](icepack/LICENSE.pdf), [COPYRIGHT.pdf](icepack/COPYRIGHT.pdf), and [DistributionPolicy.pdf](icepack/DistributionPolicy.pdf). Consult those documents for the applicable terms and attribution requirements.
