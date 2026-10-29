# Changelog

All notable changes to H2FactoryTestUtils are recorded here.

This file starts at the `refactor/interfaces` work and is not a retroactive reconstruction of the
package's earlier history. The `Unreleased` section below covers all 21 commits on that branch;
anything before it is not listed.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package has NEVER BEEN RELEASED: there are no tags, and `Project.toml` carries
`version = "0.1.0-DEV"`. So there is no released version for anything here to be breaking against,
and the `Breaking` heading below describes what changes for a consumer tracking the branch, not a
semver transition. The package is pre-1.0, so once it is tagged, breaking changes may land in a
minor release.

Consumers of this package are test suites, so a breaking change here breaks a downstream SUITE rather
than a production build. It is recorded anyway: a check that stops compiling is as disruptive to a
downstream package as one that starts failing, and harder to read off a stack trace.

## [Unreleased]

### Breaking

#### `checki2otranslation` requires a matrix reference

`truereference` is now typed `AbstractMatrix`. It previously accepted either a matrix or a callable
`x -> y`, chosen at run time by `truereference isa AbstractVecOrMat ? truereference * x :
truereference(x)`, and the docstring advertised the callable form.

The callable form was never used. It was present in the file's first commit and no caller has ever
passed one: every call site in H2Factory, iFMM and this package's own mocks passes a materialised
matrix. Rather than keep an alternative the suite documented but never exercised, the accepted shape
is now the shape that is actually tested.

A callable is not mechanically convertible, so there is no drop-in replacement: construct a dense
reference matrix equivalent to what your callable computed, and pass that. For a callable that was
already applying a known operator block, that means materialising the block; for one that computed
`y` some other way, you need the matrix whose action it reproduced.

Note the bound is `AbstractMatrix`, NOT the `AbstractVecOrMat` the old run-time check used. `x` is a
vector, so `truereference * x` never accepted a plain `AbstractVector` reference; the old predicate
admitted one and then failed on the multiply. The declared type now matches what the body requires.
`Transpose` and `Adjoint` wrappers are `AbstractMatrix` and remain accepted, which is what the
transpose and adjoint modes pass.

### Added

#### Symmetry-reduced far-field I2O fixtures

`FakeSymmetryI2O`, `fakesymmetryi2o`, `fakesymmetrymap`, `fakepetrovsymmetrymap`,
`symmetryi2osetup`, `petrovsymmetrysetup`, `symmetrytoeplitzcollection`, `symmetrytoeplitz3dsetup`,
`SymmetryDiagonalTranslator`, `symmetrypermutation`, `symmetrydirections`,
`symmetryfrequencyimages`, `symmetrykernelvalue`, `symmetryleafbasis`, `symmetrytranslatingplan` and
`permutedtoeplitzoperators`: fixtures driving H2Factory's symmetry-reduced I2O path, in Galerkin and
Petrov form and for both the dense and Toeplitz collections. The reduction is NOT diagonal-specific
and does NOT require a centred node grid; earlier drafts of these fixtures claimed both, and the
docs were corrected.

#### A genuinely three-dimensional Toeplitz fixture

`Toeplitz3DTranslator`, `toeplitz3dsetup`, `toeplitz3dtensor`, `toeplitz3dcoupling`,
`toeplitz3dnodespacing` and `checktoeplitz3di2o`. The existing Toeplitz fixture was
one-dimensional in substance, so rank-3 behaviour went unexercised by anything that used it.

#### A fixed seed for every accuracy check

`H2FactoryTestUtils.checkrng()` returns `Xoshiro(H2FactoryTestUtils.CHECKSEED)`, and the accessors in
`checkinteractions.jl` and the mocks draw from it. Callers can pass their own `rng` to sweep other
draws deliberately.

Neither name is EXPORTED; both are reached qualified, which is how the downstream suites use them.

Note the shape of this helper: `checkrng()` is FRESH per call by design, so calling it inside a loop
resets the stream and every iteration receives the identical vector. Hoist one rng above a loop and
advance it.

#### `FakeScratch` carries a leaf channel

`fakeleafscratchlength`, and `FakeScratch` now provides a leaf channel sized like the real one, so
the fakes cover the code paths that borrow from it.

### Fixed

#### The fake maps' adjoint was missing two of its four conjugations

`adjoint(A)` on the fake Galerkin and Petrov maps did not conjugate both leaf collections on top of
the coupling's own transpose. Any test that compared a real adjoint against these fakes was
comparing against a wrong reference. `test/fakemapadjoint.jl` now checks the fakes against dense
references per ingredient, so this package vouches for its own fixtures rather than assuming them.

#### Test and docs environments are portable

The `SauterSchwabQuadrature` entry in `test/Manifest.toml` pinned an absolute path to a directory
that does not exist, which prevented the resolver from running at all; `[sources]` in
`test/Project.toml` and `docs/Project.toml` used absolute paths. Both are now relative.

### Changed

- The usage guide now lives in `README.md`, with contributor instructions in
  `CONTRIBUTING.md`. Detailed function documentation remains in docstrings; the
  separate Documenter site and its deployment job were removed.
- Version moved from `1.0.0-DEV` to `0.1.0-DEV`. The package has no tags and is still landing
  documented breaking changes, and H2Factory already sits at `0.1.0-DEV`; `1.0.0-DEV` claimed a
  maturity it does not have. This also made `H2FactoryTestUtils = "0.1"` writable as a compat bound
  for the first time, which MLFMA now carries: `"0.1"` admits the `0.1.0-DEV` prerelease but not
  `1.0.0-DEV`, so the bound could not exist until the version moved.
- `H2FactoryTestUtils` now declares `H2Trees` directly instead of obtaining its module binding
  through H2Factory's former re-export. This package's own API is unchanged and downstream packages
  need no action on account of it; a package whose OWN code references `H2Trees` needs its own
  dependency, which is H2Factory's change to document and is recorded in H2Factory's changelog.
- The fake translators understand the conjugate translation direction.
- The I2O coupling protocol is reached the way a real collection reaches it, rather than through
  collection-specific entry points.
- `PermutedDiagonal` was dropped in favour of H2Factory's real builder.
- `compat`: `CompScienceMeshes` now allows `0.12`; `H2Factory` is bounded at `"0.1"`, which matches
  its `0.1.0-DEV` prerelease and let an Aqua ignore be removed. `Random` is now a declared
  dependency.
