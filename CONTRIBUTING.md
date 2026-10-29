# Contributing

Open pull requests against `main`. Follow the surrounding code style, document
new functions in their docstrings, and add tests for changes in behavior.
The README holds the usage guide; there is no separate documentation site to build.

## Tests

The test environment uses relative paths to sibling checkouts of H2Factory,
H2Trees, BEAST, and CompScienceMeshes. The expected paths are listed in
[`test/Project.toml`](test/Project.toml).

From this package's directory, run:

```sh
julia --startup-file=no --project=test -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'
```

For a new fixture, supply a reference calculation that does not call the map's
multiply. Check forward, transpose, and adjoint operations where supported.
Distinguish implementation error from the method's approximation error.

## Formatting

The suite checks formatting with JuliaFormatter and the settings in
[`.JuliaFormatter.toml`](.JuliaFormatter.toml). In a Julia session started with
`julia --project=test`, format the files you edited rather than rewriting
unrelated work across the repository:

```julia
using JuliaFormatter
format("src/interactions/checkinteractions.jl"; overwrite=true)
```
