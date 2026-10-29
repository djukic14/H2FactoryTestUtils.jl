# Toeplitz far-field I2O translation test support (Toeplitz-moment analogue of the dense harness,
# driving H2Factory's real FFT Toeplitz path: `modulate_in!` -> `toeplitzmul!` ->
# `modulate_out!`). The Toeplitz-specific machinery *is* the I2O translation step -- the
# aggregate/disaggregate traversal is identical to the dense case -- so we test that step directly,
# for every far pair, in all `LinearMaps` modes, via `checki2otranslation`.
#
# For a translation-invariant kernel `k(x-y)` and same-level boxes (which H2Factory's level-
# organized far field always pairs -- same level ⇒ same box size in a `TwoNTree`), the far-field
# coupling is Toeplitz: with equally-spaced interpolation nodes `ξ_t,μ = c_t - h/2 + (μ-1)δ`,
#     S_ts[μ,ν] = k(ξ_t,μ - ξ_s,ν) = k((c_t - c_s) + (μ-ν)δ),
# depending only on the center displacement and `μ-ν`. Its FFT-domain kernel is
# `ToeplitzMatrixRepresentation(fft!(circulant-embedding))`, exactly what H2Factory's `toeplitzmul!`
# consumes. The dense `S_ts` is the reference `checki2otranslation` compares against.

# The r×r dense Toeplitz coupling for a pair at displacement `disp`, node spacing `δ`, rank `r`.
function _toeplitzcoupling(kernelfn, disp, δ, r, ::Type{T}) where {T}
    return T[kernelfn(disp + (μ - ν) * δ) for μ in 1:r, ν in 1:r]
end

# Circulant embedding a = [g_0..g_{r-1}, g_{-(r-1)}..g_{-1}] with g_d = k(disp + dδ), shaped for the
# 3D Toeplitz path.
function _toeplitztensor(kernelfn, disp, δ, r, ::Type{T}) where {T}
    g = T[kernelfn(disp + d * δ) for d in (-(r - 1)):(r - 1)]   # length 2r-1, g[k] at displ (k-r)
    a = vcat(g[r:(2r - 1)], g[1:(r - 1)])                        # circulant embedding
    return reshape(complex(a), (2r - 1, 1, 1))
end

# FFT-domain kernel of that coupling -- what `toeplitzmul!` multiplies against.
function _toeplitzkernel(kernelfn, disp, δ, r, fftplan, ::Type{T}) where {T}
    return H2Factory.ToeplitzMatrixRepresentation(
        fftplan * _toeplitztensor(kernelfn, disp, δ, r, T)
    )
end

# Plan-free variant. `computetranslations` invokes the builder under `@tasks`, so tasks would share
# one in-place plan; an allocating `fft` keeps the builder free of shared mutable state.
function _toeplitzkernel(kernelfn, disp, δ, r, ::Nothing, ::Type{T}) where {T}
    return H2Factory.ToeplitzMatrixRepresentation(
        fft(_toeplitztensor(kernelfn, disp, δ, r, T))
    )
end

"""
    FakeToeplitzTranslator{T}(kernelfn, testtree, trialtree, rank)

Build the Toeplitz kernels used by the one-dimensional fixtures through H2Factory's
I2O collection constructor. The builder provides `eltype` and a call method with
arguments `(receivingnode, translatingnode, direction)`.

It computes the displacement from the node pair rather than using `direction`.
Node spacing comes from the receiving node's box. When the collection reuses a
translation, its representative pair therefore supplies the spacing for that level.
"""
struct FakeToeplitzTranslator{T,K,TeT,TrT}
    kernelfn::K
    testtree::TeT
    trialtree::TrT
    rank::Int
end

function FakeToeplitzTranslator{T}(kernelfn, testtree, trialtree, rank::Int) where {T}
    return FakeToeplitzTranslator{T,typeof(kernelfn),typeof(testtree),typeof(trialtree)}(
        kernelfn, testtree, trialtree, rank
    )
end

# The per-pair translation is a 1-element vector of Toeplitz operators (the `∇`-free scalar kernel
# shape), so `toeplitzmul!`'s vector method is what the traversal ends up calling. The element type
# must be the concrete `ToeplitzMatrixRepresentation{Array{...,3}}`: the extension constructor
# asks the stored operators for `toeplitzeltype`/
# `toeplitzsize`, and those only have methods for the parameterized type, not the bare UnionAll.
function _faketoeplitzrep(::Type{T}) where {T}
    return H2Factory.ToeplitzMatrixRepresentation{Array{Complex{real(T)},3}}
end

Base.eltype(::FakeToeplitzTranslator{T}) where {T} = Vector{_faketoeplitzrep(T)}

function (b::FakeToeplitzTranslator{T})(
    receivingnode::Int, translatingnode::Int, direction
) where {T}
    H2Trees.halfsize(b.testtree, receivingnode) ≈
    H2Trees.halfsize(b.trialtree, translatingnode) || error(
        "far pair ($receivingnode,$translatingnode) has mismatched box sizes; " *
        "Toeplitz coupling needs same size",
    )
    δ = _nodespacing(b.testtree, receivingnode, b.rank)
    disp = _center1d(b.testtree, receivingnode) - _center1d(b.trialtree, translatingnode)
    return _faketoeplitzrep(T)[_toeplitzkernel(b.kernelfn, disp, δ, b.rank, nothing, T)]
end

# 1D interval spacing of a node's equally-spaced rank-`r` interpolation nodes.
_nodespacing(tree, node, r) = 2 * H2Trees.halfsize(tree, node) / (r - 1)
_center1d(tree, node) = H2Trees.center(tree, node)[1]

# (testtree, trialtree, plans) for a Galerkin `TwoNTree` (both are the single tree) or a Petrov
# `BlockTree` (the two sides). Receiving nodes live in `testtree`, translating nodes in `trialtree`.
# `buildplans` itself dispatches Galerkin-vs-Petrov on the tree's `treetrait`; only the
# (testtree, trialtree) pair still needs the type-based split, since a bare `TwoNTree` has no
# `testtree`/`trialtree` accessor of its own.
function _toeplitzplanbuilder(aggn, tfit, mode)
    return H2Trees.PlanBuilder(;
        translatingnodesiterator=tfit, aggregatenode=aggn, aggregationmode=mode
    )
end

function _toeplitztreeplans(tree::H2Trees.TwoNTree, aggn, tfit, mode)
    return (
        tree, tree, H2Trees.buildplans(tree; builder=_toeplitzplanbuilder(aggn, tfit, mode))
    )
end
function _toeplitztreeplans(tree::H2Trees.BlockTree, aggn, tfit, mode)
    return (
        H2Trees.testtree(tree),
        H2Trees.trialtree(tree),
        H2Trees.buildplans(tree; builder=_toeplitzplanbuilder(aggn, tfit, mode)),
    )
end

"""
    toeplitzi2osetup(tree, kernelfn, rank; aggregationmode=H2Trees.AggregateMode())

Build Toeplitz I2O couplings for a Galerkin `TwoNTree` or Petrov `BlockTree`.
The scalar kernel is translation-invariant: `kernelfn(z)` takes `z = x - y`.
For Petrov, the receiving node belongs to the test tree and the source to the trial tree.

Return a `NamedTuple` with:

- `collection`: the `ToeplitzI2OTranslationCollection` for `checki2otranslation`.
- `channel`: the `ToeplitzI2OChannel`, including in-place FFT plans per level.
- `couplings`: dense reference matrices keyed by `(receiving, translating)`.
- `farpairs`: the list of those node pairs.
- `levels`: each pair's receiving-node level in the test tree.

Require `rank >= 2` and equal box sizes within each far pair. The box-size check uses
halfsizes, not level numbers: equal-sized boxes on the two sides of a `BlockTree` can
have different level numbers.
"""
function toeplitzi2osetup(
    tree,
    kernelfn,
    rank::Int;
    aggregationmode=H2Trees.AggregateMode(),
    nchunks::Int=Base.Threads.nthreads(),
)
    T = ComplexF64   # the FFT Toeplitz path is inherently complex-storage (FFT of a moment is
    # complex); "real" vs "complex" is a property of `kernelfn`'s values, not storage
    rank >= 2 || error("toeplitzi2osetup needs rank >= 2 (got $rank)")
    tfit = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
    aggn = H2Trees.istranslatingnode(; TranslatingNodesIterator=tfit)
    testtree, trialtree, plans = _toeplitztreeplans(tree, aggn, tfit, aggregationmode)
    schedules = H2Factory.Schedules(plans; nchunks=nchunks)

    # far pairs (recv in testtree, transl in trialtree) + level (recv's level in the test tree).
    farpairs = Tuple{Int,Int}[]
    levelof = Dict{Tuple{Int,Int},Int}()
    if plans.testdisaggregationplan isa H2Trees.DisaggregateTranslatePlan
        plan = plans.testdisaggregationplan
        for level in H2Trees.disaggregationlevels(plan)
            for node in H2Trees.disaggregationnodes(plan, level)
                for tn in H2Trees.translatingnodes(plan, node)
                    push!(farpairs, (node, tn))
                    levelof[(node, tn)] = H2Trees.level(testtree, node)
                end
            end
        end
    else
        plan = plans.trialaggregationplan
        for level in H2Trees.aggregationlevels(plan)
            for rn in H2Trees.receivingnodes(plan, level)
                for tn in H2Trees.translatingnodes(plan, rn, level)
                    push!(farpairs, (rn, tn))
                    levelof[(rn, tn)] = H2Trees.level(testtree, rn)
                end
            end
        end
    end

    # Dense reference couplings, per pair -- this is what `checki2otranslation` compares against, so
    # it stays hand-built and independent of anything H2Factory constructs.
    couplings = Dict{Tuple{Int,Int},Matrix{T}}()
    for (recv, transl) in farpairs
        H2Trees.halfsize(testtree, recv) ≈ H2Trees.halfsize(trialtree, transl) || error(
            "far pair ($recv,$transl) has mismatched box sizes; Toeplitz coupling needs same size",
        )
        δ = _nodespacing(testtree, recv, rank)
        disp = _center1d(testtree, recv) - _center1d(trialtree, transl)
        couplings[(recv, transl)] = _toeplitzcoupling(kernelfn, disp, δ, rank, T)
    end

    # The collection/channel come from H2Factory's own construction pipeline (the `ext/H2Toeplitz`
    # constructor), driven by a builder rather than hand-assembled here, so the per-level FFT-plan
    # sizing rule lives in one place. `withschedulechunks` is required: the constructor sizes each level's plan pool
    # with `nchunks(level)`, which reads the ambient schedule.
    #
    # The positional `translationtrait` is the H2Trees direction-dedup trait
    # (`DirectionInvariance`), while the
    # Toeplitz-ness is the `i2otranslationtrait` keyword (`ToeplitzI2OTranslationTrait`) -- passing
    # the I2O trait positionally would reach `H2Trees._translations`, which has no method for it.
    translator = FakeToeplitzTranslator{T}(kernelfn, testtree, trialtree, rank)
    collection, channel = H2Factory.withschedulechunks(schedules) do
        return H2Factory.ToeplitzI2OTranslationCollection(
            tree,
            plans.trialaggregationplan,
            plans.testdisaggregationplan,
            H2Trees.DirectionInvariance(),
            translator;
            i2otranslationtrait=H2Factory.ToeplitzI2OTranslationTrait(),
        )
    end

    return (
        collection=collection,
        channel=channel,
        couplings=couplings,
        farpairs=farpairs,
        levels=levelof,
    )
end

"""
    checktoeplitzi2o(tree, kernelfn, rank; aggregationmode=…, n=5, ϵ=1e-9, rng=checkrng())

Check every far pair of `tree` in forward, transpose, and adjoint modes.
[`checki2otranslation`](@ref) compares the FFT-based Toeplitz operation with its dense
coupling matrix. Identity leaf collections isolate the translation from the leaf basis.

Report assertions through `Test` and return the number of `(pair, mode)` checks.
The moments use complex storage for the FFT, even when `kernelfn` returns real values.
A real or complex kernel therefore tests the corresponding case without changing storage.
"""
function checktoeplitzi2o(
    tree,
    kernelfn,
    rank::Int;
    aggregationmode=H2Trees.AggregateMode(),
    n=5,
    ϵ=1e-9,
    nchunks::Int=Base.Threads.nthreads(),
    rng=checkrng(),
)
    T = ComplexF64
    setup = toeplitzi2osetup(
        tree, kernelfn, rank; aggregationmode=aggregationmode, nchunks=nchunks
    )
    idmc = H2Factory.ToeplitzMomentCollection1D{T}((rank, rank))
    idmc[1] .= Matrix{T}(I, rank, rank)
    buildmom() = H2Factory.ToeplitzMoment1D{T}((rank, 1, 1))
    # Seeded (see `CHECKSEED`); reproducible because `setup.farpairs` is an ordered Vector, not a
    # Dict iteration order.
    xgen() = randn(rng, T, rank)

    nchecks = 0
    for (recv, transl) in setup.farpairs
        S = setup.couplings[(recv, transl)]
        level = setup.levels[(recv, transl)]
        for (mode, ref) in (
            (H2Factory.isForward(), S),
            (H2Factory.isTranspose(), transpose(S)),
            (H2Factory.isAdjoint(), S'),
        )
            checki2otranslation(
                idmc,
                idmc,
                setup.collection,
                setup.channel,
                recv,
                transl,
                level,
                buildmom,
                buildmom,
                xgen,
                Matrix(ref);
                mode=mode,
                n=n,
                ϵ=ϵ,
                verbose=false,
            )
            nchecks += 1
        end
    end
    return nchecks
end
