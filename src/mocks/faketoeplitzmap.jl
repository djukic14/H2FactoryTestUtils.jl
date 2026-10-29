# Toeplitz version of `fakegalerkinmap`. It drives FFT modulation, Toeplitz multiplication, and
# inverse modulation through the complete multilevel traversal.
#
# Returns a workspace over a `FakeGalerkinH2Core` (its collection dicts are `Any`-typed and its
# channel type params are independent, so a `ToeplitzI2OChannel` i2o coexists with buffer-store
# i2i/o2o), so every reference/denseh2matrix/spy tool built for the dense fake applies unchanged.
# Storage is always `ComplexF64`; "real vs complex kernel" is a property of `kernelfn`'s values.

"""
    FakeToeplitzI2O(collection, couplings)

Apply I2O translations for a synthetic Toeplitz map through `collection` and record each call.
The separate `couplings` provide dense references through [`refcoupling`](@ref).
"""
struct FakeToeplitzI2O{C,K} <: H2Factory.AbstractI2OCouplingCollection
    collection::C
    couplings::K
    log::Vector{SpyRecord}
    lock::ReentrantLock
end

"""
    faketoeplitzi2o(collection, couplings)

Return a [`FakeToeplitzI2O`](@ref) with `collection`, dense reference `couplings`,
and an empty call log.
"""
function faketoeplitzi2o(collection, couplings)
    return FakeToeplitzI2O(collection, couplings, SpyRecord[], ReentrantLock())
end

_recordi2o!(w::FakeToeplitzI2O, role, a, b) = lock(w.lock) do
    return push!(w.log, SpyRecord(role, a, b))
end
records(w::FakeToeplitzI2O) = lock(w.lock) do
    return copy(w.log)
end

# The guard is forwarded, not bypassed: as a wrapper, its answer to `assertnosymmetry` is the
# wrapped collection's -- the shape `test_symmetrycollection.jl`'s `UnforwardingWrapper` exists to
# show is mandatory. `i2otrait` forwards for the same reason (below); with both declared, the
# guarded `applycoupling!` entry does the right thing without the wrapper touching the entry itself.
H2Factory.assertnosymmetry(w::FakeToeplitzI2O) = H2Factory.assertnosymmetry(w.collection)

# Records the call, then forwards to the real Toeplitz collection's kernel -- the guard has
# already run on this wrapper via the forwarding declaration above, so re-entering the entry
# would only run it twice.
#
# The role tag is derived from the direction by `_spyi2orole`, so the structural spy test can
# still tell "adjoint really applied the adjoint coupling" apart from an accidental fallthrough
# to the transpose.
function H2Factory.applycouplingkernel!(
    out,
    in,
    w::FakeToeplitzI2O,
    trait::H2Factory.AbstractI2OTranslationTrait,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    accumulation::H2Factory.AbstractAccumulationMode,
    context::H2Factory.TranslationContext,
)
    _recordi2o!(w, _spyi2orole(direction), recv, transl)
    return H2Factory.applycouplingkernel!(
        out, in, w.collection, trait, recv, transl, direction, accumulation, context
    )
end
H2Factory.i2otrait(w::FakeToeplitzI2O) = H2Factory.i2otrait(w.collection)

# Forwarded for the same reason as `assertnosymmetry`/`i2otrait` above: the wrapped collection's
# `ToeplitzModulation()` answer must reach `I2OTranslation(w)`'s auto-selection
# (`modulation(coupling)`), or this wrapper silently inherits the generic `NoModulation()` default
# meant for ordinary (non-Toeplitz) collections -- skipping the mandatory FFT into the frequency
# domain with no error, not even a wrong number close enough to look plausible.
H2Factory.modulation(w::FakeToeplitzI2O) = H2Factory.modulation(w.collection)

Base.getindex(w::FakeToeplitzI2O, key) = w.collection[key]
H2Trees.tree(w::FakeToeplitzI2O) = H2Trees.tree(w.collection)
H2Trees.level(w::FakeToeplitzI2O, node::Int) = H2Trees.level(w.collection, node)

# Reference coupling: exact dense S_ts applied to a source moment vector (used by denseh2matrix /
# ref_translate). `src` is a length-`rank` vector, `S_ts` is `rank × rank`.
function refcoupling(w::FakeToeplitzI2O, src, receiving, translating)
    return w.couplings[(receiving, translating)] * src
end

# Per-(level,chunk) Toeplitz scratch-buffer stores for I2I/O2O, shaped to match the ToeplitzMoment
# content `(rank,1,1)` so the library translators round-trip through the real multithreaded buffer-
# checkout path (a wrongly-shared buffer corrupts the result -> caught by the determinism test).
function _buildtoeplitzbufferchannels(::Type{T}, rank, levels, MomentType) where {T}
    # Keys and pool size resolved exactly as in `_buildbufferchannels` -- see the comment there on why
    # the keys are BORROW levels rather than traversed levels.
    nbuffers = _bufferpoolsize(levels)
    return Dict(
        level =>
            H2Factory.createchannel([MomentType{1,T}((rank, 1, 1)) for _ in 1:nbuffers]) for
        level in _bufferpoollevels(levels)
    )
end

"""
    faketoeplitzmap(tree, kernelfn; rank=2, i2i=identitytranslator(), o2o=identitytranslator(),
                    aggregationmode=H2Trees.AggregateMode(),
                    leafbasis=(tr,lf,r)->_onesleafbasis(ComplexF64,tr,lf,r),
                    α=one(ComplexF64), scheduler=DynamicScheduler(), usebuffers=true)

Build a synthetic Toeplitz H2 map over a `TwoNTree` for the scalar kernel `kernelfn(z)`,
where `z = x - y`. Require `rank >= 2`.

I2O uses the FFT-based collection and channel from [`toeplitzi2osetup`](@ref).
The moments are `ToeplitzMoment`s, with FFT basis conversions around I2O. Level
transfers use the supplied translators on the spatial part of each moment.

Return a workspace over a `FakeGalerkinH2Core` with `ComplexF64` storage. Its far-field
product should match `denseh2matrix(A) * x`. The kernel can return real or complex
values; the latter lets tests distinguish adjoint from transpose.
"""
function faketoeplitzmap(
    tree,
    kernelfn;
    rank::Int=2,
    i2i=identitytranslator(),
    o2o=identitytranslator(),
    aggregationmode=H2Trees.AggregateMode(),
    leafbasis=(tr, lf, r) -> _onesleafbasis(ComplexF64, tr, lf, r),
    α=one(ComplexF64),
    scheduler=DynamicScheduler(),
    usebuffers::Bool=true,
    nchunks::Union{Nothing,Int}=nothing,
    MomentType=H2Factory.ToeplitzMoment,
    MomentCollectionType=H2Factory.ToeplitzMomentCollection,
)
    T = ComplexF64
    rank >= 2 || error("faketoeplitzmap needs rank >= 2 (got $rank)")
    isnear = H2Trees.isnear()

    # Plans built exactly as `toeplitzi2osetup` builds them (same isnear/mode), so the map's far
    # pairs coincide with the i2o collection's.
    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; isnear=isnear, aggregationmode=aggregationmode)
    )
    translatingplan = H2Trees.translatingplan(
        plans.trialaggregationplan, plans.testdisaggregationplan
    )

    # The real Toeplitz i2o collection + exact dense couplings (the reference). The collection is
    # Read-only core data; the `ToeplitzI2OChannel` beside it -- FFT/iFFT plans, mutable and
    # unshareable between concurrent matvecs -- is discarded here and rebuilt per workspace by the
    # `i2o` closure below, from the same collection. One nominal chunk suffices for the discarded one.
    setup = H2Factory.withnchunks(Dict(level => 1 for level in H2Trees.levels(tree))) do
        return toeplitzi2osetup(
            tree, kernelfn, rank; aggregationmode=aggregationmode, nchunks=1
        )
    end
    i2o = faketoeplitzi2o(setup.collection, setup.couplings)

    bufferlevels = _scheduledlevels(H2Factory.Schedules(plans; nchunks=1))
    toeplitzbuffers() =
        if usebuffers
            _buildtoeplitzbufferchannels(T, rank, bufferlevels, MomentType)
        else
            H2Factory.EmptyChannels()
        end
    # A real fake leaf pool, built only to exercise the borrow path: a Toeplitz moment's leaf
    # multiply is an ordinary `mul!`, so the default `applyleafmul!` ignores the buffer it's
    # handed. `EmptyChannels` would give identical numbers while skipping the level/chunk keying
    # entirely -- exactly the part worth exercising for free.
    leafbuffers() =
        if usebuffers
            _buildleafchannels(
                T, rank, _leafbufferlevels(bufferlevels, _treeleaflevels(tree))
            )
        else
            H2Factory.EmptyChannels()
        end
    scratch = FakeScratch(
        toeplitzbuffers,
        toeplitzbuffers,
        () -> H2Factory.i2ochannels(setup.collection, translatingplan),
        leafbuffers,
    )

    core = fakegalerkincore(
        tree;
        T=T,
        rank=rank,
        i2i=i2i,
        o2o=o2o,
        i2o=i2o,
        aggregationmode=aggregationmode,
        leafbasis=leafbasis,
        α=α,
        isnear=isnear,
        MomentType=MomentType,
        MomentCollectionType=MomentCollectionType,
        # A Toeplitz moment wraps two arrays per component (the moment and the wider,
        # FFTW-alignment-class toeplitz array), so this drives the allocator's Toeplitz
        # `wrapbuffer` path and its split alignment/`alignmentslack` budget through the full
        # pooled layout, not just the standalone `test_aliasing` check. Moment array `(rank,1,1)`,
        # toeplitz array the standard `2 .* size .- 1` padding.
        momentsize=(tr, level) ->
            ((rank, 1, 1), H2Factory.standardtoeplitzsize((rank, 1, 1))),
        scratch=scratch,
    )
    return H2Workspace(core, scheduler; nchunks=nchunks)
end

"""
    fakepetrovtoeplitzmap(testpoints, trialpoints; rank=2, i2i=identitytranslator(),
                          o2o=identitytranslator(), aggregationmode=H2Trees.AggregateMode(),
                          testleafbasis=…, trialleafbasis=…, α=one(ComplexF64),
                          scheduler=DynamicScheduler(), usebuffers=true,
                          minhalfsize=nothing, testminvalues=0, trialminvalues=0)

Build the Petrov version of [`faketoeplitzmap`](@ref), with separate test and trial trees.
The forward product sends trial moments to test moments through FFT-based Toeplitz I2O.

Return a workspace over a `FakePetrovH2Core` with `ComplexF64` storage. Its far-field
matrix has shape `(Ntest, Ntrial)` and should match `denseh2matrix(A)`.
`kernelfn(z)` is a translation-invariant scalar kernel. Require `rank >= 2`.
"""
function fakepetrovtoeplitzmap(
    testpoints,
    trialpoints,
    kernelfn;
    rank::Int=2,
    i2i=identitytranslator(),
    o2o=identitytranslator(),
    aggregationmode=H2Trees.AggregateMode(),
    testleafbasis=(tr, lf, r) -> _onesleafbasis(ComplexF64, tr, lf, r),
    trialleafbasis=(tr, lf, r) -> _onesleafbasis(ComplexF64, tr, lf, r),
    α=one(ComplexF64),
    scheduler=DynamicScheduler(),
    usebuffers::Bool=true,
    nchunks::Union{Nothing,Int}=nothing,
    minhalfsize=nothing,
    testminvalues::Int=0,
    trialminvalues::Int=0,
    MomentType=H2Factory.ToeplitzMoment,
    MomentCollectionType=H2Factory.ToeplitzMomentCollection,
)
    T = ComplexF64
    rank >= 2 || error("fakepetrovtoeplitzmap needs rank >= 2 (got $rank)")
    isnear = H2Trees.isnear()

    testpts = [SVector{3,Float64}(p...) for p in testpoints]
    trialpts = [SVector{3,Float64}(p...) for p in trialpoints]
    allx = vcat(getindex.(testpts, 1), getindex.(trialpts, 1))
    mhs = minhalfsize === nothing ? (maximum(allx) - minimum(allx)) / 8 : minhalfsize
    tree = H2Trees.buildtree(
        testpts,
        trialpts;
        builder=H2Trees.BlockTreeBuilder(;
            test=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=testminvalues),
            trial=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=trialminvalues),
        ),
    )

    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; isnear=isnear, aggregationmode=aggregationmode)
    )
    translatingplan = H2Trees.translatingplan(
        plans.trialaggregationplan, plans.testdisaggregationplan
    )
    planlevels = _planlevels(
        (plans.trialaggregationplan, plans.testaggregationplan),
        (plans.testdisaggregationplan, plans.trialdisaggregationplan),
    )

    # Collection = read-only core data; its `ToeplitzI2OChannel` is discarded and rebuilt per
    # workspace. See `faketoeplitzmap`.
    setup = H2Factory.withnchunks(Dict(level => 1 for level in planlevels)) do
        return toeplitzi2osetup(
            tree, kernelfn, rank; aggregationmode=aggregationmode, nchunks=1
        )
    end
    i2o = faketoeplitzi2o(setup.collection, setup.couplings)

    bufferlevels = _scheduledlevels(H2Factory.Schedules(plans; nchunks=1))
    toeplitzbuffers() =
        if usebuffers
            _buildtoeplitzbufferchannels(T, rank, bufferlevels, MomentType)
        else
            H2Factory.EmptyChannels()
        end
    # A real fake leaf pool, built only to exercise the borrow path: a Toeplitz moment's leaf
    # multiply is an ordinary `mul!`, so the default `applyleafmul!` ignores the buffer it's
    # handed. `EmptyChannels` would give identical numbers while skipping the level/chunk keying
    # entirely -- exactly the part worth exercising for free.
    leafbuffers() =
        if usebuffers
            _buildleafchannels(
                T,
                rank,
                _leafbufferlevels(
                    bufferlevels,
                    vcat(
                        _treeleaflevels(H2Trees.testtree(tree)),
                        _treeleaflevels(H2Trees.trialtree(tree)),
                    ),
                ),
            )
        else
            H2Factory.EmptyChannels()
        end
    scratch = FakeScratch(
        toeplitzbuffers,
        toeplitzbuffers,
        () -> H2Factory.i2ochannels(setup.collection, translatingplan),
        leafbuffers,
    )

    # The one configuration that exercises everything at once: cross-side pool sharing and the
    # two-array-per-component Toeplitz `wrapbuffer` with its wider FFTW alignment, over the
    # allocator's `isBlockTree` path with separate test/trial pools on separate trees.
    core = fakepetrovcore(
        testpoints,
        trialpoints;
        T=T,
        rank=rank,
        i2i=i2i,
        o2o=o2o,
        i2o=i2o,
        aggregationmode=aggregationmode,
        testleafbasis=testleafbasis,
        trialleafbasis=trialleafbasis,
        α=α,
        isnear=isnear,
        minhalfsize=mhs,
        testminvalues=testminvalues,
        trialminvalues=trialminvalues,
        MomentType=MomentType,
        MomentCollectionType=MomentCollectionType,
        momentsize=(tr, level) ->
            ((rank, 1, 1), H2Factory.standardtoeplitzsize((rank, 1, 1))),
        scratch=scratch,
    )
    return H2Workspace(core, scheduler; nchunks=nchunks)
end

# Toeplitz interpolation mini: an accuracy oracle against the actually-sampled kernel. The
# `faketoeplitzmap` above with identity I2I/O2O is a machinery fixture (`farmul! ==
# denseh2matrix` for any kernel); testing against the actually-sampled kernel matrix
# `K[i,j] = kernel(x_i - x_j)` needs a genuine interpolation FMM with real M2M/L2L transfers.
# These reuse the dense mini's transfer/basis math (`_transfer`/`_leafbasisT`/`_interpnodes`,
# whose equispaced nodes match `toeplitzi2osetup`'s Toeplitz node convention) on the `(rank,1,1)`
# ToeplitzMoment content, so the far field reproduces `K` exactly for a kernel degree-(rank-1)
# interpolation is exact for, with rank-2 beating rank-1 for a general smooth kernel -- the dense
# mini's guarantee, now through the FFT Toeplitz path.

# The transfer uses the `tree` passed by the traversal/reference (aggregation tree for I2I,
# disaggregation tree for O2O), not a stored tree -- for a Petrov `BlockTree` those are the trial
# and test trees respectively, and a stored tree's node ids would be invalid against them (and
# silently broke for asymmetric test/trial depths). So these hold only `rank`.
struct ToeplitzMiniI2I <: MiniTranslator
    rank::Int
end
struct ToeplitzMiniO2O <: MiniTranslator
    rank::Int
end

# Apply a rank×rank transfer `M` to the (rank,1,1) moment content, round-tripping through the
# per-(level,chunk) scratch buffer (`_scratch`).
function _applytransfer!(out1, M, in1, buffer)
    return (out1 .= reshape(_scratch(buffer, M * vec(in1)), size(out1)))
end

# Level-transfer protocol (like `PetrovMiniI2I`, keyed on the passed tree, but reshaping the `(rank,1,1)`
# Toeplitz moment content). `_minii2imatrix` gives the direction-transformed matrix: forward
# `transpose(E)`, transpose `E` (reconstructing O2O for disaggregation), adjoint `conj(E)`.
function H2Factory.leveltranslation!(
    out,
    in,
    t::ToeplitzMiniI2I,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    E = _transfer(tree, parent, child, t.rank)
    _applytransfer!(out[1], _minii2imatrix(E, direction), in[1], context.buffer)
    return out
end
function H2Factory.leveltranslation!(
    out,
    in,
    t::ToeplitzMiniI2I,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    E = _transfer(tree, parent, child, t.rank)
    out[1] .=
        reshape(
            _scratch(context.buffer, _minii2imatrix(E, direction) * vec(in[1])),
            size(out[1]),
        ) .+ out[1]
    return out
end
function reftransfer(t::ToeplitzMiniI2I, src, tree, parent, child)
    return transpose(_transfer(tree, parent, child, t.rank)) * src
end

# Off the traversal path (disaggregation reconstructs O2O from the Toeplitz i2i mini); reference-only, so
# the bare-callable application was removed. `reftransfer` remains.
function reftransfer(t::ToeplitzMiniO2O, src, tree, parent, child)
    return _transfer(tree, parent, child, t.rank) * src
end

"""
    minitoeplitzh2matrix(points, kernelfn; rank=2, minhalfsize=nothing, root=1,
                         aggregationmode=H2Trees.AggregateMode())

Build the Toeplitz version of [`minih2matrix`](@ref), with one-dimensional interpolation
and FFT-based I2O. `kernelfn(z)` takes the displacement `z = x - y`. Storage is
`ComplexF64`, and `rank` must be at least 2.

Return a `NamedTuple` with `A`, `tree`, `points`, `rank`, `dense_approx`, and `dense_true`.
`dense_approx` is the dense matrix of the interpolation approximation.
`dense_true` contains exact kernel values on the same far partition.

`H2Factory.farmul(A, x)` should match `dense_approx * x` to floating-point accuracy.
Comparing `dense_approx` with `dense_true` measures the interpolation error. The
interpolation is exact, up to roundoff, for a polynomial kernel of degree at most `rank - 1`.
"""
function minitoeplitzh2matrix(
    points::AbstractVector{<:Real},
    kernelfn;
    rank::Int=2,
    minhalfsize=nothing,
    root::Int=1,
    aggregationmode=H2Trees.AggregateMode(),
    MomentType=H2Factory.ToeplitzMoment,
    MomentCollectionType=H2Factory.ToeplitzMomentCollection,
)
    rank >= 2 || error("minitoeplitzh2matrix needs rank >= 2 (got $rank)")
    coords = collect(float.(points))
    pts3d = [SVector{3,Float64}(c, 0.0, 0.0) for c in coords]
    mhs = minhalfsize === nothing ? (maximum(coords) - minimum(coords)) / 8 : minhalfsize
    tree = H2Trees.buildtree(
        pts3d; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=0, root=root)
    )

    A = faketoeplitzmap(
        tree,
        kernelfn;
        rank=rank,
        i2i=ToeplitzMiniI2I(rank),
        o2o=ToeplitzMiniO2O(rank),
        leafbasis=(tr, lf, r) -> Matrix{ComplexF64}(_leafbasisT(tr, lf, r, coords)),
        aggregationmode=aggregationmode,
        MomentType=MomentType,
        MomentCollectionType=MomentCollectionType,
    )
    kern2(x, y) = kernelfn(x - y)   # 2-arg form for the dense-mini assembly helpers
    return (
        A=A,
        tree=tree,
        points=coords,
        rank=rank,
        dense_approx=_dense_far(A, tree, rank, coords, kern2),
        dense_true=_dense_true_far(A, tree, coords, kern2),
    )
end

# Exact sampled kernel K[i,j] = kernel(x_test_i - x_trial_j) restricted to a Petrov map's far
# partition (test DOFs from the test tree, trial DOFs from the trial tree).
function _petrov_dense_true_far(A, testtree, trialtree, testcoords, trialcoords, kern2)
    Ntest = H2Trees.numberofvalues(testtree)
    Ntrial = H2Trees.numberofvalues(trialtree)
    K = zeros(ComplexF64, Ntest, Ntrial)
    for (recv, transl) in ref_farpairs(A)
        for i in H2Trees.values(testtree, recv)
            for j in H2Trees.values(trialtree, transl)
                K[i, j] += kern2(testcoords[i], trialcoords[j])
            end
        end
    end
    return K
end

"""
    minipetrovtoeplitzh2matrix(testpoints, trialpoints, kernelfn; rank=2, minhalfsize,
                               testminvalues=0, trialminvalues=0, aggregationmode=…)

Build the Petrov version of [`minitoeplitzh2matrix`](@ref), with separate test and trial
trees. Interpolation transfers aggregate on the trial tree and disaggregate on the test
tree; I2O uses the FFT-based Toeplitz operation.

Supply `minhalfsize` explicitly to set the tree geometry used by the interpolation.
Return a `NamedTuple` with `A`, `tree`, `rank`, `dense_approx`, and `dense_true`.
The latter contains `kernelfn(x_test_i - x_trial_j)` on the far partition.
Compare the product with `dense_approx` to check the implementation, and compare the
two dense matrices to measure the interpolation error.
"""
function minipetrovtoeplitzh2matrix(
    testpoints::AbstractVector{<:Real},
    trialpoints::AbstractVector{<:Real},
    kernelfn;
    rank::Int=2,
    minhalfsize::Real,
    testminvalues::Int=0,
    trialminvalues::Int=0,
    aggregationmode=H2Trees.AggregateMode(),
    MomentType=H2Factory.ToeplitzMoment,
    MomentCollectionType=H2Factory.ToeplitzMomentCollection,
)
    rank >= 2 || error("minipetrovtoeplitzh2matrix needs rank >= 2 (got $rank)")
    testcoords = collect(float.(testpoints))
    trialcoords = collect(float.(trialpoints))
    testpts = [(c, 0.0, 0.0) for c in testcoords]
    trialpts = [(c, 0.0, 0.0) for c in trialcoords]

    # The interpolation transfers read the tree the traversal passes (trial tree for I2I, test tree
    # for O2O), so no tree needs building here -- the leaf bases capture the per-side coords, and
    # `dense_true` reads the map's own trees.
    A = fakepetrovtoeplitzmap(
        testpts,
        trialpts,
        kernelfn;
        rank=rank,
        i2i=ToeplitzMiniI2I(rank),
        o2o=ToeplitzMiniO2O(rank),
        testleafbasis=(tr, lf, r) -> Matrix{ComplexF64}(_leafbasisT(tr, lf, r, testcoords)),
        trialleafbasis=(tr, lf, r) ->
            Matrix{ComplexF64}(_leafbasisT(tr, lf, r, trialcoords)),
        minhalfsize=minhalfsize,
        testminvalues=testminvalues,
        trialminvalues=trialminvalues,
        aggregationmode=aggregationmode,
        MomentType=MomentType,
        MomentCollectionType=MomentCollectionType,
    )
    kern2(x, y) = kernelfn(x - y)
    return (
        A=A,
        tree=H2Trees.tree(A),
        rank=rank,
        dense_approx=denseh2matrix(A),
        dense_true=_petrov_dense_true_far(
            A, H2Trees.testtree(A), H2Trees.trialtree(A), testcoords, trialcoords, kern2
        ),
    )
end
