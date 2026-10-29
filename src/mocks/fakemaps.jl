# Synthetic Galerkin H2 core with pluggable translators, moment rank, and leaf basis. The
# reference implementation and interpolation fixtures build on it.
#
# Field names match the generic accessors. Extra methods below cover accessors that accept
# maps but not cores.

"""
    FakeScratch(i2i, o2o, i2o, leaf)

Store four zero-argument functions that allocate scratch for I2I, O2O, I2O, and
leaf multiplication. Each workspace calls them to get its own buffers, sized for
its own schedule.
"""
struct FakeScratch{I2I,O2O,I2O,LEAF}
    i2i::I2I
    o2o::O2O
    i2o::I2O
    leaf::LEAF
end

"""
    FakeGalerkinH2Core{T} <: H2Factory.GalerkinH2Core{T}

Store the shared data for a synthetic Galerkin map, built by [`fakegalerkincore`](@ref).
The near-field matrix is zero, so the map tests the far-field operations.

The last fields describe how to allocate a workspace. To multiply, create one with
`H2Factory.H2Workspace(core, scheduling)`, or use [`fakegalerkinmap`](@ref) to build both.
"""
struct FakeGalerkinH2Core{T,TREE,ISN,I2I,O2O,I2O,FM,MT,MS,SC} <: H2Factory.GalerkinH2Core{T}
    tree::TREE
    nearinteractions::Matrix{T}
    isnear::ISN
    verbose::Bool
    α::T
    trialaggregationplan::Any
    testdisaggregationplan::Any
    testaggregationplan::Any
    trialdisaggregationplan::Any
    momentcollections::Dict{Int,Any}
    i2itranslator::I2I
    o2otranslator::O2O
    i2otranslator::I2O
    farmulmode::FM
    # --- workspace-construction metadata ---
    rank::Int
    momenttype::MT
    # How `storagemomentsplan` sizes a node's moment. A plain `(rank,)` for the dense fakes; the
    # Toeplitz fake passes the `(momentsize, toeplitzsize)` pair its two-array moments need.
    momentsize::MS
    # Allocate buffers only at scheduled levels, where chunk counts are defined.
    bufferlevels::Vector{Int}
    scratch::SC
end

# Use real per-chunk buffers so parallel tests exercise buffer borrowing.
_scheduledlevels(schedules) = sort!(collect(keys(H2Factory.schedulechunkcounts(schedules))))

# Level transfers borrow at the child's level, one above the traversed parent level.
_bufferpoollevels(levels) = sort!(unique!(vcat(collect(levels), collect(levels) .+ 1)))

# Transfer buffers have one shape; size every pool for the largest chunk count.
_bufferpoolsize(levels) = maximum(H2Factory.nchunks(level) for level in levels)

function _buildbufferchannels(::Type{T}, rank, levels, MomentType) where {T}
    nbuffers = _bufferpoolsize(levels)
    return Dict(
        level => H2Factory.createchannel([MomentType{1,T}((rank,)) for _ in 1:nbuffers]) for
        level in _bufferpoollevels(levels)
    )
end

"""
    fakeleafscratchlength(rank, level)

Return `rank + level`, the length of the leaf scratch buffer.
The length varies by level so tests can detect borrowing from the wrong level.
"""
fakeleafscratchlength(rank::Int, level::Int) = rank + level

# Leaf buffers use each leaf's level and its exact chunk count. Extra slots could hide
# an invalid chunk index.
function _buildleafchannels(::Type{T}, rank, levels) where {T}
    return Dict(
        level => H2Factory.createchannel([
            Vector{T}(undef, fakeleafscratchlength(rank, level)) for
            _ in 1:H2Factory.nchunks(level)
        ]) for level in levels
    )
end

# Allocate leaf pools only where leaves are visited, so a wrong-level borrow cannot succeed.
function _treeleaflevels(tree)
    return unique!([H2Trees.level(tree, leaf) for leaf in H2Trees.leaves(tree)])
end

_leafbufferlevels(bufferlevels, leaflevels) = sort!(intersect(bufferlevels, leaflevels))

# Default leaf momentcollection: rank×nDOF all-ones, so leaf aggregation is a plain sum and (via
# the transpose in `LinearAlgebra.mul!`) leaf disaggregation broadcasts the node moment equally to
# every DOF. A fixture wanting a real interpolation basis passes its own
# `leafbasis(tree, leaf, rank) -> AbstractMatrix` (rank×nDOF).
function _onesleafbasis(::Type{T}, tree, leaf, rank) where {T}
    n = length(H2Trees.values(tree, leaf))
    M = ones(T, rank, n)
    return M
end

"""
    fakegalerkincore(tree; T=ComplexF64, rank=1,
                     i2i=identitytranslator(), o2o=identitytranslator(), i2o=identitytranslator(),
                     aggregationmode=H2Trees.AggregateMode(),
                     leafbasis=(tr,lf,r)->_onesleafbasis(T,tr,lf,r),
                     α=one(T), isnear=H2Trees.isnear(), usebuffers=true, …)

Build a synthetic Galerkin core over a `TwoNTree`, with plans, leaf collections,
translators, and functions that allocate workspace data. Separate workspaces can share
the core. [`fakegalerkinmap`](@ref) builds a core and workspace together.

With `usebuffers=true`, each chunk gets scratch buffers that the translators write
through. This lets tests detect accidental sharing. With `false`, the channels are
`H2Factory.EmptyChannels()` instead.
"""
function fakegalerkincore(
    tree;
    T=ComplexF64,
    rank::Int=1,
    i2i=identitytranslator(),
    o2o=identitytranslator(),
    i2o=identitytranslator(),
    aggregationmode=H2Trees.AggregateMode(),
    leafbasis=(tr, lf, r) -> _onesleafbasis(T, tr, lf, r),
    α=one(T),
    isnear=H2Trees.isnear(),
    usebuffers::Bool=true,
    MomentType=H2Factory.Moment,
    MomentCollectionType=H2Factory.MomentCollection,
    momentsize=(tr, level) -> (rank,),
    # A caller with differently shaped scratch (the Toeplitz fake: Toeplitz buffer stores for i2i/o2o
    # and a `ToeplitzI2OChannel` for i2o) passes its own `FakeScratch` instead of the default stores.
    scratch=nothing,
)
    # `PlanBuilder` chains its `translatingnodesiterator`/`aggregatenode` defaults off `isnear`
    # itself (same pattern as `InteractionStrategy`), so they can't silently disagree.
    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; isnear=isnear, aggregationmode=aggregationmode)
    )

    momentcollections = Dict{Int,Any}()
    for leaf in H2Trees.leaves(tree)
        n = length(H2Trees.values(tree, leaf))
        basis = leafbasis(tree, leaf, rank)
        size(basis) == (rank, n) || error(
            "leafbasis returned $(size(basis)) for leaf $leaf; expected (rank=$rank, nDOF=$n)",
        )
        mc = MomentCollectionType{1,T}((rank, n))
        mc[1] .= basis
        momentcollections[leaf] = mc
    end

    # A one-chunk schedule identifies buffer levels; workspaces choose their own pool sizes.
    bufferlevels = _scheduledlevels(H2Factory.Schedules(plans; nchunks=1))
    thescratch = something(
        scratch,
        _bufferscratch(
            T, rank, bufferlevels, _treeleaflevels(tree), MomentType, usebuffers
        ),
    )

    N = H2Trees.numberofvalues(tree)
    # The tree-specific near/far predicate (`isnear` the kwarg is the tree-agnostic functor).
    nearpred = isnear(tree)

    # On this dense path, the plan type selects the traversal, not `farmulmode`.
    return FakeGalerkinH2Core{
        T,
        typeof(tree),
        typeof(nearpred),
        typeof(i2i),
        typeof(o2o),
        typeof(i2o),
        typeof(aggregationmode),
        typeof(MomentType),
        typeof(momentsize),
        typeof(thescratch),
    }(
        tree,
        zeros(T, N, N),
        nearpred,
        false,
        T(α),
        plans.trialaggregationplan,
        plans.testdisaggregationplan,
        plans.testaggregationplan,
        plans.trialdisaggregationplan,
        momentcollections,
        i2i,
        o2o,
        i2o,
        aggregationmode,
        rank,
        MomentType,
        momentsize,
        bufferlevels,
        thescratch,
    )
end

# Separate builders let a fixture replace one channel without changing the others.
function _bufferscratch(
    ::Type{T}, rank, bufferlevels, leaflevels, MomentType, usebuffers
) where {T}
    usebuffers || return FakeScratch(
        () -> H2Factory.EmptyChannels(),
        () -> H2Factory.EmptyChannels(),
        () -> H2Factory.EmptyChannels(),
        () -> H2Factory.EmptyChannels(),
    )
    leafbufferlevels = _leafbufferlevels(bufferlevels, leaflevels)
    build() = _buildbufferchannels(T, rank, bufferlevels, MomentType)
    buildleaf() = _buildleafchannels(T, rank, leafbufferlevels)
    return FakeScratch(build, build, build, buildleaf)
end

"""
    fakegalerkinmap(tree; scheduler=DynamicScheduler(), nchunks=nothing, kwargs...)

Build a core with [`fakegalerkincore`](@ref), then create a workspace that can multiply.
Other keywords are passed to the core builder.

The scheduler chooses the chunk count unless `nchunks` is supplied.
`OhMyThreads.SerialScheduler()` runs serially. Only one multiplication can use a
workspace at a time; concurrent calls need separate workspaces, which may share the core.
"""
function fakegalerkinmap(
    tree; scheduler=DynamicScheduler(), nchunks::Union{Nothing,Int}=nothing, kwargs...
)
    return H2Workspace(fakegalerkincore(tree; kwargs...), scheduler; nchunks=nchunks)
end

# These accessors need explicit core methods; other accessors read the matching fields.
H2Trees.isnear(A::FakeGalerkinH2Core, ::H2Factory.isForward) = A.isnear
function H2Factory.testmomentcollections(A::FakeGalerkinH2Core, ::H2Factory.isForward)
    return A.momentcollections
end
function H2Factory.trialmomentcollections(A::FakeGalerkinH2Core, ::H2Factory.isForward)
    return A.momentcollections
end

# Rebuild the plan bundle expected by `Schedules` from the four stored plans.
function H2Factory.plans(A::FakeGalerkinH2Core)
    return (
        trialaggregationplan=A.trialaggregationplan,
        testdisaggregationplan=A.testdisaggregationplan,
        testaggregationplan=A.testaggregationplan,
        trialdisaggregationplan=A.trialdisaggregationplan,
    )
end

H2Factory.schedulinglevels(A::FakeGalerkinH2Core) = H2Trees.levels(A.tree)

# Use the pooled allocator so the fixtures exercise storage reuse and aliasing.
function H2Factory.buildmoments(A::FakeGalerkinH2Core)
    momentbuffers, testmoments, trialmoments = H2Factory.storagemomentsplan(
        A.testaggregationplan,
        A.trialaggregationplan,
        A.tree,
        level -> A.momenttype{1,eltype(A)},
        A.momentsize,
    )
    _assertmomentcoverage(A, testmoments, trialmoments)
    return momentbuffers, H2Factory.Moments(testmoments, trialmoments)
end

# Check disaggregation nodes too, before a missing moment fails inside a threaded traversal.
function _assertmomentcoverage(A, testmoments, trialmoments)
    for node in _plannodes(
        (A.trialaggregationplan, A.testaggregationplan),
        (A.testdisaggregationplan, A.trialdisaggregationplan),
    )
        haskey(trialmoments, node) && haskey(testmoments, node) && continue
        error(
            "storagemomentsplan produced no moment for node $node, which the plans traverse; " *
            "the fake's node coverage and the allocator's disagree",
        )
    end
    return nothing
end

function H2Factory.buildvectorbuffers(A::FakeGalerkinH2Core)
    T = eltype(A)
    return Dict{Int,Vector{T}}(
        leaf => Vector{T}(undef, length(H2Trees.values(A.tree, leaf))) for
        leaf in H2Trees.leaves(A.tree)
    )
end

H2Factory.buildi2ichannels(A::FakeGalerkinH2Core) = A.scratch.i2i()
H2Factory.buildo2ochannels(A::FakeGalerkinH2Core) = A.scratch.o2o()
H2Factory.buildi2ochannel(A::FakeGalerkinH2Core) = A.scratch.i2o()

# Leaf-multiply scratch, bare rather than a `SidedChannels`: a Galerkin core has one tree, so its
# two sides' leaves and levels are the same, and a second copy would only alias the first -- which
# is what `trialleafchannel`/`testleafchannel` mean for a Galerkin workspace.
H2Factory.buildleafchannels(A::FakeGalerkinH2Core) = A.scratch.leaf()

# The adjoint conjugates the leaf collections and `α`, as well as transfers and couplings:
#     F = Σ transpose(MC_test) · E_test · S · E_trial · MC_trial
#     F' = Σ conj(transpose(MC_trial)) · E' · S' · conj(MC_test)
# The translation protocol handles the middle factors; these hooks handle the leaves and `α`.
function H2Factory.leafmul(::FakeGalerkinH2Core, ::H2Factory.isAdjoint)
    return H2Factory.ConjugatedCollectionMul()
end

# Conjugate at the accessor instead of storing another field. The fixture's `α` is a scalar, so
# this does not allocate in the hot loop.
H2Factory.α(A::FakeGalerkinH2Core, ::H2Factory.isAdjoint) = conj(A.α)
