# Synthetic Petrov-Galerkin H2 core with separate test and trial trees, moments, collections,
# buffers, and plans. It supports coincident, separated, and overlapping geometries.

# Separate transfers on each side make an incorrect side swap detectable.
"""
    FakePetrovH2Core{T} <: H2Factory.PetrovGalerkinH2Core{T}

Store the shared data for a synthetic Petrov-Galerkin map over a `BlockTree`.
The test and trial sides have separate trees. The near-field matrix is zero.

The last fields describe how to allocate a workspace. To multiply, create one with
`H2Factory.H2Workspace(core, scheduling)`, or use [`fakepetrovmap`](@ref) to build both.
"""
struct FakePetrovH2Core{T,TREE,ISN,TI2I,RI2I,TO2O,RO2O,I2O,FM,MT,MS,SC} <:
       H2Factory.PetrovGalerkinH2Core{T}
    tree::TREE                                   # a BlockTree
    nearinteractions::Matrix{T}                  # Ntest × Ntrial (only size(A) reads it)
    isnear::ISN
    verbose::Bool
    α::T
    trialaggregationplan::Any
    testdisaggregationplan::Any
    testaggregationplan::Any
    trialdisaggregationplan::Any
    trialmomentcollections::Dict{Int,Any}
    testmomentcollections::Dict{Int,Any}
    # Separate transfer objects for the test and trial sides.
    testi2itranslator::TI2I
    triali2itranslator::RI2I
    testo2otranslator::TO2O
    trialo2otranslator::RO2O
    i2otranslator::I2O
    farmulmode::FM
    # --- workspace-construction metadata ---
    rank::Int
    momenttype::MT
    # See `FakeGalerkinH2Core.momentsize`: a plain `(rank,)` for the dense fakes, the
    # `(momentsize, toeplitzsize)` pair for the Toeplitz one.
    momentsize::MS
    bufferlevels::Vector{Int}
    scratch::SC
end

# Nodes any of the given plans touch (aggregation and disaggregation), unioned.
function _plannodes(aggplans, disaggplans)
    nodes = Int[]
    for plan in aggplans
        for level in H2Trees.aggregationlevels(plan)
            append!(nodes, collect(H2Trees.aggregationnodes(plan, level)))
        end
    end
    for plan in disaggplans
        for level in H2Trees.disaggregationlevels(plan)
            append!(nodes, collect(H2Trees.disaggregationnodes(plan, level)))
        end
    end
    return unique!(nodes)
end

# All levels visited by aggregation or disaggregation.
function _planlevels(aggplans, disaggplans)
    levels = Int[]
    for plan in aggplans
        append!(levels, collect(H2Trees.aggregationlevels(plan)))
    end
    for plan in disaggplans
        append!(levels, collect(H2Trees.disaggregationlevels(plan)))
    end
    return sort!(unique!(levels))
end

"""
    fakepetrovcore(testpoints, trialpoints; T=ComplexF64, rank=1, …)

Build a [`FakePetrovH2Core`](@ref) with separate test and trial trees.
The forward product aggregates on the trial side, translates to the test side,
and disaggregates on the test side.

Separate workspaces can share the core. [`fakepetrovmap`](@ref) builds a core and
workspace together.
"""
function fakepetrovcore(
    testpoints,
    trialpoints;
    T=ComplexF64,
    rank::Int=1,
    i2i=identitytranslator(),
    o2o=identitytranslator(),
    i2o=identitytranslator(),
    # Override these to test which side each traversal stage uses.
    testi2i=i2i,
    triali2i=i2i,
    testo2o=o2o,
    trialo2o=o2o,
    aggregationmode=H2Trees.AggregateMode(),
    testleafbasis=(tr, lf, r) -> _onesleafbasis(T, tr, lf, r),
    trialleafbasis=(tr, lf, r) -> _onesleafbasis(T, tr, lf, r),
    α=one(T),
    isnear=H2Trees.isnear(),
    usebuffers::Bool=true,
    minhalfsize=nothing,
    testminvalues::Int=0,
    trialminvalues::Int=0,
    MomentType=H2Factory.Moment,
    MomentCollectionType=H2Factory.MomentCollection,
    momentsize=(tr, level) -> (rank,),
    scratch=nothing,
)
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
    testtree = H2Trees.testtree(tree)
    trialtree = H2Trees.trialtree(tree)

    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; isnear=isnear, aggregationmode=aggregationmode)
    )

    trialmomentcollections = Dict{Int,Any}()
    for leaf in H2Trees.leaves(trialtree)
        n = length(H2Trees.values(trialtree, leaf))
        mc = MomentCollectionType{1,T}((rank, n))
        mc[1] .= trialleafbasis(trialtree, leaf, rank)
        trialmomentcollections[leaf] = mc
    end
    testmomentcollections = Dict{Int,Any}()
    for leaf in H2Trees.leaves(testtree)
        n = length(H2Trees.values(testtree, leaf))
        mc = MomentCollectionType{1,T}((rank, n))
        mc[1] .= testleafbasis(testtree, leaf, rank)
        testmomentcollections[leaf] = mc
    end

    # A one-chunk schedule identifies buffer levels independently of pool sizes.
    bufferlevels = _scheduledlevels(H2Factory.Schedules(plans; nchunks=1))
    # The trees can have different depths; include leaf levels from both sides.
    leaflevels = vcat(_treeleaflevels(testtree), _treeleaflevels(trialtree))
    thescratch = something(
        scratch, _bufferscratch(T, rank, bufferlevels, leaflevels, MomentType, usebuffers)
    )

    Ntest = H2Trees.numberofvalues(testtree)
    Ntrial = H2Trees.numberofvalues(trialtree)
    nearpred = isnear(tree)

    return FakePetrovH2Core{
        T,
        typeof(tree),
        typeof(nearpred),
        typeof(testi2i),
        typeof(triali2i),
        typeof(testo2o),
        typeof(trialo2o),
        typeof(i2o),
        typeof(aggregationmode),
        typeof(MomentType),
        typeof(momentsize),
        typeof(thescratch),
    }(
        tree,
        zeros(T, Ntest, Ntrial),
        nearpred,
        false,
        T(α),
        plans.trialaggregationplan,
        plans.testdisaggregationplan,
        plans.testaggregationplan,
        plans.trialdisaggregationplan,
        trialmomentcollections,
        testmomentcollections,
        testi2i,
        triali2i,
        testo2o,
        trialo2o,
        i2o,
        aggregationmode,
        rank,
        MomentType,
        momentsize,
        bufferlevels,
        thescratch,
    )
end

"""
    fakepetrovmap(testpoints, trialpoints; scheduler=DynamicScheduler(), nchunks=nothing,
                  kwargs...)

Build a core with [`fakepetrovcore`](@ref), then create a workspace that can multiply.
Other keywords are passed to the core builder.

The scheduler chooses the chunk count unless `nchunks` is supplied.
Only one multiplication can use the workspace at a time.
"""
function fakepetrovmap(
    testpoints,
    trialpoints;
    scheduler=DynamicScheduler(),
    nchunks::Union{Nothing,Int}=nothing,
    kwargs...,
)
    # See `fakegalerkinmap`: `nothing` defers the pool sizing to the scheduler's own request.
    return H2Workspace(
        fakepetrovcore(testpoints, trialpoints; kwargs...), scheduler; nchunks=nchunks
    )
end

H2Trees.isnear(A::FakePetrovH2Core, ::H2Factory.isForward) = A.isnear
function H2Factory.testmomentcollections(A::FakePetrovH2Core, ::H2Factory.isForward)
    return A.testmomentcollections
end
function H2Factory.trialmomentcollections(A::FakePetrovH2Core, ::H2Factory.isForward)
    return A.trialmomentcollections
end

# Rebuild the plan bundle expected by `Schedules` from the four stored plans.
function H2Factory.plans(A::FakePetrovH2Core)
    return (
        trialaggregationplan=A.trialaggregationplan,
        testdisaggregationplan=A.testdisaggregationplan,
        testaggregationplan=A.testaggregationplan,
        trialdisaggregationplan=A.trialdisaggregationplan,
    )
end

# Both sides' raw levels, as `_planlevels` resolves them -- a `BlockTree` side can start above
# level 1, so this is not `1:numberoflevels`.
function H2Factory.schedulinglevels(A::FakePetrovH2Core)
    return _planlevels(
        (A.trialaggregationplan, A.testaggregationplan),
        (A.testdisaggregationplan, A.trialdisaggregationplan),
    )
end

# Use the pooled allocator's block-tree path, with separate test and trial storage.
function H2Factory.buildmoments(A::FakePetrovH2Core)
    T = eltype(A)
    momentbuffers, testmoments, trialmoments = H2Factory.storagemomentsplan(
        A.testaggregationplan,
        A.trialaggregationplan,
        A.tree,                                 # the BlockTree; the allocator splits test/trial itself
        level -> A.momenttype{1,T},
        level -> A.momenttype{1,T},
        A.momentsize,
        A.momentsize,
    )

    # Check disaggregation coverage separately on each side before traversal starts.
    for node in _plannodes((A.testaggregationplan,), (A.testdisaggregationplan,))
        haskey(testmoments, node) || error(
            "storagemomentsplan produced no TEST moment for node $node, which the test-side plans " *
            "traverse; the fake's node coverage and the allocator's disagree",
        )
    end
    for node in _plannodes((A.trialaggregationplan,), (A.trialdisaggregationplan,))
        haskey(trialmoments, node) || error(
            "storagemomentsplan produced no TRIAL moment for node $node, which the trial-side plans " *
            "traverse; the fake's node coverage and the allocator's disagree",
        )
    end
    return momentbuffers, H2Factory.Moments(testmoments, trialmoments)
end

function H2Factory.buildvectorbuffers(A::FakePetrovH2Core)
    T = eltype(A)
    perleaf(tr) = Dict{Int,Vector{T}}(
        leaf => Vector{T}(undef, length(H2Trees.values(tr, leaf))) for
        leaf in H2Trees.leaves(tr)
    )
    return H2Factory.VectorBuffers(
        perleaf(H2Trees.testtree(A.tree)), perleaf(H2Trees.trialtree(A.tree))
    )
end

# Each side receives separate transfer scratch buffers.
function H2Factory.buildi2ichannels(A::FakePetrovH2Core)
    return H2Factory.SidedChannels(A.scratch.i2i(), A.scratch.i2i())
end
function H2Factory.buildo2ochannels(A::FakePetrovH2Core)
    return H2Factory.SidedChannels(A.scratch.o2o(), A.scratch.o2o())
end
H2Factory.buildi2ochannel(A::FakePetrovH2Core) = A.scratch.i2o()

# Separate leaf channels make incorrect side swaps detectable.
function H2Factory.buildleafchannels(A::FakePetrovH2Core)
    return H2Factory.SidedChannels(A.scratch.leaf(), A.scratch.leaf())
end

# These accessors select a side without changing orientation. The trait-based accessors
# above them handle orientation swaps.
H2Factory.testi2itranslator(A::FakePetrovH2Core) = A.testi2itranslator
H2Factory.triali2itranslator(A::FakePetrovH2Core) = A.triali2itranslator
H2Factory.testo2otranslator(A::FakePetrovH2Core) = A.testo2otranslator
H2Factory.trialo2otranslator(A::FakePetrovH2Core) = A.trialo2otranslator

# Conjugate leaf collections and `α` for adjoint multiplication, as in the Galerkin fixture.
function H2Factory.leafmul(::FakePetrovH2Core, ::H2Factory.isAdjoint)
    return H2Factory.ConjugatedCollectionMul()
end
H2Factory.α(A::FakePetrovH2Core, ::H2Factory.isAdjoint) = conj(A.α)
