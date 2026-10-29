# Three-dimensional Toeplitz fixture for symmetry tests. The one-dimensional fixtures only
# represent reversal along one axis, so they cannot detect an incorrect inverse or an axis swap
# from the full cubic symmetry group.
#
# The grid has equal spacing and size on each axis, so every cubic symmetry maps the difference
# tensor to the same shape. It is centred so the node grid itself is also closed under the group.

"""
    toeplitz3dnodespacing(tree, node, nodesperaxis)

Return the spacing for a centered grid with `nodesperaxis` nodes along each axis.
The spacing is `2 * halfsize / (nodesperaxis - 1)`, so the outer nodes lie on the box faces.

The grid is symmetric about the box center. Lattice symmetries therefore map it to itself.
"""
function toeplitz3dnodespacing(tree, node, nodesperaxis::Int)
    return 2 * H2Trees.halfsize(tree, node) / (nodesperaxis - 1)
end

# Difference indices in the circulant embedding's own order: 0, 1, …, n-1, -(n-1), …, -1, so
# difference `d` sits at index `mod(d, 2n-1) + 1`. This is the convention `toeplitzmul!` multiplies
# against and the one `H2Factory.toeplitzpermutation` assumes.
_wraporder(n::Int) = vcat(0:(n - 1), (-(n - 1)):-1)

"""
    toeplitz3dcoupling(kernelfn, displacement, spacing, nodesperaxis, T) -> Matrix

Return the dense `n³ × n³` coupling for one far pair, where `n = nodesperaxis`.
Rows and columns follow the moment's column-major order over `(n, n, n)`.

For grid indices `I` and `J`, the entry is
`kernelfn(displacement + spacing * (I - J))`. This matrix is assembled independently
of H2Factory's Toeplitz construction and serves as the FFT operation's reference.
"""
function toeplitz3dcoupling(
    kernelfn, displacement, spacing, nodesperaxis::Int, ::Type{T}
) where {T}
    indices = vec(collect(CartesianIndices((nodesperaxis, nodesperaxis, nodesperaxis))))
    return T[
        kernelfn(displacement + spacing * SVector(Tuple(indices[i] - indices[j])...)) for
        i in eachindex(indices), j in eachindex(indices)
    ]
end

"""
    toeplitz3dtensor(kernelfn, displacement, spacing, nodesperaxis, T) -> Array{T,3}

Return the difference tensor used to embed the Toeplitz coupling in a circulant operation.
Its shape is `(2n-1, 2n-1, 2n-1)`, where `n = nodesperaxis`.
Entries are `κ[m] = kernelfn(displacement + spacing * m)`, stored in wrap-around order.
"""
function toeplitz3dtensor(
    kernelfn, displacement, spacing, nodesperaxis::Int, ::Type{T}
) where {T}
    m = _wraporder(nodesperaxis)
    return Array{T,3}([
        kernelfn(displacement + spacing * SVector(a, b, c)) for a in m, b in m, c in m
    ])
end

"""
    Toeplitz3DTranslator{T}(kernelfn, testtree, trialtree, nodesperaxis)

Build the FFT of [`toeplitz3dtensor`](@ref) for each translation requested by H2Factory.

The collection supplies a `direction` argument. With symmetry reduction, this is the
canonical direction shared by an orbit, not necessarily the direction between the
representative pair's node centers. The builder uses the supplied direction so the
stored tensor has the canonical orientation expected by the later permutations.

Node spacing still comes from the receiving node's box. It is uniform within a level.
"""
struct Toeplitz3DTranslator{T,K,TeT,TrT}
    kernelfn::K
    testtree::TeT
    trialtree::TrT
    nodesperaxis::Int
end

function Toeplitz3DTranslator{T}(kernelfn, testtree, trialtree, nodesperaxis::Int) where {T}
    return Toeplitz3DTranslator{T,typeof(kernelfn),typeof(testtree),typeof(trialtree)}(
        kernelfn, testtree, trialtree, nodesperaxis
    )
end

# The concrete representation type, for the same reason `_faketoeplitzrep` pins it: the ext
# constructor asks the stored operators for `toeplitzeltype`/`toeplitzsize`, which have methods
# only on the parameterized type. Same type as the 1-D fixture's (`Array{...,3}`), since that one
# is already three-dimensional in type and merely degenerate in shape.
function _toeplitz3drep(::Type{T}) where {T}
    return H2Factory.ToeplitzMatrixRepresentation{Array{Complex{real(T)},3}}
end

Base.eltype(::Toeplitz3DTranslator{T}) where {T} = Vector{_toeplitz3drep(T)}

function (b::Toeplitz3DTranslator{T})(
    receivingnode::Int, translatingnode::Int, direction
) where {T}
    H2Trees.halfsize(b.testtree, receivingnode) ≈
    H2Trees.halfsize(b.trialtree, translatingnode) || error(
        "far pair ($receivingnode,$translatingnode) has mismatched box sizes; " *
        "Toeplitz coupling needs same size",
    )
    spacing = toeplitz3dnodespacing(b.testtree, receivingnode, b.nodesperaxis)
    # `direction`, NOT `center(receiving) - center(translating)`. See the docstring: under the
    # symmetry trait those differ, and recomputing stores the representative pair's tensor instead
    # of the orbit's canonical one.
    tensor = toeplitz3dtensor(
        b.kernelfn, direction, spacing, b.nodesperaxis, Complex{real(T)}
    )
    return _toeplitz3drep(T)[H2Factory.ToeplitzMatrixRepresentation(FFTW.fft(tensor))]
end

"""
    toeplitz3dsetup(tree, kernelfn, nodesperaxis; aggregationmode=…, nchunks=…)

Build the three-dimensional version of [`toeplitzi2osetup`](@ref).
`kernelfn(z)` takes a three-vector displacement. Each far pair gets a Toeplitz coupling
and an independently assembled dense reference.

Return a `NamedTuple` with `collection`, `channel`, `couplings`, `farpairs`, `levels`,
`nodesperaxis`, and `rank`, where `rank = nodesperaxis^3`.

The collection is built through H2Factory's constructor. This also exercises its
allocation of FFT plans for each level and chunk.
"""
function toeplitz3dsetup(
    tree,
    kernelfn,
    nodesperaxis::Int;
    aggregationmode=H2Trees.AggregateMode(),
    nchunks::Int=Base.Threads.nthreads(),
)
    T = ComplexF64
    nodesperaxis >= 2 ||
        error("toeplitz3dsetup needs nodesperaxis >= 2 (got $nodesperaxis)")

    tfit = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
    aggn = H2Trees.istranslatingnode(; TranslatingNodesIterator=tfit)
    testtree, trialtree, plans = _toeplitztreeplans(tree, aggn, tfit, aggregationmode)
    schedules = H2Factory.Schedules(plans; nchunks=nchunks)

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

    couplings = Dict{Tuple{Int,Int},Matrix{T}}()
    for (recv, transl) in farpairs
        spacing = toeplitz3dnodespacing(testtree, recv, nodesperaxis)
        displacement = H2Trees.center(testtree, recv) - H2Trees.center(trialtree, transl)
        couplings[(recv, transl)] = toeplitz3dcoupling(
            kernelfn, displacement, spacing, nodesperaxis, T
        )
    end

    translator = Toeplitz3DTranslator{T}(kernelfn, testtree, trialtree, nodesperaxis)
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
        nodesperaxis=nodesperaxis,
        rank=nodesperaxis^3,
    )
end

"""
    checktoeplitz3di2o(tree, kernelfn, nodesperaxis; aggregationmode=…, n=3, ϵ=1e-9, rng=checkrng())

Check every far pair of `tree` in forward, transpose, and adjoint modes.
[`checki2otranslation`](@ref) compares the FFT operation with the dense `n³ × n³`
coupling, where `n = nodesperaxis`.

This is the three-dimensional counterpart of [`checktoeplitzi2o`](@ref): all axes have
multiple nodes. Identity leaf collections isolate the coupling operation. Report
assertions through `Test` and return the number of `(pair, mode)` checks.
"""
function checktoeplitz3di2o(
    tree,
    kernelfn,
    nodesperaxis::Int;
    aggregationmode=H2Trees.AggregateMode(),
    n=3,
    ϵ=1e-9,
    nchunks::Int=Base.Threads.nthreads(),
    rng=checkrng(),
)
    T = ComplexF64
    setup = toeplitz3dsetup(
        tree, kernelfn, nodesperaxis; aggregationmode=aggregationmode, nchunks=nchunks
    )
    rank = setup.rank
    idmc = H2Factory.ToeplitzMomentCollection1D{T}((rank, rank))
    idmc[1] .= Matrix{T}(I, rank, rank)
    buildmom() = H2Factory.ToeplitzMoment1D{T}((nodesperaxis, nodesperaxis, nodesperaxis))
    # Seeded (see `CHECKSEED`); reproducible because `setup.farpairs` is an ordered Vector, not a
    # Dict iteration order (`symmetrytoeplitz3dsetup` below does build `farpairs` from
    # `keys(::Dict)`, but drives no draws, so that ordering never reaches an RNG).
    xgen() = randn(rng, T, rank)

    nchecks = 0
    for (recv, transl) in setup.farpairs
        S = setup.couplings[(recv, transl)]
        level = setup.levels[(recv, transl)]
        for (mode, reference) in (
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
                Matrix(reference);
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

# Symmetry-reduced Toeplitz: the permutation has to reach inside each operator, not where
# `symmetryreducedcollection`'s default puts it. A Toeplitz collection stores a length-`D` vector
# of operators per pair (one per field component; `D == 1` for a scalar kernel), and the
# permutation acts on the difference tensor inside each operator (`(2n-1)³` entries), not on the
# vector. Wrapping the stored value directly would hand `PermutedTranslation` a length-1 vector
# and a 125-entry permutation -- caught, not silently applied: its length check throws a
# `DimensionMismatch`. The per-representation wrapping below is why this helper exists.

"""
    permutedtoeplitzoperators(operators, permutation)

Wrap each operator so its difference tensor is read through `permutation` without copying.
The outer vector still has one `ToeplitzMatrixRepresentation` per moment component.

The existing forward, transpose, and adjoint kernels read tensor entries through this
view, so they need no separate permutation kernel.
"""
function permutedtoeplitzoperators(operators, permutation)
    return [
        H2Factory.ToeplitzMatrixRepresentation(
            H2Factory.PermutedTranslation(operator.toeplitz, permutation)
        ) for operator in operators
    ]
end

"""
    symmetrytoeplitz3dsetup(tree, kernelfn, nodesperaxis; policy=H2Trees.FullLatticeSymmetry(),
                            useinverse=true, aggregationmode=…, nchunks=…)

Build symmetry-reduced and unreduced three-dimensional Toeplitz collections from the
same tree and builder, so they can be compared.

Return a `NamedTuple` with `reduced`, `plain`, `canonical`, `channel`, `couplings`,
`farpairs`, `levels`, `nodesperaxis`, `rank`, `stats`, and `plainstats`.
`canonical` still carries symmetry tags and must not be applied directly.

`reduced` has an outer `ToeplitzI2OTranslationCollection` and an inner
`H2Factory.SymmetryReducedI2OTranslationCollection`. The outer wrapper supplies the
Toeplitz application methods; the inner collection supplies pair lookup and its
coupling trait. The wrapper order matters.

`useinverse=false` deliberately uses the wrong permutation convention for a negative
control. Unlike the one-dimensional fixtures, this grid has permutations that differ
from their inverses, so the two conventions can be distinguished.
"""
function symmetrytoeplitz3dsetup(
    tree,
    kernelfn,
    nodesperaxis::Int;
    policy=H2Trees.FullLatticeSymmetry(),
    useinverse::Bool=true,
    aggregationmode=H2Trees.AggregateMode(),
    nchunks::Int=Base.Threads.nthreads(),
)
    T = ComplexF64
    tfit = H2Trees.TranslatingNodesIterator(; isnear=H2Trees.isnear())
    aggn = H2Trees.istranslatingnode(; TranslatingNodesIterator=tfit)
    testtree, trialtree, plans = _toeplitztreeplans(tree, aggn, tfit, aggregationmode)
    schedules = H2Factory.Schedules(plans; nchunks=nchunks)
    translator = Toeplitz3DTranslator{T}(kernelfn, testtree, trialtree, nodesperaxis)

    build(trait) = H2Factory.withschedulechunks(schedules) do
        return H2Factory.ToeplitzI2OTranslationCollection(
            tree,
            plans.trialaggregationplan,
            plans.testdisaggregationplan,
            trait,
            translator;
            i2otranslationtrait=H2Factory.ToeplitzI2OTranslationTrait(),
        )
    end

    canonical, channel = build(H2Trees.SymmetryDirectionInvariancePerLevel(policy))
    plain, _ = build(H2Trees.DirectionInvariance())

    inner = canonical.i2ocollection
    group = H2Factory.translationsymmetrygroup(inner)
    group === nothing && error(
        "symmetrytoeplitz3dsetup: the symmetry trait produced a collection with no tags.",
    )

    # One table per group element, over the STORED tensor size. `useinverse=false` asks for the
    # action of `S` where `S⁻¹` is meant, which is the control.
    L = 2 * nodesperaxis - 1
    tables = Dict(
        symmetry => H2Factory.toeplitzpermutation(
            (L, L, L), useinverse ? symmetry : H2Trees.inversesymmetry(symmetry)
        ) for symmetry in group
    )

    reduceddict = Dict(
        key => permutedtoeplitzoperators(
            inner[key], tables[H2Factory.translationsymmetry(inner, key[1], key[2])]
        ) for key in keys(inner)
    )
    reducedinner = H2Factory.SymmetryReducedI2OTranslationCollection(
        reduceddict, H2Factory.i2otrait(inner), H2Factory.translationstats(inner)
    )

    farpairs = collect(keys(inner))
    couplings = Dict{Tuple{Int,Int},Matrix{T}}()
    levels = Dict{Tuple{Int,Int},Int}()
    for (recv, transl) in farpairs
        spacing = toeplitz3dnodespacing(testtree, recv, nodesperaxis)
        displacement = H2Trees.center(testtree, recv) - H2Trees.center(trialtree, transl)
        couplings[(recv, transl)] = toeplitz3dcoupling(
            kernelfn, displacement, spacing, nodesperaxis, T
        )
        levels[(recv, transl)] = H2Trees.level(testtree, recv)
    end

    return (
        reduced=H2Factory.ToeplitzI2OTranslationCollection(reducedinner, tree),
        plain=plain,
        canonical=canonical,
        channel=channel,
        couplings=couplings,
        farpairs=farpairs,
        levels=levels,
        nodesperaxis=nodesperaxis,
        rank=nodesperaxis^3,
        stats=H2Factory.translationstats(inner),
        plainstats=H2Factory.translationstats(plain.i2ocollection),
    )
end
