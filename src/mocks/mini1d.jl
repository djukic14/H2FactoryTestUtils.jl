# Minimal 1D interpolation-based H2 matrix over a real `TwoNTree` partition. It supplies
# pair-dependent transfers, couplings, and leaf bases to the shared fake-map and reference code.
# The dense H2 approximation and exact kernel blocks are assembled independently.
#
# One parametrized implementation, `rank` = interpolation order: rank 1 is a constant basis
# (`V_t = 𝟙`, transfer `E = [1]`, coupling `S = K(c_t, c_s)`), structural validation with low
# approximation accuracy; rank 2 is a linear-interpolation basis at the interval endpoints, the
# first meaningful interpolation approximation.
# `e_impl` = ‖dense_approx·x − farmul!(A,x)‖ is machine precision for any kernel (both use the
# same `V S Vᵀ`). `e_mv`/`e_F` (vs `dense_true`) are approximation quality, tolerance-bearing.

# Geometry / interpolation (1D, points on the x-axis).

# Interval [a,b] of a node's box along x.
function _interval1d(tree, node)
    c = H2Trees.center(tree, node)[1]
    h = H2Trees.halfsize(tree, node)
    return (c - h, c + h)
end

# `rank` interpolation nodes on the node's interval: the center for rank 1, else uniform incl.
# endpoints (endpoints for rank 2 = the doc's linear basis).
function _interpnodes(tree, node, rank)
    a, b = _interval1d(tree, node)
    rank == 1 && return [(a + b) / 2]
    return collect(range(a, b; length=rank))
end

# ν-th Lagrange basis over `nodes`, evaluated at `x`. Empty product (rank 1) = 1 (constant basis).
function _lagrange(nodes, ν, x)
    r = one(float(eltype(nodes)))
    ξν = nodes[ν]
    for (μ, ξμ) in enumerate(nodes)
        μ == ν && continue
        r *= (x - ξμ) / (ξν - ξμ)
    end
    return r
end

# Leaf basis Vᵀ (rank × nDOF): [ν, j] = ℓ_ν(x_j). `coords[dof]` is the x-coord of DOF `dof`.
function _leafbasisT(tree, leaf, rank, coords)
    nodes = _interpnodes(tree, leaf, rank)
    dofs = H2Trees.values(tree, leaf)
    V = Matrix{Float64}(undef, rank, length(dofs))
    for (j, dof) in enumerate(dofs)
        for ν in 1:rank
            V[ν, j] = _lagrange(nodes, ν, coords[dof])
        end
    end
    return V
end

# Transfer E (rank × rank): [μ, ν] = ℓ_{parent,ν}(ξ_{child,μ}). Then V_parent|child = V_child·E.
function _transfer(tree, parent, child, rank)
    pnodes = _interpnodes(tree, parent, rank)
    cnodes = _interpnodes(tree, child, rank)
    E = Matrix{Float64}(undef, rank, rank)
    for μ in 1:rank
        for ν in 1:rank
            E[μ, ν] = _lagrange(pnodes, ν, cnodes[μ])
        end
    end
    return E
end

# Coupling S (rank × rank): [μ, ν] = K(ξ_{recv,μ}, ξ_{transl,ν}).
function _coupling(tree, recv, transl, rank, kernel)
    rnodes = _interpnodes(tree, recv, rank)
    snodes = _interpnodes(tree, transl, rank)
    return [kernel(rnodes[μ], snodes[ν]) for μ in 1:rank, ν in 1:rank]
end

# Pair-dependent interpolation translators. Standalone (not <: FakeTranslator): the generic
# FakeTranslator call operators apply a pair-independent `_apply`, which would be wrong here. We
# define exactly the methods H2Factory and the reference need.

abstract type MiniTranslator end

# The I2O ones separately, since only they are coupling collections: `MiniTranslator` is a marker
# with no dispatch of its own, so splitting the I2O role out costs nothing and keeps the claim
# honest -- a `MiniI2ITranslator` is a level transfer and has no business answering `i2otrait`.
#
# These reach the coupling protocol as any collection does (guarded `applycoupling!` entry, own
# `applycouplingkernel!`), so they must declare the two answers the entry asks for. A
# pair-dependent interpolation coupling still has no symmetry store, so no tag is left unconsumed.
abstract type MiniI2OTranslator <: H2Factory.AbstractI2OCouplingCollection end

H2Factory.assertnosymmetry(::MiniI2OTranslator) = nothing
H2Factory.i2otrait(::MiniI2OTranslator) = DirectCoupling()

"""Transfer a child's moment to its parent by applying `transpose(E)`."""
struct MiniI2ITranslator{TR} <: MiniTranslator
    tree::TR
    rank::Int
end
# The I2I mini's forward operator is `transpose(E)` (E = `_transfer`). The direction transforms
# it: forward applies `transpose(E)`, transpose applies `E` (reconstructing the O2O action so
# disaggregation never reads a materialized O2O), adjoint applies `conj(E)`, conjugate applies
# `adjoint(E)` -- all four equal to their unconjugated twins since the interpolation weights here
# are real. `t.tree` is the mini's own tree (a Galerkin map has one).
_minii2imatrix(E, ::H2Factory.ForwardTranslation) = transpose(E)
_minii2imatrix(E, ::H2Factory.TransposeTranslation) = E
_minii2imatrix(E, ::H2Factory.AdjointTranslation) = conj(E)
_minii2imatrix(E, ::H2Factory.ConjugateTranslation) = adjoint(E)

function H2Factory.leveltranslation!(
    out,
    in,
    t::MiniI2ITranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    E = _transfer(t.tree, parent, child, t.rank)
    out[1] .= _scratch(context.buffer, _minii2imatrix(E, direction) * in[1])
    return out
end
function H2Factory.leveltranslation!(
    out,
    in,
    t::MiniI2ITranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    E = _transfer(t.tree, parent, child, t.rank)
    out[1] .= _scratch(context.buffer, _minii2imatrix(E, direction) * in[1]) .+ out[1]
    return out
end
function reftransfer(t::MiniI2ITranslator, src, tree, parent, child)
    return transpose(_transfer(t.tree, parent, child, t.rank)) * src
end

"""Transfer a parent's moment to a child by applying `E`."""
struct MiniO2OTranslator{TR} <: MiniTranslator
    tree::TR
    rank::Int
end
# `MiniO2OTranslator` is off the traversal path: disaggregation reconstructs the O2O action from
# the i2i mini via `transpose(I2I)`. It exists only to supply the reference's transfer
# (`reftransfer`).
function reftransfer(t::MiniO2OTranslator, src, tree, parent, child)
    return _transfer(t.tree, parent, child, t.rank) * src
end

"""Couple well-separated boxes using `S = K(ξ_recv, ξ_transl)` at their interpolation nodes."""
struct MiniCouplingTranslator{TR,K} <: MiniI2OTranslator
    tree::TR
    rank::Int
    kernel::K
end
# All three directions in one pair of methods, because both things that distinguish them are already
# direction-dispatched in H2Factory:
#
#   * `couplingkey` supplies the node pair. Forward reads `S(recv, transl)`. The transpose operator is
#     `Tᵀ`, so a receiving node `a` gets from a translating node `b` via `Tᵀ[a,b] = S(b,a)ᵀ` -- the
#     coupling of the SWAPPED pair. The adjoint swaps likewise.
#   * `directedoperation` supplies the transform: `identity`, `transpose` (no conjugation, so
#     `farmul!(transpose(A))` matches `transpose(F)`), or `adjoint` (conjugating the coupling OPERATOR,
#     so `farmul!(adjoint(A))` matches `F'` with no vector conjugation).
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::MiniCouplingTranslator,
    ::DirectCoupling,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    a, b = H2Factory.couplingkey(recv, transl, direction)
    S = H2Factory.directedoperation(direction)(_coupling(t.tree, a, b, t.rank, t.kernel))
    out[1] .= _scratch(context.buffer, S * in[1])
    return out
end
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::MiniCouplingTranslator,
    ::DirectCoupling,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    a, b = H2Factory.couplingkey(recv, transl, direction)
    S = H2Factory.directedoperation(direction)(_coupling(t.tree, a, b, t.rank, t.kernel))
    out[1] .= _scratch(context.buffer, S * in[1]) .+ out[1]
    return out
end
function refcoupling(t::MiniCouplingTranslator, src, recv, transl)
    return _coupling(t.tree, recv, transl, t.rank, t.kernel) * src
end

# Explicit dense assembly (the independent oracle).

# Full basis V_node (nDOF_node × rank) with rows ordered as `dofs`, memoized. Leaf: transpose of the
# leaf basis. Non-leaf: vcat over children of V_child·E_child (nestedness V_parent|child = V_child·E).
function _fullbasis!(cache, tree, node, rank, coords)
    haskey(cache, node) && return cache[node]
    if H2Trees.isleaf(tree, node)
        V = permutedims(_leafbasisT(tree, node, rank, coords))          # nDOF × rank
        dofs = collect(H2Trees.values(tree, node))
        cache[node] = (V, dofs)
    else
        Vparts = Matrix{Float64}[]
        dofs = Int[]
        for child in H2Trees.children(tree, node)
            Vc, dc = _fullbasis!(cache, tree, child, rank, coords)
            E = _transfer(tree, node, child, rank)                       # rank × rank
            push!(Vparts, Vc * E)                                        # nDOF_child × rank
            append!(dofs, dc)
        end
        cache[node] = (reduce(vcat, Vparts), dofs)
    end
    return cache[node]
end

# Dense block H2 approximation Ã over the far partition: Σ_{far (t,s)} V_t S_{ts} V_sᵀ.
function _dense_far(A, tree, rank, coords, kernel)
    N = H2Trees.numberofvalues(tree)
    F = zeros(eltype(A), N, N)
    cache = Dict{Int,Tuple{Matrix{eltype(A)},Vector{Int}}}()
    for (recv, transl) in ref_farpairs(A)
        Vt, dt = _fullbasis!(cache, tree, recv, rank, coords)
        Vs, ds = _fullbasis!(cache, tree, transl, rank, coords)
        S = _coupling(tree, recv, transl, rank, kernel)
        view(F, dt, ds) .+= Vt * S * transpose(Vs)
    end
    return F
end

# The exact kernel restricted to the same far partition (for approximation-error reporting).
function _dense_true_far(A, tree, coords, kernel)
    N = H2Trees.numberofvalues(tree)
    F = zeros(eltype(A), N, N)
    for (recv, transl) in ref_farpairs(A)
        for i in H2Trees.values(tree, recv)
            for j in H2Trees.values(tree, transl)
                F[i, j] += kernel(coords[i], coords[j])
            end
        end
    end
    return F
end

"""
    COMPLEXKERNEL(x, y) = exp(im*(x - y)) / (1 + (x - y)^2)

Default smooth kernel for the one-dimensional examples. Its complex values make the
adjoint differ from the transpose. For a real-valued example, pass a kernel such as
`(x, y) -> 1 / (1 + (x - y)^2)`.
"""
COMPLEXKERNEL(x, y) = exp(im * (x - y)) / (1 + (x - y)^2)

"""
    minih2matrix(points; rank=2, kernel=COMPLEXKERNEL, minhalfsize=nothing,
                 root=1, aggregationmode=H2Trees.AggregateMode())

Build a small interpolation-based H2 map from one-dimensional `points` and a `TwoNTree`.
The kernel's output determines the element type of the map, buffers, and reference matrices.
The default kernel gives `ComplexF64`; a real kernel gives a real map.

Return a `NamedTuple` with:

- `A`: the workspace over a `FakeGalerkinH2Core`.
- `tree`: the `TwoNTree`.
- `points`: the coordinates as `Float64` values.
- `rank`: the interpolation order; 1 uses a constant basis, 2 a linear basis.
- `dense_approx`: the dense matrix of the H2 approximation, restricted to far interactions.
- `dense_true`: exact kernel values on the same far partition.

Use an input with the map's element type, for example `randn(eltype(A), N)`.
`H2Factory.farmul(A, x)` should match `dense_approx * x` to floating-point accuracy.
That checks the implementation. Comparing `dense_approx` with `dense_true` separately
measures the interpolation error.
"""
function minih2matrix(
    points::AbstractVector{<:Real};
    rank::Int=2,
    kernel=COMPLEXKERNEL,
    minhalfsize=nothing,
    root::Int=1,
    aggregationmode=H2Trees.AggregateMode(),
)
    coords = collect(float.(points))
    pts3d = [SVector{3,Float64}(c, 0.0, 0.0) for c in coords]
    mhs = minhalfsize === nothing ? (maximum(coords) - minimum(coords)) / 8 : minhalfsize
    tree = H2Trees.buildtree(
        pts3d; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=mhs, minvalues=0, root=root)
    )
    # Element type inferred from the kernel (real -> Float64, complex -> ComplexF64), used
    # consistently everywhere so the map, buffers, and dense matrices never disagree in type.
    T = typeof(kernel(coords[begin], coords[end]))

    i2i = MiniI2ITranslator(tree, rank)
    o2o = MiniO2OTranslator(tree, rank)
    i2o = MiniCouplingTranslator(tree, rank, kernel)
    A = fakegalerkinmap(
        tree;
        T=T,
        rank=rank,
        i2i=i2i,
        o2o=o2o,
        i2o=i2o,
        aggregationmode=aggregationmode,
        leafbasis=(tr, lf, r) -> _leafbasisT(tr, lf, r, coords),
    )

    return (
        A=A,
        tree=tree,
        points=coords,
        rank=rank,
        dense_approx=_dense_far(A, tree, rank, coords, kernel),
        dense_true=_dense_true_far(A, tree, coords, kernel),
    )
end

# LevelTranslator variant: same interpolation transfers as `minih2matrix`, but the I2I/O2O level
# transfers are supplied as H2Factory `LevelTranslator`s (sector-indexed matrices + a
# `SectorSelector`) instead of the callable `MiniI2ITranslator`/`MiniO2OTranslator`, exercising
# the level-transfer protocol through a different translator shape.
#
# The transfer for a `(level, sector)` is self-similar across edges (dyadic boxes, interpolation
# nodes on the box interval), so one representative edge per `(level, sector)` fixes the matrix.
# The `_transfer` matrix `E` is what `MiniO2OTranslator` applies forward; `MiniI2ITranslator`
# applies `transpose(E)`. So the i2i `LevelTranslator` stores `transpose(E)`, the o2o one stores
# `E`. The o2o slot is redundant (disaggregation applies `transpose(i2i)`), but both are provided
# so the fixture stays valid either way; `poisono2o` deliberately breaks the o2o slot to prove it
# is no longer read.
function _sectortransfers(tree, rank, D)
    i2i = Dict{Int,Vector{Matrix{Float64}}}()
    o2o = Dict{Int,Vector{Matrix{Float64}}}()
    for node in H2Trees.DepthFirstIterator(tree, H2Trees.root(tree))
        node == H2Trees.root(tree) && continue
        parent = H2Trees.parent(tree, node)
        level = H2Trees.level(tree, node)
        key = H2Trees.sector(tree, node) + 1
        E = Matrix{Float64}(_transfer(tree, parent, node, rank))
        i2ilvl = get!(() -> Vector{Matrix{Float64}}(undef, 2^D), i2i, level)
        o2olvl = get!(() -> Vector{Matrix{Float64}}(undef, 2^D), o2o, level)
        i2ilvl[key] = Matrix(transpose(E))
        o2olvl[key] = E
    end
    return i2i, o2o
end

"""
    minih2matrix_leveltranslator(points; rank=2, kernel=COMPLEXKERNEL, minhalfsize=nothing, root=1,
                                 aggregationmode=H2Trees.AggregateMode())

Build the same map as [`minih2matrix`](@ref), but store level transfers in H2Factory
`LevelTranslator` objects. `SectorSelector{3}` matches the three-dimensional tree that
holds the points on a line.

Return a `NamedTuple` with `A`, `tree`, `points`, `rank`, and `dense_approx`.
The tree, leaf basis, and I2O coupling
match `minih2matrix` for the same arguments. `dense_approx` comes from that matching
map and serves as the reference. `denseh2matrix` and `ref_farmul` cannot operate directly
on this version because its level translator has no `reftransfer` method.

With `poisono2o=true`, fill the separate O2O slot with `NaN`. Disaggregation should still
work: it derives the downward transfer from the I2I slot, not the O2O slot. This option
lets tests detect an accidental read of O2O. The I2I slot always keeps the correct matrices.
"""
function minih2matrix_leveltranslator(
    points::AbstractVector{<:Real};
    rank::Int=2,
    kernel=COMPLEXKERNEL,
    minhalfsize=nothing,
    root::Int=1,
    aggregationmode=H2Trees.AggregateMode(),
    poisono2o::Bool=false,
)
    # Twin callable mini: same geometry -> same transfers -> same dense matrix, used as the reference.
    twin = minih2matrix(
        points;
        rank=rank,
        kernel=kernel,
        minhalfsize=minhalfsize,
        root=root,
        aggregationmode=aggregationmode,
    )
    tree = twin.tree
    coords = twin.points
    T = typeof(kernel(coords[begin], coords[end]))

    i2imats, o2omats = _sectortransfers(tree, rank, 3)
    if poisono2o
        for lvl in values(o2omats)
            for k in eachindex(lvl)
                isassigned(lvl, k) && fill!(lvl[k], NaN)
            end
        end
    end
    i2i = H2Factory.LevelTranslator(i2imats, H2Factory.SectorSelector{3}())
    o2o = H2Factory.LevelTranslator(o2omats, H2Factory.SectorSelector{3}())
    i2o = MiniCouplingTranslator(tree, rank, kernel)   # I2O is the ordinary callable mini translator

    A = fakegalerkinmap(
        tree;
        T=T,
        rank=rank,
        i2i=i2i,
        o2o=o2o,
        i2o=i2o,
        aggregationmode=aggregationmode,
        leafbasis=(tr, lf, r) -> _leafbasisT(tr, lf, r, coords),
    )
    return (A=A, tree=tree, points=coords, rank=rank, dense_approx=twin.dense_approx)
end

function _petrovcoupling(testtree, trialtree, recv, transl, rank, kernel)
    rnodes = _interpnodes(testtree, recv, rank)
    snodes = _interpnodes(trialtree, transl, rank)
    return [kernel(rnodes[μ], snodes[ν]) for μ in 1:rank, ν in 1:rank]
end

struct PetrovMiniI2I <: MiniTranslator
    rank::Int
end

struct PetrovMiniO2O <: MiniTranslator
    rank::Int
end

struct PetrovMiniI2O{K,R} <: MiniI2OTranslator
    rank::Int
    kernel::K
    treeref::R
end

# Petrov I2I mini: like `MiniI2ITranslator` but keyed on the tree the traversal PASSES (the trial tree
# for aggregation, the test tree for disaggregation), since a Petrov map's two sides have different
# trees. Same direction-to-matrix logic (`_minii2imatrix`).
function H2Factory.leveltranslation!(
    out,
    in,
    t::PetrovMiniI2I,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    E = _transfer(tree, parent, child, t.rank)
    out[1] .= _scratch(context.buffer, _minii2imatrix(E, direction) * in[1])
    return out
end
function H2Factory.leveltranslation!(
    out,
    in,
    t::PetrovMiniI2I,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    E = _transfer(tree, parent, child, t.rank)
    out[1] .= _scratch(context.buffer, _minii2imatrix(E, direction) * in[1]) .+ out[1]
    return out
end
function reftransfer(t::PetrovMiniI2I, src, tree, parent, child)
    return transpose(_transfer(tree, parent, child, t.rank)) * src
end

# Off the traversal path (disaggregation reconstructs O2O from the Petrov i2i mini); reference-only, so
# the bare-callable application was removed. `reftransfer` remains.
function reftransfer(t::PetrovMiniO2O, src, tree, parent, child)
    return _transfer(tree, parent, child, t.rank) * src
end

function _petrovtrees(t::PetrovMiniI2O)
    tree = t.treeref[]
    return H2Trees.testtree(tree), H2Trees.trialtree(tree)
end

# As for `MiniCouplingTranslator`: `couplingkey` supplies the node pair (forward reads the pair as
# given, transpose and adjoint the SWAPPED pair) and `directedoperation` the transform (`identity`,
# `transpose`, or the operator-conjugating `adjoint`).
function _petrovdirectedcoupling(t::PetrovMiniI2O, recv::Int, transl::Int, direction)
    testtree, trialtree = _petrovtrees(t)
    a, b = H2Factory.couplingkey(recv, transl, direction)
    return H2Factory.directedoperation(direction)(
        _petrovcoupling(testtree, trialtree, a, b, t.rank, t.kernel)
    )
end

function H2Factory.applycouplingkernel!(
    out,
    in,
    t::PetrovMiniI2O,
    ::DirectCoupling,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    S = _petrovdirectedcoupling(t, recv, transl, direction)
    out[1] .= _scratch(context.buffer, S * in[1])
    return out
end
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::PetrovMiniI2O,
    ::DirectCoupling,
    recv::Int,
    transl::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    S = _petrovdirectedcoupling(t, recv, transl, direction)
    out[1] .= _scratch(context.buffer, S * in[1]) .+ out[1]
    return out
end

function refcoupling(t::PetrovMiniI2O, src, recv, transl)
    testtree, trialtree = _petrovtrees(t)
    return _petrovcoupling(testtree, trialtree, recv, transl, t.rank, t.kernel) * src
end

function _dense_true_far_petrov(A, testtree, trialtree, testcoords, trialcoords, kernel)
    T = eltype(A)
    F = zeros(T, size(A))
    for (recv, transl) in ref_farpairs(A)
        for i in H2Trees.values(testtree, recv)
            for j in H2Trees.values(trialtree, transl)
                F[i, j] += kernel(testcoords[i], trialcoords[j])
            end
        end
    end
    return F
end

"""
    minipetrovh2matrix(testpoints, trialpoints; rank=2, kernel=COMPLEXKERNEL,
                       minhalfsize, testminvalues=0, trialminvalues=0,
                       aggregationmode=H2Trees.AggregateMode())

Build the Petrov version of [`minih2matrix`](@ref), with separate test and trial trees.
Return a `NamedTuple` containing `A`, the two coordinate vectors, and the far-field
reference matrices `dense_approx` and `dense_true`.
"""
function minipetrovh2matrix(
    testpoints::AbstractVector{<:Real},
    trialpoints::AbstractVector{<:Real};
    rank::Int=2,
    kernel=COMPLEXKERNEL,
    minhalfsize::Real,
    testminvalues::Int=0,
    trialminvalues::Int=0,
    aggregationmode=H2Trees.AggregateMode(),
)
    testcoords = collect(float.(testpoints))
    trialcoords = collect(float.(trialpoints))
    T = typeof(kernel(testcoords[begin], trialcoords[end]))
    treeref = Ref{Any}()
    A = fakepetrovmap(
        [(c, 0.0, 0.0) for c in testcoords],
        [(c, 0.0, 0.0) for c in trialcoords];
        T=T,
        rank=rank,
        i2i=PetrovMiniI2I(rank),
        o2o=PetrovMiniO2O(rank),
        i2o=PetrovMiniI2O(rank, kernel, treeref),
        testleafbasis=(tr, lf, r) -> _leafbasisT(tr, lf, r, testcoords),
        trialleafbasis=(tr, lf, r) -> _leafbasisT(tr, lf, r, trialcoords),
        aggregationmode=aggregationmode,
        minhalfsize=minhalfsize,
        testminvalues=testminvalues,
        trialminvalues=trialminvalues,
    )
    treeref[] = H2Trees.tree(A)
    testtree = H2Trees.testtree(A)
    trialtree = H2Trees.trialtree(A)

    return (
        A=A,
        tree=H2Trees.tree(A),
        testpoints=testcoords,
        trialpoints=trialcoords,
        rank=rank,
        dense_approx=denseh2matrix(A),
        dense_true=_dense_true_far_petrov(
            A, testtree, trialtree, testcoords, trialcoords, kernel
        ),
    )
end
