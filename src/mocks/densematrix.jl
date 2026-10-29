# Builds a synthetic map's dense far-field matrix from `reftransfer` and `refcoupling`, without
# calling `mul!`. This provides a closed-form reference for forward, transpose, and adjoint tests.
#
# The far field is a sum over far pairs `(t = receiving, s = translating)`:
#     y[subtree(t)] += α · P_test(t) · S(t,s) · P_trial(s) · x[subtree(s)]
# where `P_trial(s)` (rank × nDOF_s) is the aggregation prolongation (`moment(s) = P_trial(s)·x_s`,
# from leaf momentcollections + I2I transfers), `P_test(t)` (nDOF_t × rank) is the disaggregation
# prolongation (from leaf momentcollections transposed + O2O transfers), and `S(t,s)` (rank × rank)
# is the I2O coupling matrix.

# rank × rank matrix that `reftransfer` applies for the (parent, child) pair.
function _transfermatrix(translator, tree, parent, child, rank, ::Type{T}) where {T}
    M = Matrix{T}(undef, rank, rank)
    e = zeros(T, rank)
    for k in 1:rank
        fill!(e, zero(T))
        e[k] = one(T)
        M[:, k] = reftransfer(translator, e, tree, parent, child)
    end
    return M
end

# rank × rank coupling matrix `refcoupling` applies for the (receiving, translating) pair.
function _couplingmatrix(translator, recv, transl, rank, ::Type{T}) where {T}
    M = Matrix{T}(undef, rank, rank)
    e = zeros(T, rank)
    for k in 1:rank
        fill!(e, zero(T))
        e[k] = one(T)
        M[:, k] = refcoupling(translator, e, recv, transl)
    end
    return M
end

# Aggregation prolongation P_trial(node): rank × nDOF_subtree, with `dofs` the column order.
# `mcs`/`transfer` are the aggregation-side momentcollections dict and I2I translator.
function _aggprolongation!(cache, mcs, transfer, tree, node, rank, ::Type{T}) where {T}
    haskey(cache, node) && return cache[node]
    if H2Trees.isleaf(tree, node)
        cache[node] = (Matrix{T}(mcs[node][1]), collect(H2Trees.values(tree, node)))
    else
        blocks = Matrix{T}[]
        dofs = Int[]
        for child in H2Trees.children(tree, node)
            Pc, dc = _aggprolongation!(cache, mcs, transfer, tree, child, rank, T)
            E = _transfermatrix(transfer, tree, node, child, rank, T)
            push!(blocks, E * Pc)
            append!(dofs, dc)
        end
        cache[node] = (reduce(hcat, blocks), dofs)
    end
    return cache[node]
end

# Disaggregation prolongation P_test(node): nDOF_subtree × rank, with `dofs` the row order.
# `mcs`/`transfer` are the disaggregation-side momentcollections dict and O2O translator.
function _disaggprolongation!(cache, mcs, transfer, tree, node, rank, ::Type{T}) where {T}
    haskey(cache, node) && return cache[node]
    if H2Trees.isleaf(tree, node)
        cache[node] = (
            Matrix{T}(transpose(mcs[node][1])), collect(H2Trees.values(tree, node))
        )
    else
        blocks = Matrix{T}[]
        dofs = Int[]
        for child in H2Trees.children(tree, node)
            Pc, dc = _disaggprolongation!(cache, mcs, transfer, tree, child, rank, T)
            E = _transfermatrix(transfer, tree, node, child, rank, T)
            push!(blocks, Pc * E)
            append!(dofs, dc)
        end
        cache[node] = (reduce(vcat, blocks), dofs)
    end
    return cache[node]
end

"""
    denseh2matrix(A) -> Matrix

Assemble the dense far-field matrix of a synthetic Galerkin or Petrov map `A`.
The calculation uses `reftransfer` and `refcoupling`, not A's `mul!` implementation.

Compare `H2Factory.farmul(A, x)` with `denseh2matrix(A) * x`. For transpose or adjoint
checks, apply the same operation to the reference matrix. A Petrov reference has shape
`(Ntest, Ntrial)`: trial columns supply the input, and test rows receive the output.

The near-field matrix is not included. The synthetic maps normally have a zero near
block; add it separately when comparing a map with a nonzero near block.
"""
function denseh2matrix(A)
    aggtree = _aggtree(A)                 # trial side (aggregation)
    disaggtree = _disaggtree(A)           # test side (disaggregation)
    aggmcs = _aggmomentcollections(A)
    disaggmcs = _disaggmomentcollections(A)
    T = eltype(A)
    Ntest = H2Trees.numberofvalues(disaggtree)
    Ntrial = H2Trees.numberofvalues(aggtree)
    rank = size(aggmcs[first(H2Trees.leaves(aggtree))][1], 1)
    F = zeros(T, Ntest, Ntrial)
    aggcache = Dict{Int,Tuple{Matrix{T},Vector{Int}}}()
    disaggcache = Dict{Int,Tuple{Matrix{T},Vector{Int}}}()
    for (recv, transl) in ref_farpairs(A)
        # Side-correct by accessor, not raw field: a shared i2i/o2o field would silently reuse
        # one side for both, and since this matrix is the oracle the traversal is checked
        # against, that would hide exactly the side-collapse bug it's meant to catch. Galerkin
        # maps collapse both accessors onto one field anyway, so nothing changes there.
        Pt, dt = _disaggprolongation!(
            disaggcache,
            disaggmcs,
            H2Factory.testo2otranslator(A),
            disaggtree,
            recv,
            rank,
            T,
        )
        Ps, ds = _aggprolongation!(
            aggcache, aggmcs, H2Factory.triali2itranslator(A), aggtree, transl, rank, T
        )
        S = _couplingmatrix(H2Factory.i2otranslator(A), recv, transl, rank, T)
        view(F, dt, ds) .+= H2Factory.α(A) .* (Pt * S * Ps)
    end
    return F
end

"""
    checkdensei2o(A; n=5, ϵ=1e-9, rng=checkrng())

Check every far pair of a synthetic dense Galerkin or Petrov map, in forward,
transpose, and adjoint modes. Each call to [`checki2otranslation`](@ref) compares the
single-pair translation with the map's dense coupling matrix.

Identity leaf collections isolate the coupling operation. Dense moments need no basis
conversion or scratch, so the check uses `EmptyChannels()`.

This checks individual pairs rather than the full traversal checked with
[`denseh2matrix`](@ref). Report assertions through `Test` and return the number of
`(pair, mode)` checks.
"""
function checkdensei2o(A; n=5, ϵ=1e-9, rng=checkrng())
    T = eltype(A)
    aggmcs = _aggmomentcollections(A)
    disaggtree = _disaggtree(A)
    rank = size(aggmcs[first(H2Trees.leaves(_aggtree(A)))][1], 1)
    idmc = H2Factory.MomentCollection1D{T}((rank, rank))
    idmc[1] .= Matrix{T}(I, rank, rank)
    buildmom() = H2Factory.Moment1D{T}((rank,))
    # Seeded (see `CHECKSEED`); reproducible because `ref_farpairs` returns an ordered Vector,
    # not a Dict iteration order.
    xgen() = randn(rng, T, rank)

    nchecks = 0
    for (recv, transl) in ref_farpairs(A)
        S = _couplingmatrix(H2Factory.i2otranslator(A), recv, transl, rank, T)
        level = H2Trees.level(disaggtree, recv)
        for (mode, ref) in (
            (H2Factory.isForward(), S),
            (H2Factory.isTranspose(), transpose(S)),
            (H2Factory.isAdjoint(), S'),
        )
            checki2otranslation(
                idmc,
                idmc,
                H2Factory.i2otranslator(A),
                H2Factory.EmptyChannels(),
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
