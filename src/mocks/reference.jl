# Single-threaded, buffer-free reference for aggregate, translate, and disaggregate.
#
# Forward far-field only -- transpose/adjoint duality is checked reference-free via inner
# products. The reference drives the same partition as the map under test (reads its plans, leaf
# momentcollections, and translators), so a divergence isolates to H2Factory's traversal, not to
# translator arithmetic or plan enumeration (covered separately by the mini's dense Ã and the spy
# structural checks).
#
# It shares translator arithmetic and plan enumeration with the tested map, but implements the
# traversal and accumulation independently.

# The reference applies a translator as a pure function of the source moment vector, defaulting
# to the same linear map `FakeTranslator` applies in the forward direction. A fixture with a
# pair-dependent translator (e.g. `mini1d.jl`'s per-pair interpolation/coupling) overrides these
# two methods for its own translator type.
"""
    reftransfer(translator, src, tree, parent, child)

Apply a level transfer to `src` using reference arithmetic, without H2Factory's
scratch buffers or schedulers. The reference tree passes use this method for I2I and O2O.

The default for `FakeTranslator` applies its forward linear map. This covers
[`identitytranslator`](@ref), [`scalartranslator`](@ref), and [`matrixtranslator`](@ref).
A translator whose matrix depends on the parent/child pair supplies its own method.
"""
reftransfer(t::FakeTranslator, src, tree, parent, child) = _apply(t, src)

"""
    refcoupling(translator, src, receiving, translating)

Apply an I2O coupling to `src` using reference arithmetic.
This is the coupling counterpart of [`reftransfer`](@ref).

The default applies the translator's forward linear map. A translator whose matrix
depends on the source/receiver pair supplies its own method.
"""
refcoupling(t::FakeTranslator, src, receiving, translating) = _apply(t, src)
# A spy delegates to its inner translator without recording (the reference is not being observed).
function reftransfer(t::SpyTranslator, src, tree, parent, child)
    return reftransfer(t.inner, src, tree, parent, child)
end
function refcoupling(t::SpyTranslator, src, receiving, translating)
    return refcoupling(t.inner, src, receiving, translating)
end

# Galerkin vs Petrov: aggregation runs on the trial side, disaggregation on the test side. For a
# Galerkin map both are the single shared tree/momentcollections; for Petrov they're the two
# `BlockTree` sides with separate momentcollections. Resolved through H2Factory's own accessors
# (not the concrete fake types' fields), so these helpers work on any map meeting the contract.
_aggtree(A) = H2Trees.trialtree(A)
_disaggtree(A) = H2Trees.testtree(A)
_aggmomentcollections(A) = H2Factory.trialmomentcollections(A)
_disaggmomentcollections(A) = H2Factory.testmomentcollections(A)

"""
    ref_aggregate(A, x) -> Dict{Int,Vector}

Compute reference moments for the upward pass on the trial tree.
Return one vector per node visited by `trialaggregationplan`.

At a leaf, multiply its collection by the corresponding entries of `x`. At a parent,
transfer each child's moment with I2I and add the results. No prefactor `α` or
conjugation is applied.
"""
function ref_aggregate(A, x)
    tree = _aggtree(A)
    mcs = _aggmomentcollections(A)
    plan = H2Factory.trialaggregationplan(A)
    T = eltype(A)
    agg = Dict{Int,Vector{T}}()
    for level in H2Trees.aggregationlevels(plan)
        for node in H2Trees.aggregationnodes(plan, level)
            if H2Trees.isleaf(tree, node)
                mc1 = mcs[node][1]
                xsub = Vector{T}(x[H2Trees.values(tree, node)])
                agg[node] = mc1 * xsub
            else
                acc = nothing
                for child in H2Trees.children(tree, node)
                    # Trial-side transfer: forward aggregation runs on the trial side. Via the
                    # accessor (not raw field) so a Petrov fake resolves the correct side.
                    contrib = reftransfer(
                        H2Factory.triali2itranslator(A), agg[child], tree, node, child
                    )
                    acc = acc === nothing ? copy(contrib) : acc .+ contrib
                end
                agg[node] = acc
            end
        end
    end
    return agg
end

"""
    ref_farpairs(A) -> Vector{Tuple{Int,Int}}

Return the far-field pairs in `(receiving, translating)` order.

Read them from `DisaggregateTranslatePlan` in `AggregateMode`, or from
`AggregateTranslatePlan` in `AggregateTranslateMode`.
"""
function ref_farpairs(A)
    pairs = Tuple{Int,Int}[]
    if H2Factory.testdisaggregationplan(A) isa H2Trees.DisaggregateTranslatePlan
        plan = H2Factory.testdisaggregationplan(A)
        for level in H2Trees.disaggregationlevels(plan)
            for node in H2Trees.disaggregationnodes(plan, level)
                for tn in H2Trees.translatingnodes(plan, node)
                    push!(pairs, (node, tn))
                end
            end
        end
    elseif H2Factory.trialaggregationplan(A) isa H2Trees.AggregateTranslatePlan
        plan = H2Factory.trialaggregationplan(A)
        for level in H2Trees.aggregationlevels(plan)
            for rn in H2Trees.receivingnodes(plan, level)
                for tn in H2Trees.translatingnodes(plan, rn, level)
                    push!(pairs, (rn, tn))
                end
            end
        end
    else
        error("ref_farpairs: no recognized translation plan on the map")
    end
    return pairs
end

"""
    ref_translate(A, agg) -> Dict{Int,Vector}

Apply the reference I2O couplings to the aggregation moments in `agg`.
For each far pair, add the translated source moment to the receiving test node's moment.

Return the receiving moments as a dictionary. A node with no incoming far pair has no entry.
"""
function ref_translate(A, agg)
    T = eltype(A)
    disagg = Dict{Int,Vector{T}}()
    for (recv, transl) in ref_farpairs(A)
        contrib = refcoupling(H2Factory.i2otranslator(A), agg[transl], recv, transl)
        disagg[recv] = haskey(disagg, recv) ? disagg[recv] .+ contrib : copy(contrib)
    end
    return disagg
end

"""
    ref_disaggregate(A, disagg) -> y

Run the reference downward pass and return the output vector.
Transfer each parent's moment to its children with O2O, visiting parents before children.
At a leaf, write `y_leaf = α * transpose(mc) * moment`. A leaf with no received moment
contributes zero.

The function updates `disagg` in place as it propagates moments.
"""
function ref_disaggregate(A, disagg)
    tree = _disaggtree(A)
    mcs = _disaggmomentcollections(A)
    plan = H2Factory.testdisaggregationplan(A)
    T = eltype(A)
    N = H2Trees.numberofvalues(tree)
    y = zeros(T, N)
    rootnode = H2Trees.root(tree)

    # Precomputed rather than calling `isdisaggregationnode`, which exists on a
    # `DisaggregateTranslatePlan` (AggregateMode) but not a plain `DisaggregatePlan`
    # (AggregateTranslateMode); node membership is well-defined for both.
    disaggset = Set{Int}()
    for level in H2Trees.disaggregationlevels(plan)
        for node in H2Trees.disaggregationnodes(plan, level)
            push!(disaggset, node)
        end
    end

    for level in H2Trees.disaggregationlevels(plan)
        for node in H2Trees.disaggregationnodes(plan, level)
            if node != rootnode
                parent = H2Trees.parent(tree, node)
                # Pull the parent's (now-finalized, top-down) O2O contribution into this node.
                # `haskey` reproduces the storenode accumulate-vs-overwrite split: a node with
                # existing (I2O) content accumulates, one without starts fresh from the parent.
                if parent in disaggset && haskey(disagg, parent)
                    # Test-side transfer: forward disaggregation runs on the test side.
                    contrib = reftransfer(
                        H2Factory.testo2otranslator(A), disagg[parent], tree, parent, node
                    )
                    disagg[node] =
                        haskey(disagg, node) ? disagg[node] .+ contrib : copy(contrib)
                end
            end
            if H2Trees.isleaf(tree, node) && haskey(disagg, node)
                mc1 = mcs[node][1]
                y[H2Trees.values(tree, node)] .=
                    H2Factory.α(A) .* (transpose(mc1) * disagg[node])
            end
        end
    end
    return y
end

"""
    ref_farmul(A, x) -> y

Compute the reference far-field product by calling [`ref_aggregate`](@ref),
[`ref_translate`](@ref), and [`ref_disaggregate`](@ref) in that order.
The result should agree with `H2Factory.farmul(A, x)` to floating-point tolerance.
"""
function ref_farmul(A, x)
    agg = ref_aggregate(A, x)
    disagg = ref_translate(A, agg)
    return ref_disaggregate(A, disagg)
end
