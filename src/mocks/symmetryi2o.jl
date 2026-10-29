# Symmetry-reduced far-field I2O for the synthetic maps.
#
# One translation is stored per symmetry orbit. Each interaction carries the symmetry that maps
# the stored direction to the required direction. This fixture applies that tag through complete
# forward, transpose, and adjoint traversals.
#
# The miniature representation uses 14 directions closed under the lattice group: the six face
# directions and eight cube corners of `Z^3`. The coupling for a lattice offset `q` is
# `d_q[j] = f(k̂_j ⋅ q)`; since every lattice
# symmetry is orthogonal, `k̂ ⋅ (S q) == (S⁻¹ k̂) ⋅ q`, so `d_{S q}[j] = d_q[π(j)]` where
# `k̂_{π(j)} == S⁻¹ k̂_j`. This is the structure represented by
# `H2Factory.PermutedTranslation`. A dropped inverse, level mix-up, or swapped coupling key gives
# a different result.
#
# Here `k̂ ⋅ q` is an integer dot product, so the reduced and unreduced couplings agree exactly.
#
# `symmetryfrequencyimages` probes FFT permutations on this irregular sample grid.

# Include two symmetry orbits so tests exercise more than the face directions.
"""
    symmetrydirections()

Return the six face directions and eight corner directions of the integer cube.
"""
function symmetrydirections()
    faces = [
        SVector{3,Int}(1, 0, 0),
        SVector{3,Int}(-1, 0, 0),
        SVector{3,Int}(0, 1, 0),
        SVector{3,Int}(0, -1, 0),
        SVector{3,Int}(0, 0, 1),
        SVector{3,Int}(0, 0, -1),
    ]
    corners = [SVector{3,Int}(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    return vcat(faces, corners)
end

# The scalar the coupling is built from: complex and not even in `n`, both deliberately -- a
# real `f` would make transpose and adjoint coincide, and an even one would make the antipodal
# symmetry invisible, the two ways this test could look green while checking nothing.
# `|f(n)| = |1.3 + 0.17n|` is injective on the integers (a collision needs `n + m = -1300/85`,
# not an integer), so a wrong permutation is a numerically visible wrong answer, not a coincidence.
"""
    symmetrykernelvalue(n)

Return the complex scalar used by the symmetry fixtures for lattice index `n`.
"""
symmetrykernelvalue(n::Integer) = cis(0.4 * n) * (1.3 + 0.17 * n)

"""
    SymmetryDiagonalTranslator(tree, directions)

Build the diagonal coupling `d[j] = f(k̂_j ⋅ q)` over `directions`.
`q` expresses the supplied translation direction in units of the receiving node's halfsize.

The builder provides `eltype` and a call method taking `(receiving, translating, direction)`.
The same builder works for plain and symmetry-reduced collections. A plain collection
supplies the pair's direction; a reduced collection supplies the orbit's canonical direction.
"""
struct SymmetryDiagonalTranslator{TREE,D}
    tree::TREE
    directions::D

    # Node IDs and halfsizes belong to the receiving tree, the test side of a block tree.
    function SymmetryDiagonalTranslator(tree, directions::D) where {D}
        receivetree = symmetryreceivetree(tree)
        return new{typeof(receivetree),D}(receivetree, directions)
    end
end

symmetryreceivetree(tree) = tree
symmetryreceivetree(tree::H2Trees.BlockTree) = H2Trees.testtree(tree)

Base.eltype(::SymmetryDiagonalTranslator) = Vector{ComplexF64}

function (translator::SymmetryDiagonalTranslator)(
    receivingnode::Int, translatingnode::Int, direction
)
    q = latticeoffset(translator.tree, receivingnode, direction)
    return [symmetrykernelvalue(dot(k, q)) for k in translator.directions]
end

# Express the direction in halfsize units. Reject noninteger offsets rather than round silently.
function latticeoffset(tree, receivingnode::Int, direction)
    halfsize = H2Trees.halfsize(tree, receivingnode)
    return ntuple(3) do i
        scaled = direction[i] / halfsize
        rounded = round(Int, scaled)
        isapprox(scaled, rounded; atol=1e-8) || error(
            "SymmetryDiagonalTranslator: component $i of the translation direction is " *
            "$(scaled) halfsizes, which is not an integer. The lattice symmetry reduction only " *
            "applies when every displacement is a lattice vector.",
        )
        return rounded
    end
end

"""
    symmetrypermutation(directions, symmetry; useinverse=true) -> Vector{Int}

Return the permutation that applies `symmetry` to the sampled directions.
`permutation[j]` is the position of `S⁻¹ k̂_j`, so
`d_actual[j] == d_canonical[permutation[j]]`.

`H2Factory.translationsymmetry` maps the canonical translation direction to the actual
one. The sampled directions use the inverse because `T(k̂, S d) == T(S⁻¹ k̂, d)`.
`useinverse=false` deliberately builds the wrong forward convention for a negative control.
"""
function symmetrypermutation(directions, symmetry; useinverse::Bool=true)
    applied = useinverse ? H2Trees.inversesymmetry(symmetry) : symmetry
    position = Dict(direction => j for (j, direction) in enumerate(directions))
    return [
        position[SVector{3,Int}(H2Trees.applysymmetry(applied, Tuple(direction)))] for
        direction in directions
    ]
end

"""
    FakeSymmetryI2O(collection, couplings)

Apply the diagonal couplings in `collection` and record each call.
The reference method [`refcoupling`](@ref) reads the separate, unreduced diagonals in
`couplings`, so it does not depend on the reduced collection's permutations.
"""
struct FakeSymmetryI2O{C,K} <: H2Factory.AbstractI2OCouplingCollection
    collection::C
    couplings::K
    log::Vector{SpyRecord}
    lock::ReentrantLock
end

"""
    fakesymmetryi2o(collection, couplings)

Return a [`FakeSymmetryI2O`](@ref) with `collection`, unreduced reference `couplings`,
and an empty call log.
"""
function fakesymmetryi2o(collection, couplings)
    return FakeSymmetryI2O(collection, couplings, SpyRecord[], ReentrantLock())
end

_recordi2o!(w::FakeSymmetryI2O, role, a, b) = lock(w.lock) do
    return push!(w.log, SpyRecord(role, a, b))
end
records(w::FakeSymmetryI2O) = lock(w.lock) do
    return copy(w.log)
end

# Forward the symmetry guard; wrapping a collection must not bypass its refusal.
H2Factory.assertnosymmetry(w::FakeSymmetryI2O) = H2Factory.assertnosymmetry(w.collection)

# Records the call, then forwards to the wrapped collection's KERNEL. Reached only when the guard above
# let the pair through.
function H2Factory.applycouplingkernel!(
    out,
    in,
    w::FakeSymmetryI2O,
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
H2Factory.i2otrait(w::FakeSymmetryI2O) = H2Factory.i2otrait(w.collection)
Base.getindex(w::FakeSymmetryI2O, key) = w.collection[key]

# Reference coupling: the EXACT diagonal for this pair, never the orbit's. `src` is a length-rank
# vector; the coupling is diagonal, so this is a broadcast rather than a `mul!`.
function refcoupling(w::FakeSymmetryI2O, src, receiving, translating)
    return w.couplings[(receiving, translating)] .* src
end

"""
    symmetryleafbasis(tree, leaf, rank) -> Matrix{ComplexF64}

Return a fixed `rank × nDOF` leaf basis with real values stored as complex numbers.
Entries vary by row and degree of freedom so tests can detect an incorrect coupling
permutation. An all-ones basis would give every diagonal entry the same weight.
"""
function symmetryleafbasis(tree, leaf, rank)
    dofs = collect(H2Trees.values(tree, leaf))
    basis = Matrix{ComplexF64}(undef, rank, length(dofs))
    for (column, dof) in enumerate(dofs), j in 1:rank
        basis[j, column] = cos(0.31 * j + 0.017 * j * dof) * (1 + 0.05 * j)
    end
    return basis
end

"""
    symmetrytranslatingplan(tree; isnear=H2Trees.isnear(), aggregationmode=H2Trees.AggregateMode())

Build the plan that carries I2O work, using the same options as [`fakegalerkinmap`](@ref).
This lets the collection and map use the same interaction partition.
"""
function symmetrytranslatingplan(
    tree; isnear=H2Trees.isnear(), aggregationmode=H2Trees.AggregateMode()
)
    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; isnear=isnear, aggregationmode=aggregationmode)
    )
    return H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)
end

"""
    symmetryi2osetup(tree, translatingplan; policy=H2Trees.FullLatticeSymmetry(),
                     directions=symmetrydirections(), useinverse=true)

Build plain and symmetry-reduced collections over one tree, using the same
[`SymmetryDiagonalTranslator`](@ref).

Return a `NamedTuple` with:

- `plain`: one diagonal per direction, using `DirectionInvariancePerLevel`.
- `canonical`: one diagonal per symmetry orbit, with a side table of symmetry tags.
  Applying this collection directly is an error because the tags have not been handled.
- `reduced`: the canonical diagonals wrapped for each pair by
  `H2Factory.symmetryreducedcollection`. This is the collection the map applies.
- `couplings`: the plain per-pair diagonals, used as the reference.
- `directions`, `rank`, and `policy`: the construction settings.
- `stats` and `plainstats`: statistics for the canonical and plain collections.

`reduced` is a `H2Factory.SymmetryReducedI2OTranslationCollection`. Each stored value
already applies its pair's permutation, so no symmetry tags remain for the caller to
handle. Its `TranslationStats` still reports the canonical construction's orbit count.

`useinverse=false` deliberately uses `S` instead of `S⁻¹` for the negative control
described in [`symmetrypermutation`](@ref).
"""
function symmetryi2osetup(
    tree,
    translatingplan;
    policy=H2Trees.FullLatticeSymmetry(),
    directions=symmetrydirections(),
    useinverse::Bool=true,
    verbose::Bool=false,
)
    builder = SymmetryDiagonalTranslator(tree, directions)

    # Build the tagged collection first, so its lattice guard runs before the fixture's own check.
    canonical, _ = H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        H2Trees.SymmetryDirectionInvariancePerLevel(policy),
        builder;
        verbose=verbose,
    )
    plain, _ = H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        H2Trees.DirectionInvariancePerLevel(),
        builder;
        verbose=verbose,
    )

    group = H2Factory.translationsymmetrygroup(canonical)
    group === nothing && error(
        "symmetryi2osetup: the symmetry trait produced a collection with no symmetry tags, so " *
        "there is nothing to permute and the reduced collection would silently be the canonical " *
        "one.",
    )

    # One table per group element, shared by every pair carrying that element -- the group order,
    # not the interaction count. The direction set is fixed across levels; that does not affect
    # the permutation convention being tested.
    tables = [symmetrypermutation(directions, s; useinverse=useinverse) for s in group]
    idbyelement = Dict(symmetry => id for (id, symmetry) in enumerate(group))

    # The direction set is fixed, so permutation tables do not depend on the receiving node.
    reduced = H2Factory.symmetryreducedcollection(
        canonical, (receivingnode, symmetry) -> tables[idbyelement[symmetry]]
    )

    return (
        plain=plain,
        canonical=canonical,
        reduced=reduced,
        couplings=Dict(key => plain[key] for key in keys(plain)),
        directions=directions,
        rank=length(directions),
        policy=policy,
        stats=H2Factory.translationstats(canonical),
        plainstats=H2Factory.translationstats(plain),
    )
end

"""
    fakesymmetrymap(tree; setup=nothing, reduced=true, policy=H2Trees.FullLatticeSymmetry(),
                    useinverse=true, kwargs...)

Build a [`fakegalerkinmap`](@ref) with the diagonal I2O couplings from
[`symmetryi2osetup`](@ref). The direction set determines `rank`; it cannot be overridden.

By default, use the reduced collection. `reduced=false` uses the plain collection of
the same couplings. Both should give the same far-field matrix as `denseh2matrix`.
`useinverse=false` selects the deliberately wrong permutation convention.

Pass a prebuilt `setup` to share couplings and construction statistics across maps.
Otherwise the function builds one using the map's `isnear` and `aggregationmode` settings.
"""
function fakesymmetrymap(
    tree;
    setup=nothing,
    reduced::Bool=true,
    policy=H2Trees.FullLatticeSymmetry(),
    directions=symmetrydirections(),
    useinverse::Bool=true,
    aggregationmode=H2Trees.AggregateMode(),
    isnear=H2Trees.isnear(),
    # Not the fakes' all-ones default: that would make every moment constant along the rank index,
    # so a diagonal coupling contributes only its sum and every permutation is indistinguishable.
    # See `symmetryleafbasis`.
    leafbasis=symmetryleafbasis,
    kwargs...,
)
    if setup === nothing
        setup = symmetryi2osetup(
            tree,
            symmetrytranslatingplan(tree; isnear=isnear, aggregationmode=aggregationmode);
            policy=policy,
            directions=directions,
            useinverse=useinverse,
        )
    end
    return fakegalerkinmap(
        tree;
        T=ComplexF64,
        rank=setup.rank,
        i2o=fakesymmetryi2o(reduced ? setup.reduced : setup.plain, setup.couplings),
        aggregationmode=aggregationmode,
        isnear=isnear,
        leafbasis=leafbasis,
        kwargs...,
    )
end

"""
    petrovsymmetrysetup(testpoints, trialpoints; policy, directions, useinverse, kwargs...)

Build a [`symmetryi2osetup`](@ref) for the `BlockTree` that
[`fakepetrovsymmetrymap`](@ref) creates with the same arguments.
Reduced and unreduced maps can share this setup, including its couplings and statistics.

The two roots must have an offset allowed by the translation lattice. Otherwise the
function throws `H2Trees.NonLatticeTranslationError`. This depends on the built boxes,
not just the point coordinates: root halfsizes are rounded to power-of-two multiples
of `minhalfsize`.
"""
function petrovsymmetrysetup(
    testpoints,
    trialpoints;
    policy=H2Trees.FullLatticeSymmetry(),
    directions=symmetrydirections(),
    useinverse::Bool=true,
    aggregationmode=H2Trees.AggregateMode(),
    isnear=H2Trees.isnear(),
    kwargs...,
)
    probe = fakepetrovcore(
        testpoints,
        trialpoints;
        T=ComplexF64,
        rank=length(directions),
        aggregationmode=aggregationmode,
        isnear=isnear,
        kwargs...,
    )
    # `probe.tree`, not `H2Trees.tree(probe)`: a core is deliberately neither an `AbstractH2Map`
    # nor a `GalerkinH2Map`, so the generic accessor doesn't reach it. The tree comes from a core
    # built exactly as the map's will be, rather than a second `BlockTreeBuilder` call here
    # repeating the same recipe -- the "two independent sources assumed to agree" trap.
    return symmetryi2osetup(
        probe.tree,
        H2Trees.translatingplan(probe.trialaggregationplan, probe.testdisaggregationplan);
        policy=policy,
        directions=directions,
        useinverse=useinverse,
    )
end

"""
    fakepetrovsymmetrymap(testpoints, trialpoints; setup=nothing, reduced=true, minhalfsize, …)

Build the Petrov version of [`fakesymmetrymap`](@ref), with separate test and trial trees.
The setup requires lattice-aligned translations; off-lattice roots raise
`H2Trees.NonLatticeTranslationError`. Pass a [`petrovsymmetrysetup`](@ref) result to share
the same couplings between reduced and unreduced maps.
"""
function fakepetrovsymmetrymap(
    testpoints,
    trialpoints;
    setup=nothing,
    reduced::Bool=true,
    policy=H2Trees.FullLatticeSymmetry(),
    directions=symmetrydirections(),
    useinverse::Bool=true,
    aggregationmode=H2Trees.AggregateMode(),
    isnear=H2Trees.isnear(),
    testleafbasis=symmetryleafbasis,
    trialleafbasis=symmetryleafbasis,
    kwargs...,
)
    if setup === nothing
        setup = petrovsymmetrysetup(
            testpoints,
            trialpoints;
            policy=policy,
            directions=directions,
            useinverse=useinverse,
            aggregationmode=aggregationmode,
            isnear=isnear,
            kwargs...,
        )
    end

    return fakepetrovmap(
        testpoints,
        trialpoints;
        T=ComplexF64,
        rank=setup.rank,
        i2o=fakesymmetryi2o(reduced ? setup.reduced : setup.plain, setup.couplings),
        aggregationmode=aggregationmode,
        isnear=isnear,
        testleafbasis=testleafbasis,
        trialleafbasis=trialleafbasis,
        kwargs...,
    )
end

"""
    symmetrytoeplitzcollection(tree, translatingplan; policy=H2Trees.FullLatticeSymmetry(), …)

Build a collection with symmetry tags and `H2Factory.ToeplitzI2OTranslationTrait`.
The tags have not been applied, so the translation guard should reject this collection.

This fixture tests rejection of unhandled tags. It does not imply that Toeplitz
couplings cannot use symmetry reduction.
"""
function symmetrytoeplitzcollection(
    tree,
    translatingplan;
    policy=H2Trees.FullLatticeSymmetry(),
    directions=symmetrydirections(),
    verbose::Bool=false,
)
    collection, _ = H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        H2Trees.SymmetryDirectionInvariancePerLevel(policy),
        SymmetryDiagonalTranslator(tree, directions);
        i2otranslationtrait=H2Factory.ToeplitzI2OTranslationTrait(),
        verbose=verbose,
    )
    return collection
end

"""
    symmetryfrequencyimages(symmetry, q; directions=symmetrydirections())

Sample the kernel at `q` and its symmetry image, then compute both FFTs.
Return the samples and spectra, along with sorted samples and sorted FFT magnitudes.

Sorting removes the effect of ordering. If the sorted FFT magnitudes differ, no
permutation of frequency entries can relate those two spectra.

This check uses an irregular direction set, not the regular grid used by Toeplitz
couplings. Some cases, including the identity symmetry, have equal spectra.
"""
function symmetryfrequencyimages(symmetry, q; directions=symmetrydirections())
    qimage = H2Trees.applysymmetry(symmetry, Tuple(q))
    sample = [symmetrykernelvalue(dot(k, Tuple(q))) for k in directions]
    sampleimage = [symmetrykernelvalue(dot(k, qimage)) for k in directions]
    spectrum = FFTW.fft(sample)
    spectrumimage = FFTW.fft(sampleimage)
    return (
        sample=sample,
        sampleimage=sampleimage,
        spectrum=spectrum,
        spectrumimage=spectrumimage,
        samplemultiset=sort(sample; by=x -> (real(x), imag(x))),
        imagemultiset=sort(sampleimage; by=x -> (real(x), imag(x))),
        spectrummultiset=sort(abs.(spectrum)),
        spectrumimagemultiset=sort(abs.(spectrumimage)),
    )
end
