module TestFakeMapAdjoint

using Test
using Random
using LinearAlgebra
using StaticArrays
using H2Factory, H2Trees
using H2FactoryTestUtils

# ARE THE FAKE MAPS THEMSELVES CORRECT? Until this file, nothing here asked.
#
# This package's product IS the mocks, and its own suite tested only `_scaledmaxerror` plus the
# quality gates -- it never built a fake map. Their correctness was verified exclusively downstream,
# in H2Factory's suite, which means a change to `fakemaps.jl` could break every consumer while this
# package's gate stayed green. That is the wrong way round: a test utility should be able to vouch
# for itself.
#
# WHAT IT COST: the fakes' adjoint was missing two of its four conjugations and nothing here noticed.
# An adjoint is the transpose with EVERY factor conjugated,
#     F   = Σ transpose(MC_test) · E_test · S · E_trial · MC_trial
#     F'  = Σ conj(transpose(MC_trial)) · E' · S' · conj(MC_test)
# H2Factory supplies `E'` and `S'` itself; the LEAF COLLECTIONS and `α` are the map's, and the fakes
# supplied neither. Measured against `denseh2matrix`: a complex leaf basis was out by a relative 1.43
# and a complex `α` by 0.894, while forward and transpose stayed exact at ~6e-16 -- the signature of a
# missing conjugation, since transposing needs none.
#
# THE MAP BELOW IS DELIBERATELY MEANINGLESS. It approximates no operator and its numbers stand for
# nothing; every ingredient is complex and non-symmetric purely so that forward, transpose and adjoint
# are three DIFFERENT answers and a dropped conjugation cannot hide. That is sound because
# `denseh2matrix` assembles the fake's exact matrix from the `reftransfer`/`refcoupling` interface,
# independently of `mul!` -- so the pair is self-validating without either side having to be physical.

# Complex, and not merely a real basis stored in a complex array: with `cos` in place of `cis` every
# assertion here passes with the leaf conjugation removed.
function complexleafbasis(tree, leaf, rank)
    return ComplexF64[
        cis(0.31 * j + 0.017 * j * dof) * (1 + 0.05 * j) for
        j in 1:rank, dof in collect(H2Trees.values(tree, leaf))
    ]
end

# A complex NON-SYMMETRIC coupling, so `transpose(M)`, `adjoint(M)` and `conj(M)` are three distinct
# matrices and the transfer/coupling conjugations are exercised too, not just the leaf ones.
function skewedmatrix(rank)
    return ComplexF64[
        cis(0.7 * i - 0.23 * j) * (1 + 0.11 * i + 0.03 * j) for i in 1:rank, j in 1:rank
    ]
end

# A CONSISTENT transfer pair. See the note in the first testset: the `o2o` slot must be the transpose
# of `i2i`, because the traversal reconstructs O2O that way and only the oracle reads the slot.
_transferpair(M) = (; i2i=matrixtranslator(M), o2o=matrixtranslator(transpose(M)))

function linetree(n=64; minhalfsize=0.03)
    points = [SVector((i + 0.5) / n, 0.0, 0.0) for i in 0:(n - 1)]
    return H2Trees.buildtree(
        points;
        builder=H2Trees.TwoNTreeBuilder(; minhalfsize=minhalfsize, minvalues=0, root=1),
    ),
    n
end

const RANK = 4

@testset "Galerkin fake: everything complex, all three modes agree with the exact matrix" begin
    tree, n = linetree()
    M = skewedmatrix(RANK)
    A = fakegalerkinmap(
        tree;
        T=ComplexF64,
        rank=RANK,
        i2i=matrixtranslator(M),
        # `transpose(M)`, NOT `M`. The traversal reconstructs the O2O action as `transpose(I2I)` and
        # never reads the `o2o` slot, while `denseh2matrix` builds its disaggregation prolongation
        # from `testo2otranslator(A)` -- the slot. So the two describe the same operator only when
        # `o2o == transpose(i2i)`. Identity satisfies that trivially, which is why every existing
        # fixture gets away without stating it; passing `M` to both here made all THREE modes wrong
        # by O(1), forward included, before this comment existed.
        o2o=matrixtranslator(transpose(M)),
        i2o=matrixtranslator(skewedmatrix(RANK)),
        leafbasis=complexleafbasis,
        α=2.0 - 1.0im,
    )
    F = denseh2matrix(A)
    x = randn(MersenneTwister(3), ComplexF64, n)

    # NON-VACUITY. If the three modes were not genuinely different operators, agreeing with all three
    # would say nothing about conjugation. The thresholds are set against the tolerance the assertions
    # below actually use -- `isapprox`'s default rtol, ~1.5e-8 -- rather than against a round number.
    #
    # THE TRANSPOSE MARGIN IS THE NARROW ONE, measured at 1.9e-4 on the matrix and 2.5e-4 on the
    # matvec, and that is structural rather than a weak fixture: a GALERKIN map has the same leaf
    # basis and the same transfers on both sides, and `matrixtranslator`'s coupling is
    # pair-independent, so `F` is symmetric apart from what `S - transpose(S)` contributes. Four
    # orders above the comparison tolerance is still decisive for a forward/transpose swap. The
    # adjoint is O(1) away from both, which is the case this file exists for.
    @test norm(F - transpose(F)) / norm(F) > 1e-6
    @test norm(F - adjoint(F)) / norm(F) > 1e-1
    @test norm(transpose(F) - adjoint(F)) / norm(F) > 1e-1

    @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
    @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x
    @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x
end

@testset "Each complex ingredient on its own" begin
    # One at a time, so a failure names the factor rather than just "the adjoint is wrong". Each of
    # these is a no-op for a real value, which is exactly why the suite never reached them before.
    tree, n = linetree()
    x = randn(MersenneTwister(4), ComplexF64, n)

    # THE BASELINE IS FULLY REAL -- identity transfers and a REAL scalar coupling -- so each case
    # below turns exactly one thing complex. A complex SCALAR coupling would be a poor default here:
    # `ScalarTranslator`'s forward and transpose actions are both `c .* in`, identical, so it cannot
    # distinguish those two modes at all. The coupling case therefore uses a non-symmetric MATRIX,
    # for which `_apply`, `_applyT`, `_applyA` and `_applyC` are four different maps.
    base = (; T=ComplexF64, rank=RANK)

    for (name, extra) in (
        ("everything real", (; i2o=scalartranslator(2.0))),
        ("complex matrix coupling", (; i2o=matrixtranslator(skewedmatrix(RANK)))),
        ("complex leaf basis", (; i2o=scalartranslator(2.0), leafbasis=complexleafbasis)),
        ("complex α", (; i2o=scalartranslator(2.0), α=2.0 - 1.0im)),
        (
            "complex transfers",
            (; i2o=scalartranslator(2.0), _transferpair(skewedmatrix(RANK))...),
        ),
        (
            "all four",
            (;
                i2o=matrixtranslator(skewedmatrix(RANK)),
                leafbasis=complexleafbasis,
                α=2.0 - 1.0im,
                _transferpair(skewedmatrix(RANK))...,
            ),
        ),
    )
        A = fakegalerkinmap(tree; base..., extra...)
        F = denseh2matrix(A)
        @testset "$name" begin
            @test H2Factory.farmul!(similar(x), A, x) ≈ F * x
            @test H2Factory.farmul!(similar(x), transpose(A), x) ≈ transpose(F) * x
            @test H2Factory.farmul!(similar(x), adjoint(A), x) ≈ F' * x
        end
    end
end

@testset "Petrov fake: separate trees, everything complex" begin
    # The Petrov core has its own accessors, so it needs its own overrides -- a Galerkin-only fix
    # leaves this silently wrong in exactly the same way, which is why it is checked separately.
    u = [(i + 0.5) / 32 for i in 0:31]
    testpoints = [(p, 0.0, 0.0) for p in u]
    trialpoints = [(p + 0.5, 0.0, 0.0) for p in u]
    A = fakepetrovmap(
        testpoints,
        trialpoints;
        T=ComplexF64,
        rank=RANK,
        _transferpair(skewedmatrix(RANK))...,
        i2o=matrixtranslator(skewedmatrix(RANK)),
        testleafbasis=complexleafbasis,
        trialleafbasis=complexleafbasis,
        α=2.0 - 1.0im,
        minhalfsize=0.06,
    )
    F = denseh2matrix(A)
    x = randn(MersenneTwister(5), ComplexF64, size(A, 2))
    w = randn(MersenneTwister(6), ComplexF64, size(A, 1))

    @test norm(F - adjoint(transpose(F))) / norm(F) > 1e-3   # genuinely complex
    @test H2Factory.farmul!(similar(x, size(A, 1)), A, x) ≈ F * x
    @test H2Factory.farmul!(similar(w, size(A, 2)), transpose(A), w) ≈ transpose(F) * w
    @test H2Factory.farmul!(similar(w, size(A, 2)), adjoint(A), w) ≈ F' * w
end

@testset "The two adjoint hooks are actually selected" begin
    # The narrowest statement, so a regression points at the accessor rather than at a matvec.
    tree, _ = linetree()
    core = fakegalerkincore(tree; T=ComplexF64, rank=RANK, α=2.0 - 1.0im)
    @test H2Factory.α(core, H2Factory.isForward()) == 2.0 - 1.0im
    @test H2Factory.α(core, H2Factory.isAdjoint()) == 2.0 + 1.0im
    @test H2Factory.leafmul(core, H2Factory.isForward()) isa H2Factory.LinAlgebraMulFunctor
    @test H2Factory.leafmul(core, H2Factory.isAdjoint()) isa
        H2Factory.ConjugatedCollectionMul
end

end # module TestFakeMapAdjoint
