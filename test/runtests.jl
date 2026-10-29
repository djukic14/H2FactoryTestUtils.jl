using H2FactoryTestUtils
using Test
using LinearAlgebra
using Random
using StaticArrays
using H2Trees
using H2Factory: H2Factory
include("partitionfixtures.jl")

@testset "H2FactoryTestUtils.jl" begin
    @testset "_scaledmaxerror: the α/β branch metric" begin
        # `check{near,far,}mul`'s α/β branch seeds both vectors with the exact reference and compares
        #     y   = α·Ref·x + β·(Ref·x) = (α+β)·Ref·x
        #     yH2 = α·H2·x  + β·(Ref·x)
        # so `y - yH2 = α·(Ref·x - H2·x)`. Dividing that by `maximum(abs, y)` -- which carries
        # `|α+β|` -- reports `(|α|/|α+β|)·(true error)`, and random `α+β` near zero inflated healthy
        # results into real spurious failures. `_scaledmaxerror` normalizes by `|α|·max|Ref·x|`
        # instead. These tests pin both halves of that claim: it must be α/β-independent, and it must
        # still catch a genuinely wrong α/β implementation.
        Random.seed!(11)
        n = 64
        Fx = randn(ComplexF64, n)                     # the exact reference result
        H2x = Fx .+ 1e-6 .* randn(ComplexF64, n)      # an approximation with a known error
        baseline = maximum(abs, Fx .- H2x) / maximum(abs, Fx)
        refscale = maximum(abs, Fx)

        # 1. Invariance. The reported error must equal the true approximation error for ANY α, β the
        #    helpers actually draw (`randn(ComplexF64)`, so both O(1)) -- and, critically, for the
        #    adversarial `β = -α + tiny` that drove the observed failures. If this held only for
        #    benign draws the metric would still be sampling-dependent and nothing was fixed.
        #    `α+β ≈ 0` is exact here: `y - yH2` reduces to `Fx - H2x` with nothing to cancel.
        for (α, β) in (
            (randn(ComplexF64), randn(ComplexF64)),
            (randn(ComplexF64), randn(ComplexF64)),
            (1.0 + 0im, 0.0 + 0im),
            (2.0 + 0im, 3.0 + 0im),
            (1.0 + 0im, -1.0 + 1e-12im),      # α+β ≈ 0: the pathological case, the whole point
        )
            y = (α + β) .* Fx
            yH2 = α .* H2x .+ β .* Fx
            @test H2FactoryTestUtils._scaledmaxerror(y, yH2, α, refscale) ≈ baseline rtol =
                1e-8
        end

        # 1b. Extreme |β|/|α| is a floating-point limit, NOT a flaw in the metric -- worth pinning so
        #     the distinction is not rediscovered later. The β terms cancel *exactly* only in exact
        #     arithmetic: `y` and `yH2` are both ~|β|·|Fx| while their difference is only ~|α|·error,
        #     so a ratio of 1e8 costs ~8 digits in the subtraction and no metric can recover them.
        #     The helpers never hit this (both scalars are O(1) draws); the tolerance here is loose
        #     because the arithmetic, not the formula, sets the floor.
        for (α, β) in ((1e-8 + 0im, 1.0 + 0im), (1.0 + 0im, 1e8 + 0im))
            y = (α + β) .* Fx
            yH2 = α .* H2x .+ β .* Fx
            @test H2FactoryTestUtils._scaledmaxerror(y, yH2, α, refscale) ≈ baseline rtol =
                1e-2
        end

        # 2. The old metric really was broken on this exact data -- otherwise test 1 proves only that
        #    some formula is self-consistent, not that the reported bug is gone.
        let α = 1.0 + 0im, β = -1.0 + 1e-6im
            y = (α + β) .* Fx
            yH2 = α .* H2x .+ β .* Fx
            @test H2FactoryTestUtils._maxerror(y, yH2) > 1e4 * baseline   # inflated by |α|/|α+β|
            @test H2FactoryTestUtils._scaledmaxerror(y, yH2, α, refscale) ≈ baseline rtol =
                1e-8
        end

        # 3. A wrong α/β implementation is STILL caught. This is the assertion that matters: a metric
        #    that normalized real failures away alongside the artifact would satisfy tests 1-2 too.
        #    Here the map forgets the `β·y` accumulation -- the realistic 5-arg `mul!` bug. The
        #    residual is then dominated by the whole missing `β·Ref·x` term rather than by the 1e-6
        #    approximation error, so it must land far above any sane tolerance.
        let α = randn(ComplexF64), β = randn(ComplexF64)
            y = (α + β) .* Fx
            broken = α .* H2x                                   # missing `+ β .* Fx`
            @test H2FactoryTestUtils._scaledmaxerror(y, broken, α, refscale) > 1e-3
        end

        # 4. Zero guard: a reference that maps x to nothing leaves no scale to normalize by. Report
        #    the raw gap rather than dividing by zero and handing back NaN/Inf.
        @test H2FactoryTestUtils._scaledmaxerror(
            zeros(ComplexF64, 3), zeros(ComplexF64, 3), 0.0 + 0im, 0.0
        ) == 0.0
        @test isfinite(
            H2FactoryTestUtils._scaledmaxerror(
                ones(ComplexF64, 3), zeros(ComplexF64, 3), 0.0 + 0im, 0.0
            ),
        )
    end

    @testset "Zero reference results" begin
        z = zeros(ComplexF64, 4)
        @test H2FactoryTestUtils._maxerror(z, z) == 0
        @test H2FactoryTestUtils._maxerror(z, ones(ComplexF64, 4)) == 1

        Z = zeros(ComplexF64, 4, 3)
        @test estimatenorm(Z; verbose=false) == 0
        @test estimaterelnormdifference(Z, Z; verbose=false) == 0
        @test isinf(estimaterelnormdifference(Z, ones(size(Z)); verbose=false))

        collection = H2Factory.MomentCollection{1,ComplexF64}((2, 2))
        collection[1] .= Matrix{ComplexF64}(I, 2, 2)
        buildmoment() = H2Factory.Moment{1,ComplexF64}((2,))
        coupling = matrixtranslator(zeros(ComplexF64, 2, 2))
        for mode in (H2Factory.isForward(), H2Factory.isTranspose(), H2Factory.isAdjoint())
            @test checki2otranslation(
                collection,
                collection,
                coupling,
                H2Factory.EmptyChannels(),
                1,
                2,
                1,
                buildmoment,
                buildmoment,
                () -> ones(ComplexF64, 2),
                zeros(ComplexF64, 2, 2);
                mode=mode,
                n=1,
                verbose=false,
            ) == 0
        end
    end

    @testset "Known operator norms" begin
        A = Matrix(3.0I, 4, 4)
        @test estimatenorm(A; verbose=false) ≈ 3
        @test estimatenorm(A; itmin=1, itmax=1, verbose=false) ≈ 3
        @test estimatenorm(Diagonal([1.0, 4.0]); verbose=false) ≈ 4 rtol = 1e-4
        @test estimaterelnormdifference(A, A; verbose=false) == 0
        @test estimaterelnormdifference(A, 1.25A; verbose=false) ≈ 0.25
        @test estimaterelnormdifference(A, 1.25A; itmin=1, itmax=1, verbose=false) ≈ 0.25
        @test_throws ArgumentError estimatenorm(A; itmin=0, verbose=false)
        @test_throws ArgumentError estimaterelnormdifference(A, A; itmax=0, verbose=false)
    end

    @testset "Well-separated partition" begin
        points = [SVector((i + 0.5) / 32, 0.0, 0.0) for i in 0:31]
        tree = H2Trees.buildtree(
            points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.06, minvalues=0, root=1)
        )
        @test H2FactoryTestUtils.testwellseparatedness(tree)

        A = fakepetrovmap(points, points .+ Ref(SVector(0.5, 0.0, 0.0)); minhalfsize=0.06)
        @test H2FactoryTestUtils.testwellseparatedness(H2Trees.tree(A))
        for (near, far, valid) in (
            (true, false, true),
            (false, true, true),
            (false, false, false),
            (true, true, false),
        )
            fixture = PartitionFixtures.Tree(near, far)
            @test H2FactoryTestUtils.testwellseparatedness(fixture) == valid
            block = PartitionFixtures.BlockTree(fixture, fixture)
            @test H2FactoryTestUtils.testwellseparatedness(block) == valid
        end
    end

    @testset "Petrov symmetry setup rank" begin
        points = [SVector((i + 0.5) / 32, 0.0, 0.0) for i in 0:31]
        setup = petrovsymmetrysetup(
            points, points; directions=symmetrydirections()[1:6], minhalfsize=0.06
        )
        x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, length(points))
        for reduced in (false, true)
            A = fakepetrovsymmetrymap(
                points, points; setup=setup, reduced=reduced, minhalfsize=0.06
            )
            @test H2Factory.core(A).rank == setup.rank == 6
            reference = denseh2matrix(A)
            for operation in (identity, transpose, adjoint)
                @test H2Factory.farmul(operation(A), x) ≈ operation(reference) * x rtol =
                    1e-12
            end
        end
    end

    # The mocks are this package's product, so its own suite must exercise them directly.
    @testset "Fake maps: adjoint conjugations" begin
        include("fakemapadjoint.jl")
    end

    @testset "Code quality (Aqua.jl)" begin
        using Aqua
        # `persistent_tasks`: this check spins up a throwaway environment and re-resolves
        # H2FactoryTestUtils's dependencies from scratch, which fails because H2Factory is
        # unregistered. Not testable until H2Factory is published or registered.
        #
        # `deps_compat` no longer needs its `ignore=[:H2Factory]`: H2Factory now HAS a compat entry
        # (`"0.1"`, see Project.toml). The reason it lacked one was a measurement error, recorded
        # there: a normal compat entry DOES match a prerelease of its lower bound, so such an entry
        # was never the resolution hazard the old comment claimed. Being unregistered is a real and
        # separate constraint, which is why `persistent_tasks` stays off.
        Aqua.test_all(H2FactoryTestUtils; persistent_tasks=false)
    end

    @testset "Code formatting (JuliaFormatter.jl)" begin
        using JuliaFormatter
        @test JuliaFormatter.format(pkgdir(H2FactoryTestUtils); overwrite=false)
    end
end
