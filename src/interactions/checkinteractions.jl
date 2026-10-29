"""
    checkinteractions(A, A_H2; n=10, ϵ=1e-5, T=eltype(A_H2), verbose=true,
                       mul=true, prod=true, nearmul=true, farmul=true)

Compare the H2 map `A_H2` with the dense reference matrix `A`.

First check their sizes and near/far partition. Then check the matrix-vector products.
Set `mul`, `prod`, `nearmul`, or `farmul` to `false` to skip that product check.
The function reports failures through `Test` and returns `true` after the checks.
"""
function checkinteractions(
    A,
    A_H2;
    n=10,
    ϵ=1e-5,
    T::DataType=eltype(A_H2),
    verbose=true,
    mul=true,
    prod=true,
    nearmul=true,
    farmul=true,
)
    @test size(A) == size(A_H2)

    Anear = H2Factory.nearinteractions(A, A_H2)
    @test Anear ≈ H2Factory.nearinteractions(A_H2)

    Afar = H2Factory.farinteractions(A, A_H2)
    @test Afar ≈ (A - Anear) rtol = ϵ

    if mul
        verbose && @info "Checking LinearAlgebra.mul!"
        checkmul(A, A_H2; n=n, ϵ=ϵ, T=T, verbose=verbose)
    end
    if prod
        verbose && @info "Checking Matrix-Vector product"
        checkproduct(A, A_H2; n=n, ϵ=ϵ, T=T, verbose=verbose)
    end
    if nearmul
        verbose && @info "Checking H2Factory.nearmul"
        checknearmul(Anear, A_H2; n=n, ϵ=ϵ, T=T, verbose=verbose)
    end
    if farmul
        verbose && @info "Checking H2Factory.farmul"
        checkfarmul(Afar, A_H2; n=n, ϵ=ϵ, T=T, verbose=verbose)
    end

    return true
end

"""
    CHECKSEED

Default seed for the accuracy helpers. Pass an `rng` to helpers that accept it to try
different input vectors.
"""
const CHECKSEED = 0x5eed1a2b3c4d

"""
    checkrng()

Return a new random-number generator seeded with [`CHECKSEED`](@ref).
Each call starts the same sequence without changing another check's generator.
"""
checkrng() = Xoshiro(CHECKSEED)

# Shared checks for plain and scaled mul!, plus an optional allocating product.
function _checkmulcore(
    ref,
    A_H2,
    h2mul!;
    n=10,
    ϵ=1e-5,
    T::DataType=eltype(A_H2),
    verbose=true,
    h2op=nothing,
    rng=checkrng(),
)
    y = Vector{promote_type(T, eltype(ref))}(undef, size(ref, 1))
    yH2 = Vector{promote_type(T, eltype(ref))}(undef, size(ref, 1))

    for _ in 1:n
        x = randn(rng, T, size(ref, 2))
        α = randn(rng, T)
        β = randn(rng, T)

        LinearAlgebra.mul!(y, ref, x)
        h2mul!(yH2, A_H2, x)

        @test y ≈ yH2 rtol = ϵ
        error = _maxerror(y, yH2)
        @test error < ϵ
        verbose && println("Maximum relative error: $error")

        # The scale of the quantity being approximated, taken before the α/β call overwrites `y`.
        refscale = maximum(abs, y)

        yH2 .= y

        LinearAlgebra.mul!(y, ref, x, α, β)
        h2mul!(yH2, A_H2, x, α, β)

        # `_maxerror` divides by `|α+β|·max|ref·x|`, which blows up near a random `α+β ≈ 0` draw.
        # `_scaledmaxerror` normalizes by `|α|·max|ref·x|` instead -- see its docstring.
        error = _scaledmaxerror(y, yH2, α, refscale)
        @test error < ϵ
        verbose && println("Maximum relative error: $error")

        if h2op !== nothing
            # Fresh variables: `ref * x`/`h2op(...)` allocate regardless, so reusing y/yH2 here
            # would just make the preallocated buffers pointless.
            yproduct = ref * x
            yH2product = h2op(A_H2, x)

            @test yproduct ≈ yH2product rtol = ϵ
            error = _maxerror(yproduct, yH2product)
            @test error < ϵ
            verbose && println("Maximum relative error: $error")
        end
    end

    return true
end

"""
    checknearmul(Anear, A_H2; n=10, ϵ=1e-5, T=eltype(A_H2), verbose=true)

Compare the near-field product of `A_H2` with `Anear` using `n` input vectors.
Check both forms of `nearmul!`, with and without `α` and `β`, and the allocating `nearmul`.
"""
function checknearmul(Anear, A_H2; n=10, ϵ=1e-5, T::DataType=eltype(A_H2), verbose=true)
    return _checkmulcore(
        Anear,
        A_H2,
        H2Factory.nearmul!;
        n=n,
        ϵ=ϵ,
        T=T,
        verbose=verbose,
        h2op=H2Factory.nearmul,
    )
end

"""
    checkfarmul(Afar, A_H2; n=10, ϵ=1e-5, T=eltype(A_H2), verbose=true)

Compare the far-field product of `A_H2` with `Afar` using `n` input vectors.
Check both forms of `farmul!`, with and without `α` and `β`, and the allocating `farmul`.

The `α`/`β` check uses [`_scaledmaxerror`](@ref), so a small `α + β` does not inflate
the reported error.
"""
function checkfarmul(Afar, A_H2; n=10, ϵ=1e-5, T::DataType=eltype(A_H2), verbose=true)
    return _checkmulcore(
        Afar, A_H2, H2Factory.farmul!; n=n, ϵ=ϵ, T=T, verbose=verbose, h2op=H2Factory.farmul
    )
end

"""
    checkmul(A, A_H2; n=10, ϵ=1e-5, T=eltype(A_H2), verbose=true)

Compare `LinearAlgebra.mul!` on `A_H2` with the reference matrix `A` using `n` input vectors.
Check both forms, with and without `α` and `β`.

The `α`/`β` check uses [`_scaledmaxerror`](@ref), so a small `α + β` does not inflate
the reported error.
"""
function checkmul(A, A_H2; n=10, ϵ=1e-5, T::DataType=eltype(A_H2), verbose=true)
    return _checkmulcore(A, A_H2, LinearAlgebra.mul!; n=n, ϵ=ϵ, T=T, verbose=verbose)
end

"""
    checkproduct(A, A_H2; n=10, ϵ=1e-5, T=eltype(A_H2), verbose=true, rng=checkrng())

Compare `A_H2 * x` with `A * x` for `n` random input vectors.
Pass `rng` to choose a different sequence of inputs.
"""
function checkproduct(
    A, A_H2; n=10, ϵ=1e-5, T::DataType=eltype(A_H2), verbose=true, rng=checkrng()
)
    for _ in 1:n
        x = randn(rng, T, size(A, 2))

        y = A * x
        yH2 = A_H2 * x

        @test y ≈ yH2 rtol = ϵ
        error = _maxerror(y, yH2)
        @test error < ϵ
        verbose && println("Maximum relative error: $error")
    end
    return true
end

"""
    _maxerror(y, yH2)

Return `maximum(abs, y - yH2) / maximum(abs, y)`.
If the reference vector `y` is zero, return the absolute error instead.
The `α`/`β` checks use [`_scaledmaxerror`](@ref).
"""
function _maxerror(y, yH2)
    denominator = maximum(abs, y)
    error = maximum(abs, y - yH2)
    return iszero(denominator) ? error : error / denominator
end

"""
    _scaledmaxerror(y, yH2, α, refscale)

Return the error for a multiplication check with `α` and `β`:
`maximum(abs, y - yH2) / (abs(α) * refscale)`. Return the absolute error if that
denominator is zero.

The helpers initialize both output vectors with `Ref * x` before the scaled multiply.
In exact arithmetic, their shared `β * Ref * x` term cancels from the difference.
The remaining difference is `α * (Ref * x - H2 * x)` when the scaled multiply is correct.

Dividing by `maximum(abs, y)` would also divide by `abs(α + β)`, which may be very
small. This function instead uses `refscale = maximum(abs, Ref * x)`, saved before
the scaled call. An incorrect `α` or `β` update still contributes to the error.
"""
function _scaledmaxerror(y, yH2, α, refscale)
    denominator = abs(α) * refscale
    # Only reachable if the reference maps x to zero (or α underflows); there is then no scale to
    # normalize by, so report the raw gap rather than divide by zero.
    iszero(denominator) && return maximum(abs.(y .- yH2))
    return maximum(abs.(y .- yH2)) / denominator
end

"""
    estimatenorm(A; tol=1e-4, itmax=1000, itmin=3, adjointA=adjoint(A),
                 T=eltype(A), verbose=true, rng=checkrng())

Estimate the spectral norm of `A` by power iteration on `adjointA * A`.

Run at least `itmin` iterations. Stop when successive norm estimates agree within
the relative tolerance `tol`, or when `itmax` is reached. Return zero if the probe
is mapped to zero. The result is an estimate, not a guaranteed upper bound.

Pass `rng` to choose another starting vector. `adjointA` can supply an adjoint
operation when `adjoint(A)` is not available.
"""
function estimatenorm(
    A;
    tol=1e-4,
    itmax=1000,
    itmin=3,
    adjointA=adjoint(A),
    T::DataType=eltype(A),
    verbose=true,
    rng=checkrng(),
)
    1 <= itmin <= itmax || throw(ArgumentError("require 1 <= itmin <= itmax"))
    v = rand(rng, T, size(A, 2))

    v = v / norm(v)

    σnew = 0.0
    for i in 1:itmax
        σold = σnew
        w = Vector(A * v)
        x = Vector(adjointA * w)
        iszero(norm(x)) && return zero(real(eltype(x)))
        σnew = norm(x)
        verbose && @info "Norm estimate" iteration = i estimate = sqrt(σnew)
        i >= itmin && abs(sqrt(σold) - sqrt(σnew)) <= tol * sqrt(σold) && break
        v = x / σnew
    end
    return sqrt(σnew)
end

"""
    estimaterelnormdifference(A, Aapprox; tol=1e-4, itmin=3, verbose=true, itmax=1000,
                               adjointA=adjoint(A), adjointAapprox=adjoint(Aapprox),
                               normrefmat=nothing, rng=checkrng())

Estimate `‖Aapprox - A‖₂ / ‖A‖₂` without forming the difference matrix.
Power iteration uses products with the two matrices and their adjoints.

Pass `normrefmat` to supply the reference norm. Otherwise it is estimated with
[`estimatenorm`](@ref). Pass `rng` to choose another starting vector.

Return zero if the difference maps the probe to zero, including when both matrices
are zero. A nonzero difference estimate divided by a zero reference norm returns `Inf`.
"""
function estimaterelnormdifference(
    A,
    Aapprox;
    tol=1e-4,
    itmin=3,
    verbose=true,
    itmax=1000,
    adjointA=adjoint(A),
    adjointAapprox=adjoint(Aapprox),
    normrefmat=nothing,
    rng=checkrng(),
)
    @assert size(Aapprox) == size(A)
    1 <= itmin <= itmax || throw(ArgumentError("require 1 <= itmin <= itmax"))

    v = rand(rng, promote_type(eltype(A), eltype(Aapprox)), size(Aapprox, 2))

    v = v / norm(v)
    σnew = 0.0
    for i in 1:itmax
        σold = σnew
        w = Aapprox * v - A * v
        x = (adjointAapprox * w) - (adjointA * w)
        if iszero(norm(x))
            return zero(real(eltype(x)))
        end
        σnew = norm(x)
        verbose && @info "Difference norm estimate" iteration = i estimate = sqrt(σnew)
        i >= itmin && abs(sqrt(σold) - sqrt(σnew)) <= tol * sqrt(σold) && break
        v = x / σnew
    end
    normrefmat = if isnothing(normrefmat)
        verbose && @info "Estimate norm of reference matrix"
        estimatenorm(
            A;
            tol=tol,
            itmin=itmin,
            itmax=itmax,
            adjointA=adjointA,
            verbose=verbose,
            rng=rng,
        )
    else
        normrefmat
    end
    return sqrt(σnew) / normrefmat
end
