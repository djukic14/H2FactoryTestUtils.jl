# Small, pluggable translators with hand-computable results.
#
# Every translator here supports all three roles, so the same object can be plugged as i2i, o2o,
# or i2o:
#   * I2I/O2O (level transfers) dispatch through `H2Factory.leveltranslation!`, which also takes
#     `ConjugateTranslation` (used by an adjoint map's aggregation pass).
#   * I2O (cross-node far-field coupling) dispatches through `H2Factory.applycouplingkernel!`,
#     reached via the guarded `applycoupling!` entry, with `direction` one of
#     `ForwardTranslation`/`TransposeTranslation`/`AdjointTranslation`.
#   Both also take an `accumulation` of `Overwrite`/`Accumulate`.
#
# Moments are single-component: content lives in `moment[1]`, a length-`rank` vector, and a
# translator applies a fixed linear map to it. Pair-dependent couplings (a real interpolation
# kernel, e.g. the mini H2 matrix in `mini1d.jl`) are not in this library -- a fixture that needs
# them defines its own translator with per-pair matrices; this library covers the
# pair-independent building blocks (identity, scalar, fixed matrix) plus a recording wrapper.

"""
    DirectCoupling()

Select the kernel for a synthetic translator that applies its linear map directly,
rather than looking up a block by node pair.

`applycoupling!` reads this trait through `i2otrait` before calling the kernel.
The synthetic translator therefore uses the same symmetry check as other collections.
"""
struct DirectCoupling <: H2Factory.AbstractI2OTranslationTrait end

# `<: AbstractI2OCouplingCollection` so these mocks reach the I2O protocol through the guarded
# `applycoupling!` entry (supplying `applycouplingkernel!`), the way a real collection does --
# every translator here genuinely is an I2O coupling collection (see the header).
abstract type FakeTranslator <: H2Factory.AbstractI2OCouplingCollection end

# The two questions `applycoupling!` asks of any coupling collection, declared (not inherited --
# H2Factory has no permissive fallback) once for the family. A fixed linear map has no symmetry
# store, so there is never an unconsumed tag to refuse.
H2Factory.assertnosymmetry(::FakeTranslator) = nothing
H2Factory.i2otrait(::FakeTranslator) = DirectCoupling()

struct IdentityTranslator <: FakeTranslator end

"""
    identitytranslator()

Return a translator that copies the source moment unchanged.
With this translator, I2I sums child moments, O2O copies a parent moment to its
children, and I2O copies between source and receiver.
"""
identitytranslator() = IdentityTranslator()

struct ScalarTranslator{T} <: FakeTranslator
    c::T
end

"""
    scalartranslator(c)

Return a translator that multiplies the source moment by `c`.
A complex `c` helps distinguish the adjoint operation from the transpose.
"""
scalartranslator(c) = ScalarTranslator(c)

struct MatrixTranslator{M} <: FakeTranslator
    m::M
end

"""
    matrixtranslator(M)

Return a translator that applies `M` to the source moment: `out[1] = M * in[1]`.
Forward uses `M`, transpose uses `transpose(M)`, and adjoint uses `adjoint(M)`.
These operations act on the matrix, not by conjugating the input moment.

A complex, nonsymmetric matrix helps tests distinguish all three operations.
"""
matrixtranslator(M) = MatrixTranslator(M)

# The linear map each translator applies, returned as a fresh vector (harness allocates freely).
_apply(::IdentityTranslator, in) = copy(in)
_apply(t::ScalarTranslator, in) = t.c .* in
_apply(t::MatrixTranslator, in) = t.m * in

# Round-trips a computed vector `v` through the per-(level,thread) scratch `buffer` so the real
# multithreaded buffer-checkout machinery is exercised rather than bypassed: written then read
# back, or passed through when the map uses `EmptyChannels`. A buffer wrongly shared between two
# concurrent tasks would corrupt the result here -- what the harness's determinism test catches.
_scratch(buffer, v) = (buffer[1].=v; return buffer[1])
_scratch(::H2Factory.EmptyChannels, v) = v
# `checki2otranslation` drives one translation step with a `nothing` translate buffer -> pass through.
_scratch(::Nothing, v) = v

# Transpose-direction map (`TransposeTranslation`): plain `transpose(M)`, NO conjugation.
_applyT(::IdentityTranslator, in) = copy(in)
_applyT(t::ScalarTranslator, in) = t.c .* in
_applyT(t::MatrixTranslator, in) = transpose(t.m) * in

# Adjoint-direction map (`AdjointTranslation`): `adjoint(M) = conj(transpose(M))`. The adjoint
# conjugates the coupling operator here rather than relying on the traversal to conjugate
# moments, so a dropped/misplaced conjugation shows up numerically -- why `ScalarTranslator`
# carries a genuinely complex `c`.
_applyA(::IdentityTranslator, in) = copy(in)
_applyA(t::ScalarTranslator, in) = conj(t.c) .* in
_applyA(t::MatrixTranslator, in) = adjoint(t.m) * in

# Conjugate-direction map (`ConjugateTranslation`): `conj(M)`, conjugated but not transposed.
# Applied by an adjoint map's aggregation, which runs child-to-parent like forward and so reverses
# nothing while still needing the conjugation. Distinct from `_applyA` in general -- only
# accidentally equal for the identity/scalar, which is why `MatrixTranslator` is what lets this
# mock catch a traversal applying the wrong one.
_applyC(::IdentityTranslator, in) = copy(in)
_applyC(t::ScalarTranslator, in) = conj(t.c) .* in
_applyC(t::MatrixTranslator, in) = conj(t.m) * in

# I2I/O2O role: level transfers route through `leveltranslation!`. `direction` selects the
# operator at the transfer boundary -- forward `_apply`, transpose `_applyT`, adjoint `_applyA`,
# conjugate `_applyC` -- so a level transfer carries no hidden moment conjugation. Aggregation
# applies `ForwardTranslation` (or `ConjugateTranslation` for an adjoint map); disaggregation
# applies `TransposeTranslation`/`AdjointTranslation` on the same side's translator (O2O ==
# transpose(I2I)), so all four directions must be right here. Routed through the per-thread
# scratch buffer (`_scratch`) to exercise the real buffer-checkout path.
_directionapply(t::FakeTranslator, ::H2Factory.ForwardTranslation, in) = _apply(t, in)
_directionapply(t::FakeTranslator, ::H2Factory.TransposeTranslation, in) = _applyT(t, in)
_directionapply(t::FakeTranslator, ::H2Factory.AdjointTranslation, in) = _applyA(t, in)
_directionapply(t::FakeTranslator, ::H2Factory.ConjugateTranslation, in) = _applyC(t, in)

# Args fully typed to match the generic `leveltranslation!` (leveltranslator.jl) plus the
# `FakeTranslator`/accumulation specializations, so these are strictly more specific (no ambiguity).
function H2Factory.leveltranslation!(
    out,
    in,
    t::FakeTranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    out[1] .= _scratch(context.buffer, _directionapply(t, direction, in[1]))
    return out
end
function H2Factory.leveltranslation!(
    out,
    in,
    t::FakeTranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    out[1] .= _scratch(context.buffer, _directionapply(t, direction, in[1])) .+ out[1]
    return out
end

# I2O role: implements the coupling kernel, never the `applycoupling!` entry -- implementing the
# entry would shadow H2Factory's guarded method, and a test fixture is the last place that should
# model that bypass. `_directionapply` selects the operator per direction (forward `_apply`,
# transpose `_applyT`, adjoint `_applyA`), exactly as the level-transfer role does. Split by
# accumulation type so a `SpyTranslator` method below is strictly more specific than the
# `FakeTranslator` method of the same accumulation.
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::FakeTranslator,
    ::DirectCoupling,
    receiving::Int,
    translating::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    out[1] .= _scratch(context.buffer, _directionapply(t, direction, in[1]))
    return out
end
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::FakeTranslator,
    ::DirectCoupling,
    receiving::Int,
    translating::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    out[1] .= _scratch(context.buffer, _directionapply(t, direction, in[1])) .+ out[1]
    return out
end

"""
    SpyRecord

One recorded translator call, with a `role` and two node indices `(a, b)`.

- `:transfer`: aggregation transfer; `(a, b) = (parent, child)`.
- `:transferdisagg`: transpose or adjoint transfer during disaggregation;
  `(a, b) = (parent, child)`.
- `:i2o`: forward coupling; `(a, b) = (receiving, translating)`.
- `:o2i`: transpose coupling, with the same node convention.
- `:adjointo2i`: adjoint coupling, with the same node convention.

Disaggregation reuses the I2I translator with a transpose or adjoint operation, so
a spy in the I2I slot records both transfer roles. Separate coupling roles let tests
distinguish an adjoint call from a transpose call.
"""
struct SpyRecord
    role::Symbol
    a::Int
    b::Int
end

"""
    SpyTranslator

Wrap a translator and record each call while leaving its arithmetic to the inner object.
Recording is protected by a lock, so several traversal tasks can share the spy.

Read the log with [`records`](@ref). Tests use it to check traversal order and which
node pairs were translated. The records describe when calls start, not when they finish.
"""
struct SpyTranslator{Inner<:FakeTranslator} <: FakeTranslator
    inner::Inner
    log::Vector{SpyRecord}
    lock::ReentrantLock
end

"""
    spytranslator(inner)

Return a [`SpyTranslator`](@ref) around `inner` with an empty call log.
Retrieve the recorded transfers and couplings with [`records`](@ref).
"""
spytranslator(inner::FakeTranslator) = SpyTranslator(inner, SpyRecord[], ReentrantLock())

"""
    records(t::SpyTranslator)

Return a copy of the call log. Each call is recorded before its arithmetic starts.
The order is the order in which calls acquired the lock, not their completion order.
"""
records(t::SpyTranslator) = lock(t.lock) do
    return copy(t.log)
end

_record!(t::SpyTranslator, role, a, b) = lock(t.lock) do
    return push!(t.log, SpyRecord(role, a, b))
end

# Spy delegates arithmetic to its inner translator, tagging each call with its role. The
# level-transfer form is used for both I2I and O2O (same signature, indistinguishable at this
# level) -- disambiguate via recorded parent/child against the plan, or wrap separate spies for
# the i2i/o2o slots (the builder does the latter). The same i2i translator serves both
# aggregation (forward) and disaggregation (transpose/adjoint, reconstructing O2O as
# transpose(I2I) rather than a separate slot), so a spy on the i2i slot sees both; the role is
# derived from the direction to keep them distinguishable.
_spytransferrole(::H2Factory.ForwardTranslation) = :transfer
_spytransferrole(::H2Factory.AbstractTranslationDirection) = :transferdisagg

# Split by accumulation type (not a generic `AbstractAccumulationMode`), so each method stays
# strictly more specific than the matching `FakeTranslator` method -- otherwise "more specific
# translator, less specific accumulation" is ambiguous.
function H2Factory.leveltranslation!(
    out,
    in,
    t::SpyTranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    accumulation::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    _record!(t, _spytransferrole(direction), parent, child)
    return H2Factory.leveltranslation!(
        out, in, t.inner, tree, parent, child, direction, accumulation, context
    )
end
function H2Factory.leveltranslation!(
    out,
    in,
    t::SpyTranslator,
    tree,
    parent::Int,
    child::Int,
    direction::H2Factory.AbstractTranslationDirection,
    accumulation::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    _record!(t, _spytransferrole(direction), parent, child)
    return H2Factory.leveltranslation!(
        out, in, t.inner, tree, parent, child, direction, accumulation, context
    )
end
# The I2O role's tag derives from the direction, exactly as `_spytransferrole` derives the
# transfer role. The `:adjointo2i` tag still catches a matvec that reaches the coupling with
# `TransposeTranslation` where it should have passed `AdjointTranslation`: that still records
# `:o2i`, distinguishably wrong.
_spyi2orole(::H2Factory.ForwardTranslation) = :i2o
_spyi2orole(::H2Factory.TransposeTranslation) = :o2i
_spyi2orole(::H2Factory.AdjointTranslation) = :adjointo2i

function H2Factory.applycouplingkernel!(
    out,
    in,
    t::SpyTranslator,
    trait::DirectCoupling,
    receiving::Int,
    translating::Int,
    direction::H2Factory.AbstractTranslationDirection,
    accumulation::H2Factory.Overwrite,
    context::H2Factory.TranslationContext,
)
    _record!(t, _spyi2orole(direction), receiving, translating)
    return H2Factory.applycouplingkernel!(
        out, in, t.inner, trait, receiving, translating, direction, accumulation, context
    )
end
function H2Factory.applycouplingkernel!(
    out,
    in,
    t::SpyTranslator,
    trait::DirectCoupling,
    receiving::Int,
    translating::Int,
    direction::H2Factory.AbstractTranslationDirection,
    accumulation::H2Factory.Accumulate,
    context::H2Factory.TranslationContext,
)
    _record!(t, _spyi2orole(direction), receiving, translating)
    return H2Factory.applycouplingkernel!(
        out, in, t.inner, trait, receiving, translating, direction, accumulation, context
    )
end
