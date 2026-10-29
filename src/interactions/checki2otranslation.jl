"""
    _i2omodeconfig(
        mode, testmomentcollection, trialmomentcollection, testnode, trialnode,
        buildtestmoment, buildtrialmoment,
    )

Choose the collections, nodes, and builders used by `checki2otranslation` for `mode`.
Return the tuple `(incollection, outcollection, receivingnode, translatingnode,
direction, populate!, buildin, buildout)`.

- Forward sends trial moments to test moments, with nodes `(testnode, trialnode)`.
- Transpose sends test moments to trial moments, with nodes `(trialnode, testnode)`.
- Adjoint uses the same node order as transpose, with `AdjointTranslation()`.

The test and trial builders are separate because their moments can have different shapes.
They swap input/output roles along with the collections. All modes populate the input
with plain `mul!`; the adjoint direction conjugates the coupling operator.
"""
function _i2omodeconfig(
    ::H2Factory.isForward,
    testmomentcollection,
    trialmomentcollection,
    testnode,
    trialnode,
    buildtestmoment,
    buildtrialmoment,
)
    return (
        trialmomentcollection,
        testmomentcollection,
        testnode,
        trialnode,
        H2Factory.ForwardTranslation(),
        LinearAlgebra.mul!,
        buildtrialmoment,
        buildtestmoment,
    )
end

function _i2omodeconfig(
    ::H2Factory.isTranspose,
    testmomentcollection,
    trialmomentcollection,
    testnode,
    trialnode,
    buildtestmoment,
    buildtrialmoment,
)
    return (
        testmomentcollection,
        trialmomentcollection,
        trialnode,
        testnode,
        H2Factory.TransposeTranslation(),
        LinearAlgebra.mul!,
        buildtestmoment,
        buildtrialmoment,
    )
end

function _i2omodeconfig(
    ::H2Factory.isAdjoint,
    testmomentcollection,
    trialmomentcollection,
    testnode,
    trialnode,
    buildtestmoment,
    buildtrialmoment,
)
    return (
        testmomentcollection,
        trialmomentcollection,
        trialnode,
        testnode,
        # Same swapped node order as transpose; `AdjointTranslation` conjugates the coupling
        # operator, so the leaf multiply is still plain `mul!`.
        H2Factory.AdjointTranslation(),
        LinearAlgebra.mul!,
        buildtestmoment,
        buildtrialmoment,
    )
end

"""
    checki2otranslation(
        testmomentcollection, trialmomentcollection,
        i2ocollection, i2ochannel,
        testnode, trialnode, level,
        buildtestmoment, buildtrialmoment, xgenerator,
        truereference::AbstractMatrix;
        mode=H2Factory.isForward(), α=true,
        scheduler=OhMyThreads.SerialScheduler(), n=10, ϵ=1e-3, verbose=true,
    )

Compare one I2O translation with the reference matrix `truereference` using `n` input vectors.
The check runs `modulate_in!`, `applyi2o!`, and `modulate_out!`, then reconstructs the
output through the leaf collection.

`buildtestmoment()` and `buildtrialmoment()` create fresh moments for their respective
sides. The shapes may differ. `xgenerator()` supplies each input vector.

`mode` selects forward, transpose, or adjoint translation. For transpose and adjoint,
the input/output sides and node order swap. The adjoint direction conjugates the coupling;
all modes use plain `mul!` for the leaf operations. Supply a reference matrix already
oriented for the chosen mode; `Transpose` and `Adjoint` wrappers are accepted.

`α` scales the final reconstruction. It can be a scalar or one value per moment
component, and must match the operator and mode being checked.

Return the largest relative error among the probes, measured as
`maximum(abs, ytrue - y) / maximum(abs, ytrue)`. Use the absolute error when the
reference output is zero. Each probe also checks `error < ϵ` through `Test`.
"""
function checki2otranslation(
    testmomentcollection,
    trialmomentcollection,
    i2ocollection,
    i2ochannel,
    testnode,
    trialnode,
    level,
    buildtestmoment,
    buildtrialmoment,
    xgenerator,
    truereference::AbstractMatrix;
    mode=H2Factory.isForward(),
    α=true,
    scheduler=OhMyThreads.SerialScheduler(),
    n=10,
    ϵ=1e-3,
    verbose=true,
)
    inmomentcollection, outmomentcollection, receivingnode, translatingnode, direction, populate!, buildin, buildout = _i2omodeconfig(
        mode,
        testmomentcollection,
        trialmomentcollection,
        testnode,
        trialnode,
        buildtestmoment,
        buildtrialmoment,
    )

    maxerror = 0
    for trial in 1:n
        x = xgenerator()
        ytrue = truereference * x

        in_ = buildin()
        populate!(in_, inmomentcollection, x)
        out = buildout()

        # Single-pair driver (no chunked traversal), so borrow chunk 1's buffer directly. Drives
        # the public I2O protocol (`modulate_in!`/`applyi2o!`/`modulate_out!`), not the
        # collection-specific entry points, so it works whether `i2ocollection` is a bare
        # coupling collection, a self-modulating collection, or a composed `I2OTranslation`.
        context = H2Factory.TranslationContext(scheduler, nothing)
        buffer1 = H2Factory.borrowbuffer(i2ochannel, level, 1, mode)
        H2Factory.modulate_in!(
            in_,
            level,
            mode,
            H2Factory.modulation(i2ocollection),
            H2Factory.coupling(i2ocollection),
            i2ochannel,
            scheduler,
            buffer1,
        )
        H2Factory.applyi2o!(
            out,
            in_,
            i2ocollection,
            receivingnode,
            translatingnode,
            direction,
            H2Factory.Overwrite(),
            context,
        )
        buffer2 = H2Factory.borrowbuffer(i2ochannel, level, 1, mode)
        H2Factory.modulate_out!(
            out,
            level,
            mode,
            H2Factory.modulation(i2ocollection),
            H2Factory.coupling(i2ocollection),
            i2ochannel,
            scheduler,
            buffer2,
        )

        y = zero(ytrue)
        LinearAlgebra.mul!(y, outmomentcollection, out, α, false)
        # No trailing `conj!(y)`: adjoint conjugates the coupling operator (`AdjointTranslation`),
        # so `out` already carries `adjoint(coupling)*in` and `y` is adjoint-correct as-is.

        er = _maxerror(ytrue, y)
        @test er < ϵ
        verbose && println("trial $trial ($mode): relative error = $er")
        maxerror = max(maxerror, er)
    end
    return maxerror
end
