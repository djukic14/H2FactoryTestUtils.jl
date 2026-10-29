module H2FactoryTestUtils
using H2Factory
using H2Trees
using CompScienceMeshes
using SparseArrays
using Test
using LinearAlgebra
using RandomizedLinAlg
using OhMyThreads
using Random
using StaticArrays
using FFTW
include("interactions/checkinteractions.jl")
include("interactions/checki2otranslation.jl")
include("tree/wellseparatedness.jl")
include("mocks/translators.jl")
include("mocks/fakemaps.jl")
include("mocks/fakepetrovmap.jl")
include("mocks/reference.jl")
include("mocks/densematrix.jl")
include("mocks/mini1d.jl")
include("mocks/toeplitz.jl")
include("mocks/faketoeplitzmap.jl")
include("mocks/toeplitz3d.jl")
include("mocks/symmetryi2o.jl")

# Synthetic aggregation/disaggregation fixtures with pluggable translators.
export identitytranslator, scalartranslator, matrixtranslator, spytranslator
export SpyTranslator, SpyRecord, records
export FakeGalerkinH2Core, fakegalerkincore, fakegalerkinmap
export FakePetrovH2Core, fakepetrovcore, fakepetrovmap
export FakeScratch, fakeleafscratchlength
# Naive reference oracle.
export ref_aggregate, ref_translate, ref_disaggregate, ref_farmul, ref_farpairs
export reftransfer, refcoupling
# Complete dense far-field matrix (closed-form, independent of mul!).
export denseh2matrix
# Per-pair dense I2O translation check (via checki2otranslation), Galerkin + Petrov.
export checkdensei2o
# Mini 1D interpolation-based H2 matrix.
export minih2matrix, minipetrovh2matrix, minih2matrix_leveltranslator
# Toeplitz far-field I2O translation tests (drive the real FFT Toeplitz path via checki2otranslation).
export toeplitzi2osetup, checktoeplitzi2o
# Fake Toeplitz H2 matrix: full aggregate/translate/disaggregate over the real FFT Toeplitz path.
export faketoeplitzmap, fakepetrovtoeplitzmap, FakeToeplitzI2O, faketoeplitzi2o
# Toeplitz interpolation mini: genuine interpolation FMM (real transfers) -> reproduces the sampled kernel.
export minitoeplitzh2matrix, minipetrovtoeplitzh2matrix
# Genuinely 3-D Toeplitz: a non-degenerate (2n-1)³ tensor, needed for the symmetry reduction but
# not provided by the 1-D fixtures above.
export toeplitz3dsetup, checktoeplitz3di2o, Toeplitz3DTranslator
export toeplitz3dcoupling, toeplitz3dtensor, toeplitz3dnodespacing
# Symmetry-reduced Toeplitz: one difference tensor per lattice-symmetry orbit, read through a
# permutation of the tensor inside each operator (not the per-component vector).
export symmetrytoeplitz3dsetup, permutedtoeplitzoperators
# Symmetry-reduced far-field I2O: one diagonal coupling per lattice-symmetry orbit, read through
# a permutation.
export symmetrydirections, symmetrykernelvalue, symmetrypermutation
export SymmetryDiagonalTranslator
export FakeSymmetryI2O, fakesymmetryi2o, symmetryi2osetup, fakesymmetrymap
export symmetrytranslatingplan, symmetryleafbasis
export fakepetrovsymmetrymap, petrovsymmetrysetup
export symmetrytoeplitzcollection
export symmetryfrequencyimages

export nearinteractions, farinteractions
export checkinteractions, checkmul, checkproduct, checknearmul, checkfarmul
export checki2otranslation
export estimatenorm, estimaterelnormdifference

end # module H2FactoryTestUtils
