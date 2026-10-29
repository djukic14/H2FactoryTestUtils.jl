module PartitionFixtures
using H2Trees: H2Trees

struct Tree
    near::Bool
    far::Bool
end

H2Trees.treetrait(::Tree) = nothing
H2Trees.levels(::Tree) = (1,)
H2Trees.LevelIterator(::Tree, level::Int) = (1, 2)
H2Trees.ParentUpwardsIterator(::Tree, node::Int) = ()
H2Trees.isnear(tree::Tree, testnode::Int, trialnode::Int, ::Nothing) = tree.near
H2Trees.TranslatingNodesIterator(tree::Tree, trialnode::Int) = tree.far ? (1, 2) : ()

struct BlockTree
    test::Tree
    trial::Tree
end

H2Trees.treetrait(::BlockTree) = H2Trees.isBlockTree()
H2Trees.testtree(tree::BlockTree) = tree.test
H2Trees.trialtree(tree::BlockTree) = tree.trial
H2Trees.isnear(test::Tree, trial::Tree, testnode::Int, trialnode::Int) = test.near
function H2Trees.TranslatingNodesIterator(test::Tree, trial::Tree, trialnode::Int)
    return test.far ? (1, 2) : ()
end
end
