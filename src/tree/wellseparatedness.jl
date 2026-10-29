"""
    testwellseparatedness(tree) -> Bool

Check that each node pair at the same level is accounted for exactly once: as a near
interaction, a direct translation, or a translation between ancestors.

Return `true` if every pair passes. Otherwise print the offending pair and return `false`.
For a single tree, skip pairs where both nodes are the same.
"""
function testwellseparatedness(tree)
    return testwellseparatedness(tree, H2Trees.treetrait(tree))
end

function testwellseparatedness(tree, ::Any)
    for level in H2Trees.levels(tree)
        for trialnode in H2Trees.LevelIterator(tree, level)
            for testnode in H2Trees.LevelIterator(tree, level)
                testnode == trialnode && continue

                connections = 0

                neartranslations = 0
                directtranslations = 0
                parenttranslations = 0

                if H2Trees.isnear(tree, testnode, trialnode, H2Trees.treetrait(tree))
                    connections += 1
                    neartranslations += 1
                end

                if testnode in H2Trees.TranslatingNodesIterator(tree, trialnode)
                    connections += 1
                    directtranslations += 1
                end

                trialparent = trialnode

                for testparent in H2Trees.ParentUpwardsIterator(tree, testnode)
                    testparent == 0 && break

                    trialparent = H2Trees.parent(tree, trialparent)

                    if testparent in H2Trees.TranslatingNodesIterator(tree, trialparent)
                        connections += 1
                        parenttranslations += 1
                    end
                end

                if connections != 1
                    println(
                        "Detected connection error: parenttranslations: $parenttranslations directtranslations: $directtranslations neartranslations: $neartranslations for testnode $testnode and trialnode $trialnode on level $level",
                    )

                    return false
                end
            end
        end
    end
    return true
end

function testwellseparatedness(tree, ::H2Trees.isBlockTree)
    trialtree = H2Trees.trialtree(tree)
    testtree = H2Trees.testtree(tree)

    for level in H2Trees.levels(trialtree)
        for trialnode in H2Trees.LevelIterator(trialtree, level)
            for testnode in H2Trees.LevelIterator(testtree, level)
                connections = 0
                neartranslations = 0
                directtranslations = 0
                parenttranslations = 0

                if H2Trees.isnear(testtree, trialtree, testnode, trialnode)
                    connections += 1
                    neartranslations += 1
                end

                if testnode in
                    H2Trees.TranslatingNodesIterator(testtree, trialtree, trialnode)
                    connections += 1
                    directtranslations += 1
                end

                trialparent = trialnode

                for testparent in H2Trees.ParentUpwardsIterator(testtree, testnode)
                    testparent == 0 && break

                    trialparent = H2Trees.parent(trialtree, trialparent)
                    trialparent == 0 && break

                    if testparent in
                        H2Trees.TranslatingNodesIterator(testtree, trialtree, trialparent)
                        connections += 1
                        parenttranslations += 1
                    end
                end
                if connections != 1
                    println(
                        "Detected connection error: parenttranslations: $parenttranslations directtranslations: $directtranslations neartranslations: $neartranslations for testnode $testnode and trialnode $trialnode on level $level",
                    )

                    return false
                end
            end
        end
    end
    return true
end
