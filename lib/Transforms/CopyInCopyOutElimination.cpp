// We explicitly do not use push / pop for diagnostic in
// order to propagate pragma further on
#include "mlir/Analysis/AliasAnalysis.h"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-copyincopyout-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_COPYINCOPYOUTELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct CopyInCopyOutEliminationPass
    : public P4::P4MLIR::impl::CopyInCopyOutEliminationBase<CopyInCopyOutEliminationPass> {
    CopyInCopyOutEliminationPass() = default;
    void runOnOperation() override;
};

template <typename... EffectTypes>
bool hasNoInterveningEffect(Operation *start, Operation *end, Value ref,
                            llvm::function_ref<bool(Value, Value)> mayAlias) {
    // A boolean representing whether an intervening operation could have impacted
    // `ref`.
    bool hasSideEffect = false;

    // Check whether the effect on ref can be caused by a given operation op.
    std::function<void(Operation *)> checkOperation = [&](Operation *op) {
        // If the effect has alreay been found, early exit,
        if (hasSideEffect) return;

        if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
            SmallVector<MemoryEffects::EffectInstance, 1> effects;
            memEffect.getEffects(effects);

            bool opMayHaveEffect = false;
            for (auto effect : effects) {
                if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

                // If op causes EffectType on a potentially aliasing location for
                // memOp, mark as having the effect.
                if (isa<EffectTypes...>(effect.getEffect())) {
                    if (effect.getValue() && effect.getValue() != ref &&
                        !mayAlias(effect.getValue(), ref))
                        continue;
                    opMayHaveEffect = true;
                    break;
                }
            }

            if (!opMayHaveEffect) return;

            // We have an op with a memory effect and we cannot prove if it
            // intervenes.
            llvm::dbgs() << "Case 1: " << ref << " ";
            op->dump();
            hasSideEffect = true;
            return;
        }

        if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
            // Recurse into the regions for this op and check whether the internal
            // operations may have the side effect `EffectType` on memOp.
            for (Region &region : op->getRegions())
                for (Block &block : region)
                    for (Operation &op : block) checkOperation(&op);
            return;
        }

        // Otherwise, conservatively assume generic operations have the effect
        // on the operation
        llvm::dbgs() << "Case 2: " << ref << " ";
        op->dump();
        hasSideEffect = true;
    };

    // Check all paths from ancestor op `parent` to the operation `to` for the
    // effect. It is known that `to` must be contained within `parent`.
    auto until = [&](Operation *parent, Operation *to) {
        // TODO check only the paths from `parent` to `to`.
        // Currently we fallback and check the entire parent op, rather than
        // just the paths from the parent path, stopping after reaching `to`.
        // This is conservatively correct, but could be made more aggressive.
        assert(parent->isAncestor(to));
        checkOperation(parent);
    };

    // Check for all paths from operation `from` to operation `untilOp` for the
    // given memory effect on the value `ref`.
    std::function<void(Operation *, Operation *)> recur = [&](Operation *from, Operation *untilOp) {
        assert(from->getParentRegion()->isAncestor(untilOp->getParentRegion()) &&
               "Checking for side effect between two operations without a common "
               "ancestor");

        // If the operations are in different regions, recursively consider all
        // path from `from` to the parent of `to` and all paths from the parent
        // of `to` to `to`.
        if (from->getParentRegion() != untilOp->getParentRegion()) {
            recur(from, untilOp->getParentOp());
            until(untilOp->getParentOp(), untilOp);
            return;
        }

        // Now, assuming that `from` and `to` exist in the same region, perform
        // a CFG traversal to check all the relevant operations.

        // Additional blocks to consider.
        SmallVector<Block *, 2> todoBlocks;
        {
            // First consider the parent block of `from` an check all operations
            // after `from`.
            for (auto iter = std::next(from->getIterator()), end = from->getBlock()->end();
                 iter != end && &*iter != untilOp; ++iter) {
                checkOperation(&*iter);
            }

            // If the parent of `from` doesn't contain `to`, add the successors
            // to the list of blocks to check.
            if (untilOp->getBlock() != from->getBlock())
                for (Block *succ : from->getBlock()->getSuccessors()) todoBlocks.push_back(succ);
        }

        SmallPtrSet<Block *, 4> done;
        // Traverse the CFG until hitting `to`.
        while (!todoBlocks.empty()) {
            Block *blk = todoBlocks.pop_back_val();
            if (done.count(blk)) continue;
            done.insert(blk);
            for (auto &op : *blk) {
                if (&op == untilOp) break;
                checkOperation(&op);
                if (&op == blk->getTerminator())
                    for (Block *succ : blk->getSuccessors()) todoBlocks.push_back(succ);
            }
        }
    };

    recur(start, end);
    return !hasSideEffect;
}

/// Attempt to eliminate loadOp by replacing it with a value stored into memory
/// which the load is guaranteed to retrieve. This check involves three
/// components: 1) The store and load must be on the same location 2) The store
/// must dominate (and therefore must always occur prior to) the load 3) No
/// other operations will overwrite the memory loaded between the given load
/// and store.  If such a value exists, the replaced `loadOp` will be added to
/// `loadOpsToErase` and its memref will be added to `memrefsToErase`.
static void forwardStoreToLoad(P4HIR::ReadOpInterface loadOp,
                               llvm::SmallVectorImpl<Operation *> &loadOpsToErase,
                               llvm::SmallPtrSetImpl<Value> &memrefsToErase,
                               mlir::DominanceInfo &domInfo,
                               llvm::function_ref<bool(Value, Value)> mayAlias) {
    // The store op candidate for forwarding that satisfies all conditions
    // to replace the load, if any.
    mlir::Operation *lastWriteStoreOp = nullptr;

    for (auto *user : loadOp.getMemRef().getUsers()) {
        auto storeOp = dyn_cast<P4HIR::WriteOpInterface>(user);
        if (!storeOp) continue;
        auto srcAccess = storeOp.getMemRef();
        auto destAccess = loadOp.getMemRef();

        // 1. Check if the store and the load have mathematically equivalent
        // affine access functions; this implies that they statically refer to the
        // same single memref element. As an example this filters out cases like:
        //     store %A[%i0 + 1]
        //     load %A[%i0]
        //     store %A[%M]
        //     load %A[%N]
        // Use the AffineValueMap difference based memref access equality checking.
        if (srcAccess != destAccess) continue;

        llvm::dbgs() << "loadOp candidate " << domInfo.dominates(storeOp, loadOp) << " "; loadOp->dump();
        llvm::dbgs() << "storeOp candidate "; storeOp->dump();

        // 2. The store has to dominate the load op to be candidate.
        if (!domInfo.dominates(storeOp, loadOp)) continue;

        llvm::dbgs() << "hasNoInterveningEffect "
                     << hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp,
                                                                     loadOp.getMemRef(), mayAlias)
                     << "\n";

        // 4. Ensure there is no intermediate operation which could replace the
        // value in memory.
        if (!hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp, loadOp.getMemRef(),
                                                          mayAlias))
            continue;

        // We now have a candidate for forwarding.
        assert(lastWriteStoreOp == nullptr && "multiple simultaneous replacement stores");
        lastWriteStoreOp = storeOp;
    }

    if (!lastWriteStoreOp) return;

    // Perform the actual store to load forwarding.
    mlir::Value storeVal = cast<P4HIR::WriteOpInterface>(lastWriteStoreOp).getValue();
    // Check if 2 values have the same shape. This is needed for affine vector
    // loads and stores.
    if (storeVal.getType() != loadOp.getValue().getType()) return;

    llvm::dbgs() << "Replace " << loadOp.getValue() << " with " << storeVal << "\n";
    loadOp.getValue().replaceAllUsesWith(storeVal);
    // Record the memref for a later sweep to optimize away.
    memrefsToErase.insert(loadOp.getMemRef());
    // Record this to erase later.
    loadOpsToErase.push_back(loadOp);
}

// This attempts to find stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates writeA.
// 3) There is no potential read between writeA and writeB.
static void findUnusedStore(P4HIR::WriteOpInterface writeA,
                            llvm::SmallVectorImpl<Operation *> &opsToErase,
                            mlir::PostDominanceInfo &postDominanceInfo,
                            llvm::function_ref<bool(Value, Value)> mayAlias) {
    for (Operation *user : writeA.getMemRef().getUsers()) {
        // Only consider writing operations.
        auto writeB = dyn_cast<P4HIR::WriteOpInterface>(user);
        if (!writeB) continue;

        // The operations must be distinct.
        if (writeB == writeA) continue;

        // Both operations must lie in the same region.
        if (writeB->getParentRegion() != writeA->getParentRegion()) continue;

        // Both operations must write to the same memory.
        auto srcAccess = writeB.getMemRef();
        auto destAccess = writeA.getMemRef();

        if (srcAccess != destAccess) continue;

        // writeB must postdominate writeA.
        if (!postDominanceInfo.postDominates(writeB, writeA)) continue;

        // There cannot be an operation which reads from memory between
        // the two writes.
        if (!hasNoInterveningEffect<MemoryEffects::Read>(writeA, writeB, writeB.getMemRef(),
                                                         mayAlias))
            continue;

        opsToErase.push_back(writeA);
        break;
    }
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB.
static void loadCSE(P4HIR::ReadOpInterface loadA,
                    llvm::SmallVectorImpl<Operation *> &loadOpsToErase,
                    mlir::DominanceInfo &domInfo, llvm::function_ref<bool(Value, Value)> mayAlias) {
    llvm::SmallVector<P4HIR::ReadOpInterface, 4> loadCandidates;
    for (auto *user : loadA.getMemRef().getUsers()) {
        auto loadB = dyn_cast<P4HIR::ReadOpInterface>(user);
        if (!loadB || loadB == loadA) continue;

        auto srcAccess = loadB.getMemRef();
        auto destAccess = loadA.getMemRef();

        // 1. The accesses should be to be to the same location.
        if (srcAccess != destAccess) {
            continue;
        }

        // 2. loadB should dominate loadA.
        if (!domInfo.dominates(loadB, loadA)) continue;

        // 3. There should not be a write between loadA and loadB.
        if (!hasNoInterveningEffect<MemoryEffects::Write>(loadB.getOperation(), loadA,
                                                          loadA.getMemRef(), mayAlias))
            continue;

        // Check if two values have the same shape. This is needed for affine vector
        // loads.
        if (loadB.getValue().getType() != loadA.getValue().getType()) continue;

        loadCandidates.push_back(loadB);
    }

    // Of the legal load candidates, use the one that dominates all others
    // to minimize the subsequent need to loadCSE
    mlir::Value loadB;
    for (P4HIR::ReadOpInterface option : loadCandidates) {
        if (llvm::all_of(loadCandidates, [&](P4HIR::ReadOpInterface depStore) {
                return depStore == option ||
                       domInfo.dominates(option.getOperation(), depStore.getOperation());
            })) {
            loadB = option.getValue();
            break;
        }
    }

    if (loadB) {
        loadA.getValue().replaceAllUsesWith(loadB);
        // Record this to erase later.
        loadOpsToErase.push_back(loadA);
    }
}

// The store to load forwarding and load CSE rely on three conditions:
//
// 1) store/load providing a replacement value and load being replaced need to
// have mathematically equivalent affine access functions (checked after full
// composition of load/store operands); this implies that they access the same
// single memref element for all iterations of the common surrounding loop,
//
// 2) the store/load op should dominate the load op,
//
// 3) no operation that may write to memory read by the load being replaced can
// occur after executing the instruction (load or store) providing the
// replacement value and before the load being replaced (thus potentially
// allowing overwriting the memory read by the load).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
static void scalarReplace(mlir::Operation *f, DominanceInfo &domInfo,
                          PostDominanceInfo &postDomInfo, AliasAnalysis &aliasAnalysis) {
    // Load op's whose results were replaced by those forwarded from stores.
    llvm::SmallVector<Operation *, 8> opsToErase;

    // A list of memref's that are potentially dead / could be eliminated.
    llvm::SmallPtrSet<Value, 4> memrefsToErase;

    auto mayAlias = [&](Value val1, Value val2) -> bool {
        return !aliasAnalysis.alias(val1, val2).isNo();
    };

    // Walk all load's and perform store to load forwarding.
    f->walk([&](P4HIR::ReadOpInterface loadOp) {
        forwardStoreToLoad(loadOp, opsToErase, memrefsToErase, domInfo, mayAlias);
    });
    for (auto *op : opsToErase) op->erase();
    opsToErase.clear();

    // Walk all store's and perform unused store elimination
    f->walk([&](P4HIR::WriteOpInterface storeOp) {
        findUnusedStore(storeOp, opsToErase, postDomInfo, mayAlias);
    });
    for (auto *op : opsToErase) op->erase();
    opsToErase.clear();

    // Check if the store fwd'ed memrefs are now left with only stores and
    // deallocs and can thus be completely deleted. Note: the canonicalize pass
    // should be able to do this as well, but we'll do it here since we collected
    // these anyway.
    for (auto memref : memrefsToErase) {
        // If the memref hasn't been locally alloc'ed, skip.
        mlir::Operation *defOp = memref.getDefiningOp();
        if (!defOp) continue;
        auto users = defOp->getUsers();
        auto firstNonAssignOp =
            llvm::find_if(users, [](auto *user) { return !mlir::isa<P4HIR::AssignOp>(user); });

        if (firstNonAssignOp == users.end()) {
            // Completely remove variable if it is only written to.
            for (auto *user : llvm::make_early_inc_range(users)) user->erase();
            defOp->erase();
        }
    }

    // To eliminate as many loads as possible, run load CSE after eliminating
    // stores. Otherwise, some stores are wrongly seen as having an intervening
    // effect.
    f->walk([&](P4HIR::ReadOpInterface loadOp) { loadCSE(loadOp, opsToErase, domInfo, mayAlias); });
    for (auto *op : opsToErase) op->erase();
}

class CopyOutElimination : public mlir::OpRewritePattern<P4HIR::VariableOp> {
 public:
    CopyOutElimination(MLIRContext *context, AliasAnalysis &aliasAnalysis)
        : OpRewritePattern(context), aliasAnalysis(aliasAnalysis) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp alias,
                                        mlir::PatternRewriter &rewriter) const override {
        auto *block = alias->getBlock();
        auto aliasUsers = llvm::to_vector(alias->getUsers());

        auto mayAlias = [&](Value val1, Value val2) -> bool {
            return !aliasAnalysis.alias(val1, val2).isNo();
        };

        // The variable should have only 2 uses:
        //   - The instruction that writes to it (function / extern / action call)
        //   - The read out
        // For now we assume that all uses are within the same BB. This could be
        // changed with dominance condition later on if necessary.
        if (llvm::size(aliasUsers) != 2 || alias->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(
                alias, "alias variable does not have out alias use pattern");

        llvm::sort(aliasUsers,
                   [&](mlir::Operation *a, mlir::Operation *b) { return a->isBeforeInBlock(b); });

        // Last user must be read which, in turn, must have a single use in the current block.
        auto writeAliasOp = dyn_cast<MemoryEffectOpInterface>(aliasUsers.front());
        auto readOp = dyn_cast<P4HIR::ReadOp>(aliasUsers.back());
        if (!writeAliasOp || !readOp || !readOp->hasOneUse() || readOp->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(alias, "invalid alias use");

        // Find the read destination
        auto writeAliaseeOp = dyn_cast<P4HIR::AssignOp>(*readOp->getUsers().begin());
        if (!writeAliaseeOp)
            return rewriter.notifyMatchFailure(alias, "invalid alias value assignment");

        // Ensure that writeOp really writes to alias
        auto aliasee = writeAliaseeOp.getRef();

        // Now we are having the following set of ops:
        //  %alias = p4hir.variable
        //  ...
        //  <writeOp> op1, ..., %alias, ... opN
        //  ...
        //  %alias.val = p4hir.read %alias
        //  ...
        //  p4hir.assign %alias.val, %aliasee
        //
        // We want to transform this into:
        //  <writeOp> op1, ..., %aliasee, ... opN
        // eliminating variable and copies
        //
        // In order to do this we need to ensure:
        //   - None of op1, ..., opN alias %aliasee and <writeOp> only writes to the value
        //   - There is no intervening write to or read from %aliasee between writeOp and assign
        //   - Note that %aliasee might be a field of struct, header or array, so we
        //     need to check for sub- and super-field writes

        // Check for aliasing & memory effects of <writeOp>
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        writeAliasOp.getEffects(effects);
        for (const auto &effect : effects) {
            // Skip non-default resources, these never affect / alias normal values
            if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

            // <writeOp> should only write to %alias, reading is disallowed as it will be
            // an uninitialized read
            if (effect.getValue() == alias) {
                if (!mlir::isa<MemoryEffects::Write>(effect.getEffect()))
                    return rewriter.notifyMatchFailure(alias, "unsupported alias value write op");
                continue;
            }

            if (mayAlias(aliasee, effect.getValue()))
                return rewriter.notifyMatchFailure(alias, [&](auto &diag) {
                    diag << aliasee << " may alias " << effect.getValue();
                });
        }

        // Check for intervening memory effects on %aliasee
        if (!hasNoInterveningEffect<MemoryEffects::Write, MemoryEffects::Read>(
                writeAliasOp, writeAliaseeOp, aliasee, mayAlias))
            return rewriter.notifyMatchFailure(alias, "intervening write to the value");

        // We should be good now:
        //  - Replace %alias with %aliasee
        //  - Kill read out of %alias
        //  - Kill write to %aliasee
        rewriter.replaceOp(alias, aliasee);
        rewriter.eraseOp(writeAliaseeOp);
        rewriter.eraseOp(readOp);

        return mlir::success();
    }

 private:
    AliasAnalysis &aliasAnalysis;
};

class CopyInOutElimination : public mlir::OpRewritePattern<P4HIR::VariableOp> {
 public:
    CopyInOutElimination(MLIRContext *context, AliasAnalysis &aliasAnalysis)
        : OpRewritePattern(context), aliasAnalysis(aliasAnalysis) {}

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp alias,
                                        mlir::PatternRewriter &rewriter) const override {
        auto *block = alias->getBlock();
        auto aliasUsers = llvm::to_vector(alias->getUsers());

        auto mayAlias = [&](Value val1, Value val2) -> bool {
            return !aliasAnalysis.alias(val1, val2).isNo();
        };

        // The variable should have only 3 uses:
        //   - The read in
        //   - The instruction that writes to it (function / extern / action call)
        //   - The read out
        // For now we assume that all uses are within the same BB. This could be
        // changed with dominance condition later on if necessary.
        if (llvm::size(aliasUsers) != 3 || alias->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(
                alias, "alias variable does not have inout alias use pattern");

        llvm::sort(aliasUsers,
                   [&](mlir::Operation *a, mlir::Operation *b) { return a->isBeforeInBlock(b); });

        // Last user must be read which, in turn, must have a single use in the current block.
        auto readInOp = dyn_cast<P4HIR::AssignOp>(aliasUsers.front());
        auto writeAliasOp = dyn_cast<MemoryEffectOpInterface>(aliasUsers[1]);
        auto readOutOp = dyn_cast<P4HIR::ReadOp>(aliasUsers.back());
        if (!writeAliasOp || !readInOp || !readOutOp || !readOutOp->hasOneUse() ||
            readOutOp->isUsedOutsideOfBlock(block))
            return rewriter.notifyMatchFailure(alias, "invalid alias use");

        // Find the read out destination (aliasee)
        auto writeAliaseeOp = dyn_cast<P4HIR::AssignOp>(*readOutOp->getUsers().begin());
        if (!writeAliaseeOp)
            return rewriter.notifyMatchFailure(alias, "invalid alias value assignment");

        auto aliasee = writeAliaseeOp.getRef();

        // Ensure that read in originates from aliasee
        auto readAliaseeOp = readInOp.getValue().getDefiningOp<P4HIR::ReadOp>();
        if (!readAliaseeOp || !readAliaseeOp->hasOneUse() || readAliaseeOp.getRef() != aliasee)
            return rewriter.notifyMatchFailure(alias, "invalid aliasee value read");

        // Now we are having the following set of ops:
        //  %alias = p4hir.variable
        //  ...
        //  %aliasee.val = p4hir.read %aliasee
        //  ...
        //  p4hir.assign %aliasee.val, %alias
        //  ...
        //  <writeOp> op1, ..., %alias, ... opN
        //  ...
        //  %alias.val = p4hir.read %alias
        //  ...
        //  p4hir.assign %alias.val, %aliasee
        //
        // We want to transform this into:
        //  <writeOp> op1, ..., %aliasee, ... opN
        // eliminating variable and copies
        //
        // In order to do this we need to ensure:
        //   - None of op1, ..., opN alias %aliasee and <writeOp> both reads writes to the value
        //   - There is no intervening write to or read from %aliasee between read in and assign
        //     (this is a bit conservative, we can allow reads before <writeOp>).
        //   - Note that %alias might be a field of struct, header or array, so we
        //     need to check for sub- and super-field writes

        // Check for aliasing & memory effects of <writeOp>
        SmallVector<MemoryEffects::EffectInstance, 1> effects;
        writeAliasOp.getEffects(effects);
        for (const auto &effect : effects) {
            // Skip non-default resources, these never affect / alias normal values
            if (effect.getResource() != SideEffects::DefaultResource::get()) continue;

            if (effect.getValue() == alias) {
                if (!mlir::isa<MemoryEffects::Write, MemoryEffects::Read>(effect.getEffect()))
                    return rewriter.notifyMatchFailure(alias, "unsupported alias value write op");
                continue;
            }

            if (mayAlias(aliasee, effect.getValue())) {
                return rewriter.notifyMatchFailure(alias, [&](auto &diag) {
                    diag << aliasee << " may alias " << effect.getValue();
                });
            }
        }

        // Check for intervening memory effects on %aliasee
        if (!hasNoInterveningEffect<MemoryEffects::Write, MemoryEffects::Read>(
                writeAliasOp, writeAliaseeOp, aliasee, mayAlias))
            return rewriter.notifyMatchFailure(
                alias, "intervening read from or write to the aliasee after write op");

        if (!hasNoInterveningEffect<MemoryEffects::Write>(readAliaseeOp, writeAliasOp, aliasee,
                                                          mayAlias))
            return rewriter.notifyMatchFailure(alias,
                                               "intervening read from the aliasee before write op");

        // We should be good now:
        //  - Replace %alias with %aliasee
        //  - Kill read out of %alias
        //  - Kill write to %aliasee
        rewriter.replaceOp(alias, aliasee);
        rewriter.eraseOp(writeAliaseeOp);
        rewriter.eraseOp(readInOp);
        rewriter.eraseOp(readOutOp);
        rewriter.eraseOp(readAliaseeOp);

        return mlir::success();
    }

 private:
    AliasAnalysis &aliasAnalysis;
};

void CopyInCopyOutEliminationPass::runOnOperation() {
    getOperation()->walk([&](mlir::Operation *op) {
        if (mlir::isa<P4HIR::FuncOp, P4HIR::ParserOp>(op))
            scalarReplace(op, getAnalysis<DominanceInfo>(), getAnalysis<PostDominanceInfo>(),
                            getAnalysis<AliasAnalysis>());
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<CopyOutElimination, CopyInOutElimination>(patterns.getContext(),
                                                           getAnalysis<AliasAnalysis>());

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createCopyInCopyOutEliminationPass() {
    return std::make_unique<CopyInCopyOutEliminationPass>();
}
