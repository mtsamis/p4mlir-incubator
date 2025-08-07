#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::detail {

class IRUtils {
 public:
    // Helper to create a new sub-state for `state` and move `ops` in it.
    P4HIR::ParserStateOp createSubState(mlir::IRRewriter &rewriter, P4HIR::ParserStateOp state,
                                        llvm::StringRef suffix, mlir::Block *ops = nullptr) {
        auto parser = state->getParentOfType<P4HIR::ParserOp>();
        std::string basename = (llvm::Twine(state.getSymName()) + "_" + suffix).str();
        std::string name = basename;

        size_t counter = 0;
        while (parser.lookupSymbol(name)) name = basename + "_" + std::to_string(counter++);

        auto newState =
            rewriter.create<P4HIR::ParserStateOp>(state.getLoc(), name, mlir::DictionaryAttr());
        mlir::Block *newStateBB =
            rewriter.createBlock(&newState.getBody(), newState.getBody().begin());
        if (ops) rewriter.inlineBlockBefore(ops, newStateBB, newStateBB->end());
        return newState;
    };

    // Inline `scopeOp`'s body to its parent.
    static void inlineScope(mlir::IRRewriter &rewriter, P4HIR::ScopeOp scopeOp) {
        mlir::Block *block = &scopeOp.getScopeRegion().front();
        mlir::Operation *terminator = block->getTerminator();
        mlir::ValueRange results = terminator->getOperands();
        rewriter.inlineBlockBefore(block, scopeOp, /*blockArgs=*/{});
        rewriter.replaceOp(scopeOp, results);
        rewriter.eraseOp(terminator);
    }

    // Return true if it's valid to call `splitBlockAt` for the given arguments.
    static bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op) {
        assert(block->findAncestorOpInBlock(*op) != nullptr);

        while (true) {
            if (op->getBlock() == block) return true;
            op = op->getParentOp();
            if (!op || !mlir::isa<P4HIR::ScopeOp>(op)) return false;
        }
    }

    // Split `block` that has as ancestor `op` in three:
    // One with operations before `op`, one with `op` and one with operations after `op`.
    static std::array<mlir::Block *, 3> splitBlockAt(mlir::IRRewriter &rewriter, mlir::Block *block,
                                                     mlir::Operation *op) {
        assert(canSplitBlockAt(block, op));

        // Inline all scopes surrounding `op`.
        block->walk<mlir::WalkOrder::PostOrder>(
            [&](P4HIR::ScopeOp scopeOp) { inlineScope(rewriter, scopeOp); });

        // Split resulting block.
        mlir::Block *before = op->getBlock();
        mlir::Block *middle = rewriter.splitBlock(before, op->getIterator());
        mlir::Block *after = rewriter.splitBlock(middle, ++op->getIterator());
        return {before, middle, after};
    }

    using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;

    // Returns blocks that contain uses of val and are not ancestors of `block`.
    static BlockSet getEscapingBlocks(mlir::Block *block, mlir::Value val) {
        BlockSet escapingBlocks;
        for (mlir::Operation *user : val.getUsers())
            if (user->getBlock() != block && !block->findAncestorOpInBlock(*user))
                escapingBlocks.insert(user->getBlock());
        return escapingBlocks;
    }

    // Fix up operations in `block` with uses in other blocks due to splitting.
    // The rewriter's insertion point is the location where new variables may be created.
    static void adjustBlockUses(mlir::IRRewriter &rewriter, mlir::Block *block) {
        // Iterate in reverse so we process uses before their definition in this block.
        // This is needed to properly handle operands after `copyOp` calls.
        for (mlir::Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
            if (mlir::isa<P4HIR::VariableOp>(op)) {
                // Promote variables to a parser locals.
                rewriter.moveOpBefore(&op, rewriter.getInsertionBlock(),
                                      rewriter.getInsertionPoint());
            } else if (mlir::isa<P4HIR::ConstOp, P4HIR::StructExtractRefOp,
                                 P4HIR::ArrayElementRefOp, P4HIR::SliceRefOp>(op)) {
                // Copy RefOps and constants.
                auto escapingBlocks = getEscapingBlocks(block, op.getResult(0));
                copyOp(rewriter, &op, escapingBlocks);
            } else {
                // Promote values to variables.
                for (auto val : op.getResults()) {
                    auto escapingBlocks = getEscapingBlocks(block, val);

                    if (!escapingBlocks.empty()) promoteValToVar(rewriter, val, escapingBlocks);
                }
            }
        }
    }

    // Promote `val` to a local variable.
    // The rewriter's insertion point is the location where the new variable is created.
    static void promoteValToVar(mlir::IRRewriter &rewriter, mlir::Value val,
                                const BlockSet &escapingBlocks,
                                llvm::StringRef varName = "promoted_local") {
        // Create new variable to hold `val`.
        auto newVar = rewriter.create<P4HIR::VariableOp>(
            val.getLoc(), P4HIR::ReferenceType::get(val.getType()), varName);

        mlir::OpBuilder::InsertionGuard guard(rewriter);

        // Insert a new read in all escaping blocks and replace uses of `val` in that block.
        for (mlir::Block *block : escapingBlocks) {
            rewriter.setInsertionPointToStart(block);
            auto newVal = rewriter.create<P4HIR::ReadOp>(val.getLoc(), newVar);
            rewriter.replaceUsesWithIf(val, newVal, [block](mlir::OpOperand &use) {
                return use.getOwner()->getBlock() == block;
            });
        }

        // Assign the new variable after `val`'s definition.
        rewriter.setInsertionPointAfterValue(val);
        rewriter.create<P4HIR::AssignOp>(val.getLoc(), val, newVar);
    }

    // Replace uses of `op` with a copy in all escaping blocks.
    static void copyOp(mlir::IRRewriter &rewriter, mlir::Operation *op,
                       const BlockSet &escapingBlocks) {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        for (mlir::Block *block : escapingBlocks) {
            rewriter.setInsertionPointToStart(block);
            auto newOp = rewriter.clone(*op);
            rewriter.replaceOpUsesWithinBlock(op, newOp->getResults(), block);
        }
    }
};

};  // namespace P4::P4MLIR::detail

#endif  // P4MLIR_IMPL_IR_UTILS_H
