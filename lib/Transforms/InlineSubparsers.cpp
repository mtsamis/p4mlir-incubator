#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-inline-subparsers"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_INLINESUBPARSERS
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct InlineSubparsersPass : public P4::P4MLIR::impl::InlineSubparsersBase<InlineSubparsersPass> {
    InlineSubparsersPass() = default;
    void runOnOperation() override;

 private:
    mlir::LogicalResult inlineSubparserCall(P4HIR::ApplyOp applyOp,
                                            mlir::IRRewriter &rewriter) const {
        auto ctx = rewriter.getContext();
        auto mod = applyOp->getParentOfType<mlir::ModuleOp>();
        auto parser = applyOp->getParentOfType<P4HIR::ParserOp>();
        auto state = applyOp->getParentOfType<P4HIR::ParserStateOp>();

        // We cannot split the state if there are any scopes.
        // Promote any variables to parser locals and inline all scopes.
        state->walk<mlir::WalkOrder::PostOrder>([&](P4HIR::ScopeOp scopeOp) {
            for (mlir::Operation &op :
                 llvm::make_early_inc_range(scopeOp.getScopeRegion().front())) {
                if (mlir::isa<P4HIR::VariableOp>(op)) {
                    rewriter.moveOpBefore(&op, parser.getStartState());
                } else if (mlir::isa<P4HIR::YieldOp>(op)) {
                    break;
                } else {
                    rewriter.moveOpBefore(&op, scopeOp);
                }
            }

            rewriter.eraseOp(scopeOp);
        });

        // Helper to create a new sub-state for `state` and move `ops` in it.
        auto createSubState = [&](llvm::StringRef suffix, mlir::Block *ops) {
            auto stateName =
                mlir::StringAttr::get(ctx, llvm::Twine(state.getSymName()) + "_" + suffix);
            auto newState = rewriter.create<P4HIR::ParserStateOp>(state.getLoc(), stateName,
                                                                  mlir::DictionaryAttr());
            mlir::Block *newStateBB =
                rewriter.createBlock(&newState.getBody(), newState.getBody().begin());
            rewriter.inlineBlockBefore(ops, newStateBB, newStateBB->end());
            return std::pair(newState, newStateBB);
        };

        // Create "pre" and "post" states that will hold the operations before and after the
        // subparser call. Any initialization code of the subparser will also be appended to the
        // "pre" state. The "post" state is also the accept state of the subparser once inlined. The
        // original state is left only with one p4hir.apply operation.
        auto [beforeBB, applyBB, afterBB] = threeWaySplit(rewriter, applyOp);
        rewriter.setInsertionPointAfter(state);
        auto [preState, preStateBB] = createSubState("pre", beforeBB);
        rewriter.setInsertionPointAfter(preState);
        auto [postState, postStateBB] = createSubState("post", afterBB);

        // Call the inliner.
        auto instOp = mod.lookupSymbol<P4HIR::InstantiateOp>(applyOp.getCallee());
        auto calleeParser = mod.lookupSymbol<P4HIR::ParserOp>(instOp.getCallee());
        ParserInliner inliner(ctx, rewriter, instOp.getSymName(), parser, preStateBB, postState);
        [[maybe_unused]] auto res =
            mlir::inlineRegion(inliner, calleeParser.getCallableRegion(), state,
                               applyOp.getArgOperands(), {}, applyOp.getLoc());

        // If isLegalToInline always returns true and we match arguments / results properly
        // the inliner will succeed. If that stops to be the case we need to refactor the code
        // above so that the IR is left unchanged on failure.
        assert(res.succeeded());

        // Replace the p4hir.apply with a transition to the "pre" state.
        // Any unnecessary states created are to be cleaned by subsequent passes.
        rewriter.setInsertionPoint(applyOp);
        rewriter.create<P4HIR::ParserTransitionOp>(applyOp.getLoc(), preState.getSymbolRef());
        rewriter.eraseOp(applyOp);

        // Subparser was entirely inlined, remove the p4hir.instantiate op.
        rewriter.eraseOp(instOp);

        // After inlining, due to the splitting states, we may have values that have uses in other
        // states. We need to promote these values to parser local variables.
        for (mlir::Operation &op : llvm::make_early_inc_range(*preStateBB)) {
            for (auto val : op.getResults()) {
                bool valEscaped = false;
                for (mlir::Operation *user : val.getUsers()) {
                    if (user->getBlock() != preStateBB) {
                        valEscaped = true;
                        break;
                    }
                }

                if (valEscaped) {
                    if (mlir::isa<P4HIR::VariableOp>(op)) {
                        rewriter.moveOpBefore(&op, parser.getStartState());
                        break;
                    } else {
                        promoteToVar(rewriter, parser, val);
                    }
                }
            }
        }

        return mlir::success();
    }

    // Split the block that contains `op` in three blocks:
    // One with operations before `op`, one with `op` and one with operations after `op`.
    std::array<mlir::Block *, 3> threeWaySplit(mlir::IRRewriter &rewriter,
                                               mlir::Operation *op) const {
        mlir::Block *before = op->getBlock();
        mlir::Block *middle = rewriter.splitBlock(before, op->getIterator());
        mlir::Block *after = rewriter.splitBlock(middle, ++op->getIterator());
        return {before, middle, after};
    }

    // Promote `val` to a local variable in `parser`.
    void promoteToVar(mlir::IRRewriter &rewriter, P4HIR::ParserOp parser, mlir::Value val) const {
        rewriter.setInsertionPoint(parser.getStartState());
        auto newVar = rewriter.create<P4HIR::VariableOp>(
            val.getLoc(), P4HIR::ReferenceType::get(val.getType()), std::string{"promoted_local"});

        for (mlir::Operation *user : val.getUsers()) {
            rewriter.setInsertionPoint(user);
            auto newVal = rewriter.create<P4HIR::ReadOp>(user->getLoc(), newVar);

            rewriter.modifyOpInPlace(user, [&] {
                for (mlir::OpOperand &operand : user->getOpOperands()) {
                    if (operand.get() == val) {
                        operand.assign(newVal);
                    }
                }
            });
        }

        rewriter.setInsertionPointAfterValue(val);
        rewriter.create<P4HIR::AssignOp>(val.getLoc(), val, newVar);
    }

    struct ParserInliner : public mlir::InlinerInterface {
        ParserInliner(mlir::MLIRContext *context, mlir::IRRewriter &rewriter,
                      mlir::StringRef prefix, P4HIR::ParserOp destParser, mlir::Block *initBB,
                      P4HIR::ParserStateOp postState)
            : mlir::InlinerInterface(context),
              rewriter(rewriter),
              prefix(prefix),
              destParser(destParser),
              initBB(initBB),
              postState(postState) {}

        mlir::IRRewriter &rewriter;
        mlir::StringRef prefix;
        P4HIR::ParserOp destParser;
        mlir::Block *initBB;
        P4HIR::ParserStateOp postState;

        bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool wouldBeCloned,
                             mlir::IRMapping &valueMapping) const override {
            return true;
        }
        bool isLegalToInline(mlir::Operation *op, mlir::Region *dest, bool wouldBeCloned,
                             mlir::IRMapping &valueMapping) const override {
            return true;
        }
        void handleTerminator(mlir::Operation *op, mlir::ValueRange valuesToRepl) const override {
            assert(mlir::isa<P4HIR::ParserTransitionOp>(op));
            // The transition to subparser's start has been cloned to the "pre" state.
            // We just want the original op removed, which is handled by InliningUtils.
        }

        // Update names and operation positions after inlining.
        void processInlinedBlocks(
            llvm::iterator_range<mlir::Region::iterator> inlinedBlocks) override {
            for (mlir::Block &block : inlinedBlocks)
                block.walk([&](mlir::Operation *op) { updateNamesAndRefs(op); });

            for (mlir::Block &block : inlinedBlocks) {
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    proccessInlinedOp(&op);
                }
            }
        }

        mlir::StringAttr updateName(mlir::StringAttr attr) {
            if (attr.getValue().empty()) return attr;
            return updateName(attr.getValue());
        }

        mlir::StringAttr updateName(mlir::StringRef ref) {
            return mlir::StringAttr::get(rewriter.getContext(), prefix + "::" + ref);
        }

        void updateNamesAndRefs(mlir::Operation *op) {
            if (auto name = op->getAttrOfType<mlir::StringAttr>("sym_name"))
                op->setAttr("sym_name", updateName(name));
            if (auto name = op->getAttrOfType<mlir::StringAttr>("name"))
                op->setAttr("name", updateName(name));
            if (auto stateSymbol = op->getAttrOfType<mlir::SymbolRefAttr>("state")) {
                auto parserAttr =
                    mlir::StringAttr::get(rewriter.getContext(), destParser.getSymName());
                mlir::StringRef leaf = stateSymbol.getLeafReference();
                auto leafSymbol = mlir::SymbolRefAttr::get(updateName(leaf));
                auto newSymbol = mlir::SymbolRefAttr::get(parserAttr, {leafSymbol});
                op->setAttr("state", newSymbol);
            }
        }

        // Move inlined operation `op` to its desired position.
        void proccessInlinedOp(mlir::Operation *op) {
            if (mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::InstantiateOp>(op)) {
                // Move constants and variable declarations to dest parser.
                rewriter.moveOpBefore(op, destParser.getStartState());
            } else if (auto stateOp = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
                // Move states to dest parser but make accept transition to the "post" state.
                if (stateOp.isAccept()) {
                    auto acceptOp = mlir::cast<P4HIR::ParserAcceptOp>(stateOp.getNextTransition());
                    rewriter.setInsertionPoint(acceptOp);
                    rewriter.create<P4HIR::ParserTransitionOp>(acceptOp.getLoc(),
                                                               postState.getSymbolRef());
                    rewriter.eraseOp(acceptOp);
                } else {
                    rewriter.moveOpBefore(op, postState);
                }
            } else if (mlir::isa<P4HIR::ParserTransitionOp>(op)) {
                // Make the subparsers's start transition the terminator of the init block.
                // We need to clone instead of move because otherwise the inlined block will
                // be left without a terminator and assert.
                rewriter.setInsertionPointToEnd(initBB);
                rewriter.clone(*op);
            } else {
                // Move other operations (e.g. reads) to the init block.
                rewriter.moveOpBefore(op, initBB, initBB->end());
            }
        }
    };
};

}  // namespace

void InlineSubparsersPass::runOnOperation() {
    mlir::IRRewriter rewriter(&getContext());

    // P4 guarantees that subparser definitions come before their use so inlining them
    // in order will be equivalent to a DFS on the subparser call graph.
    llvm::SmallVector<P4HIR::ApplyOp, 4> subparserCalls;
    getOperation()->walk([&](P4HIR::ApplyOp applyOp) {
        if (applyOp.isSubparserCall()) subparserCalls.push_back(applyOp);
    });

    for (P4HIR::ApplyOp applyOp : subparserCalls) {
        [[maybe_unused]] auto res = inlineSubparserCall(applyOp, rewriter);
    }
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineSubparsersPass() {
    return std::make_unique<InlineSubparsersPass>();
}
