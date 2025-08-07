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
#include "p4mlir/Transforms/IRUtils.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-remove-parser-control-flow"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_REMOVEPARSERCONTROLFLOW
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct RemoveParserControlFlowPass
    : public P4::P4MLIR::impl::RemoveParserControlFlowBase<RemoveParserControlFlowPass>,
      protected detail::IRUtils {
    RemoveParserControlFlowPass() = default;
    void runOnOperation() override;

 private:
    mlir::LogicalResult replaceIfWithTransitions(P4HIR::IfOp ifOp, mlir::IRRewriter &rewriter) {
        auto loc = ifOp.getLoc();
        auto state = ifOp->getParentOfType<P4HIR::ParserStateOp>();
        auto parser = ifOp->getParentOfType<P4HIR::ParserOp>();

        if (!canSplitBlockAt(&state.getBody().back(), ifOp))
            return rewriter.notifyMatchFailure(loc, "Cannot split state at given position.");

        // Create "pre" and "post" states with the operations before and after the if statement.
        auto [beforeBB, ifBB, afterBB] = splitBlockAt(rewriter, &state.getBody().back(), ifOp);
        rewriter.setInsertionPointAfter(state);
        auto preState = createSubState(rewriter, state, "pre", beforeBB);
        rewriter.setInsertionPointAfter(preState);
        auto postState = createSubState(rewriter, state, "post", afterBB);

        // Create a "then" state and move the then region code in it.
        rewriter.setInsertionPointAfter(preState);
        auto thenState = createSubState(rewriter, state, "then", &ifOp.getThenRegion().front());

        // If the ifOp has an else block, then similarly create an "else" state for it.
        P4HIR::ParserStateOp elseState;
        mlir::Block *elseStateBB;
        if (!ifOp.getElseRegion().empty()) {
            rewriter.setInsertionPointAfter(thenState);
            elseState = createSubState(rewriter, state, "else", &ifOp.getElseRegion().front());
            elseStateBB = elseState.getBlock();
        } else {
            // Otherwise the else state is the "post" state.
            elseState = postState;
            elseStateBB = nullptr;
        }

        // Add a (join) transition going from the "then"/"else" states to "post".
        for (auto newStateBB : {thenState.getBlock(), elseStateBB}) {
            if (newStateBB) {
                rewriter.eraseOp(newStateBB->getTerminator());
                rewriter.setInsertionPointToEnd(newStateBB);
                rewriter.create<P4HIR::ParserTransitionOp>(loc, postState.getSymbolRef());
            }
        }

        // Create a select statement that transitions to then/else base on ifOp's condition.
        rewriter.setInsertionPointToEnd(preState.getBlock());
        createBoolTransitionSelect(rewriter, loc, ifOp.getCondition(), thenState, elseState);

        // Replace the empty if statement with a transition to the "pre" state.
        // Any unnecessary states created are to be cleaned by subsequent passes.
        rewriter.setInsertionPoint(ifOp);
        rewriter.create<P4HIR::ParserTransitionOp>(loc, preState.getSymbolRef());
        rewriter.eraseOp(ifOp);

        // Due to the splitting states, we may have values with uses in other states.
        rewriter.setInsertionPoint(parser.getStartState());
        adjustBlockUses(rewriter, preState.getBlock());

        return mlir::success();
    }

    void createBoolTransitionSelect(mlir::IRRewriter &rewriter, mlir::Location loc,
                                    mlir::Value cond, P4HIR::ParserStateOp thenState,
                                    P4HIR::ParserStateOp elseState) {
        auto selectOp = rewriter.create<P4HIR::ParserTransitionSelectOp>(loc, cond);
        mlir::Block *selectBB = &selectOp.getBody().emplaceBlock();
        rewriter.setInsertionPointToStart(selectBB);

        auto buildBoolCase = [&](bool val, P4HIR::ParserStateOp state) {
            rewriter.create<P4HIR::ParserSelectCaseOp>(
                loc,
                [&](mlir::OpBuilder &b, mlir::Location) {
                    auto valAttr = P4HIR::BoolAttr::get(rewriter.getContext(), val);
                    auto caseLabel = b.create<P4HIR::ConstOp>(loc, valAttr);
                    b.create<P4HIR::YieldOp>(loc, mlir::ValueRange(caseLabel));
                },
                state.getSymbolRef());
        };

        buildBoolCase(true, thenState);
        buildBoolCase(false, elseState);
    }
};

}  // namespace

void RemoveParserControlFlowPass::runOnOperation() {
    mlir::IRRewriter rewriter(&getContext());

    // Walk in pre-order because we need to trasnform the outter IfOp in order to enable
    // trasnforming any nested ones.
    llvm::SmallVector<P4HIR::IfOp, 4> ifStatements;
    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](P4HIR::IfOp applyOp) { ifStatements.push_back(applyOp); });

    for (P4HIR::IfOp ifStatement : ifStatements)
        [[maybe_unused]] auto res = replaceIfWithTransitions(ifStatement, rewriter);
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createRemoveParserControlFlowPass() {
    return std::make_unique<RemoveParserControlFlowPass>();
}
