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

#define DEBUG_TYPE "p4hir-inline-parsers"

namespace P4::P4MLIR {
#define GEN_PASS_DEF_INLINEPARSERS
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct InlineParsersPass : public P4::P4MLIR::impl::InlineParsersBase<InlineParsersPass>,
                           protected detail::IRUtils {
    InlineParsersPass() = default;
    void runOnOperation() override;

 private:
    // Struct to hold info for each subparser apply call.
    struct InlineCallInfo {
        InlineCallInfo(P4HIR::ApplyOp applyOp) : applyOp(applyOp) {}

        P4HIR::ParserStateOp getState() const {
            return applyOp->getParentOfType<P4HIR::ParserStateOp>();
        }
        P4HIR::ParserOp getCallerParser() const {
            return applyOp->getParentOfType<P4HIR::ParserOp>();
        }

        P4HIR::ApplyOp applyOp;
        P4HIR::ParserStateOp preState;
        P4HIR::ParserStateOp postState;
    };

    static mlir::StringAttr updateName(mlir::StringAttr attr, mlir::StringRef prefix) {
        if (attr.getValue().empty()) return attr;
        auto newName = (prefix + "." + attr.getValue()).str();
        return mlir::StringAttr::get(attr.getContext(), newName);
    }

    // Append `prefix` to names and refs after inlining `op` to `callerParser`.
    static void updateNamesAndRefs(P4HIR::ParserOp callerParser, llvm::StringRef prefix,
                                   mlir::Operation *op) {
        if (auto name = op->getAttrOfType<mlir::StringAttr>("sym_name"))
            op->setAttr("sym_name", updateName(name, prefix));
        if (auto name = op->getAttrOfType<mlir::StringAttr>("name"))
            op->setAttr("name", updateName(name, prefix));
        if (auto stateSymbol = op->getAttrOfType<mlir::SymbolRefAttr>("state")) {
            auto parserAttr = callerParser.getSymNameAttr();
            auto leaf = stateSymbol.getLeafReference();
            auto leafSymbol = mlir::SymbolRefAttr::get(updateName(leaf, prefix));
            auto newSymbol = mlir::SymbolRefAttr::get(parserAttr, {leafSymbol});
            op->setAttr("state", newSymbol);
        }
    }

    // Given a subparser instantiation `instOp` find all its apply calls and try to inline them.
    // Leaves the IR unchanged in case of failure.
    mlir::LogicalResult inlineSubparser(P4HIR::InstantiateOp instOp, mlir::IRRewriter &rewriter) {
        auto ctx = rewriter.getContext();
        auto mod = instOp->getParentOfType<mlir::ModuleOp>();
        auto parser = instOp->getParentOfType<P4HIR::ParserOp>();
        auto calleeRegion =
            mod.lookupSymbol<P4HIR::ParserOp>(instOp.getCallee()).getCallableRegion();
        auto instName = instOp.getSymName();

        if (!instOp.getArgOperands().empty())
            return rewriter.notifyMatchFailure(
                instOp.getLoc(), "Cannot inline subparser with constructor arguments.");

        llvm::SmallVector<InlineCallInfo, 4> inliningInfo;
        // Collect all ApplyOps for this InstantiateOp.
        parser.walk([&](P4HIR::ApplyOp applyOp) {
            auto applyInstOp = mod.lookupSymbol<P4HIR::InstantiateOp>(applyOp.getCallee());
            if (instOp && applyInstOp == instOp) inliningInfo.emplace_back(applyOp);
        });

        for (auto &info : inliningInfo)
            if (!canSplitBlockAt(info.getState().getBlock(), info.applyOp))
                return rewriter.notifyMatchFailure(info.applyOp.getLoc(),
                                                   "Cannot split state at given position.");

        // Although we are calling the inliner once per apply, parser local declarations must be
        // cloned once per instantiation. The IRMapper argument in InliningUtils cannot be used to
        // map whole operations within the cloned region so we must take care to de-duplicate parser
        // locals ourselves.
        // For each operation in the callee's top level block, `declMapping` holds an operation if we
        // must remmap the I-th operation and nullptr otherwise. We have to use a vector of indices
        // instead of a map for two reasons:
        // 1) When the inliner runs we don't have access to the original operation, only the cloned one.
        // 2) If something fails we must erase these declarations in reverse order.
        // Copying locals here and not in the inliner also makes sure to preserve side effects if we
        // have an instantiation and zero apply calls.
        llvm::SmallVector<mlir::Operation *> declMapping;
        mlir::IRMapping mapper;
        rewriter.setInsertionPoint(parser.getStartState());

        for (mlir::Operation &op : calleeRegion->front()) {
            if (mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::InstantiateOp>(op)) {
                mlir::Operation *newOp = rewriter.clone(op, mapper);
                updateNamesAndRefs(parser, instName, newOp);
                declMapping.push_back(newOp);
            } else {
                declMapping.push_back(nullptr);
            }
        }

        // Try to inline all subparser calls.
        mlir::LogicalResult status = mlir::success();
        size_t apply_counter = 0;
        for (auto &info : inliningInfo) {
            auto applyOp = info.applyOp;
            auto state = info.getState();

            // Create two new sub-states per subparser apply:
            // "pre" will hold the code up to the apply call plus any subparser initialization.
            // "post" will hold the code after the apply call and is the subparser's accept state.
            rewriter.setInsertionPointAfter(state);
            info.preState = createSubState(rewriter, state, "pre");
            rewriter.setInsertionPointAfter(info.preState);
            info.postState = createSubState(rewriter, state, "post");

            std::string prefix = instName.str();
            if (inliningInfo.size() > 1)
                prefix += std::string("#") + std::to_string(apply_counter++);

            ParserInliner inliner(ctx, rewriter, prefix, declMapping, info);
            status = mlir::inlineRegion(inliner, calleeRegion, state, applyOp.getArgOperands(), {},
                                        applyOp.getLoc());

            if (status.failed()) break;
        }

        if (status.failed()) {
            // Erase all newly introduced states and operations from inlining.
            for (auto &info : inliningInfo) {
                if (!info.preState || !info.postState) break;

                auto createdStates =
                    llvm::make_range(info.preState->getIterator(), ++info.postState->getIterator());
                for (mlir::Operation &op : llvm::make_early_inc_range(createdStates))
                    rewriter.eraseOp(&op);
            }

            // Erase all cloned parser locals.
            for (mlir::Operation *op : llvm::make_early_inc_range(llvm::reverse(declMapping)))
                if (op) rewriter.eraseOp(op);

            return status;
        }

        for (auto &info : inliningInfo) {
            auto applyOp = info.applyOp;
            auto stateBB = info.getState().getBlock();
            auto preStateBB = info.preState.getBlock();
            auto postStateBB = info.postState.getBlock();

            // Split code before and after the apply op and move it to "pre"/"post".
            auto [beforeBB, applyBB, afterBB] = splitBlockAt(rewriter, stateBB, applyOp);
            rewriter.inlineBlockBefore(beforeBB, preStateBB, preStateBB->begin());
            rewriter.inlineBlockBefore(afterBB, postStateBB, postStateBB->begin());

            // Replace the apply op with a transition to the "pre" state.
            rewriter.setInsertionPoint(applyOp);
            rewriter.create<P4HIR::ParserTransitionOp>(applyOp.getLoc(),
                                                       info.preState.getSymbolRef());
            rewriter.eraseOp(applyOp);

            // Due to the splitting states, we may have values with uses in other states.
            rewriter.setInsertionPoint(parser.getStartState());
            adjustBlockUses(rewriter, preStateBB);
        }

        // Inlining successful, finally erase the instantiate op.
        rewriter.eraseOp(instOp);

        return mlir::success();
    }

    struct ParserInliner : public mlir::InlinerInterface {
        ParserInliner(mlir::MLIRContext *context, mlir::IRRewriter &rewriter,
                      mlir::StringRef prefix, llvm::SmallVector<mlir::Operation *> &declMapping,
                      InlineCallInfo &info)
            : mlir::InlinerInterface(context),
              rewriter(rewriter),
              prefix(prefix),
              declMapping(declMapping),
              info(info) {}

        mlir::IRRewriter &rewriter;
        mlir::StringRef prefix;
        llvm::SmallVector<mlir::Operation *> &declMapping;
        InlineCallInfo &info;

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
            for (mlir::Block &block : inlinedBlocks) {
                size_t index = 0;
                for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
                    proccessInlinedOp(index, &op);
                    index++;
                }
            }
        }

        // Move inlined operation `op` to its desired position.
        void proccessInlinedOp(size_t index, mlir::Operation *op) {
            // Remmap parser locals if needed.
            if (declMapping[index]) {
                rewriter.replaceOp(op, declMapping[index]);
                return;
            }

            assert((!mlir::isa<P4HIR::ConstOp, P4HIR::VariableOp, P4HIR::InstantiateOp>(op)) &&
                   "Should be handled by caller");

            op->walk(
                [&](mlir::Operation *o) { updateNamesAndRefs(info.getCallerParser(), prefix, o); });

            P4HIR::ParserStateOp postState = info.postState;
            mlir::Block *preStateBB = &info.preState.getBody().back();

            if (auto stateOp = mlir::dyn_cast<P4HIR::ParserStateOp>(op)) {
                rewriter.moveOpBefore(stateOp, postState);

                // Make the subparser's accept transition to the "post" state.
                if (stateOp.isAccept()) {
                    auto acceptOp = mlir::cast<P4HIR::ParserAcceptOp>(stateOp.getNextTransition());
                    rewriter.setInsertionPoint(acceptOp);
                    rewriter.create<P4HIR::ParserTransitionOp>(acceptOp.getLoc(),
                                                               postState.getSymbolRef());
                    rewriter.eraseOp(acceptOp);
                }
            } else if (mlir::isa<P4HIR::ParserTransitionOp>(op)) {
                // Make the subparsers's start transition the terminator of the init block.
                // We need to clone instead of move because otherwise the inlined block will
                // be left without a terminator and assert.
                rewriter.setInsertionPointToEnd(preStateBB);
                rewriter.clone(*op);
            } else {
                // Move other operations (e.g. reads) to the init block.
                rewriter.moveOpBefore(op, preStateBB, preStateBB->end());
            }
        }
    };
};

}  // namespace

void InlineParsersPass::runOnOperation() {
    auto mod = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    // P4 guarantees that subparser definitions come before their use so inlining them
    // in order will be equivalent to a DFS on the subparser call graph.
    llvm::SmallVector<P4HIR::InstantiateOp, 4> subparserInsts;
    getOperation()->walk([&](P4HIR::InstantiateOp instOp) {
        if (mod.lookupSymbol<P4HIR::ParserOp>(instOp.getCallee())) subparserInsts.push_back(instOp);
    });

    for (P4HIR::InstantiateOp instOp : subparserInsts)
        [[maybe_unused]] auto res = inlineSubparser(instOp, rewriter);
}

std::unique_ptr<mlir::Pass> P4::P4MLIR::createInlineParsersPass() {
    return std::make_unique<InlineParsersPass>();
}
