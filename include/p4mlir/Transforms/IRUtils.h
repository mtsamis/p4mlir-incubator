// SPDX-FileCopyrightText: 2025 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

#ifndef P4MLIR_IMPL_IR_UTILS_H
#define P4MLIR_IMPL_IR_UTILS_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "p4mlir/Dialect/P4HIR/FieldPath.h"
#include "p4mlir/Dialect/P4HIR/Matchers.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_OpInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

namespace P4::P4MLIR::IRUtils {

/// Make `val` accessible to `op` regardless of wherher `op` is inside an action.
mlir::Value getAccessibleValue(mlir::RewriterBase &rewriter, mlir::Operation *op, mlir::Value val);

// Inline `scopeOp`'s body to its parent.
void inlineScope(mlir::RewriterBase &rewriter, P4HIR::ScopeOp scopeOp);

// If `op` is an operation somewhere within `block`, check if we can split it in three parts:
// One with operations before `op`, one with `op` and one with operations after `op`.
// This function should be called to check if it's possible to use `splitBlockAt`.
bool canSplitBlockAt(mlir::Block *block, mlir::Operation *op);

// Split block in three parts: One with operations before `op`, one with `op` and one with
// operations after `op`. Any scopes surrounding `op` may be inlined to perform the split.
std::array<mlir::Block *, 3> splitBlockAt(mlir::RewriterBase &rewriter, mlir::Block *block,
                                          mlir::Operation *op);

// Fix up operations in `block` with uses in other blocks due to splitting.
// The rewriter's insertion point is the location where new variables may be created.
void adjustBlockUses(mlir::RewriterBase &rewriter, mlir::Block *block);

// Helper to create a new empty sub-state for `state`.
P4HIR::ParserStateOp createSubState(mlir::RewriterBase &rewriter, P4HIR::ParserStateOp state,
                                    const llvm::Twine &suffix);

// Helper class to replace one operation in a state with multiple new states.
// During init it creates two "pre" and "post" states in which the code before and after the split
// point is put. Then any number of additional states between pre and post can be created with the
// createXXXState functions. Once done the user should either call finalize to commit changes or
// cancel and any changes will be undone.
class SplitStateRewriter {
 public:
    SplitStateRewriter(mlir::RewriterBase &rewriter, mlir::Operation *op)
        : rewriter(rewriter), op(op), step(CREATED) {}
    SplitStateRewriter(SplitStateRewriter &&) = delete;
    SplitStateRewriter &operator=(SplitStateRewriter &&) = delete;
    SplitStateRewriter(const SplitStateRewriter &) = delete;
    SplitStateRewriter &operator=(const SplitStateRewriter &) = delete;
    ~SplitStateRewriter() {
        assert((step == CANCELED || step == FINALIZED) && "Incorrect state on destruction");
    }

    // Initialize and check if splitting is feasible.
    mlir::LogicalResult init();

    // Create new intermediate state that transitions to `transitionTo` and move `ops` in it.
    // If `ops` is non-null the terminator of the block is erased once moved.
    P4HIR::ParserStateOp createSubState(const llvm::Twine &suffix,
                                        P4HIR::ParserStateOp transitionTo = {},
                                        mlir::Block *ops = nullptr);

    // Same as createSubState but transition to the "post" state specifically.
    P4HIR::ParserStateOp createJoinSubState(const llvm::Twine &suffix, mlir::Block *ops = nullptr) {
        return createSubState(suffix, postState, ops);
    }

    // Undo any changes done so far.
    void cancel();

    // Finalize and commit all changes.
    void finalize();

    mlir::Operation *getSplitOp() { return op; }
    P4HIR::ParserStateOp getState() { return op->getParentOfType<P4HIR::ParserStateOp>(); }

    P4HIR::ParserStateOp getPreState() {
        assert(step == INITIALIZED);
        return preState;
    }

    P4HIR::ParserStateOp getPostState() {
        assert(step == INITIALIZED);
        return postState;
    }

    void setStateInsertionPointAfter(P4HIR::ParserStateOp afterState) {
        stateCreationPoint = afterState;
    }

 private:
    enum RewriteStep { CREATED, INITIALIZED, CANCELED, FINALIZED };

    mlir::RewriterBase &rewriter;
    mlir::Operation *op;
    RewriteStep step;

    P4HIR::ParserStateOp preState;
    P4HIR::ParserStateOp postState;
    mlir::Operation *stateCreationPoint;
};

class PathWalker {
 public:
    using IndirectUsesMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;
    using NodeCallbackFn = std::function<mlir::WalkResult(mlir::Value, P4HIR::FieldPath)>;
    using LeafCallbackFn =
        std::function<mlir::WalkResult(mlir::Operation *, mlir::OpOperand &, P4HIR::FieldPath)>;

    PathWalker() {}
    virtual ~PathWalker() {}

    /// Return an indirect use mapping for P4HIR `control_local` and `symbol_ref`-based accesses.
    static IndirectUsesMap getIndirectSymbolUses(P4HIR::ControlOp control) {
        IndirectUsesMap result;
        control.walk([&](P4HIR::SymToValueOp symbolRef) {
            if (auto referencedControlLocal = symbolRef.getDeclOp<P4HIR::ControlLocalOp>()) {
                auto [it, ins] = result.insert({referencedControlLocal.getVal(), {}});
                it->second.push_back(symbolRef.getResult());
            }
        });
        return result;
    }

    PathWalker &setIndirectUsesMap(const IndirectUsesMap *indirectUsesMap) {
        indirectUses = indirectUsesMap;
        return *this;
    }

    PathWalker &setLeafCallback(LeafCallbackFn cb) {
        leafCb = std::move(cb);
        return *this;
    }

    PathWalker &setNodeCallback(NodeCallbackFn cb) {
        nodeCb = std::move(cb);
        return *this;
    }

    mlir::WalkResult walk(mlir::Value root, mlir::Type rootType = mlir::Type()) {
        if (!rootType) rootType = root.getType();

        P4HIR::FieldPath rootPath(P4HIR::maybeUnref(rootType));
        return walkImpl(root, rootPath);
    }

 protected:
    // Value can only be `root` or otherwise a result of a field access operation.
    mlir::WalkResult walkImpl(mlir::Value value, P4HIR::FieldPath path) {
        auto status = nodeCb(value, path);
        if (status.wasSkipped()) return mlir::WalkResult::advance();
        if (status.wasInterrupted()) return mlir::WalkResult::interrupt();

        for (auto &use : value.getUses()) {
            mlir::Operation *user = use.getOwner();

            auto [resValue, resPath] = lookThroughUser(user, use, path);
            if (resValue) {
                if (walkImpl(resValue, resPath).wasInterrupted())
                    return mlir::WalkResult::interrupt();
            } else {
                if (leafCb(user, use, path).wasInterrupted()) return mlir::WalkResult::interrupt();
            }
        }

        if (indirectUses) {
            auto it = indirectUses->find(value);
            if (it != indirectUses->end()) {
                for (mlir::Value alias : it->second)
                    if (walkImpl(alias, path).wasInterrupted())
                        return mlir::WalkResult::interrupt();
            }
        }

        return mlir::WalkResult::advance();
    }

    /// Helper to look through a P4HIR `struct_extract` or `struct_field_ref` operation.
    std::pair<mlir::Value, P4HIR::FieldPath> lookThroughStructAccess(mlir::Operation *op,
                                                                     mlir::OpOperand &operand,
                                                                     P4HIR::FieldPath path) {
        return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
            .Case<P4HIR::StructExtractOp, P4HIR::StructFieldRefOp>([&](auto structAccessOp) {
                assert((operand.get() == structAccessOp.getInput()) && "Unexpected operand");
                return std::pair{structAccessOp.getResult(), path[structAccessOp.getFieldIndex()]};
            })
            .Default(
                [](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
    }

    /// Helper to look through a P4HIR `array_get` or `array_element_ref` operation.
    std::pair<mlir::Value, P4HIR::FieldPath> lookThroughArrayAccess(mlir::Operation *op,
                                                                    mlir::OpOperand &operand,
                                                                    P4HIR::FieldPath path) {
        return llvm::TypeSwitch<mlir::Operation *, std::pair<mlir::Value, P4HIR::FieldPath>>(op)
            .Case<P4HIR::ArrayGetOp, P4HIR::ArrayElementRefOp>(
                [&](auto arrayAccessOp) -> std::pair<mlir::Value, P4HIR::FieldPath> {
                    assert((operand.get() == arrayAccessOp.getInput()) && "Unexpected operand");

                    unsigned cstIndex;
                    if (matchPattern(arrayAccessOp.getIndex(), m_ConstantInt(&cstIndex)))
                        return {arrayAccessOp.getResult(), path[cstIndex]};

                    return {mlir::Value(), P4HIR::FieldPath()};
                })
            .Default(
                [](mlir::Operation *) { return std::pair{mlir::Value(), P4HIR::FieldPath()}; });
    }

    /// `operand` is an operand in `op` that is used to access the root value's field that
    /// corresponds to `path`. If `op` is a field access operation and the resulting path can be
    /// determined based on `path`, then this function should return the resulting new value and
    /// path. Otherwise an empty value must be returned.
    virtual std::pair<mlir::Value, P4HIR::FieldPath> lookThroughUser(mlir::Operation *op,
                                                                     mlir::OpOperand &operand,
                                                                     P4HIR::FieldPath path) {
        // Default implementation to handle P4HIR operations. Users should override this to also
        // support operations from other dialects.
        auto lookThroughStruct = lookThroughStructAccess(op, operand, path);
        if (lookThroughStruct.first) return lookThroughStruct;

        auto lookThroughArray = lookThroughArrayAccess(op, operand, path);
        if (lookThroughArray.first) return lookThroughStruct;

        return {mlir::Value(), P4HIR::FieldPath()};
    }

    const IndirectUsesMap *indirectUses = nullptr;
    NodeCallbackFn nodeCb;
    LeafCallbackFn leafCb;
};

class IndexableValueRewriter {
    /// A helper class to replace indexable-typed values with new values that may have entirely
    /// different layouts, using P4HIR::FieldPath. Users of an indexable-typed value may be
    /// field-access operations, which produce new FieldPaths based on their operands, or leaf
    /// operations which consume values. A successful transformation using this class should make
    /// the original value transitively dead: it may only have other field-access operations as
    /// users, which themselves are transitively dead.
    /// The root value may either be a reference or a value of an indexable-type and the values to
    /// replace are references or values accordingly.

    IndexableValueRewriter(mlir::RewriterBase &rewriter, mlir::Value root, mlir::Type type,
                           mlir::Value newRoot, mlir::Type newType)
        : rewriter(rewriter), root(root), newRoot(newRoot), indirectUses(nullptr) {
        isRef = mlir::isa<P4HIR::ReferenceType>(type);
        rootType = P4HIR::maybeUnref(type);
        assert(mlir::isa<P4HIR::IndexableTypeInterface>(rootType) &&
               "Expected indexable root types");

        if (newType) {
            assert((mlir::isa<P4HIR::ReferenceType>(newType) == isRef) &&
                   "Expected both refs or values");
            newRootType = P4HIR::maybeUnref(newType);
        }
    }

 public:
    using IndirectUsesMap = llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>;

    /// Minimal constructor.
    IndexableValueRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue)
        : IndexableValueRewriter(rewriter, rootValue, rootValue.getType(), mlir::Value(),
                                 mlir::Type()) {}

    /// Constructor friendly for operation result replacement.
    IndexableValueRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue,
                           mlir::Value newRootValue)
        : IndexableValueRewriter(rewriter, rootValue, rootValue.getType(), newRootValue,
                                 newRootValue.getType()) {}

    /// Constructor friendly for operation argument replacement.
    IndexableValueRewriter(mlir::RewriterBase &rewriter, mlir::Value rootValue,
                           mlir::Type origRootType)
        : IndexableValueRewriter(rewriter, rootValue, origRootType, rootValue,
                                 rootValue.getType()) {}

    IndexableValueRewriter(IndexableValueRewriter &&) = delete;
    IndexableValueRewriter &operator=(IndexableValueRewriter &&) = delete;
    IndexableValueRewriter(const IndexableValueRewriter &) = delete;
    IndexableValueRewriter &operator=(const IndexableValueRewriter &) = delete;
    virtual ~IndexableValueRewriter() {}

    IndexableValueRewriter &setIndirectUsesMap(const IndirectUsesMap *indirectUsesMap) {
        indirectUses = indirectUsesMap;
        return *this;
    }

    /// This function must build the necessary operation to access the field that corresponds to
    /// `field` from `value`, using `loc` as the location of access.
    virtual mlir::Value getFromField(mlir::Location loc, mlir::Value val,
                                     P4HIR::IndexedField field) {
        // Default implementation to handle P4HIR operations. Users should override this to also
        // support operations from other dialects.
        bool isRef = mlir::isa<P4HIR::ReferenceType>(val.getType());
        if (mlir::isa<P4HIR::StructLikeTypeInterface>(field.getParentType())) {
            if (isRef)
                return P4HIR::StructFieldRefOp::create(rewriter, loc, val, field.getIndex());
            else
                return P4HIR::StructExtractOp::create(rewriter, loc, val, field.getIndex());
        } else if (mlir::isa<P4HIR::ArrayType>(field.getParentType())) {
            auto idxType = P4HIR::BitsType::get(rewriter.getContext(), 32, false);
            auto idx = P4HIR::ConstOp::create(rewriter, loc,
                                              P4HIR::IntAttr::get(idxType, field.getIndex()));
            if (isRef)
                return P4HIR::ArrayElementRefOp::create(rewriter, loc, val, idx);
            else
                return P4HIR::ArrayGetOp::create(rewriter, loc, val, idx);
        } else {
            llvm_unreachable("Unexpected type");
            return mlir::Value();
        }
    }

    /// This function builds the necessary operations to access the field that corresponds to
    /// `path` from `value`, using `loc` as the location of access.
    mlir::Value getFromPath(mlir::Location loc, mlir::Value val, P4HIR::FieldPath path) {
        for (auto field : path.iterFields()) val = getFromField(loc, val, field);
        return val;
    }

    /// `op` is an operation that has some operands accessing root value's fields. If it is possible
    /// to make `op` not use these operands then this should be done and then mlir::success() must
    /// be returned. Otherwise mlir::failure() must be returned. These operands and their paths can
    /// be found by using `getPath` on `op`'s arguments.
    virtual mlir::LogicalResult replaceUsesIn(mlir::Operation *op) = 0;

    /// `value` is a value used to access the root value's field that corresponds `path`. If it is
    /// possible to replace `value` with a new equivalent that doesn't depend on the root value,
    /// then this new value should be returned. Otherwise mlir::Value() should be returned.
    virtual mlir::Value getReplacement(mlir::Value value, P4HIR::FieldPath path) {
        return mlir::Value();
    }

    /// If `value` is accessing a field of the root value then return the corresponding field path,
    /// otherwise return the empty path.
    P4HIR::FieldPath getPath(mlir::Value value) {
        auto it = valueToPath.find(value);
        return it != valueToPath.end() ? it->second : P4HIR::FieldPath();
    }

    // Assuming that `op` can only have a single operand with a corresponding field path, return
    // that operand and path.
    std::pair<mlir::OpOperand *, P4HIR::FieldPath> getOperandWithPath(mlir::Operation *op) {
        P4HIR::FieldPath resPath;
        mlir::OpOperand *resOperand = nullptr;
        for (mlir::OpOperand &operand : op->getOpOperands()) {
            if (auto path = getPath(operand.get()); !path.isEmpty()) {
                assert(!resOperand && "Unexpected operation with multiple used paths");
                resPath = path;
                resOperand = &operand;
            }
        }
        assert(resOperand && "A path must exist for leaf operations");
        return {resOperand, resPath};
    }

    /// Make `root` transitively unused by transitively replacing or making users dead.
    mlir::LogicalResult replace() {
        /// Use a vector and deduplicate with a set so we get deterministic order.
        llvm::SmallVector<mlir::Operation *> leafOps;
        mlir::DenseSet<mlir::Operation *> seenLeafOps;

        PathWalker()
            .setIndirectUsesMap(indirectUses)
            .setNodeCallback([&](mlir::Value value, P4HIR::FieldPath path) {
                if (mlir::Operation *def = value.getDefiningOp()) rewriter.setInsertionPoint(def);
                mlir::Value valReplacement = getReplacement(value, path);

                if (valReplacement) {
                    // Stop early: replace `value` with a new one that doesn't depend on the root.
                    rewriter.replaceAllUsesWith(value, valReplacement);
                    return mlir::WalkResult::skip();
                }

                valueToPath.insert({value, path});
                return mlir::WalkResult::advance();
            })
            .setLeafCallback(
                [&](mlir::Operation *user, mlir::OpOperand &operand, P4HIR::FieldPath path) {
                    // It is reasonable to expect that leaf operations may have multiple uses of
                    // root fields, so we don't directly ask for their replacement. Store `user` and
                    // process once we have discovered all leaf paths.
                    if (seenLeafOps.insert(user).second) leafOps.push_back(user);
                    return mlir::WalkResult::advance();
                })
            .walk(root, rootType);

        for (mlir::Operation *op : leafOps) {
            rewriter.setInsertionPoint(op);
            if (failed(replaceUsesIn(op))) return mlir::failure();
        }

        return mlir::success();
    }

 protected:
    mlir::RewriterBase &rewriter;

    /// The value whose uses are being replaced.
    mlir::Value root;
    /// The plain (without p4hir.ref) original type of `root`.
    /// Note that it may differ from `root.getType()` (e.g. when doing dialect ceonversion).
    mlir::Type rootType;
    /// Optional helper: A new root value to use while doing the replacement or `root`.
    mlir::Value newRoot;
    /// Optional helper: A plain (without p4hir.ref) new root type for `value` after replacement.
    mlir::Type newRootType;
    /// True if doing reference replacement, false if doing value replacement.
    bool isRef;

    /// Maintains a mapping from values that are used in leaf operations to field paths.
    llvm::DenseMap<mlir::Value, P4HIR::FieldPath> valueToPath;

    /// Optional mapping of indirect value uses. An indirect use is when a value is accessed in a
    /// way other than normal SSA def-use chains (e.g. through symbols).
    const IndirectUsesMap *indirectUses = nullptr;
};

}  // namespace P4::P4MLIR::IRUtils

#endif  // P4MLIR_IMPL_IR_UTILS_H
