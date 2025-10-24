#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-enum-elimination"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_ENUMELIMINATION
#include "p4mlir/Transforms/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct EnumEliminationPass : public P4::P4MLIR::impl::EnumEliminationBase<EnumEliminationPass> {
    EnumEliminationPass() = default;
    void runOnOperation() override;
};

void EnumEliminationPass::runOnOperation() {
    mlir::ModuleOp module = getOperation();
    MLIRContext &context = getContext();

    P4HIRTypeConverter converter;
    ConversionTarget target(context);

    converter.addConversion([](P4HIR::EnumType enumType) -> mlir::Type {
        auto repr = mlir::dyn_cast<P4HIR::EnumRepresentationInterface>(enumType);
        if (!repr || !repr.shouldConvert()) return enumType;

        auto underlyingType = mlir::cast<P4HIR::BitsType>(repr.getUnderlyingType());
        auto fields = enumType.getFields().getAsRange<mlir::StringAttr>();
        auto serFields = llvm::map_to_vector(fields, [&](const auto &field) {
            return mlir::NamedAttribute(
                field, P4HIR::IntAttr::get(underlyingType, repr.getEncoding(field)));
        });

        return P4HIR::SerEnumType::get(enumType.getName(), underlyingType, serFields,
                                       enumType.getAnnotations());
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
        if (auto func = dyn_cast<FunctionOpInterface>(op))
            return converter.isLegal(func.getFunctionType());

        if (!converter.isLegal(op->getOperandTypes()) || !converter.isLegal(op->getResultTypes()))
            return false;

        mlir::AttrTypeWalker walker;
        walker.addWalk([&](Type type) {
            return converter.isLegal(type) ? WalkResult::advance() : WalkResult::interrupt();
        });

        for (auto attr : op->getAttrs())
            if (walker.walk(attr.getValue()).wasInterrupted()) return false;

        return true;
    });

    RewritePatternSet patterns(&context);
    populateTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> P4::P4MLIR::createEnumEliminationPass() {
    return std::make_unique<EnumEliminationPass>();
}