#include "p4mlir/Dialect/P4HIR/P4HIRToLLVMIRTranslation.h"

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Dialect.h"

using namespace P4::P4MLIR;

namespace {

class P4HIRDialectLLVMIRTranslationInterface : public mlir::LLVMTranslationDialectInterface {
 public:
    using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

    /// Translates the given operation to LLVM IR using the provided IR builder
    /// and saving the state in `moduleTranslation`.
    mlir::LogicalResult convertOperation(
        mlir::Operation *op, llvm::IRBuilderBase &builder,
        mlir::LLVM::ModuleTranslation &moduleTranslation) const final {
        return mlir::failure();
    }
};
}  // namespace

void P4HIR::registerP4HIRDialectTranslation(mlir::DialectRegistry &registry) {
    registry.insert<P4HIR::P4HIRDialect>();
    registry.addExtension(+[](mlir::MLIRContext *ctx, P4HIR::P4HIRDialect *dialect) {
        dialect->addInterfaces<P4HIRDialectLLVMIRTranslationInterface>();
    });
}

void P4HIR::registerP4HIRDialectTranslation(mlir::MLIRContext &ctx) {
    mlir::DialectRegistry registry;
    P4HIR::registerP4HIRDialectTranslation(registry);
    ctx.appendDialectRegistry(registry);
}
