#ifndef P4MLIR_DIALECT_P4HIRTOLLVMIRTRANSLATION_H
#define P4MLIR_DIALECT_P4HIRTOLLVMIRTRANSLATION_H

namespace mlir {
class DialectRegistry;
class MLIRContext;
}  // namespace mlir

namespace P4::P4MLIR::P4HIR {

void registerP4HIRDialectTranslation(mlir::DialectRegistry &registry);

void registerP4HIRDialectTranslation(mlir::MLIRContext &context);

}  // namespace P4::P4MLIR::P4HIR

#endif  // P4MLIR_DIALECT_P4HIRTOLLVMIRTRANSLATION_H
