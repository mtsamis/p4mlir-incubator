#include "llvm/Support/Casting.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "p4mlir/Conversion/ConversionPatterns.h"
#include "p4mlir/Conversion/Passes.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-lower-to-mlir-core"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOMLIRCORE
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct LowerP4HIRToMLIRCorePass
    : public P4::P4MLIR::impl::LowerP4HIRToMLIRCoreBase<LowerP4HIRToMLIRCorePass> {
    LowerP4HIRToMLIRCorePass() = default;
    void runOnOperation() override;
};

struct FuncOpConversion : public mlir::OpConversionPattern<P4HIR::FuncOp> {
    using OpConversionPattern<P4HIR::FuncOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::FuncOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // TODO
        FailureOr<Operation *> newOp =
            doTypeConversion(op, adaptor.getOperands(), rewriter, getTypeConverter());
        if (failed(newOp)) return failure();

        op = mlir::cast<P4HIR::FuncOp>(*newOp);

        auto type = mlir::FunctionType::get(rewriter.getContext(), op.getFunctionType().getInputs(),
                                            op.getFunctionType().getReturnTypes());
        auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymNameAttr(), type);
        rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.getBody().end());
        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnOpConversion : public mlir::OpConversionPattern<P4HIR::ReturnOp> {
    using OpConversionPattern<P4HIR::ReturnOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ReturnOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

struct ConstOpConversion : public mlir::OpConversionPattern<P4HIR::ConstOp> {
    using OpConversionPattern<P4HIR::ConstOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto newAttr = typeConverter->convertTypeAttribute(op.getValue().getType(), op.getValue());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, mlir::cast<mlir::TypedAttr>(*newAttr));
        return success();
    }
};

struct VariableOpConversion : public mlir::OpConversionPattern<P4HIR::VariableOp> {
    using OpConversionPattern<P4HIR::VariableOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Type convertedElementType =
            typeConverter->convertType(op.getRef().getType().getObjectType());
        mlir::MemRefType memrefType = mlir::MemRefType::get({}, convertedElementType);
        // TODO if (op->hasAttr("init"))?
        rewriter.replaceOpWithNewOp<memref::AllocaOp>(op, memrefType);
        return mlir::success();
    }
};

struct ReadOpConversion : public mlir::OpConversionPattern<P4HIR::ReadOp> {
    using OpConversionPattern<P4HIR::ReadOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ReadOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<memref::LoadOp>(op, adaptor.getOperands()[0]);
        return mlir::success();
    }
};

struct AssignOpConversion : public mlir::OpConversionPattern<P4HIR::AssignOp> {
    using OpConversionPattern<P4HIR::AssignOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<memref::StoreOp>(op, adaptor.getOperands()[0],
                                                     adaptor.getOperands()[1]);
        return mlir::success();
    }
};

struct ScopeOpConversion : public mlir::OpConversionPattern<P4HIR::ScopeOp> {
    using OpConversionPattern<P4HIR::ScopeOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ScopeOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto scope = rewriter.create<memref::AllocaScopeOp>(op.getLoc(), mlir::TypeRange());
        rewriter.inlineRegionBefore(op.getRegion(), scope.getBodyRegion(),
                                    scope.getBodyRegion().end());
        rewriter.eraseOp(op);
        return success();
    }
};

struct YieldOpConversion : public mlir::OpConversionPattern<P4HIR::YieldOp> {
    using OpConversionPattern<P4HIR::YieldOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::YieldOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        if (mlir::isa<memref::AllocaScopeOp>(op->getParentOp())) {
            rewriter.replaceOpWithNewOp<memref::AllocaScopeReturnOp>(op, mlir::ValueRange());
        } else {
            return mlir::failure();
        }

        return mlir::success();
    }
};

struct IfOpConversion : public mlir::OpConversionPattern<P4HIR::IfOp> {
    using OpConversionPattern<P4HIR::IfOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::IfOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        bool hasElse = !op.getElseRegion().empty();
        auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), mlir::TypeRange(),
                                               adaptor.getOperands()[0], hasElse);

        mlir::Block *thenBlock = &op.getThenRegion().front();
        mlir::Operation *thenTerminator = thenBlock->getTerminator();
        rewriter.inlineBlockBefore(thenBlock, ifOp.getThenRegion().front().getTerminator());
        rewriter.eraseOp(thenTerminator);

        if (hasElse) {
            mlir::Block *elseBlock = &op.getElseRegion().front();
            mlir::Operation *elseTerminator = elseBlock->getTerminator();
            rewriter.inlineBlockBefore(elseBlock, ifOp.getElseRegion().front().getTerminator());
            rewriter.eraseOp(elseTerminator);
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct UnaryOpConversion : public mlir::OpConversionPattern<P4HIR::UnaryOp> {
    using OpConversionPattern<P4HIR::UnaryOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::UnaryOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        switch (op.getKind()) {
            case P4HIR::UnaryOpKind::Neg:
            case P4HIR::UnaryOpKind::Cmpl: {
                auto type = mlir::cast<mlir::IntegerType>(typeConverter->convertType(op.getType()));
                auto minusOneAttr =
                    mlir::IntegerAttr::get(type, llvm::APInt::getAllOnes(type.getWidth()));
                auto minusOneValue = rewriter.create<arith::ConstantOp>(op.getLoc(), minusOneAttr);
                if (op.getKind() == P4HIR::UnaryOpKind::Neg)
                    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getOperands()[0],
                                                               minusOneValue);
                else
                    rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getOperands()[0],
                                                               minusOneValue);
                break;
            }
            // TODO P4HIR::UnaryOpKind::LNot
            default:
                return mlir::failure();
        }
        return mlir::success();
    }
};

struct BinOpConversion : public mlir::OpConversionPattern<P4HIR::BinOp> {
    using OpConversionPattern<P4HIR::BinOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::BinOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        switch (op.getKind()) {
            case P4HIR::BinOpKind::Add:
                rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Sub:
                rewriter.replaceOpWithNewOp<arith::SubIOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Mul:
                rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Or:
                rewriter.replaceOpWithNewOp<arith::OrIOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Xor:
                rewriter.replaceOpWithNewOp<arith::XOrIOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::And:
                rewriter.replaceOpWithNewOp<arith::AndIOp>(op, adaptor.getOperands());
                break;
            // TODO
            // case P4HIR::BinOpKind::Div:
            // case P4HIR::BinOpKind::Mod:
            // case P4HIR::BinOpKind::SAdd:
            // case P4HIR::BinOpKind::SSub:
            default:
                return mlir::failure();
        }
        return mlir::success();
    }
};

class MLIRCoreTypeConverter : public P4HIRTypeConverter {
 public:
    MLIRCoreTypeConverter() {
        addConversion([&](P4HIR::BitsType bitsType) {
            return mlir::IntegerType::get(
                bitsType.getContext(), bitsType.getWidth()
                /* , bitsType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned */);
        });

        addConversion([&](P4HIR::BoolType boolType) {
            return mlir::IntegerType::get(boolType.getContext(), 1);
        });

        addTypeAttributeConversion([&](mlir::Type type, P4HIR::IntAttr attr) {
            return mlir::IntegerAttr::get(convertType(type), attr.getValue());
        });

        addTypeAttributeConversion([&](mlir::Type type, P4HIR::BoolAttr attr) {
            return mlir::BoolAttr::get(attr.getContext(), attr.getValue());
        });

        addConversion([&](P4HIR::ReferenceType refType) {
            return mlir::MemRefType::get({}, convertType(refType.getObjectType()));
        });
    }
};

void LowerP4HIRToMLIRCorePass::runOnOperation() {
    mlir::MLIRContext &context = getContext();
    mlir::ModuleOp mod = getOperation();

    mlir::ConversionTarget target(context);
    mlir::RewritePatternSet patterns(&context);
    MLIRCoreTypeConverter typeConverter;

    target.addLegalDialect<mlir::BuiltinDialect, arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, scf::SCFDialect>();

    P4HIR::populateP4HIRToMLIRCoreConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}

}  // end anonymous namespace

std::unique_ptr<Pass> P4::P4MLIR::createLowerP4HIRToMLIRCorePass() {
    return std::make_unique<LowerP4HIRToMLIRCorePass>();
}

void P4HIR::populateP4HIRToMLIRCoreConversionPatterns(mlir::TypeConverter &converter,
                                                      mlir::RewritePatternSet &patterns) {
    mlir::MLIRContext *ctx = patterns.getContext();
    patterns.add<FuncOpConversion, ReturnOpConversion, ConstOpConversion, VariableOpConversion,
                 ReadOpConversion, AssignOpConversion, ScopeOpConversion, YieldOpConversion,
                 IfOpConversion, UnaryOpConversion, BinOpConversion>(converter, ctx);
}
