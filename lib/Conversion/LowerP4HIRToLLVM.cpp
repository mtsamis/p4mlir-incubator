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
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

#define DEBUG_TYPE "p4hir-lower-to-llvm"

using namespace mlir;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

using namespace P4::P4MLIR;

namespace {
struct LowerP4HIRToLLVMPass : public P4::P4MLIR::impl::LowerP4HIRToLLVMBase<LowerP4HIRToLLVMPass> {
    LowerP4HIRToLLVMPass() = default;
    void runOnOperation() override;
};

template <typename OpTy>
struct FuncLikeOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto funcType = op.getFunctionType();
        auto retType = funcType.getOptionalReturnType();
        auto mlirFuncType =
            mlir::FunctionType::get(rewriter.getContext(), funcType.getInputs(),
                                    retType ? mlir::TypeRange(retType) : mlir::TypeRange());

        TypeConverter::SignatureConversion result(op.getNumArguments());
        auto llvmFuncType =
            this->getTypeConverter()->convertFunctionSignature(mlirFuncType, false, false, result);

        if (!llvmFuncType) return rewriter.notifyMatchFailure(op, "signature conversion failed");

        // TODO see what attributes we want to handle.
        auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), op.getName(), llvmFuncType);
        cast<FunctionOpInterface>(newFuncOp.getOperation()).setVisibility(op.getVisibility());

        rewriter.inlineRegionBefore(op.getFunctionBody(), newFuncOp.getBody(), newFuncOp.end());

        // Convert just the entry block.
        // Arguments in other blocks are converted in BrOp/CondBrOp.
        if (!newFuncOp.getBody().empty())
            rewriter.applySignatureConversion(&newFuncOp.getBody().front(), result,
                                              this->getTypeConverter());

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ExternOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ExternOp> {
    using ConvertOpToLLVMPattern<P4HIR::ExternOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ExternOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

template <typename OpTy>
struct CallLikeOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto converter = this->getTypeConverter();
        auto res = converter->convertType(op.getResult().getType());
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, res ? mlir::TypeRange(res) : mlir::TypeRange(),
            adaptor.getCallee().getLeafReference(), adaptor.getArgOperands());
        return mlir::success();
    }
};

struct ReturnOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ReturnOp> {
    using ConvertOpToLLVMPattern<P4HIR::ReturnOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ReturnOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

bool isMaterializableConstant(P4HIR::ConstOp op) {
    return mlir::isa<P4HIR::BoolType, P4HIR::BitsType>(op.getType());
}

struct ConstOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ConstOp> {
    using ConvertOpToLLVMPattern<P4HIR::ConstOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ConstOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        if (isMaterializableConstant(op)) {
            auto newAttr = getTypeConverter()->convertTypeAttribute(op.getType(), op.getValue());
            rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op,
                                                          mlir::cast<mlir::TypedAttr>(*newAttr));
        } else {
            rewriter.eraseOp(op);
        }
        return success();
    }
};

struct VariableOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::VariableOp> {
    using ConvertOpToLLVMPattern<P4HIR::VariableOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::VariableOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto one = rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getIntegerType(64),
                                                     rewriter.getI64IntegerAttr(1));
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto elmType = getTypeConverter()->convertType(op.getType().getObjectType());
        rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(op, ptrType, elmType, one);
        return mlir::success();
    }
};

mlir::Value extOrTrunc(mlir::OpBuilder &builder, mlir::Location loc, mlir::Type dstType,
                       mlir::Value srcValue, bool isSigned = false) {
    auto srcLen = mlir::cast<mlir::IntegerType>(srcValue.getType()).getWidth();
    auto dstLen = mlir::cast<mlir::IntegerType>(dstType).getWidth();
    if (dstLen < srcLen)
        return builder.create<LLVM::TruncOp>(loc, dstType, srcValue);
    else if (dstLen > srcLen && isSigned)
        return builder.create<LLVM::SExtOp>(loc, dstType, srcValue);
    else if (dstLen > srcLen && !isSigned)
        return builder.create<LLVM::ZExtOp>(loc, dstType, srcValue);
    else
        return srcValue;
}

struct CastOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::CastOp> {
    using ConvertOpToLLVMPattern<P4HIR::CastOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::CastOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto examineType = [](mlir::Type type) -> std::optional<std::pair<unsigned, bool>> {
            if (auto boolType = mlir::dyn_cast<P4HIR::BoolType>(type)) return std::pair{1, false};
            if (auto bitsType = mlir::dyn_cast<P4HIR::BitsType>(type))
                return std::pair{bitsType.getWidth(), bitsType.isSigned()};
            return std::nullopt;
        };

        auto srcType = examineType(op.getSrc().getType());
        if (!srcType) return mlir::failure();

        auto [srcLen, srcSigned] = srcType.value();
        mlir::Type destType = getTypeConverter()->convertType(op.getType());

        auto extVal = extOrTrunc(rewriter, op.getLoc(), destType, adaptor.getSrc(), srcSigned);
        rewriter.replaceOp(op, extVal);
        return mlir::success();
    }
};

struct UnaryOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::UnaryOp> {
    using ConvertOpToLLVMPattern<P4HIR::UnaryOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::UnaryOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto type = mlir::cast<mlir::IntegerType>(getTypeConverter()->convertType(op.getType()));

        switch (op.getKind()) {
            case P4HIR::UnaryOpKind::LNot:
            case P4HIR::UnaryOpKind::Cmpl: {
                auto minusOne = rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), type, rewriter.getIntegerAttr(type, -1));
                rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, adaptor.getInput(), minusOne);
                break;
            }
            case P4HIR::UnaryOpKind::Neg: {
                auto zero = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type,
                                                              rewriter.getIntegerAttr(type, 0));
                rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, zero, adaptor.getInput());
                break;
            }
            default:
                return mlir::failure();
        }
        return mlir::success();
    }
};

struct BinOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::BinOp> {
    using ConvertOpToLLVMPattern<P4HIR::BinOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::BinOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        bool isSigned = mlir::cast<P4HIR::BitsType>(op.getLhs().getType()).isSigned();
        switch (op.getKind()) {
            case P4HIR::BinOpKind::Add:
                rewriter.replaceOpWithNewOp<LLVM::AddOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Sub:
                rewriter.replaceOpWithNewOp<LLVM::SubOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Mul:
                rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Or:
                rewriter.replaceOpWithNewOp<LLVM::OrOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Xor:
                rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::And:
                rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Div:
                if (isSigned)
                    rewriter.replaceOpWithNewOp<LLVM::SDivOp>(op, adaptor.getOperands());
                else
                    rewriter.replaceOpWithNewOp<LLVM::UDivOp>(op, adaptor.getOperands());
                break;
            case P4HIR::BinOpKind::Mod:
                if (isSigned)
                    rewriter.replaceOpWithNewOp<LLVM::SRemOp>(op, adaptor.getOperands());
                else
                    rewriter.replaceOpWithNewOp<LLVM::URemOp>(op, adaptor.getOperands());
                break;
            // TODO
            // case P4HIR::BinOpKind::SAdd:
            // case P4HIR::BinOpKind::SSub:
            default:
                return mlir::failure();
        }
        return mlir::success();
    }
};

template <typename OpTy>
struct ShiftOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // Converter values are signless so get signedness from original type.
        bool isSigned = mlir::cast<P4HIR::BitsType>(op.getLhs().getType()).isSigned();
        auto lhs = adaptor.getLhs();
        auto type = mlir::cast<mlir::IntegerType>(lhs.getType());
        // LLVM shifts require both types to have same width, so extend Rhs to match Lhs.
        auto rhs = extOrTrunc(rewriter, op.getLoc(), type, adaptor.getRhs());
        auto lhsWidth = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), type, rewriter.getIntegerAttr(type, type.getWidth()));
        auto overflowCond =
            rewriter.create<LLVM::ICmpOp>(op.getLoc(), LLVM::ICmpPredicate::sge, rhs, lhsWidth);
        mlir::Value shiftVal;
        mlir::Value overflowVal;

        // There are many ways to lower the overflow semantics, we choose a `select` based approach
        // since it's intuitive. LLVM is able to figure this pattern and will produce suitable code.
        if constexpr (std::is_same_v<OpTy, P4HIR::ShlOp>) {
            shiftVal = rewriter.create<LLVM::ShlOp>(op.getLoc(), lhs, rhs);
            overflowVal = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type,
                                                            rewriter.getIntegerAttr(type, 0));
        } else {
            if (isSigned) {
                shiftVal = rewriter.create<LLVM::AShrOp>(op.getLoc(), lhs, rhs);
                auto maxShift = rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), type, rewriter.getIntegerAttr(type, type.getWidth() - 1));
                overflowVal =
                    rewriter.create<LLVM::AShrOp>(op.getLoc(), adaptor.getLhs(), maxShift);
            } else {
                shiftVal = rewriter.create<LLVM::LShrOp>(op.getLoc(), lhs, rhs);
                overflowVal = rewriter.create<LLVM::ConstantOp>(op.getLoc(), type,
                                                                rewriter.getIntegerAttr(type, 0));
            }
        }

        rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, overflowCond, overflowVal, shiftVal);
        return mlir::success();
    }
};

struct ConcatOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ConcatOp> {
    using ConvertOpToLLVMPattern<P4HIR::ConcatOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ConcatOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto lhsType = mlir::cast<mlir::IntegerType>(adaptor.getLhs().getType());
        auto rhsType = mlir::cast<mlir::IntegerType>(adaptor.getRhs().getType());
        auto resType =
            mlir::IntegerType::get(rewriter.getContext(), lhsType.getWidth() + rhsType.getWidth());
        auto lhsExt = rewriter.create<LLVM::ZExtOp>(op.getLoc(), resType, adaptor.getLhs());
        auto rhsExt = rewriter.create<LLVM::ZExtOp>(op.getLoc(), resType, adaptor.getRhs());
        auto shiftAmt = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), resType, rewriter.getIntegerAttr(resType, rhsType.getWidth()));
        auto high = rewriter.create<LLVM::ShlOp>(op.getLoc(), lhsExt, shiftAmt);
        rewriter.replaceOpWithNewOp<LLVM::XOrOp>(op, high, rhsExt);
        return mlir::success();
    }
};

struct CmpOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::CmpOp> {
    using ConvertOpToLLVMPattern<P4HIR::CmpOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::CmpOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto lhsType = mlir::dyn_cast<P4HIR::BitsType>(op.getLhs().getType());
        bool isSigned = lhsType && lhsType.isSigned();
        LLVM::ICmpPredicate pred;

        switch (op.getKind()) {
            case P4HIR::CmpOpKind::Eq:
                pred = LLVM::ICmpPredicate::eq;
                break;
            case P4HIR::CmpOpKind::Ne:
                pred = LLVM::ICmpPredicate::ne;
                break;
            case P4HIR::CmpOpKind::Lt:
                pred = isSigned ? LLVM::ICmpPredicate::slt : LLVM::ICmpPredicate::ult;
                break;
            case P4HIR::CmpOpKind::Le:
                pred = isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule;
                break;
            case P4HIR::CmpOpKind::Gt:
                pred = isSigned ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::ugt;
                break;
            case P4HIR::CmpOpKind::Ge:
                pred = isSigned ? LLVM::ICmpPredicate::sge : LLVM::ICmpPredicate::uge;
                break;
            default:
                return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, pred, adaptor.getLhs(), adaptor.getRhs());
        return mlir::success();
    }
};

FailureOr<Block *> getConvertedBlock(ConversionPatternRewriter &rewriter,
                                     const TypeConverter *converter, Operation *branchOp,
                                     Block *block, TypeRange expectedTypes) {
    assert(converter && "expected non-null type converter");
    assert(!block->isEntryBlock() && "entry blocks have no predecessors");

    // There is nothing to do if the types already match.
    if (block->getArgumentTypes() == expectedTypes) return block;

    // Compute the new block argument types and convert the block.
    std::optional<TypeConverter::SignatureConversion> conversion =
        converter->convertBlockSignature(block);
    if (!conversion)
        return rewriter.notifyMatchFailure(branchOp, "could not compute block signature");
    if (expectedTypes != conversion->getConvertedTypes())
        return rewriter.notifyMatchFailure(
            branchOp, "mismatch between adaptor operand types and computed block signature");
    return rewriter.applySignatureConversion(block, *conversion, converter);
}

struct BrOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::BrOp> {
    using ConvertOpToLLVMPattern<P4HIR::BrOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::BrOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        FailureOr<Block *> convertedBlock = getConvertedBlock(
            rewriter, getTypeConverter(), op, op.getDest(), mlir::TypeRange(adaptor.getOperands()));
        if (failed(convertedBlock)) return failure();
        rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, adaptor.getOperands(), *convertedBlock);
        return mlir::success();
    }
};

struct CondBrOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::CondBrOp> {
    using ConvertOpToLLVMPattern<P4HIR::CondBrOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::CondBrOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        FailureOr<Block *> convertedTrueBlock =
            getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestTrue(),
                              mlir::TypeRange(adaptor.getDestOperandsTrue()));
        if (failed(convertedTrueBlock)) return failure();

        FailureOr<Block *> convertedFalseBlock =
            getConvertedBlock(rewriter, getTypeConverter(), op, op.getDestFalse(),
                              mlir::TypeRange(adaptor.getDestOperandsFalse()));
        if (failed(convertedFalseBlock)) return failure();

        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
            op, adaptor.getCond(), *convertedTrueBlock, adaptor.getDestOperandsTrue(),
            *convertedFalseBlock, adaptor.getDestOperandsFalse());
        return mlir::success();
    }
};

struct ReadOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ReadOp> {
    using ConvertOpToLLVMPattern<P4HIR::ReadOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ReadOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Type destType = getTypeConverter()->convertType(op.getType());
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, destType, adaptor.getRef());
        return mlir::success();
    }
};

struct AssignOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::AssignOp> {
    using ConvertOpToLLVMPattern<P4HIR::AssignOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::AssignOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), adaptor.getRef());
        return mlir::success();
    }
};

template <typename OpTy>
struct StructLikeOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto type = this->getTypeConverter()->convertType(op.getType());
        mlir::Value result = rewriter.create<LLVM::UndefOp>(op.getLoc(), type);
        unsigned structPos = 0;
        for (auto val : adaptor.getInput())
            result = rewriter.create<LLVM::InsertValueOp>(op.getLoc(), result, val, structPos++);
        rewriter.replaceOp(op, mlir::ValueRange(result));
        return mlir::success();
    }
};

struct ArrayGetOpConversion : public ConvertOpToLLVMPattern<P4HIR::ArrayGetOp> {
    using ConvertOpToLLVMPattern<P4HIR::ArrayGetOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ArrayGetOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // TODO we have an issue with value arrays here.
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto elmType = getTypeConverter()->convertType(op.getResult().getType());
        auto elmPtr =
            rewriter.create<LLVM::GEPOp>(op.getLoc(), ptrType, elmType, adaptor.getInput(),
                                         ArrayRef<LLVM::GEPArg>{adaptor.getIndex()});
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elmType, elmPtr);
        return mlir::success();
    }
};

struct ArrayElementRefOpConversion : public ConvertOpToLLVMPattern<P4HIR::ArrayElementRefOp> {
    using ConvertOpToLLVMPattern<P4HIR::ArrayElementRefOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ArrayElementRefOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto elmType = getTypeConverter()->convertType(op.getResult().getType());
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, ptrType, elmType, adaptor.getInput(),
                                                 ArrayRef<LLVM::GEPArg>{adaptor.getIndex()});
        return mlir::success();
    }
};

template <typename OpTy>
struct StructExtractLikeOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, adaptor.getInput(),
                                                          adaptor.getFieldIndex());
        return mlir::success();
    }
};

struct StructExtractRefOpConversion : public ConvertOpToLLVMPattern<P4HIR::StructExtractRefOp> {
    using ConvertOpToLLVMPattern<P4HIR::StructExtractRefOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::StructExtractRefOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto fieldType = getTypeConverter()->convertType(op.getFieldType());
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, ptrType, fieldType, adaptor.getInput(),
                                                 ArrayRef<LLVM::GEPArg>{adaptor.getFieldIndex()});
        return mlir::success();
    }
};

static std::string nameFor(mlir::Operation *op) {
    if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
        return (llvm::Twine("_p_") + parserOp.getSymName()).str();
    } else if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
        return (llvm::Twine("_c_") + controlOp.getSymName()).str();
    } else {
        llvm_unreachable("Invalid op");
        return {};
    }
}

static std::string memberFn(llvm::StringRef parent, llvm::StringRef child) {
    return (parent + "_m_" + child).str();
}

static std::string memberFn(llvm::StringRef parent, P4HIR::ParserStateOp stateOp) {
    return memberFn(parent, stateOp.getSymName());
}

static std::string memberFn(llvm::StringRef parent, P4HIR::FuncOp funcOp) {
    return memberFn(parent, funcOp.getSymName());
}

static std::string specialInitFn(llvm::StringRef parent) { return memberFn(parent, "_init"); }

static std::string specialInitFn(mlir::Operation *parent) { return specialInitFn(nameFor(parent)); }

static std::string specialApplyFn(llvm::StringRef parent) { return memberFn(parent, "_apply"); }

static std::string specialApplyFn(mlir::Operation *parent) {
    return specialApplyFn(nameFor(parent));
}

// Helper to transform parsers and controls.
// TODO find appropriate name and generalize further.
struct PCHelper {
    PCHelper(const TypeConverter *converter, mlir::ConversionPatternRewriter &rewriter,
               mlir::Operation *op)
        : converter(converter), rewriter(rewriter), op(op), loc(op->getLoc()) {
        ctx = rewriter.getContext();
        mod = op->getParentOfType<mlir::ModuleOp>();
    }

    void addLocal(mlir::Operation *op) {
        if (auto variableOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            localVars.push_back(variableOp);
            auto objType =
                mlir::cast<P4HIR::ReferenceType>(variableOp.getRef().getType()).getObjectType();
            localTypes.push_back(objType);
        } else {
            // ControlLocalOp
        }
    }

    void add(mlir::Operation *op) {
        if (mlir::isa<P4HIR::VariableOp, P4HIR::ControlLocalOp>(op)) {
            addLocal(op);
        } else if (mlir::isa<P4HIR::ConstOp>(op)) {
            constantOps.push_back(op);
        } else if (mlir::isa<P4HIR::ParserStateOp, P4HIR::FuncOp>(op)) {
            methodOps.push_back(op);
        } else if (mlir::isa<P4HIR::ParserTransitionOp, P4HIR::ControlApplyOp>(op)) {
            applyOps.push_back(op);
        } else {
            initOps.push_back(op);
        }
    }

    void init() {
        ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

        if (mlir::isa<P4HIR::ParserOp>(op)) {
            retTy = mlir::IntegerType::get(ctx, 1);
        } else {
            retTy = LLVM::LLVMVoidType::get(ctx);
        }

        // stateTy = LLVM::LLVMStructType::getNewIdentified(ctx, "parser_state", localTypes, true);
    }

    std::pair<LLVM::LLVMFuncOp, mlir::IRMapping> createFn(mlir::Type returnType,
                                                          llvm::StringRef name, bool addArguments) {
        llvm::SmallVector<mlir::Type, 8> signature;

        // `this` state pointer.
        signature.push_back(ptrTy);

        if (addArguments) {
            for (auto arg : arguments) {
                signature.push_back(converter->convertType(arg.getType()));
            }
        }

        auto funcType = LLVM::LLVMFunctionType::get(returnType, signature);

        auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, name, funcType);
        func.setPrivate();

        func->setAttr("member_of", mlir::StringAttr::get(ctx, nameFor(op)));

        Block *entryBlock = func.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);

        mlir::IRMapping mapping;

        auto thisPtr = func.getArgument(0);
        for (size_t i = 0; i < localVars.size(); i++) {
            auto localVar = localVars[i];
            auto type = localTypes[i];
            auto fieldType = converter->convertType(type);
            auto gepOp = rewriter.create<LLVM::GEPOp>(localVar->getLoc(), ptrTy, fieldType, thisPtr,
                                                      ArrayRef<LLVM::GEPArg>{i});
            auto refType = P4HIR::ReferenceType::get(type);
            auto ucOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
                localVar->getLoc(), refType, gepOp->getResults());
            mapping.map(localVar->getResults(), ucOp->getResults());
        }

        if (addArguments) {
            for (size_t i = 0; i < arguments.size(); i++) {
                auto arg = arguments[i];
                auto ucOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
                    arg.getLoc(), arg.getType(), func.getArgument(i + 1));
                assert(ucOp->getNumResults() == 1);
                mapping.map(arg, ucOp->getResult(0));
            }
        }

        // Clone all constants and let the canonicalizer clean up.
        for (auto constOp : constantOps) {
            auto newConstOp = rewriter.clone(*constOp);
            mapping.map(constOp->getResults(), newConstOp->getResults());
        }

        return {func, std::move(mapping)};
    }

    void createInitFn() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());

        auto [func, mapping] = createFn(LLVM::LLVMVoidType::get(ctx), specialInitFn(op), true);

        for (mlir::Operation *op : initOps) {
            rewriter.clone(*op, mapping);
        }

        rewriter.create<LLVM::ReturnOp>(loc, mlir::ValueRange());
    }

    void createApplyFn() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());

        auto [func, mapping] = createFn(retTy, specialApplyFn(op), true);

        for (mlir::Operation *op : applyOps) {
            rewriter.clone(*op, mapping);
        }
    }

    void createMemberFns() {
        auto getName = [&](mlir::Operation *fn) -> std::string {
            if (auto stateOp = mlir::dyn_cast<P4HIR::ParserStateOp>(fn)) {
                return memberFn(nameFor(op), stateOp);
            } else if (auto funcOp = mlir::dyn_cast<P4HIR::FuncOp>(fn)) {
                return memberFn(nameFor(op), funcOp);
            } else {
                llvm_unreachable("Invalid member fn");
                return {};
            }
        };

        for (mlir::Operation *fn : methodOps) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(mod.getBody());

            auto [func, mapping] = createFn(retTy, getName(fn), true);

            rewriter.cloneRegionBefore(fn->getRegion(0), func.getBody(), func.getBody().end(),
                                       mapping);

            auto it = func.getBody().begin();
            rewriter.setInsertionPointToEnd(&*it);
            rewriter.create<LLVM::BrOp>(func.getLoc(), &*std::next(it));
        }
    }

    mlir::Type stateTy;
    mlir::Type retTy;
    mlir::Type ptrTy;

    mlir::ValueRange arguments;

    llvm::SmallVector<mlir::Operation *, 4> localVars;
    llvm::SmallVector<mlir::Type, 4> localTypes;
    llvm::SmallVector<mlir::Operation *, 16> initOps;
    llvm::SmallVector<mlir::Operation *, 16> constantOps;
    llvm::SmallVector<mlir::Operation *, 16> applyOps;
    llvm::SmallVector<mlir::Operation *, 16> methodOps;

    const TypeConverter *converter;
    mlir::ConversionPatternRewriter &rewriter;
    mlir::Operation *op;

    mlir::Location loc;
    mlir::MLIRContext *ctx;
    mlir::ModuleOp mod;
};

struct ParserOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ParserOp> {
    using ConvertOpToLLVMPattern<P4HIR::ParserOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        llvm::SmallVector<mlir::Type, 4> parserLocalTypes;

        PCHelper pca(getTypeConverter(), rewriter, op);
        pca.arguments = op.getArguments();

        for (mlir::Operation &op : op.getBody().front()) {
            pca.add(&op);
        }

        pca.init();
        pca.createInitFn();
        pca.createApplyFn();
        pca.createMemberFns();

        rewriter.eraseOp(op);

        return mlir::success();
    }
};

mlir::Operation *createTailCallStateTransition(mlir::Operation *op, mlir::SymbolRefAttr state,
                                               mlir::ConversionPatternRewriter &rewriter) {
    auto parent = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!parent) return {};

    auto parentName = parent->getAttrOfType<mlir::StringAttr>("member_of");
    if (!parentName) return {};

    auto callee = memberFn(parentName.getValue(), state.getLeafReference());
    auto callOp = rewriter.create<LLVM::CallOp>(op->getLoc(), parent.getResultTypes(), callee,
                                                parent.getArguments());
    return rewriter.create<LLVM::ReturnOp>(op->getLoc(), mlir::ValueRange(callOp.getResult()));
}

struct ParserTransitionOpConversion
    : public mlir::ConvertOpToLLVMPattern<P4HIR::ParserTransitionOp> {
    using ConvertOpToLLVMPattern<P4HIR::ParserTransitionOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto returnOp = createTailCallStateTransition(op, op.getState(), rewriter);
        if (!returnOp) return mlir::failure();
        rewriter.replaceOp(op, returnOp);
        return mlir::success();
    }
};

struct ParserTransitionSelectOpConversion
    : public mlir::ConvertOpToLLVMPattern<P4HIR::ParserTransitionSelectOp> {
    using ConvertOpToLLVMPattern<P4HIR::ParserTransitionSelectOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserTransitionSelectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Block *termBlock = op->getBlock();
        mlir::Region *region = termBlock->getParent();
        auto loc = op.getLoc();

        auto convertSelectCase = [&](mlir::Block *block, P4HIR::ParserSelectCaseOp caseOp) {
            mlir::Block *continueBlock = nullptr;
            rewriter.setInsertionPointToEnd(block);

            if (!caseOp.isDefault()) {
                mlir::Block *thenBlock = rewriter.createBlock(region, ++Region::iterator(block));
                continueBlock = rewriter.createBlock(region, ++Region::iterator(thenBlock));

                auto yield = mlir::cast<P4HIR::YieldOp>(caseOp.getBody()->getTerminator());
                rewriter.inlineBlockBefore(caseOp.getBody(), block, block->begin());

                rewriter.setInsertionPoint(yield);
                auto cond = createKeyCondition(yield.getOperand(0), op.getSelect(), rewriter);
                rewriter.create<LLVM::CondBrOp>(caseOp.getLoc(), cond, thenBlock, continueBlock);
                rewriter.eraseOp(yield);
                rewriter.setInsertionPointToStart(thenBlock);
            }

            createTailCallStateTransition(op, caseOp.getState(), rewriter);

            return continueBlock;
        };

        mlir::Block *newBlock = rewriter.createBlock(region, ++Region::iterator(termBlock));
        rewriter.setInsertionPointToEnd(termBlock);
        rewriter.create<LLVM::BrOp>(loc, newBlock);

        mlir::Block *block = newBlock;
        for (auto caseOp : op.selects()) {
            block = convertSelectCase(block, caseOp);
        }

        // Remaining cases not covered by a default statement.
        if (block) {
            // TODO check specification.
            rewriter.setInsertionPointToEnd(block);
            mlir::Value retVal =
                rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(false));
            rewriter.create<LLVM::ReturnOp>(loc, mlir::ValueRange(retVal));
        }

        rewriter.eraseOp(op);

        return mlir::success();
    }

    // Helper to create a condition for select arguments and the yield key values.
    mlir::Value createKeyCondition(mlir::Value selectKey, mlir::Value selectArg,
                                   mlir::ConversionPatternRewriter &rewriter) const {
        auto intType =
            mlir::cast<mlir::IntegerType>(typeConverter->convertType(selectArg.getType()));

        // The argument will be materialized in the future, insert UnrealizedConversionCastOp.
        auto asUnmaterialized = [&](mlir::Type desiredType, mlir::Value value) {
            mlir::Operation *ucc = rewriter.create<mlir::UnrealizedConversionCastOp>(
                value.getLoc(), desiredType, mlir::ValueRange(value));
            return ucc->getResult(0);
        };

        selectArg = asUnmaterialized(intType, selectArg);

        if (auto constOp = selectKey.getDefiningOp<P4HIR::ConstOp>()) {
            mlir::Value cmpVal;

            if (auto setAttr = mlir::dyn_cast<P4HIR::SetAttr>(constOp.getValue())) {
                if (setAttr.getKind() == P4HIR::SetKind::Constant) {
                    auto setVal = P4HIR::getConstantInt(setAttr.getMembers()[0]).value();
                    auto attr = mlir::IntegerAttr::get(intType, setVal);
                    cmpVal = rewriter.create<LLVM::ConstantOp>(selectKey.getLoc(), attr);
                }
            }

            return rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), LLVM::ICmpPredicate::eq,
                                                 selectArg, cmpVal);
        } else if (auto setOp = selectKey.getDefiningOp<P4HIR::SetOp>()) {
            assert(setOp.getInput().size() == 1);
            auto cmpVal = asUnmaterialized(intType, setOp.getInput()[0]);
            rewriter.eraseOp(setOp);
            return rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), LLVM::ICmpPredicate::eq,
                                                 selectArg, cmpVal);
        } else if (auto rangeOp = selectKey.getDefiningOp<P4HIR::RangeOp>()) {
            bool isSigned = mlir::cast<P4HIR::BitsType>(selectArg.getType()).isSigned();
            auto lhs = asUnmaterialized(intType, rangeOp.getLhs());
            auto rhs = asUnmaterialized(intType, rangeOp.getRhs());
            auto le = isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule;
            auto lowerBound = rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), le, lhs, selectArg);
            auto upperBound = rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), le, selectArg, rhs);
            return rewriter.create<LLVM::AndOp>(selectKey.getLoc(), lowerBound, upperBound);
        } else {
            llvm_unreachable("Unsupported yield value");
            return {};
        }
    }
};

struct ParserAcceptOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ParserAcceptOp> {
    using ConvertOpToLLVMPattern<P4HIR::ParserAcceptOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserAcceptOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value retVal =
            rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getBoolAttr(true));
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, retVal);
        return mlir::success();
    }
};

struct ParserRejectOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ParserRejectOp> {
    using ConvertOpToLLVMPattern<P4HIR::ParserRejectOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ParserRejectOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value retVal =
            rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getBoolAttr(false));
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, retVal);
        return mlir::success();
    }
};

void configureP4HIRToLLVMTypeConverter(mlir::TypeConverter &converter) {
    configureP4HIRTypeConverter(converter);

    converter.addConversion([&](P4HIR::BitsType bitsType) {
        return mlir::IntegerType::get(bitsType.getContext(), bitsType.getWidth());
    });

    converter.addConversion(
        [&](P4HIR::BoolType boolType) { return mlir::IntegerType::get(boolType.getContext(), 1); });

    converter.addTypeAttributeConversion([&](mlir::Type type, P4HIR::IntAttr attr) {
        return mlir::IntegerAttr::get(converter.convertType(type), attr.getValue());
    });

    converter.addTypeAttributeConversion([&](mlir::Type type, P4HIR::BoolAttr attr) {
        return mlir::BoolAttr::get(attr.getContext(), attr.getValue());
    });

    converter.addConversion([&](P4HIR::ReferenceType refType) {
        return LLVM::LLVMPointerType::get(refType.getContext());
    });

    converter.addConversion([&](P4HIR::StructLikeTypeInterface structType) {
        auto types = llvm::map_to_vector(structType.getFields(), [&](const auto &member) {
            return converter.convertType(member.type);
        });
        return LLVM::LLVMStructType::getLiteral(structType.getContext(), types, true);
    });

    converter.addConversion([&](mlir::TupleType tupleType) {
        auto types = llvm::map_to_vector(tupleType.getTypes(), [&](const auto &member) {
            return converter.convertType(member);
        });
        return LLVM::LLVMStructType::getLiteral(tupleType.getContext(), types, true);
    });

    converter.addConversion([&](P4HIR::ArrayType arrayType) {
        auto newElementType = converter.convertType(arrayType.getElementType());
        return LLVM::LLVMArrayType::get(newElementType, arrayType.getSize());
    });

    converter.addConversion([&](P4HIR::ValidBitType validBitType) {
        // We may need to adjust this once we actually use the validity bit.
        return mlir::IntegerType::get(validBitType.getContext(), 1);
    });

    auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType, ValueRange inputs,
                                          Location loc) -> Value {
        if (inputs.size() != 1) return Value();

        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
    };

    converter.addSourceMaterialization(materializeAsUnrealizedCast);
    converter.addTargetMaterialization(materializeAsUnrealizedCast);
}

void LowerP4HIRToLLVMPass::runOnOperation() {
    mlir::ModuleOp mod = getOperation();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    mlir::LLVMTypeConverter typeConverter(&getContext());
    configureP4HIRToLLVMTypeConverter(typeConverter);

    mlir::RewritePatternSet patterns(&getContext());
    P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}
}  // end anonymous namespace

void P4HIR::populateP4HIRToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                                  mlir::RewritePatternSet &patterns) {
    patterns
        .add<FuncLikeOpConversion<P4HIR::FuncOp>, CallLikeOpConversion<P4HIR::CallOp>,
             ExternOpConversion, ReturnOpConversion, ConstOpConversion, VariableOpConversion,
             UnaryOpConversion, BrOpConversion, CondBrOpConversion, BinOpConversion,
             ShiftOpConversion<P4HIR::ShlOp>, ShiftOpConversion<P4HIR::ShrOp>, CmpOpConversion,
             CastOpConversion, ReadOpConversion, AssignOpConversion, ArrayGetOpConversion,
             ArrayElementRefOpConversion, StructExtractLikeOpConversion<P4HIR::StructExtractOp>,
             StructExtractLikeOpConversion<P4HIR::TupleExtractOp>, ConcatOpConversion,
             StructLikeOpConversion<P4HIR::ArrayOp>, StructLikeOpConversion<P4HIR::StructOp>,
             StructLikeOpConversion<P4HIR::TupleOp>, StructExtractRefOpConversion,
             ParserOpConversion, ParserTransitionOpConversion, ParserTransitionSelectOpConversion,
             ParserAcceptOpConversion, ParserRejectOpConversion>(converter);
}

std::unique_ptr<Pass> P4::P4MLIR::createLowerP4HIRToLLVMPass() {
    return std::make_unique<LowerP4HIRToLLVMPass>();
}

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//
namespace {
/// Implement the interface to convert P4HIR to LLVM.
struct P4HIRToLLVMDialectInterface : public mlir::ConvertToLLVMPatternInterface {
    using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
    void loadDependentDialects(MLIRContext *context) const final {
        context->loadDialect<P4HIR::P4HIRDialect>();
    }

    /// Hook for derived dialect interface to provide conversion patterns
    /// and mark dialect legal for the conversion target.
    void populateConvertToLLVMConversionPatterns(ConversionTarget &target,
                                                 LLVMTypeConverter &typeConverter,
                                                 RewritePatternSet &patterns) const final {
        P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);
    }
};
}  // namespace

void P4HIR::registerConvertP4HIRToLLVMInterface(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, P4HIR::P4HIRDialect *dialect) {
        dialect->addInterfaces<P4HIRToLLVMDialectInterface>();
    });
}
