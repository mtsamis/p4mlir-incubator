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
#include "p4mlir/Conversion/P4ToLLVM.h"
#include "p4mlir/Conversion/Passes.h"
// Temporary include CoreLib.
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
//
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "p4hir-lower-to-llvm"

using namespace mlir;
using namespace P4::P4MLIR;
using namespace P4::P4ToLLVM;

namespace P4::P4MLIR {
#define GEN_PASS_DEF_LOWERP4HIRTOLLVM
#include "p4mlir/Conversion/Passes.cpp.inc"
}  // namespace P4::P4MLIR

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
        if (!funcType.getTypeArguments().empty()) {
            // TODO temporary.
            rewriter.eraseOp(op);
            return mlir::success();
            // return rewriter.notifyMatchFailure(op, "Cannot handle type arguments");
        }

        auto retType = funcType.getOptionalReturnType();
        auto mlirFuncType =
            mlir::FunctionType::get(rewriter.getContext(), funcType.getInputs(),
                                    retType ? mlir::TypeRange(retType) : mlir::TypeRange());

        TypeConverter::SignatureConversion result(op.getNumArguments());
        auto llvmFuncType =
            this->getTypeConverter()->convertFunctionSignature(mlirFuncType, false, false, result);

        if (!llvmFuncType) return rewriter.notifyMatchFailure(op, "Signature conversion failed");

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

template <typename OpTy>
struct CallLikeOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
    using OpAdaptor = typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto converter = this->getTypeConverter();
        auto mod = op->template getParentOfType<mlir::ModuleOp>();

        auto callee = adaptor.getCallee();
        auto rootOp = mod.lookupSymbol(callee.getRootReference());
        auto newOperands = llvm::to_vector(adaptor.getArgOperands());
        mlir::StringAttr newCallee;

        if (mlir::isa<P4HIR::ControlOp>(rootOp)) {
            // Adjust calls within controls to the corresponding control member function.
            // This should probably be moved to Control Conversion.
            auto parent = op->template getParentOfType<LLVM::LLVMFuncOp>();
            if (!parent) return mlir::failure();

            auto name = memberFn(nameFor(rootOp), callee.getLeafReference());
            newCallee = rewriter.getStringAttr(name);
            newOperands.insert(newOperands.begin(), parent.getArguments().begin(),
                               parent.getArguments().end());
        } else {
            newCallee = callee.getLeafReference();
        }

        if (op.getResult()) {
            auto resType = converter->convertType(op.getResult().getType());
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, resType, newCallee, newOperands);
        } else {
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, mlir::TypeRange(), newCallee,
                                                      newOperands);
        }

        return mlir::success();
    }
};

// TODO
struct CallMethodOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::CallMethodOp> {
    using ConvertOpToLLVMPattern<P4HIR::CallMethodOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::CallMethodOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct ExternOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ExternOp> {
    using ConvertOpToLLVMPattern<P4HIR::ExternOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ExternOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto parentName = nameFor(op);
        rewriter.startOpModification(op);
        op.walk([&](P4HIR::FuncOp fn) { fn.setSymName(memberFn(parentName, fn.getSymName())); });
        rewriter.finalizeOpModification(op);

        rewriter.inlineBlockBefore(&op.getBody().front(), op);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct OverloadSetOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::OverloadSetOp> {
    using ConvertOpToLLVMPattern<P4HIR::OverloadSetOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::OverloadSetOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.inlineBlockBefore(&op.getBody().front(), op);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// Handled through InstantiateOpConversion.
struct ConstructOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ConstructOp> {
    using ConvertOpToLLVMPattern<P4HIR::ConstructOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ConstructOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// Handled through InstantiateOpConversion.
struct PackageOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::PackageOp> {
    using ConvertOpToLLVMPattern<P4HIR::PackageOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::PackageOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

struct InstantiateOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::InstantiateOp> {
    InstantiateOpConversion(P4LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern(converter), converter(&converter) {}

    llvm::SmallVector<mlir::Value, 4> instantiate(mlir::ConversionPatternRewriter &rewriter,
                                                  mlir::ModuleOp mod, mlir::Location loc,
                                                  mlir::Operation *op,
                                                  mlir::ValueRange arguments) const {
        if (mlir::isa<P4HIR::ParserOp, P4HIR::ControlOp>(op)) {
            llvm::SmallVector<mlir::Value, 4> args;

            auto obj = converter->getObjType(nameFor(op));
            auto objPtr = obj->allocaLLVM(rewriter, loc);
            args.push_back(objPtr);

            // There are still some questions around argument allocation.
            // This loop will be completely replaced once we have proper type -> p4obj etc.
            auto funcType = mlir::cast<P4HIR::FuncType>(
                mlir::cast<mlir::FunctionOpInterface>(op).getFunctionType());

            for (mlir::Type argType : funcType.getInputs()) {
                mlir::Type allocaType = argType;

                if (auto refType = mlir::dyn_cast<P4HIR::ReferenceType>(argType)) {
                    allocaType = refType.getObjectType();
                }

                P4Obj argObj;
                // Total workaround for proof-of-concept testing.
                if (mlir::isa<P4CoreLib::PacketInType>(allocaType)) {
                    argObj = *converter->getObjType("_core_packet_in");
                } else {
                    argObj = P4Obj::create(getTypeConverter(), allocaType);
                }
                auto argPtr = argObj.allocaLLVM(rewriter, loc);

                if (mlir::isa<P4HIR::ReferenceType>(argType)) {
                    args.push_back(argObj.thisObj(argPtr).getPtrLLVM(rewriter, loc));
                } else {
                    args.push_back(argObj.thisObj(argPtr).getValueLLVM(rewriter, loc));
                }
            }

            // Call initializer.
            auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
            rewriter.create<LLVM::CallOp>(loc, voidType, specialInitFn(op), args);

            return args;
        } else if (auto externOp = mlir::dyn_cast<P4HIR::ExternOp>(op)) {
            // TODO Call constructor if present.
            return {};
        } else if (auto constructOp = mlir::dyn_cast<P4HIR::ConstructOp>(op)) {
            auto target = mod.lookupSymbol(constructOp.getCallee());
            if (!target) return {};
            return instantiate(rewriter, mod, target->getLoc(), target,
                               constructOp.getArgOperands());
        } else if (auto packageOp = mlir::dyn_cast<P4HIR::PackageOp>(op)) {
            // llvm::SmallVector<mlir::Value, 4> stateInsts;

            for (mlir::Value arg : arguments) {
                auto target = arg.getDefiningOp();
                auto stateInst = instantiate(rewriter, mod, loc, target, {});
                // stateInsts.push_back(stateInst);
            }

            // We need to place the arguments forwarding logic etc here.
            // Call apply...

            return {};
        } else {
            llvm_unreachable("Impossible instantiation target");
            return {};
        }
    }

    mlir::LogicalResult matchAndRewrite(P4HIR::InstantiateOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto mod = op->getParentOfType<mlir::ModuleOp>();
        auto instTarget = mod.lookupSymbol(op.getCallee());
        if (!instTarget) return mlir::failure();

        if (op.getSymName() == "main") {
            mod.dump();
            // Create a main function for the main package.
            auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
            auto funcType = LLVM::LLVMFunctionType::get(voidType, {});
            auto func = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), "main", funcType);
            Block *entryBlock = func.addEntryBlock(rewriter);
            rewriter.setInsertionPointToStart(entryBlock);

            instantiate(rewriter, mod, op.getLoc(), instTarget, op.getArgOperands());

            rewriter.create<LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange());
        }

        rewriter.eraseOp(op);
        return mlir::success();
    }

 private:
    P4LLVMTypeConverter *converter;
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
        // Constants that cannot be materialzed (e.g. set types) are expected to be eliminated from
        // other conversions.
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

// Similar to ControlFlow.BranchOp
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

// Similar to ControlFlow.CondBranchOp
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
        // Convert struct/tuple like initializer.
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
        // Currently an RHS expression like a[idx].b emits a a[idx] as a value rather than a ref.
        // We need to fix this in the translator or emit bad code and hope LLVM optimizes it.
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

// Helper to transform parsers and controls.
struct ParserOrControlConversionHelper {
    ParserOrControlConversionHelper(P4LLVMTypeConverter *converter,
                                    mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
                                    mlir::ValueRange arguments)
        : converter(converter),
          rewriter(rewriter),
          op(op),
          arguments(arguments),
          loc(op->getLoc()) {
        ctx = rewriter.getContext();
        mod = op->getParentOfType<mlir::ModuleOp>();
    }

    // Categorize operations within a parser/control.
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

    void addLocal(mlir::Operation *op) {
        if (auto variableOp = mlir::dyn_cast<P4HIR::VariableOp>(op)) {
            localVarDefs.push_back(variableOp);
        } else if (auto controlLocalOp = mlir::dyn_cast<P4HIR::ControlLocalOp>(op)) {
            auto val = controlLocalOp.getVal();
            assert((mlir::isa<mlir::BlockArgument>(val) ||
                    mlir::isa<P4HIR::VariableOp>(val.getDefiningOp())) &&
                   "Unexpected control local variable kind");
            symMapping.insert({controlLocalOp.getSymName(), val});
        }
    }

    void init() {
        ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

        if (mlir::isa<P4HIR::ParserOp>(op)) {
            retTy = mlir::IntegerType::get(ctx, 1);
        } else {
            retTy = LLVM::LLVMVoidType::get(ctx);
        }

        llvm::SmallVector<mlir::Type, 4> localVarTypes;
        for (mlir::Operation *localVarDef : localVarDefs) {
            auto variableOp = mlir::cast<P4HIR::VariableOp>(localVarDef);
            auto refType = mlir::cast<P4HIR::ReferenceType>(variableOp.getRef().getType());
            localVarTypes.push_back(refType.getObjectType());
        }

        auto objDef = P4Obj::createAggregate(ctx, nameFor(op), converter, localVarTypes);
        obj = converter->registerObjType(std::move(objDef));
    }

    // Create a new member function for this parser/control.
    // For parsers this corresponds to state functions.
    // For controls this corresponds to actions.
    LLVM::LLVMFuncOp createMemberFunction(
        mlir::Type returnType, llvm::StringRef name, bool addArguments,
        llvm::function_ref<void(LLVM::LLVMFuncOp, mlir::IRMapping &)> createFunctionBody) {
        llvm::SmallVector<mlir::Type, 8> signature;

        // First argument is a `this` state pointer.
        signature.push_back(ptrTy);

        if (addArguments)
            for (auto arg : arguments) signature.push_back(converter->convertType(arg.getType()));

        auto funcType = LLVM::LLVMFunctionType::get(returnType, signature);
        auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, name, funcType);
        func.setPrivate();
        // Use a 'member_of' attribute to keep track that this is a member function of a particular
        // parser/control.
        func->setAttr("member_of", mlir::StringAttr::get(ctx, nameFor(op)));

        // We need to map:
        //   1) Original values to the arguments in the newly created function.
        //   2) Original local variables to members in the `this` state object.
        //   3) Original constants to newly cloned constants within the function.
        //   4) p4hir.symbol_ref operations in control actions to members in the `this` state
        //   object.
        // 1, 2 and 3 are done with `irMapping` while 4 is done with `symMapping`.
        mlir::IRMapping irMapping;

        Block *entryBlock = func.addEntryBlock(rewriter);
        rewriter.setInsertionPointToStart(entryBlock);
        auto thisPtr = func.getArgument(0);

        for (size_t i = 0; i < localVarDefs.size(); i++) {
            auto localVar = mlir::cast<P4HIR::VariableOp>(localVarDefs[i]);
            auto memerVar = obj->getMember(thisPtr, i).getPtrP4(rewriter, localVar.getLoc());
            irMapping.map(localVar.getRef(), memerVar);
        }

        if (addArguments) {
            // Use unrealized casts to convert arguments to their original types, since we haven't
            // converted the body yet.
            for (size_t i = 0; i < arguments.size(); i++) {
                auto arg = arguments[i];
                auto ucOp = rewriter.create<mlir::UnrealizedConversionCastOp>(
                    arg.getLoc(), arg.getType(), func.getArgument(i + 1));
                assert(ucOp->getNumResults() == 1);
                irMapping.map(arg, ucOp->getResult(0));
            }
        }

        // Clone all constants and let the canonicalizer clean up.
        for (auto constOp : constantOps) {
            auto newConstOp = rewriter.clone(*constOp);
            irMapping.map(constOp->getResults(), newConstOp->getResults());
        }

        createFunctionBody(func, irMapping);

        // Replace p4hir.symbol_ref in the function body.
        func->walk([&](mlir::Operation *op) {
            if (auto symOp = mlir::dyn_cast<P4HIR::SymToValueOp>(op)) {
                // TODO check root reference?
                mlir::StringAttr sym = symOp.getDecl().getLeafReference();
                auto it = symMapping.find(sym);
                assert(it != symMapping.end());
                auto val = irMapping.lookupOrDefault(it->second);
                rewriter.replaceOp(symOp, mlir::ValueRange(val));
            }
        });

        return func;
    }

    void createInitFunction() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());

        auto cb = [&](LLVM::LLVMFuncOp, mlir::IRMapping &mapping) {
            for (mlir::Operation *op : initOps) {
                rewriter.clone(*op, mapping);
            }
        };

        createMemberFunction(LLVM::LLVMVoidType::get(ctx), specialInitFn(op), true, cb);

        rewriter.create<LLVM::ReturnOp>(loc, mlir::ValueRange());
    }

    void createApplyFunction() {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(mod.getBody());

        auto cb = [&](LLVM::LLVMFuncOp, mlir::IRMapping &mapping) {
            for (mlir::Operation *op : applyOps) {
                rewriter.clone(*op, mapping);
            }
        };

        createMemberFunction(retTy, specialApplyFn(op), true, cb);
    }

    void createMemberFunctions() {
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

            auto cb = [&](LLVM::LLVMFuncOp func, mlir::IRMapping &mapping) {
                rewriter.cloneRegionBefore(fn->getRegion(0), func.getBody(), func.getBody().end(),
                                           mapping);
            };

            auto func = createMemberFunction(retTy, getName(fn), true, cb);

            mlir::Block &entry = func.getBody().front();
            rewriter.setInsertionPointToEnd(&entry);
            rewriter.create<LLVM::BrOp>(func.getLoc(), entry.getNextNode());
        }
    }

    void convert() {
        init();
        createInitFunction();
        createApplyFunction();
        createMemberFunctions();
    }

 private:
    P4LLVMTypeConverter *converter;
    mlir::ConversionPatternRewriter &rewriter;
    mlir::Operation *op;
    mlir::ValueRange arguments;

    mlir::Location loc;
    mlir::MLIRContext *ctx;
    mlir::ModuleOp mod;
    mlir::Type retTy;
    mlir::Type ptrTy;

    llvm::DenseMap<mlir::StringRef, mlir::Value> symMapping;
    llvm::SmallVector<mlir::Operation *, 4> localVarDefs;
    P4Obj *obj;

    llvm::SmallVector<mlir::Operation *, 16> initOps;
    llvm::SmallVector<mlir::Operation *, 16> constantOps;
    llvm::SmallVector<mlir::Operation *, 16> applyOps;
    llvm::SmallVector<mlir::Operation *, 16> methodOps;
};

template <typename OpTy>
struct ParserOrControlConversion : public ConvertOpToLLVMPattern<OpTy> {
    ParserOrControlConversion(P4LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern<OpTy>(converter), converter(&converter) {}

    mlir::LogicalResult matchAndRewrite(OpTy op,
                                        typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        ParserOrControlConversionHelper pca(converter, rewriter, op, op.getArguments());

        for (mlir::Operation &op : op.getBody().front()) {
            pca.add(&op);
        }

        pca.convert();

        rewriter.eraseOp(op);

        return mlir::success();
    }

 private:
    P4LLVMTypeConverter *converter;
};

struct ControlApplyOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ControlApplyOp> {
    using ConvertOpToLLVMPattern<P4HIR::ControlApplyOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ControlApplyOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        auto parent = op->getParentOfType<LLVM::LLVMFuncOp>();
        if (!parent) return mlir::failure();

        if (!op.getBody().empty()) {
            rewriter.inlineRegionBefore(op.getBody(), parent.getBody(), parent.end());
            rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, op->getBlock()->getNextNode());
        } else {
            rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, mlir::ValueRange());
        }

        // TODO we should address terminators in control_apply in general (see flatten-cfg).
        if (!parent.getBody().back().mightHaveTerminator()) {
            rewriter.setInsertionPointToEnd(&parent.getBody().back());
            rewriter.create<LLVM::ReturnOp>(op->getLoc(), mlir::ValueRange());
        }

        return mlir::success();
    }
};

struct ControlLocalOpConversion : public mlir::ConvertOpToLLVMPattern<P4HIR::ControlLocalOp> {
    using ConvertOpToLLVMPattern<P4HIR::ControlLocalOp>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(P4HIR::ControlLocalOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // Locals are handled during ParserOrControlConversion. Erase them once we're done.
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

// Helper function to do a parser state transition with a tail call.
// Can only be called from parser states that have been converted to LLVM functions.
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
        auto parent = op->getParentOfType<LLVM::LLVMFuncOp>();
        if (!parent) return mlir::failure();

        auto parentName = parent->getAttrOfType<mlir::StringAttr>("member_of");
        if (!parentName) return mlir::failure();

        mlir::Region *region = op->getParentRegion();
        mlir::Block *block = &region->back();
        assert((block == op->getBlock()) && "Expected terminator to be in the last block");

        mlir::Block *newBlock = rewriter.createBlock(region, region->end());
        rewriter.setInsertionPointToEnd(block);
        rewriter.create<LLVM::BrOp>(op.getLoc(), newBlock);
        block = newBlock;

        for (auto caseOp : op.selects()) {
            block = convertSelectCase(block, op.getSelect(), caseOp, rewriter);
            if (!block) break;
        }

        if (block) {
            // Handles remaining cases not covered by a default statement.
            // TODO check specification.
            rewriter.setInsertionPointToEnd(block);
            mlir::Value retVal =
                rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getBoolAttr(false));
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange(retVal));
        }

        rewriter.eraseOp(op);

        return mlir::success();
    }

    // Helper to convert the select case `caseOp`.
    // `block` should be an unterminated block where the code for this case will be appended.
    // If not a default case, a new unterminated block is returned.
    mlir::Block *convertSelectCase(mlir::Block *block, mlir::Value selectArg,
                                   P4HIR::ParserSelectCaseOp caseOp,
                                   mlir::ConversionPatternRewriter &rewriter) const {
        mlir::Region *region = block->getParent();
        mlir::Block *continueBlock = nullptr;

        if (caseOp.isDefault()) {
            // The default state transition will be added as a terminator in the last block.
            rewriter.setInsertionPointToEnd(block);
        } else {
            // Create if-then for current case.
            mlir::Block *thenBlock = rewriter.createBlock(region, region->end());
            continueBlock = rewriter.createBlock(region, region->end());

            // Inline keyset expression to last block.
            auto yield = mlir::cast<P4HIR::YieldOp>(caseOp.getBody()->getTerminator());
            rewriter.inlineBlockBefore(caseOp.getBody(), block, block->begin());

            // Replace yield with conditional jump.
            rewriter.setInsertionPoint(yield);
            auto cond = createKeyCondition(yield.getOperand(0), selectArg, rewriter);
            rewriter.create<LLVM::CondBrOp>(caseOp.getLoc(), cond, thenBlock, continueBlock);
            rewriter.eraseOp(yield);

            // The state transition will be added in the then block.
            rewriter.setInsertionPointToStart(thenBlock);
        }

        createTailCallStateTransition(caseOp, caseOp.getState(), rewriter);

        return continueBlock;
    }

    // Helper to create a condition for select arguments and the yield key values.
    mlir::Value createKeyCondition(mlir::Value selectKey, mlir::Value selectArg,
                                   mlir::ConversionPatternRewriter &rewriter) const {
        auto intType =
            mlir::cast<mlir::IntegerType>(getTypeConverter()->convertType(selectArg.getType()));

        // Use unrealized conversion cast in order to emit keyset expressions in LLVM Dialect rather
        // than P4HIR.
        auto asUnrealized = [&](mlir::Type desiredType, mlir::Value value) {
            mlir::Operation *ucc = rewriter.create<mlir::UnrealizedConversionCastOp>(
                value.getLoc(), desiredType, mlir::ValueRange(value));
            return ucc->getResult(0);
        };

        selectArg = asUnrealized(intType, selectArg);

        if (auto constOp = selectKey.getDefiningOp<P4HIR::ConstOp>()) {
            mlir::Value cmpVal;

            if (auto setAttr = mlir::dyn_cast<P4HIR::SetAttr>(constOp.getValue())) {
                if (setAttr.getKind() == P4HIR::SetKind::Constant) {
                    auto setVal = P4HIR::getConstantInt(setAttr.getMembers()[0]).value();
                    auto attr = mlir::IntegerAttr::get(intType, setVal);
                    cmpVal = rewriter.create<LLVM::ConstantOp>(selectKey.getLoc(), attr);
                } else {
                    // TODO support all.
                    llvm_unreachable("Unsupported constant set attribute");
                    return {};
                }
            }

            return rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), LLVM::ICmpPredicate::eq,
                                                 selectArg, cmpVal);
        } else if (auto setOp = selectKey.getDefiningOp<P4HIR::SetOp>()) {
            assert(setOp.getInput().size() == 1);
            auto cmpVal = asUnrealized(intType, setOp.getInput()[0]);
            rewriter.eraseOp(setOp);
            return rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), LLVM::ICmpPredicate::eq,
                                                 selectArg, cmpVal);
        } else if (auto rangeOp = selectKey.getDefiningOp<P4HIR::RangeOp>()) {
            bool isSigned = mlir::cast<P4HIR::BitsType>(selectArg.getType()).isSigned();
            auto lhs = asUnrealized(intType, rangeOp.getLhs());
            auto rhs = asUnrealized(intType, rangeOp.getRhs());
            auto le = isSigned ? LLVM::ICmpPredicate::sle : LLVM::ICmpPredicate::ule;
            auto lowerBound = rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), le, lhs, selectArg);
            auto upperBound = rewriter.create<LLVM::ICmpOp>(selectKey.getLoc(), le, selectArg, rhs);
            return rewriter.create<LLVM::AndOp>(selectKey.getLoc(), lowerBound, upperBound);
        } else {
            // TODO support all.
            llvm_unreachable("Unsupported yield value");
            return {};
        }
    }
};

template <typename OpTy, bool Result>
struct ParserAcceptRejectOpConversion : public ConvertOpToLLVMPattern<OpTy> {
    using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

    mlir::LogicalResult matchAndRewrite(OpTy op,
                                        typename ConvertOpToLLVMPattern<OpTy>::OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        mlir::Value retVal =
            rewriter.create<LLVM::ConstantOp>(op.getLoc(), rewriter.getBoolAttr(Result));
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, retVal);
        return mlir::success();
    }
};

void LowerP4HIRToLLVMPass::runOnOperation() {
    mlir::ModuleOp mod = getOperation();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    P4LLVMTypeConverter typeConverter(&getContext());
    configureP4HIRTypeConverter(typeConverter);
    P4HIR::configureP4HIRToLLVMTypeConverter(typeConverter);

    mlir::RewritePatternSet patterns(&getContext());
    P4HIR::populateP4HIRToLLVMConversionPatterns(typeConverter, patterns);

    if (mlir::failed(mlir::applyPartialConversion(mod, target, std::move(patterns))))
        signalPassFailure();
}
}  // end anonymous namespace

void P4HIR::populateP4HIRToLLVMConversionPatterns(P4LLVMTypeConverter &converter,
                                                  mlir::RewritePatternSet &patterns) {
    patterns.add<FuncLikeOpConversion<P4HIR::FuncOp>, CallLikeOpConversion<P4HIR::CallOp>,
                 CallMethodOpConversion, ExternOpConversion, OverloadSetOpConversion,
                 PackageOpConversion, InstantiateOpConversion, ConstructOpConversion,
                 ReturnOpConversion, ConstOpConversion, VariableOpConversion, UnaryOpConversion,
                 BrOpConversion, CondBrOpConversion, BinOpConversion,
                 ShiftOpConversion<P4HIR::ShlOp>, ShiftOpConversion<P4HIR::ShrOp>, CmpOpConversion,
                 CastOpConversion, ReadOpConversion, AssignOpConversion, ArrayGetOpConversion,
                 ArrayElementRefOpConversion, StructExtractLikeOpConversion<P4HIR::StructExtractOp>,
                 StructExtractLikeOpConversion<P4HIR::TupleExtractOp>, ConcatOpConversion,
                 StructLikeOpConversion<P4HIR::ArrayOp>, StructLikeOpConversion<P4HIR::StructOp>,
                 StructLikeOpConversion<P4HIR::TupleOp>, StructExtractRefOpConversion,
                 ParserOrControlConversion<P4HIR::ParserOp>,
                 ParserOrControlConversion<P4HIR::ControlOp>, ControlApplyOpConversion,
                 ControlLocalOpConversion, ParserTransitionOpConversion,
                 ParserTransitionSelectOpConversion,
                 ParserAcceptRejectOpConversion<P4HIR::ParserAcceptOp, true>,
                 ParserAcceptRejectOpConversion<P4HIR::ParserRejectOp, false>>(converter);
}

void P4HIR::configureP4HIRToLLVMTypeConverter(P4LLVMTypeConverter &converter) {
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
        return LLVM::LLVMStructType::getNewIdentified(structType.getContext(), structType.getName(),
                                                      types, true);
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

    converter.addConversion([&](P4HIR::StringType stringType) {
        // Assuming we convert string to null terminated pointers.
        return LLVM::LLVMPointerType::get(stringType.getContext());
    });

    converter.addConversion([&](P4HIR::ErrorType errorType) {
        // This is temporary.
        return mlir::IntegerType::get(errorType.getContext(), 32);
    });

    converter.addConversion([&](P4HIR::ExternType externType) {
        // This is temporary.
        auto name = (llvm::Twine("_e_") + externType.getName()).str();
        return LLVM::LLVMStructType::getIdentified(externType.getContext(), name);
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

std::unique_ptr<Pass> P4::P4MLIR::createLowerP4HIRToLLVMPass() {
    return std::make_unique<LowerP4HIRToLLVMPass>();
}
