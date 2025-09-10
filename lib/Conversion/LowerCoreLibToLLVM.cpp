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
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Dialect.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Ops.h"
#include "p4mlir/Dialect/P4CoreLib/P4CoreLib_Types.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "p4mlir/Transforms/Passes.h"

#define DEBUG_TYPE "core-lib-lower-to-llvm"

using namespace mlir;
using namespace P4::P4MLIR;
using namespace P4::P4ToLLVM;

namespace {

struct PacketExtractOpConversion : public mlir::ConvertOpToLLVMPattern<P4CoreLib::PacketExtractOp> {
    PacketExtractOpConversion(P4LLVMTypeConverter &converter)
        : ConvertOpToLLVMPattern(converter), converter(&converter) {}

    size_t extractMemberImpl(mlir::RewriterBase &rewriter, mlir::Location loc,
                             mlir::Value packetBytesPtr, size_t bitOffset,
                             const P4Obj::ObjAccess &dest) const {
        if (mlir::isa<P4HIR::ValidBitType>(dest->getTypeP4())) {
            // Validity bit is lowered as a one-element struct.
            auto ptr = dest.getPtrLLVM(rewriter, loc);
            auto validityBitObj = P4Obj::fromStructLLVM(rewriter.getContext(), dest->getTypeLLVM());
            auto innerBit = validityBitObj.getMember(ptr, 0);
            auto constTrue = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getBoolAttr(true));
            innerBit.setValueLLVM(rewriter, loc, constTrue);
            return 0;
        }

        size_t destWidthBits = mlir::cast<mlir::IntegerType>(dest->getTypeLLVM()).getWidth();
        size_t destWidthBytes = (destWidthBits + 7) / 8;
        size_t byteOffset = bitOffset / 8;
        size_t bytesToRead = llvm::PowerOf2Ceil(destWidthBytes);

        [[maybe_unused]] std::string debugExpr;
        LLVM_DEBUG(llvm::dbgs() << "Read " << destWidthBits << " bit field from offset "
                                << bitOffset << " with read width " << bytesToRead << " bytes\n");
        LLVM_DEBUG(debugExpr = (llvm::Twine("(u") + std::to_string(bytesToRead * 8) + ") data[" +
                                std::to_string(byteOffset) + "]")
                                   .str());

        auto byteType = rewriter.getIntegerType(8);
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto src = rewriter.create<LLVM::GEPOp>(loc, ptrType, byteType, packetBytesPtr,
                                                ArrayRef<LLVM::GEPArg>{byteOffset});
        auto srcType = rewriter.getIntegerType(bytesToRead * 8);
        auto srcVal = rewriter.create<LLVM::LoadOp>(loc, srcType, src).getRes();

        if ((bitOffset % 8) != 0) {
            auto shiftAmount = rewriter.create<LLVM::ConstantOp>(
                loc, rewriter.getIntegerAttr(srcType, bitOffset % 8));
            srcVal = rewriter.create<LLVM::LShrOp>(loc, srcVal, shiftAmount);
            LLVM_DEBUG(
                debugExpr =
                    (llvm::Twine("(") + debugExpr + ") >> " + std::to_string(bitOffset % 8)).str());
        }

        if ((bytesToRead * 8) != destWidthBits) {
            srcVal = rewriter.create<LLVM::TruncOp>(loc, dest->getTypeLLVM(), srcVal);
            LLVM_DEBUG(debugExpr = (llvm::Twine("(u") + std::to_string(destWidthBits) + ") (" +
                                    debugExpr + ")")
                                       .str());
        }

        LLVM_DEBUG(llvm::dbgs() << "With expr " << debugExpr << "\n");

        dest.setValueLLVM(rewriter, loc, srcVal);

        return destWidthBits;
    }

    size_t extractImpl(mlir::RewriterBase &rewriter, mlir::Location loc, mlir::Value packetBytesPtr,
                       size_t bitOffset, P4Obj::ObjAccess dest) const {
        if (dest->isAggregate()) {
            size_t bitsConsumed = 0;
            for (size_t i = 0; i < dest->getMemberCount(); i++) {
                bitsConsumed += extractImpl(rewriter, loc, packetBytesPtr, bitOffset + bitsConsumed,
                                            dest.getMember(i));
            }
            return bitsConsumed;
        } else {
            return extractMemberImpl(rewriter, loc, packetBytesPtr, bitOffset, dest);
        }
    }

    mlir::LogicalResult matchAndRewrite(P4CoreLib::PacketExtractOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter &rewriter) const override {
        // bitsToExtract = sizeofInBits(headerLValue);
        // lastBitNeeded = this.nextBitIndex + bitsToExtract;
        // ParserModel.verify(this.lengthInBits >= lastBitNeeded, error.PacketTooShort);
        // headerLValue = this.data.extractBits(this.nextBitIndex, bitsToExtract);
        // headerLValue.valid$ = true;
        // if headerLValue.isNext$ {
        //     verify(headerLValue.nextIndex$ < headerLValue.size, error.StackOutOfBounds);
        //     headerLValue.nextIndex$ = headerLValue.nextIndex$ + 1;
        // }
        // this.nextBitIndex += bitsToExtract;

        auto loc = op.getLoc();
        auto packetInObj = converter->getObjType("_core_packet_in");
        auto headerObj = converter->getObjType(removeRef(op.getHdr().getType()));
        auto packetIn = packetInObj->thisObj(adaptor.getPacketIn());
        auto header = headerObj->thisObj(adaptor.getHdr());

        auto nextBitIndexMember = packetIn.getMember(2);

        auto packetBytesPtr = packetIn.getMember(0).getValueLLVM(rewriter, loc);
        auto nextBitIndex = nextBitIndexMember.getValueLLVM(rewriter, loc);
        auto bitsPerByteAttr = rewriter.getIntegerAttr(nextBitIndex.getType(), 8);
        auto bitsPerByte = rewriter.create<LLVM::ConstantOp>(loc, bitsPerByteAttr);
        auto nextByteIndex = rewriter.create<LLVM::UDivOp>(loc, nextBitIndex, bitsPerByte);
        auto byteType = rewriter.getIntegerType(8);
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        packetBytesPtr = rewriter.create<LLVM::GEPOp>(
            loc, ptrType, byteType, packetBytesPtr, ArrayRef<LLVM::GEPArg>{nextByteIndex.getRes()});

        size_t bitsConsumed = extractImpl(rewriter, op.getLoc(), packetBytesPtr, 0, header);
        assert((bitsConsumed % 8 == 0) && "Cannot extract non-byte-aligned data from packet_in");

        auto bitsConsumedVal = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), rewriter.getIntegerAttr(nextBitIndex.getType(), bitsConsumed));
        auto newNextBitIndex =
            rewriter.create<LLVM::AddOp>(op.getLoc(), nextBitIndex, bitsConsumedVal);
        nextBitIndexMember.setValueLLVM(rewriter, op.getLoc(), newNextBitIndex);

        rewriter.eraseOp(op);

        return mlir::success();
    }

 private:
    P4LLVMTypeConverter *converter;
};

}  // end anonymous namespace

void P4HIR::populateCoreLibToLLVMConversionPatterns(P4LLVMTypeConverter &converter,
                                                    mlir::RewritePatternSet &patterns) {
    patterns.add<PacketExtractOpConversion>(converter);
}

void P4HIR::configureCoreLibToLLVMTypeConverter(P4LLVMTypeConverter &converter) {
    converter.addConversion([&](P4CoreLib::PacketInType packetInType) {
        return LLVM::LLVMPointerType::get(packetInType.getContext());
    });

    converter.addConversion([&](P4CoreLib::PacketOutType packetOutType) {
        return LLVM::LLVMPointerType::get(packetOutType.getContext());
    });

    // struct packet_in {
    //     byte[] data;
    //     unsigned lengthInBits;
    //     unsigned nextBitIndex;
    // }
    auto ptrTy = LLVM::LLVMPointerType::get(&converter.getContext());
    auto unsignedTy = mlir::IntegerType::get(&converter.getContext(), 32);
    auto packetInObj = P4Obj::createAggregate(&converter.getContext(), "_core_packet_in",
                                              {ptrTy, unsignedTy, unsignedTy});
    converter.registerObjType(std::move(packetInObj));
}
