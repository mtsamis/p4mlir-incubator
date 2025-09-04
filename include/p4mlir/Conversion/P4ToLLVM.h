#ifndef P4MLIR_CONVERSION_P4TOLLVM_H
#define P4MLIR_CONVERSION_P4TOLLVM_H

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "mlir/Transforms/DialectConversion.h"
#include <type_traits>
#include "llvm/Support/Casting.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Ops.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_TypeInterfaces.h"
#include "p4mlir/Dialect/P4HIR/P4HIR_Types.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace P4::P4ToLLVM {

using namespace mlir;
using namespace P4::P4MLIR;

inline std::string nameFor(mlir::Operation *op) {
    if (auto parserOp = mlir::dyn_cast<P4HIR::ParserOp>(op)) {
        return (llvm::Twine("_p_") + parserOp.getSymName()).str();
    } else if (auto controlOp = mlir::dyn_cast<P4HIR::ControlOp>(op)) {
        return (llvm::Twine("_c_") + controlOp.getSymName()).str();
    } else if (auto externOp = mlir::dyn_cast<P4HIR::ExternOp>(op)) {
        return (llvm::Twine("_e_") + externOp.getSymName()).str();
    } else {
        llvm_unreachable("Invalid op");
        return {};
    }
}

inline std::string memberFn(llvm::StringRef parent, llvm::StringRef child) {
    return (parent + "_m_" + child).str();
}

inline std::string memberFn(llvm::StringRef parent, P4HIR::ParserStateOp stateOp) {
    return memberFn(parent, stateOp.getSymName());
}

inline std::string memberFn(llvm::StringRef parent, P4HIR::FuncOp funcOp) {
    return memberFn(parent, funcOp.getSymName());
}

inline std::string specialInitFn(llvm::StringRef parent) { return memberFn(parent, "_init"); }

inline std::string specialInitFn(mlir::Operation *parent) { return specialInitFn(nameFor(parent)); }

inline std::string specialApplyFn(llvm::StringRef parent) { return memberFn(parent, "_apply"); }

inline std::string specialApplyFn(mlir::Operation *parent) {
    return specialApplyFn(nameFor(parent));
}

class P4Obj {
 public:
    static P4Obj create(const TypeConverter *converter, mlir::Type p4Type) {
        P4Obj res;
        res.p4Type = p4Type;
        res.llvmType = converter->convertType(p4Type);
        return res;
    }

    static P4Obj create(mlir::Type llvmType) {
        P4Obj res;
        res.llvmType = llvmType;
        return res;
    }

    static P4Obj createAggregateImpl(mlir::MLIRContext *ctx, const std::string &name, mlir::TypeRange types, llvm::function_ref<P4Obj(mlir::Type)> createFn) {
        P4Obj res;
        res.name = name;
        for (mlir::Type type : types)
            res.members.push_back(createFn(type));
        
        auto llvmTypes = llvm::map_to_vector(res.members, [&](auto &member) { return member.llvmType; });
        res.llvmType = LLVM::LLVMStructType::getNewIdentified(ctx, name, llvmTypes, true);
        
        return res;
    }

    static P4Obj createAggregate(mlir::MLIRContext *ctx, const std::string &name, mlir::TypeRange llvmTypes) {
        return createAggregateImpl(ctx, name, llvmTypes, [&](mlir::Type llvmType) {
            return create(llvmType);
        });
    }

    static P4Obj createAggregate(mlir::MLIRContext *ctx, const std::string &name, const TypeConverter *converter, mlir::TypeRange p4Types) {
        return createAggregateImpl(ctx, name, p4Types, [&](mlir::Type p4Type) {
            return create(converter, p4Type);
        });
    }

    struct ObjAccess {
        template<typename... Indices,
                 typename std::enable_if_t<(std::is_convertible_v<Indices, size_t> && ...), bool> = true>
        ObjAccess getMember(Indices... indices) {
            size_t tempArr[] = {indices...};
            return getMember(tempArr, sizeof...(Indices));
        }

        ObjAccess getMember(SmallVectorImpl<size_t> &indices) {
            return getMember(indices.data(), indices.size());
        }

        ObjAccess getMember(const size_t *indices, size_t count) {
            ObjAccess res;
            res.root = root;
            res.rootPtr = rootPtr;
            res.gepArgs = gepArgs;
            res.member = member;
            for (size_t i = 0; i < count; i++) {
                res.member = &res.member->members[indices[i]];
                res.gepArgs.push_back(i);
            }
            return res;
        }

        mlir::Value getPtrLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) {
            auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
            auto gepOp = rewriter.create<LLVM::GEPOp>(loc, ptrType, root->llvmType, rootPtr, gepArgs);
            return gepOp.getRes();
        }

        mlir::Value getValueLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) {
            auto ptr = getPtrLLVM(rewriter, loc);
            auto loadOp = rewriter.create<LLVM::LoadOp>(loc, member->llvmType, ptr);
            return loadOp.getRes();
        }

        mlir::Value getPtrP4(mlir::RewriterBase &rewriter, mlir::Location loc) {
            assert(member->p4Type && "Missing P4 type");
            auto resType = P4HIR::ReferenceType::get(rewriter.getContext(), member->p4Type);
            auto llvmPtr = getPtrLLVM(rewriter, loc);
            auto res = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, resType, llvmPtr);
            assert(res->getNumResults() == 1);
            return res->getResult(0);
        }

        mlir::Value getValueP4(mlir::RewriterBase &rewriter, mlir::Location loc) {
            assert(member->p4Type && "Missing P4 type");
            auto llvmValue = getValueLLVM(rewriter, loc);
            auto res = rewriter.create<mlir::UnrealizedConversionCastOp>(loc, member->p4Type, llvmValue);
            assert(res->getNumResults() == 1);
            return res->getResult(0);
        }

        operator const P4Obj *() const { return member; }
        const P4Obj *operator->() const { return member; }

        const P4Obj *member;
        const P4Obj *root;
        mlir::Value rootPtr;
        llvm::SmallVector<LLVM::GEPArg, 4> gepArgs;
    };

    ObjAccess thisObj(mlir::Value ptr) const {
        ObjAccess res;
        res.rootPtr = ptr;
        res.member = this;
        res.root = this;
        res.gepArgs.push_back(0);
        return res;
    }

    template<typename... Args>
    ObjAccess getMember(mlir::Value ptr, Args... args) const {
        return thisObj(ptr).getMember(std::forward<Args>(args)...);
    }

    mlir::Value allocaLLVM(mlir::RewriterBase &rewriter, mlir::Location loc) const {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(1));
        auto allocaOp = rewriter.create<LLVM::AllocaOp>(loc, ptrType, llvmType, one);
        auto thisPtr = allocaOp->getResult(0);
        construct(thisPtr);
        return thisPtr;
    }

    static void constructHelper(ObjAccess obj) {
        if (obj->isAggregate()) {
            for (size_t i = 0; i < obj->members.size(); i++) {
                constructHelper(obj.getMember(i));
            }
        } else {
            // Call constructor etc.
        }
    }

    void construct(mlir::Value thisPtr) const {
        constructHelper(thisObj(thisPtr));
    }

    bool isAggregate() const { return !members.empty(); }
    size_t getMemberCount() const { return members.size(); }
    llvm::StringRef getName() const { return name; }

 private:
    std::vector<P4Obj> members;
    std::string name;
    mlir::Type p4Type;
    mlir::Type llvmType;
};

class P4LLVMTypeConverter : public mlir::LLVMTypeConverter {
 public:
    using mlir::LLVMTypeConverter::LLVMTypeConverter;

    P4Obj *registerObjType(P4Obj obj) {
        auto objPtr = std::make_unique<P4Obj>(std::move(obj));
        llvm::StringRef nameRef = objPtr->getName();
        [[maybe_unused]] auto [it, ins] = defToObj.insert({nameRef, std::move(objPtr)});
        assert(ins && "Cannot register with duplicate name");
        return it->second.get();
    }

    P4Obj *getObjType(llvm::StringRef name) {
        auto it = defToObj.find(name);
        return (it != defToObj.end())? it->second.get() : nullptr;
    }

 private:
    llvm::DenseMap<llvm::StringRef, std::unique_ptr<P4Obj>> defToObj;
};

}  // namespace P4::P4ToLLVM

#endif  // P4MLIR_CONVERSION_P4TOLLVM_H
