set(LIBS
  P4MLIR_P4HIR
  P4MLIR_P4CoreLib
  P4MLIRConversion
  P4MLIRTransforms

  MLIRFuncDialect

  MLIRFuncAllExtensions
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMIRToLLVMTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRExecutionEngineUtils
)

add_llvm_executable(p4mlir-ebpf p4mlir-ebpf.cpp)

llvm_update_compile_flags(p4mlir-ebpf)
target_link_libraries(p4mlir-ebpf PRIVATE ${LIBS})

mlir_check_all_link_libraries(p4mlir-ebpf)
