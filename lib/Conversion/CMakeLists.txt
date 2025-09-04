add_mlir_dialect_library(P4MLIRConversion
  ConversionPatterns.cpp
  P4HIRToCoreLib.cpp
  LowerP4HIRToLLVM.cpp
  LowerCoreLibToLLVM.cpp
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/p4mlir/Conversion

  DEPENDS
  P4MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRTransforms
  MLIRLLVMCommonConversion
  MLIRBuiltinToLLVMIRTranslation
)
