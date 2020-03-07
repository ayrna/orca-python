from distutils.core import setup, Extension

svmTrain_python_extension = Extension(
    name="svmtrain",
    language = "c++",
    sources = ["svmtrainmodule.cpp", "svm_model_python.cpp", "svm.cpp"],
    extra_compile_args=["-lefence", "-Wno-unused-result"]
)
svmPredict_python_extension = Extension(
    name="svmpredict",
    language = "c++",
    sources = ["svmpredictmodule.cpp", "svm_model_python.cpp", "svm.cpp"],
    extra_compile_args=["-lefence", "-Wno-unused-result"]
)
setup(
    name = "redsvm", 
    ext_modules=[svmTrain_python_extension, svmPredict_python_extension]
)
