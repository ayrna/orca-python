from distutils.core import setup, Extension

svorexTrain_python_extension = Extension(
    name="svorextrain",
    language = "c",
    sources = ["svorex_train.c", "alphas.c", "cachelist.c", "datalist.c", 
               "def_settings.c", "loadfile.c", "ordinal_takestep.c", 
               "setandfi.c", "smo_kernel.c", "smo_routine.c", "smo_settings.c", 
               "smo_timer.c", "svc_predict.c", "smo_model_python.c", "smo_loadproblem_python.c",
               "smo_routine_python.c"],
    extra_compile_args=["-lefence", "-Wno-unused-result"]
)
svorexPredict_python_extension = Extension(
    name="svorexpredict",
    language = "c",
    sources = ["svorex_predict.c", "alphas.c", "cachelist.c", "datalist.c", 
               "def_settings.c", "loadfile.c", "ordinal_takestep.c", 
               "setandfi.c", "smo_kernel.c", "smo_routine.c", "smo_settings.c", 
               "smo_timer.c", "svc_predict.c", "smo_model_python.c"],
    extra_compile_args=["-lefence", "-Wno-unused-result"]
)
setup(
    name = "svorex", 
    ext_modules=[svorexTrain_python_extension, svorexPredict_python_extension]
)