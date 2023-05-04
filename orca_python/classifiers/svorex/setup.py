from distutils.core import setup, Extension

svorex_python_extension = Extension(
    name="svorex",
    language = "c",
    sources = ["svorex_module.c", "svorex_train.c", "svorex_predict.c", "alphas.c", "cachelist.c", "datalist.c", 
               "def_settings.c", "loadfile.c", "ordinal_takestep.c", 
               "setandfi.c", "smo_kernel.c", "smo_routine.c", "smo_settings.c", 
               "smo_timer.c", "svc_predict.c", "smo_model_python.c", "smo_loadproblem_python.c",
               "smo_routine_python.c"],
    extra_compile_args=["-lefence", "-Wno-unused-result"]
)
setup(
    name = "svorex", 
    ext_modules=[svorex_python_extension]
)
