# This Makefile is used under Linux

.PHONY: clean svorex_c svorex_python default

default: svorex_python

svorex_c: main.c alphas.c cachelist.c datalist.c def_settings.c loadfile.c ordinal_takestep.c setandfi.c smo_kernel.c smo_routine.c smo_settings.c smo_timer.c svc_predict.c
	gcc -o svorex main.c alphas.c cachelist.c datalist.c def_settings.c loadfile.c ordinal_takestep.c setandfi.c smo_kernel.c smo_routine.c smo_settings.c smo_timer.c svc_predict.c -lm -O2 -Wall

svorex_python: setup.py
	python setup.py build_ext --inplace && rm -f svorex.cpp && rm -Rf build

clean:
	rm *.so svorex
