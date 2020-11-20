LIBSVM_DIR = classifiers/libsvmRank/python
SVOREX_DIR = classifiers/svorex

.PHONY: clean subdirs

subdirs: $(LIBSVM_DIR) $(SVOREX_DIR)
	$(MAKE) -e -C $(LIBSVM_DIR)
	$(MAKE) -e -C $(SVOREX_DIR)
	
clean: $(LIBSVM_DIR) $(SVOREX_DIR)
	$(MAKE) -e -C $(LIBSVM_DIR) clean
	$(MAKE) -e -C $(SVOREX_DIR) clean
