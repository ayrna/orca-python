.. _esl_dataset:

ESL dataset
-----------

**Data Set Characteristics:**

:Number of Instances: 488
:Number of Attributes: 4 ordinal predictive attributes
:Attribute Information:
    - in1, in2, in3, in4: ordinal scores from psychometric tests and
      interviews assigned by expert psychologists
    - out: overall ordinal fitness score for the position (1 = lowest
      fitness ... 9 = highest fitness)
:Class Distribution: (2, 12, 38, 100, 116, 135, 62, 19, 4)
:Missing Attribute Values: None
:Donor: Arie Ben-David, Holon Academic Inst. of Technology, Israel.
:Original data owner: "Yoav Ganzah, Business Administration School,
    Tel Aviv Univerity" (verbatim from the WEKA donation header; the
    academic publication record uses the spelling "Yoav Ganzach",
    Tel Aviv University).

The Employee Selection (ESL) dataset contains profiles of candidates for
industrial jobs, scored along four ordinal dimensions.  The ordinal
target is the overall fitness rating issued by an expert.

This dataset is not catalogued in the UCI Machine Learning Repository.
It is part of the ESL/ERA/LEV/SWD family of ordinal benchmarks donated
by Arie Ben-David (Holon Academic Inst. of Technology, Israel — now
Holon Institute of Technology, HIT) and distributed as
``datasets-arie_ben_david.tar.gz`` through the WEKA project
(https://waikato.github.io/weka-wiki/datasets/). The four datasets are
mirrored on OpenML (ESL: d/1027, ERA: d/1030, LEV: d/1029, SWD: d/1028).
The TOC-UCO repository (Ayllón-Gavilán et al. 2025,
https://github.com/ayrna/tocuco) curates this family among its 46
tabular ordinal-classification benchmarks.

References
----------
- Ben-David, A. (1992). "Automatic generation of symbolic multiattribute
  ordinal knowledge-based DSSs: methodology and applications".  Decision
  Sciences, 23(6), 1357-1372.
- OpenML records: ESL at https://www.openml.org/d/1027; see also ERA
  (d/1030), LEV (d/1029), SWD (d/1028).
