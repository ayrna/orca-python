.. _lev_dataset:

LEV dataset
-----------

**Data Set Characteristics:**

:Number of Instances: 1000
:Number of Attributes: 4 ordinal predictive attributes
:Attribute Information:
    - in1, in2, in3, in4: four ordinal student-rating attributes capturing
      different dimensions of a university course evaluation (the WEKA
      donation does not restrict to any particular degree programme)
    - out: overall ordinal lecturer rating (1 = lowest ... 5 = highest)
:Class Distribution: (93, 280, 403, 197, 27)
:Missing Attribute Values: None
:Donor: Arie Ben-David, Holon Academic Inst. of Technology, Israel.

The Lecturers Evaluation (LEV) dataset collects anonymous student
evaluations of university lecturers scored along four ordinal dimensions;
the target is the overall lecturer rating on a 1–5 scale.

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
- OpenML records: LEV at https://www.openml.org/d/1029; see also ESL
  (d/1027), ERA (d/1030), SWD (d/1028).
