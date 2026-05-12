.. _era_dataset:

ERA dataset
-----------

**Data Set Characteristics:**

:Number of Instances: 1000
:Number of Attributes: 4 ordinal predictive attributes
:Attribute Information:
    - in1, in2, in3, in4: four ordinal input attributes describing a job
      candidate (the original WEKA donation does not name the individual
      attributes)
    - out: overall ordinal acceptance score from 1 (strongly reject) to 9
      (strongly accept)
:Class Distribution: (92, 142, 181, 172, 158, 118, 88, 31, 18)
:Missing Attribute Values: None
:Donor: Arie Ben-David, Holon Academic Inst. of Technology, Israel.

The Employee Rejection/Acceptance (ERA) dataset was donated by Arie
Ben-David in the context of his work on automatic generation of symbolic
multiattribute ordinal knowledge-based decision support systems
(Ben-David, 1992). It is described in the ordinal-classification
literature as a student-survey dataset capturing the willingness to hire
a candidate as a function of four ordinal candidate features. The target
is an overall ordinal acceptance score on a 1-9 scale.

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
- OpenML records: ERA at https://www.openml.org/d/1030; see also ESL
  (d/1027), LEV (d/1029), SWD (d/1028).
