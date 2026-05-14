.. _swd_dataset:

SWD dataset
-----------

**Data Set Characteristics:**

:Number of Instances: 1000
:Number of Attributes: 10 ordinal predictive attributes
:Attribute Information:
    - in1..in10: ordinal risk-assessment features filled in by qualified
      social workers
    - out: ordinal level of risk used in family-court decisions (1 =
      lowest risk ... 4 = highest risk)
:Class Distribution: (32, 352, 399, 217)
:Missing Attribute Values: None
:Donor: Arie Ben-David, Holon Academic Inst. of Technology, Israel.

The Social Workers' Decisions (SWD) dataset records real-world risk
assessments by qualified social workers regarding whether an allegedly
abused or neglected child should remain at home.  The target is the
ordinal risk level used in family-court decisions.

This dataset is not catalogued in the UCI Machine Learning Repository.
It is part of the ESL/ERA/LEV/SWD family of ordinal benchmarks donated
by Arie Ben-David (Holon Academic Inst. of Technology, Israel — now
Holon Institute of Technology, HIT) and distributed as
``datasets-arie_ben_david.tar.gz`` through the WEKA project
(https://waikato.github.io/weka-wiki/datasets/). The four datasets are
mirrored on OpenML (ESL: d/1027, ERA: d/1030, LEV: d/1029, SWD: d/1028).
The TOC-UCO repository (Ayllón-Gavilán et al. 2026,
https://github.com/ayrna/tocuco) curates this family among its 46
tabular ordinal-classification benchmarks.

References
----------
- Ben-David, A. (1992). "Automatic generation of symbolic multiattribute
  ordinal knowledge-based DSSs: methodology and applications".  Decision
  Sciences, 23(6), 1357-1372.
- OpenML records: SWD at https://www.openml.org/d/1028; see also ESL
  (d/1027), ERA (d/1030), LEV (d/1029).
