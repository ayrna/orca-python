.. _balance_scale_dataset:

Balance Scale dataset
---------------------

**Data Set Characteristics:**

:Number of Instances: 625
:Number of Attributes: 4 numeric, predictive attributes
:Attribute Information:
    - left_weight: integer weight on the left pan (1-5)
    - left_distance: integer distance from fulcrum on the left pan (1-5)
    - right_weight: integer weight on the right pan (1-5)
    - right_distance: integer distance from fulcrum on the right pan (1-5)
    - class: which way the scale tips
        - L (left)
        - B (balanced)
        - R (right)
:Class Distribution: L=288, B=49, R=288
:Missing Attribute Values: None
:Source: R. S. Siegler (1976), modelled in a synthetic dataset donated to
    the UCI Machine Learning Repository by Tim Hume on 22 April 1994.

A synthetic dataset modelling Piaget-style balance-scale experiments.
The class is determined by comparing the products of left_weight ×
left_distance and right_weight × right_distance.  The target is treated
as an ordinal target in the ordinal-classification literature, following
the convention of the AYRNA group (Gutiérrez et al. 2016) and the
TOC-UCO repository (Ayllón-Gavilán et al. 2026), with the order
``L < B < R`` along the tilt axis.  Outside the ordinal-classification
community Balance Scale is more often treated as nominal.

References
----------
- Siegler, R. S. (1976). "Three aspects of cognitive development".
  Cognitive Psychology, 8, 481-520.  (No issue number is given in the
  UCI ARFF header; the volume-only form is standard.)
- Siegler, R. (1976). Balance Scale [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5488X
- UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/12/balance+scale
- Gutiérrez, P. A., Pérez-Ortiz, M., Sánchez-Monedero, J., Fernández-
  Navarro, F., & Hervás-Martínez, C. (2016). Ordinal regression methods:
  survey and experimental study.  IEEE Transactions on Knowledge and Data
  Engineering, 28(1), 127-146.
- Ayllón-Gavilán, R., Guijo-Rubio, D., Gómez-Orellana, A. M.,
  Bérchez-Moreno, F., Vargas, V. M., & Gutiérrez, P. A. (2026).
  TOC-UCO: a comprehensive repository of tabular ordinal classification
  datasets.  Neurocomputing, 133528.
