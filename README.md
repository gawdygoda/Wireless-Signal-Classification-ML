# Wireless-Signal-Classification-ML

This repository uses an SVM Classification algorithm to detect the signal type of simulated modulated signals.

Run main.py to use Curtis's set of features
, run main2.py to use Jake's set of features

There are other slight differences between the two, in terms of plotting and test code.

You will need a "data" directory at the top level with the RadioML Pickel File: [RML2016.10a_dict.pkl](data/RML2016.10a_dict.pkl)

## Tools & Technologies

- Language - [**Python (v3.10.11)**](https://www.python.org)
  - Libraries
    - scikit-learn (v1.5.2)
    - PyWavelets (1.7.0)
    - Pandas (1.4.4)
    - seaborn (0.13.2)
    - matplotlib (3.9.2)

_NOTE: These were the version used during testing, it is likely a later minor version should be fine. Be careful with older versions or major newer versions._
