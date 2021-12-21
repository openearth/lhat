.. lsm documentation master file, created by
   sphinx-quickstart on Tue Nov 30 09:23:14 2021.


Welcome to LHAT's documentation!
********************************

Landslide Hazard Assessment Tool (LHAT) is a rapid assessment tool for landslide hazards around the world.

This online documentation describes the tool, installation and its usage to
derive landslide susceptibility maps.


What is LHAT?
=============

LHAT is a GIS-based tool for data-driven forecasting of landslide susceptibilities.
LHAT relies on globally and publicly available data sources as input for the data
model. The user can choose between three machine learning models: Support Vector,
Logistic Regression and Random Forest. These models are auto-parameterised using
GridSearch and scored based on accuracy. Currently, the data inputs are considered
static. LHAT has the option to be integrated into a forecasting framework for
Early Warning Systems, and can receive dynamic and near real-time inputs such as
precipitation data, inferometric Synthetic Aperture Radar (InSAR) maps, etc..


.. toctree::
   :maxdepth: 2
   :caption: Contents:

setup
lhat
modules


Contact
=======
For further enquiries, please approach the following developers: Giorgio Santinelli
(giorgio.santinelli@deltares.nl) and Robyn Gwee (robyn.gwee@deltares.nl). LHAT
has been developed with support from the Deltares Natural Hazards Strategic
Research Program.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
