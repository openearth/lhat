.. _LHAT-background:

Landslide Hazard Assessment Tool
*********************************

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


Why use LHAT?
=============

LHAT is a flexible tool that is capable of taking in both user-defined inputs (
for example, high-resolution LiDAR data) as well as keeping the option to rely
on global and/or publicly available assets. LHAT was built with the GHIRAF
Framework in mind. GHIRAF is a Globally-applicable High-resolution Integrated
Risk Assessment Framework, allowing for provision of globally available data (
local when possible), rapid risk assessment and flexibility. Building LHAT in
this manner allows its inputs to remain flexible, and enables inter-operability
between other risk assessment tool (such as Delft-FIAT, RA2CE, Ri2de or Criticality
tool).

LHAT has been tested on areas within Bangladesh and Jamaica thus far, with accuracies
of more than 80%.


When should I use LHAT?
=======================

A main requirement of the LHAT tool is the need for historical occurrences of
landslides in an area (in the form of coordinates). The sufficiency and
representation of the landslide data, both in time and space, is a main
determining factor on the accuracy of the output susceptibility map.

The final output of the LHAT tool is a raster with two bands; the first band displays the probability that a landslide will not occur, while the second band displays the probability that a landslide will occur. For a given pixel, the sum of band 1 and band 2 is 1. 
occurrences.

.. figure:: images/results.png

        Fig. 1: Example results from each of the models


How do I get started?
=====================

:ref:`setup`.
