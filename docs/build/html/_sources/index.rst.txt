.. lsm documentation master file, created by
   sphinx-quickstart on Tue Nov 30 09:23:14 2021.


Welcome to LHAT's documentation!
********************************

Landslide Hazard Assessment Tool (LHAT) is a rapid assessment tool for landslide hazards around the world.

This online documentation describes the tool, its usage and the model choices that can be used to derive landslide susceptibility maps.
For further enquiries, please approach the following developers: Giorgio Santinelli (giorgio.santinelli@deltares.nl) and Robyn Gwee (robyn.gwee@deltares.nl)


Installing LHAT
===============

Clone the LHAT repository locally from https://github.com/openearth/lhat

.. code-block:: text

      >> git clone https://github.com/openearth/lhat.git

Navigate to the directory where you cloned the repository and create a conda environment from the yml file.

.. code-block:: text

      >> conda env create -f environment.yml

An example file has been made to run LHAT.

.. code-block:: text
      >> python example.py

.. automodule:: lhat
    :imported-members:
    :members:
    :undoc-members:
    :show-inheritance:


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
