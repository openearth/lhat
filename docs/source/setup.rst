.. _setup:

Setting up
***********

Installation
=============

Clone the LHAT repository locally from https://github.com/openearth/lhat

.. code-block:: text

      >> git clone https://github.com/openearth/lhat.git

Navigate to the directory where you cloned the repository and create a conda environment from the yml file.

.. code-block:: text

      >> conda env create -f environment.yml

Once the environment is created, activate it and import lhat. Ensure your working
directory is the same root folder of the cloned repository.

Activate the conda environment
.. code-block:: text

      >> conda activate lhat


Import LHAT as so below:

.. code-block:: python

      >>> import lhat


Run the example script in your command line

.. code-block:: text

      >> python example.py




Parameterising LHAT
===================

The LHAT tool requires some parameters. The following arguments are necessary:
* Name of project
* Coordinate Referencing System (crs)
* Path to where your landslide point dataset is (accepts JSON or .shp format)
* A random state (necessary for reproducability of data)
* Bounding box for clipping public assets
* inputs (dictionary)
* no_data values (can be a list or single value)
* Pixel resolution (important for the retrieval of online datasets)
* Kernel size (default 3x3): necessary for defining an area as 'landslide', since
a landslide does not occur as locally as a point but as an area affected.

.. note::
        Not all input data have an online source. For those that do not, using
        the 'online' option will return nothing.

The following code snippet can be used for the initial parameterisation, also
available in `example.py` that is placed in the root of the lhat repository.

.. literalinclude:: ../../example.py
        :language: python
        :caption: Example of parameterising inputs
        :lines: 11-77



Array harmonisation
====================

Once the inputs have been defined, the tool harmonises all the input datasets
into a stack of arrays by reprojecting and resampling them into the same grid
size. The resampling is performed using nearest neighbour, and all datasets are
reprojected into the crs defined in `project.io.inputs()`. Subsequently, any
pixel from any input dataset that has no data becomes masked for the entire
stack of arrays, leading to a final output consisting of an array where all valid
data exists across all input datasets.


Data engineering step
======================

Once the valid set of arrays are generated, the pixels that intersect with the
landslide points are selected, as well as a 3x3 kernel window around the pixel.
These points are marked as landslides areas, and are then selected across the
arrays and flattened into a single dimension (for each type of input dataset).
For the same number of landslide points, the same number of non-landslide points
are then randomly selected in the stack of arrays and subsequently flattened as
well. The flattened data, in the form of a `pandas.DataFrame` object, serves as
input for the next steps, i.e. machine learning. Using the `generate_xy()`
method, two dataframes are exported: the first consists of the flattened pixel
values from each input dataset that coincide with the landslide point and the
kernel window around it, and the second consists of landslide classes, where
0 indicates no landslide and 1 indicates landslide.

.. literalinclude:: ../../example.py
        :language: python
        :caption: Example of parameterising inputs
        :lines: 79-82
        :linenos:

If the data is categorical, you will need to call `pd.get_dummies`
