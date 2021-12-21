.. _improvements:

Future Developments
********************


Dynamic data ingestion
=======================

Although LHAT can theoretically take dynamic inputs, certain datasets may need
expert judgement or statistically derived thresholds to determine the
contribution of a variable (eg. rainfall) to the susceptibility of landslides
occurring. Determining thresholds may be challenging as one region may have
different thresholds compared to others. However, this is important to address
in order to support Early Warning Systems and enable rapid action for eg.
providing humanitarian aid. Other types of dynamic data to consider also
include InSAR data, which can help detect deformations on the surface. By
detecting significant deformations, this will contribute to the risk assessment.


Model refinement
=================

Currently, model parameterisation is performed automatically using GridSearch
by providing the best combination of model parameters that generate the
highest accuracy. A caveat of this process is that this may lead to overfitting.
Future developments will include looking into an alternative factor to rank the
performance of the various model parameters.

Secondly, model refinement can also be performed by providing data on the
return period of landslides. However, this is a challenging activity as many
publicly available datasets either scarce, incomplete or insufficiently
representative of landslide occurrences in such areas. Although it remains a
priority to generate return periods of landslide occurrences, this is largely
hindered by lack of sufficient data.

Do you have a suggestion on potential data sources that can help us? Feel free
to contact one of us (giorgio.santinelli@deltares.nl or robyn.gwee@deltares.nl)


Making an executable
=====================

To improve accessibility to LHAT, an executable and/or an API (Application
Programming Interface) can be made to enable inter-operability between other tools
and improve access by non-python users.


Impact Assessments: Damage functions and coupling with impact assessment tools
===============================================================================

The output of LHAT is a machine-readable format, which can be read by other
Deltares impact assessment tools such as RA2CE, Delft-FIAT Accelerator and Ri2de.
Next logical step is to generate impact assessments from these tools. To
empower the analysis, damage functions of landslide hazards can be made as an
estimate of potential economic damage.
