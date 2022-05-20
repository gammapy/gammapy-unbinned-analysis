.. include:: ../../references.txt

.. _pig-023:

**********************************
PIG 23 - Unbinned Dataset
**********************************

* Author: Giacomo D'Amico, Julia Djuvsland, Tim Unbehaun (in alphabetical order)
* Created: May 9, 2022
* Accepted: -
* Status: draft
* Discussion: 

Abstract
========

Our goal is to be able to perform an unbinned analysis using gammapy.
To this end we like to propose to add new Dataset classes with a dedicated fit
statistic and an EventList in addition to the features already implemented for the 
other gammapy Datasets.

Motivation
==============
Unbinned data analyses can provide several advantages compared to binned fits. Firstly, 
they can be useful when very narrow features of a spectrum are expected. Then an unbinned 
fit can provide more information than a binned one, as the latter one can't be more precise 
than the bin width. Admittedly, this advantage is small when narrow bins are chosen especially
as the IRFs are binned.
Secondly, using an unbinned data set can save computing time in case of sparsely populated spectra.
While the computing time is dependent on the number of bins of the Dataset in the binned case, 
it is dependent on the number of events in the unbinned case. Introducing unbinned versions of the 
currently existing gammapy Datasets (as we propose here) therefore gives the user the freedom to 
choose the appropriate data structure according to their needs with the potential to save computing costs.


Alternatives
==============
One conceivable alternative would be to extend the existing Datasets by an EventList.
This has the disadvantage of increasing the size of the objects by information not needed
in the binned analyses. In addition one would have to think about a different way to handle
the fit statistics (stat_sum) as the likelihood for the unbinned fit is different than for 
the binned version.

Status
==============
We are working on it :)


Outlook
==============


Decision
==============