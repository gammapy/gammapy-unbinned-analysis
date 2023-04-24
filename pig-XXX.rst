.. include:: ../../references.txt

.. _pig-025:


**************************
PIG 025 - Unbinned Dataset
**************************

* Author: Giacomo D'Amico, Julia Djuvsland, Tim Unbehaun (in alphabetical order)
* Created: May 9, 2022
* Accepted: -
* Status: draft
* Discussion: 

Abstract
========

Our goal is to be able to perform an unbinned analysis using gammapy.
To this end we like to propose to add a new EventDataset class with a dedicated fit
statistic and an EventDatasetEvaluator to compute the differential model prediction (flux) at each event's reconstructed coordinates.

Motivation
==========
Unbinned data analyses can provide several advantages compared to binned analyses. Firstly, 
they can be useful when very narrow features of a spectrum are expected. Then an unbinned 
fit can provide more information than a binned one, as the latter one can't be more precise 
than the bin width. Admittedly, this advantage is small when narrow bins are chosen especially
as the IRFs are binned.
Secondly, when performing time analyses the instruments response can be assumed to be perfect. 
So all features of the light curve can be fully taken into account with the unbinned analysis.  
Thirdly, using an unbinned data set can save computing time in case of low event numbers.
While the computing time is dependent on the number of bins of the binned dataset, 
it is dependent on the number of events in the unbinned case. Introducing unbinned versions of the 
currently existing gammapy datasets (as we propose here) therefore gives the user the freedom to 
choose the appropriate data structure according to their needs with the potential to save computing costs.

Use cases
=========

The EventDataset still contains all the reconstructed properties of the events (energy, position, time) and can therefore be used for

****************************************
****************************************

* Spectral analysis with narrow features
* Pulsar analysis
* Flare detection
* Time variablity
* Energy-temporal analysis
* ..


Class requirements 
==================
* Individual event information (position, energy, time); therefore the ``EventList`` seems to be a good choice
* IRF information (should support time dependence) 
  General requirement: Memory usage of the IRFs should not be too big, e.g. we only want to store the IRFs at the resolution of the instrument. Also, building of the kernel should be fast for many events and fine integration grids. 
  Open question: Do we want projected IRFs?
  Pros/Cons: 
        + Could use existing classes
        + Memory consumption is under control
        - Would require binning
        - Implementation of time dependence is not straightforward
        - Information loss due to interpolations
      In case we use projected IRFs we should only support the ``EDispMap`` and ``PSFMap`` and not the "kernel" version of those for simplicity and precision.
     
 Alternatives: 
 1. Event-wise IRFs: Interpolate unprojected IRFs to the event coordinates
        + Processing for the UnbinnedEvaluator would be fast
        + No information loss 
        - Classes would need to be implemented
        - Might be too memory intensive esp. in cases of many events that are close to each other (who could use the same binned IRF)
 2. Unprojected IRFs: Information of the observation
        + No information loss
        + Classes exist
        + Fast to build the Dataset
        - Slow to build the kernels for each event
        - Cannot inherit from MapDataset (might complicate stacking)
* Want to store and evaluate models using an UnbinnedEvaluator. 
* Which "convenience fucntions" similar to the ``MapDataset``: 
  ** copy: Can be inherited?
  ** create: Modification needed wrt ``MapDataset``
  ** cutout: Modification needed wrt ``MapDataset``
  ** downsample: Only meaningful in case we use projected IRFs; then modifications needed
  ** fake: Yes, but use EventSampler
  ** I/O operations: Yes, if we support stacking
  ** from_geoms: Only meaningful in case we use projected IRFs)
  ** npred (npred_*): Need to return a list with event response and summed number of events in FoV.
  ** pad: not needed 
  ** residuals/visualisation: Want to have "on the fly" binning of the events and inherit the functions
  ** resample_energyAxis: Can be inherited
  ** slice_by_energy: Needs slight modifications
  ** slice_by_index: Could be done based on the binning of the mask
  ** stack: Makes sense in case of many observations under similar IRFs with few events (e.g. high energy).
            Note: Stacking is not expected to increase the speed of the analysis. Thus it mostly makes sense if you want to serialise. 
  ** stat_array + stat_sum: Modification needed wrt ``MapDataset``
  ** to_image: only for visualisation
  ** to_masked: Modification needed wrt ``MapDataset`` and would require stack to be implemented. 
  ** to_region_map_dataset/ to_spectrum_dataset: Want to have links to binned datasets as well as 1D datasets. 
  All in all, many methods are useful but need adaptation. Need to discuss how to avoid code duplication while maintaining transparency. 
  

Implementation
==============
EventDataset:

**********************************
**********************************

* DL4 (Eventlist + projected IRFs)
* We need a maker class
* Models
* unbinned likelihood (stat_sum)
`$-2 \\log \\mathcal{L} =  2 N_{pred} - 2 \\sum_{i} \\log \\phi( E_i, \\vec{r}_i )$`

* Binned Dataset functionality: create, downsample (the IRFs), pad, plotting, .to and .from methods, ...
* No need for slices

EventDatasetEvaluator:

**********************************
**********************************

* Takes: One model + IRFs + Events
* Returns differential model flux at event's position, the total model flux inside the mask
* Uses Event kernels for the integration grid which are computed (ideally) once and stored

(Alternatives)
==============
One conceivable alternative would be to extend the existing Datasets by an EventList.
This has the disadvantage of increasing the size of the objects by information not needed
in the binned analyses. In addition one would have to think about a different way to handle
the fit statistics (stat_sum) as the likelihood for the unbinned fit is different than for 
the binned version.

Status
======
We are working on it :)


Outlook
=======


Decision
========
