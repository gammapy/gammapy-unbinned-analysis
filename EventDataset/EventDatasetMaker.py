# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from astropy.table import Table
from regions import PointSkyRegion
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map
from gammapy.makers import Maker, MapDatasetMaker, SafeMaskMaker
from gammapy.datasets import MapDataset
from gammapy.makers.utils import (
    make_edisp_kernel_map,
    make_edisp_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_psf_map,
)
from EventDataset import EventDataset

__all__ = ["EventDatasetMaker"]

log = logging.getLogger(__name__)


class EventDatasetMaker(Maker):
    """Make event dataset for a single IACT observation.

    """

    tag = "EventDatasetMaker"

 #   def __init__(
 #       self,
 #   ):
       
    @staticmethod
    def make_meta_table(observation):
        """Make info meta table.
        Parameters
        ----------
        observation : `~gammapy.data.Observation`
            Observation
        Returns
        -------
        meta_table: `~astropy.table.Table`
        """
        meta_table = Table()
        meta_table["TELESCOP"] = [observation.aeff.meta.get("TELESCOP", "Unknown")]
        meta_table["OBS_ID"] = [observation.obs_id]
        meta_table["RA_PNT"] = [observation.pointing_radec.icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing_radec.icrs.dec.deg] * u.deg

        return meta_table    


    def run(self, emptyMapDs, obs, safeMaskMaker=None):
        """Make the EventDataset.

        Parameters
        ----------
        emptyMapDs : MapDataset
               Empty MapDataset specifying the geometries for the IRFs
        obs : `~gammapy.data.Observation`
            Observation to build the EventDataset from 
        safeMaskMaker : `~gammapy.makers.SafeMaskMaker`
            SafeMaskMaker in case events should be excluded from the dataset
            
        Returns
        -------
        dataset : `~gammapy.datasets.EventDataset`
            EventDataset.
        """
        kwargs = {}
        kwargs["meta_table"] = self.make_meta_table(obs)
        
        mapDsMaker = MapDatasetMaker(selection=("exposure", "psf", "edisp"))
        dataset = mapDsMaker.run(emptyMapDs, obs)
        
        if safeMaskMaker: dataset = safeMaskMaker.run(dataset, obs)

        event_dataset = EventDataset(events=obs.events, exposure=dataset.exposure, edisp=dataset.edisp, psf=dataset.psf, mask_fit=dataset.mask)
        
        kwargs["exposure"] = dataset.exposure
        kwargs["psf"] = dataset.psf
        kwargs["edisp"] = dataset.edisp
                          
        return event_dataset.__class__(name=dataset.name, **kwargs)
