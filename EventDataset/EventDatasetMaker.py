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
    available_selection = ["exposure", "background", "psf", "edisp"]

    def __init__(
            self,
            selection=None,
            safe_mask_maker=None,
            **maker_kwargs,
        ):
         
            if selection is None:
                selection = self.available_selection

            selection = set(selection)

            if not selection.issubset(self.available_selection):
                difference = selection.difference(self.available_selection)
                raise ValueError(f"{difference} is not a valid method.")

            self.selection = selection
            self.map_ds_maker = MapDatasetMaker(selection=selection, **maker_kwargs)
            self.safe_mask_maker = safe_mask_maker

       
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
#         meta_table["RA_PNT"] = [observation.pointing_radec.icrs.ra.deg] * u.deg
#         meta_table["DEC_PNT"] = [observation.pointing_radec.icrs.dec.deg] * u.deg
        meta_table["RA_PNT"] = [observation.pointing.fixed_icrs.ra.deg] * u.deg
        meta_table["DEC_PNT"] = [observation.pointing.fixed_icrs.dec.deg] * u.deg

        return meta_table    


    def run(self, emptyMapDs, obs):
        """Make the EventDataset.

        Parameters
        ----------
        emptyMapDs : MapDataset
               Empty MapDataset specifying the geometries for the IRFs
        obs : `~gammapy.data.Observation`
            Observation to build the EventDataset from 
        
        Returns
        -------
        dataset : `~gammapy.datasets.EventDataset`
            EventDataset.
        """
        kwargs = {}
        kwargs["meta_table"] = self.make_meta_table(obs)
        kwargs["events"] = obs.events
        
        dataset = self.map_ds_maker.run(emptyMapDs, obs)
        
        if self.safe_mask_maker: 
            dataset = self.safe_mask_maker.run(dataset, obs)
            kwargs["mask_safe"] = dataset.mask_safe
        
        for key in self.selection:
            kwargs[key] = getattr(dataset, key, None)
            
        kwargs['gti'] = dataset.gti
                          
        return EventDataset(name=dataset.name, **kwargs)
