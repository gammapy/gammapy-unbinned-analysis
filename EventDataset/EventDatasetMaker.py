# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import astropy.units as u
from astropy.table import Table
from regions import PointSkyRegion
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map
from gammapy.makers import Maker
from gammapy.makers.utils import (
    make_counts_rad_max,
    make_edisp_kernel_map,
    make_edisp_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_psf_map,
)

__all__ = ["EventDatasetMaker"]

log = logging.getLogger(__name__)


class EventDatasetMaker(Maker):
    """Make event dataset for a single IACT observation.

    Parameters
    ----------
    background_oversampling : int
        Background evaluation oversampling factor in energy.
    background_interp_missing_data: bool
        Interpolate missing values in background 3d map.
        Default is True, have to be set to True for CTA IRF.

    """

    tag = "EventDatasetMaker"

    def __init__(
        self,
        background_oversampling=None,
        background_interp_missing_data=True,
    ):
        self.background_oversampling = background_oversampling
        self.background_interp_missing_data = background_interp_missing_data

    
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


    def _init_mask_safe(self, dataset):
        """
        create a mask for events which are not part of any evaluator
        and will be discarded in the analysis
        """
        coords = dataset.obs.events.map_coord(self.mask_safe.geom)
        dataset.mask_safe = dataset.mask_safe.get_by_coord(coords)==1
    
    

    def run(self, dataset, observation):
        """Make EventDataset.

        Parameters
        ----------
        dataset : `~gammapy.datasets.EventDataset`
            Reference dataset.
        observation : `~gammapy.data.Observation`
            Observation

        Returns
        -------
        dataset : `~gammapy.datasets.EventDataset`
            Map dataset.
        """
        kwargs = {}
        kwargs["meta_table"] = self.make_meta_table(observation)
        
        
        # should we use the _init_mask_safe function from above?
        # I think this code does the same, doesn't it?
        mask_safe = Map.from_geom(dataset.geom, dtype=bool)
        mask_safe.data[...] = True        
        kwargs["mask_safe"] = mask_safe        
        
        # exposure = self.make_exposure(dataset.geom, observation)
        # kwargs["exposure"] = exposure

       # kwargs["background"] = self.make_background(
       #   dataset.geom, observation
       # )

       #psf = self.make_psf(dataset.psf.psf_map.geom, observation)
       # kwargs["psf"] = psf

        # if dataset.edisp.edisp_map.geom.axes[0].name.upper() == "MIGRA":
        #      edisp = self.make_edisp(dataset.edisp.edisp_map.geom, observation)
        # else:
        #      edisp = self.make_edisp_kernel(
        #          dataset.edisp.edisp_map.geom, observation
        #      )
        # kwargs["edisp"] = edisp
                  
        kwargs["observation"] = observation

        return dataset.__class__(name=dataset.name, **kwargs)
