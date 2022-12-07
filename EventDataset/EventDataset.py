import gammapy
import numpy as np
import astropy.units as u
from gammapy.maps import MapAxis, WcsGeom, Map, MapAxes, WcsNDMap
from gammapy.modeling.models import (
    Model,
    Models,
    FoVBackgroundModel,
    DatasetModels
)
from gammapy.utils.integrate import trapz_loglog
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.fits import HDULocation, LazyFitsData
from gammapy.irf import EDispKernelMap, EDispMap, PSFKernel, PSFMap
import gammapy.makers.utils 
from gammapy.data import GTI

PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg

RAD_MAX = 0.66
RAD_AXIS_DEFAULT = MapAxis.from_bounds(
    0, RAD_MAX, nbin=66, node_type="edges", name="rad", unit="deg"
)
MIGRA_AXIS_DEFAULT = MapAxis.from_bounds(
    0.2, 5, nbin=48, node_type="edges", name="migra"
)

BINSZ_IRF_DEFAULT = 0.2


class EventDataset(gammapy.datasets.Dataset):
    """
    Bundle together event list, background, IRFs, models and compute a likelihood.
    Uses unbinned statistics by default.
     
    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference target geometry in reco energy, used for the background maps
    models : `~gammapy.modeling.models.Models`
        Source sky models.  
    background : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Background cube
    mask_fit : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask to apply to the likelihood for fitting.
    mask_safe : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask defining the safe data range.
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    observation: `~gammapy.data.Observation` 
         Observation containing the data (EventList) and IRFs
    name : str
         Name of the returned dataset.
   
    ?If an `HDULocation` is passed the map is loaded lazily. This means the
    ?map data is only loaded in memory as the corresponding data attribute
    ?on the MapDataset is accessed. If it was accessed once it is cached for
    ?the next time.        
    """
    
    stat_type = "unbinned"
    tag = "EventDataset"
    background = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)
    
    _lazy_data_members = [
        "background",
        "mask_fit",
        "mask_safe",
    ]
    
    def __init__(
        self,
        geom=None,
        models=None,
        background=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        observation=None,
        name=None,
    ): 
 
        self._name = make_name(name)
        self.background = background #WSCNDmap
        self.background_model = None #FoVBGmodel
        self._background_cached = None        
        self._response_background_cached = None
        self._background_parameters_cached = None
        self._background_parameters_cached_prev = None
        
        self.geom=geom
        
        self.mask_fit = mask_fit
        self.mask_safe = mask_safe
        
        self._models = models
        self.meta_table = meta_table
        
        #self.events = observation.events
        self.obs = observation
        self._evaluators=None
            
    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")
        str_ += "\t{:32}: {{events}} \n".format("Event list")
        #str_ += "\t{:32}: {{background:.2f}}\n".format("Total background counts")

        # likelihood section
        str_ += "\t{:32}: {{stat_type}}\n".format("Fit statistic type")
        str_ += "\t{:32}: {{stat_sum:.2f}}\n\n".format(
            "Fit statistic value (-2 log(L))"
        )

        info = self.info_dict()
        str_ = str_.format(**info)

        # model section
        n_models, n_pars, n_free_pars = 0, 0, 0
        if self._models is not None:
            n_models = len(self._models)
            n_pars = len(self._models.parameters)
            n_free_pars = len(self._models.parameters.free_parameters)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)
        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self._models is not None:
            str_ += "\t" + "\n\t".join(str(self._models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)
   
    @property
    def evaluators(self):
        """Model evaluators"""
        return self._evaluators    
    
    def stat_array(self):
        pass
    
    @classmethod
    def create(
        cls,
        geom,
        observation=None,
        name=None,
        meta_table=None,
        **kwargs,
    ):
        """Create an EventDataset object

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Reference target geometry in reco energy, used for the background maps
        name : str
            Name of the returned dataset.
        meta_table : `~astropy.table.Table`
            Table listing information on observations used to create the dataset.
            One line per observation for stacked datasets.

        Returns
        -------
        empty_ds : `EventDataset`
            An empty EventDataset
        """
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name
        kwargs["geom"] = geom

        kwargs["mask_safe"] = Map.from_geom(geom, unit="", dtype=bool)

        return cls(**kwargs)

    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models   

    
    @models.setter
    def models(self, models):    

        """Set UnbinnedEvaluator(s)"""
        self._evaluators = {}

        if self._models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            
            irfs={'psf':self.obs.psf, 'edisp':self.obs.edisp, 'exposure':self.exposure}
            events = self.events.select_row_subset(self.mask_safe)
            
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = UnbinnedEvaluator(
                        model=model,
                        irfs=irfs,
                        events=events,
                        pointing=self.obs.pointing_radec,
                        acceptance = self.acceptance
                    )
                    self._evaluators[model.name] = evaluator
        
        self._models = models             
                    
                        
    def stat_sum(self, response_only=False):
        """
        compute the unbinned TS value
        """
        response, npred_sum = self.response_background() # start with the bkg (all events)
        # add the models
        for ev in self._evaluators.values():
            npred, mask, s = ev.compute_response()
            response[mask] += npred
            npred_sum += s
        
        if response_only:
            return np.log(response)
        if np.all(response>0): 
            # valid response - no event has npred <= 0
            logL = np.sum(np.log(response)) - npred_sum
            return -2 * logL
        else:
            # invalid response, reject the model
            return np.inf

    def response_background(self):
        """
        compute the response of the background.
        returns: interpolated bkg value for all events, sum of bkg counts
        """
        #self._background_parameters_changed needs be copied from the MapDataset implementation
        if self._background_cached is not None and (self.background_model is not None or not self._background_parameters_changed):
            return self._response_bkg_cached

        # case of bkg and extra model
        if self.background_model and self.background:
            if self._background_parameter_norm_only_changed:
                self._response_bkg_cached[0] *= self.bkg_renorm()
                self._response_bkg_cached[1] *= self.bkg_renorm()
                return self._response_bkg_cached
            elif self._background_parameters_changed:
                values = self.background_model.evaluate_geom(geom=background.geom)
                bkg_map = self.background * values
                # interpolate and sum the bkg values
                bkg_sum = bkg_map.data[self.ds.mask_safe.data].sum()
                events = self.events.select_row_subset(self.mask_safe)
                coords = events.map_coord(background.geom)
                # we need to interpolate the differential bkg npred
                bkg_map.quantity /= bkg_map.geom.bin_volume()
                self._response_bkg_cached = bkg_map.interp_by_coord(coords, method='linear'), bkg_sum
                return self._response_bkg_cached
    
        # case of bkg but no extra model        
        elif self.background:
            bkg_map = self.background
            # interpolate and sum the bkg values
            bkg_sum = bkg_map.data[self.ds.mask_safe.data].sum()
            events = self.events.select_row_subset(self.mask_safe)
            coords = events.map_coord(background.geom)
            # we need to interpolate the differential bkg npred
            bkg_map.quantity /= bkg_map.geom.bin_volume()
            self._response_bkg_cached = bkg_map.interp_by_coord(coords, method='linear'), bkg_sum
            return self._response_bkg_cached
    
        # case of no bkg at all
        else: 
            if(self.obs.events != None): return (np.zeros(len(self.obs.events.table)), 0)
            else: return (np.zeros(0),0)

    def _background_parameters_changed(self):
        values = self.background_model.parameters.value
        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._background_parameters_cached == values)
        if changed:
            self._background_parameters_cached = values
        return changed

    @property
    def _background_parameter_norm_only_changed(self):
        """Only norm parameter changed"""
        norm_only_changed = False
        idx = self._bkg_norm_idx()
    
        values = self.background_model.parameters.value
        if idx and self._background_parameters_cached is not None:
            changed = self._background_parameters_cached_previous == values
            norm_only_changed = sum(changed) == 1 and changed[idx]

        if not norm_only_changed:
            self._background_parameters_cached_previous = values
        return norm_only_changed

    def bkg_renorm(self):
        value = self.background_model.parameters.value[self._bkg_norm_idx]
        value_cached = self._background_parameters_cached_previous[self._bkg_norm_idx]
        return value / value_cached

    @property
    def _bkg_norm_idx(self):
        """norm index"""
        names = self.background_model.parameters.names
        idx = [idx for idx, name in enumerate(names) if name in ["norm", "amplitude"]]
        if len(idx) == 1:
            idx = idx[0]
        else:
            idx = None

            
    def info_dict(self):
        """Info dict with summary statistics

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = {}
        info["name"] = self.name

        ontime = u.Quantity(np.nan, "s")
        if self.obs:
            if self.obs.gti: ontime = self.obs.gti.time_sum
            if self.obs.events: 
                energies = self.obs.events.energy
                info["events"] = unit_array_to_string(energies)
            else: info["events"] = None
        info["events"] = None
        info["ontime"] = ontime

        info["stat_type"] = self.stat_type

        stat_sum = np.nan
        if self._models is not None:
            stat_sum = self.stat_sum()

        info["stat_sum"] = float(stat_sum)

        return info