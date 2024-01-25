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
from gammapy.datasets import MapDataset
from UnbinnedEvaluator import UnbinnedEvaluator
from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE

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
    exposure : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Exposure cube   
    psf : `~gammapy.irf.PSFMap` or `~gammapy.utils.fits.HDULocation`
        PSF kernel
    edisp : `~gammapy.irf.EDispMap` or `~gammapy.utils.fits.HDULocation`
        Energy dispersion kernel    
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.
    events: `~gammapy.data.EventList` 
         EventList containing the recorded photons
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
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)   
    psf = LazyFitsData(cache=True)
    
    _lazy_data_members = [
        "background",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "mask_safe",
    ]
    
    def __init__(
        self,
        geom=None,
        models=None,
        exposure=None,
        psf=None,
        edisp=None,
        background=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        name=None,
        events=None,
        gti=None
    ): 
 
        self._name = make_name(name)
        self.background = background #WSCNDmap
        self._response_bkg_cached = None
        self._background_parameters_cached = None
        self.exposure = exposure     
        self._geom=geom
        self.events = events 
        self.edisp = edisp
        self.gti = gti
        self._evaluators=None
        
        self.mask_fit = mask_fit
        self.mask_safe = mask_safe
        
        self.models = models
        self.meta_table = meta_table
        
        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )

        self.psf = psf

        if edisp and not isinstance(edisp, (EDispMap, EDispKernelMap, HDULocation)):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )

            
    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")
        
        str_ += "\t{:32}: {{events}} \n".format("Events in EventList")      
        str_ += "\t{:32}: {{signal:.2f}}\n".format("Predicted excess counts")
        str_ += "\t{:32}: {{background:.2f}}\n\n".format(
            "Predicted background counts"
        )

        str_ += "\t{:32}: {{exposure_min:.2e}}\n".format("Exposure min")
        str_ += "\t{:32}: {{exposure_max:.2e}}\n\n".format("Exposure max")
             
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
    def event_mask(self):
        """Entry for each event whether it is inside the mask or not"""
        if self.mask is None:
            return np.ones(len(self.events.table), dtype=bool)
        coords = self.events.map_coord(self.mask.geom)
        return self.mask.get_by_coord(coords)==1
    
    @property
    def events_in_mask(self):
        return self.events.select_row_subset(self.event_mask)
                
    @property
    def geoms(self):
        """Map geometries

        Returns
        -------
        geoms : dict
            Dict of map geometries involved in the dataset.
        """
        geoms = {}
        
        if self._geom is not None:
            geoms["geom"] = self._geom
        else:
            if self.mask:
                geoms["geom"] = self.mask.geom
            if self.exposure:
                geom = self.exposure.geom.to_image()
                axis = self.exposure.geom.axes["energy_true"].copy()
                axis._name = "energy"
                geoms['geom'] = geom.to_cube([axis])

        if self.exposure:
            geoms["geom_exposure"] = self.exposure.geom

        if self.psf:
            geoms["geom_psf"] = self.psf.psf_map.geom

        if self.edisp:
            geoms["geom_edisp"] = self.edisp.edisp_map.geom

        return geoms    
    
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
        events=None,
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

        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = UnbinnedEvaluator(model=model, gti=self.gti)
                    self._evaluators[model.name] = evaluator
        
        self._models = models             
                    
    @property
    def background_model(self):
        try:
            return self.models[f"{self.name}-bkg"]
        except (ValueError, TypeError):
            pass
        
    def stat_sum(self):
        """
        compute the unbinned TS value
        """
        response_bkg, bkg_sum = self.response_background() # start with the bkg (all events)
        response_signal, sig_sum = self.response_signal()
        response = response_bkg + response_signal
        total = bkg_sum + sig_sum
        response = np.where(response <= TRUNCATION_VALUE, TRUNCATION_VALUE, response)
        
        logL = np.sum(np.log(response)) - total
        return -2 * logL
        
    def response_signal(self, model_name=None):
        """Model predicted signal counts (differential) at the events coordinates.
        If a model name is passed, predicted counts from that component are returned.
        Else, the total signal counts are returned.

        Parameters
        ----------
        model_name: str
            Name of  SkyModel for which to compute the npred for.
            If none, the sum of all components (minus the background model)
            is returned

        Returns
        -------
        response_sig, sum_sig: array, float
            array with the differential predicted counts and the total number of predicted counts inside the mask"""
        
        response = np.zeros(len(self.events_in_mask.table))
        total = 0.0
        
        evaluators = self.evaluators
        if model_name is not None:
            evaluators = {model_name: self.evaluators[model_name]}

        for evaluator in evaluators.values():
            if evaluator.needs_update:
                evaluator.update(
                    self.events_in_mask,
                    self.exposure,
                    self.psf,
                    self.edisp,
                    self.mask,
                    use_modelpos=True,
                    geom=self.geom,
                )

            if evaluator.contributes:
                r,s = evaluator.compute_npred()
                response[evaluator.event_mask] += r
                total += s

        return response, total

    def response_background(self):
        """
        compute the response of the background.
        returns: interpolated bkg value for all events, sum of bkg counts
        """
        #self._background_parameters_changed needs be copied from the MapDataset implementation
        if self._response_bkg_cached is not None and (self.background_model is None or not self._background_parameters_changed):
#             print('using bkg cache')
            return self._response_bkg_cached

        # case of bkg and extra model
        if self.background_model and self.background:
            norm_value = self.background_model.parameters.value[self._bkg_norm_idx]
            if norm_value == 0 or not np.isfinite(norm_value):
                # just return 0, don't update cache or cached parameters
                return (np.zeros(len(self.events_in_mask.table)), 0.0)
            elif self._background_parameter_norm_only_changed and self._response_bkg_cached is not None:
                self._response_bkg_cached[0] *= self.bkg_renorm()
                self._response_bkg_cached[1] *= self.bkg_renorm()
                if not np.isfinite(self._response_bkg_cached[1]):
                    print(f'renorm: {self.bkg_renorm()}, {self.background_model.parameters.value[self._bkg_norm_idx]}, {self._background_parameters_cached[self._bkg_norm_idx]}')
#                 print('simply renorming bkg')
                self._background_parameters_cached = self.background_model.parameters.value
                return self._response_bkg_cached
            elif self._background_parameters_changed or self._response_bkg_cached is None:
                values = self.background_model.evaluate_geom(geom=self.background.geom)
                bkg_map = self.background * values
                # interpolate and sum the bkg values
                bkg_sum = bkg_map.data[self.mask].sum()
                coords = self.events_in_mask.map_coord(self.background.geom)
                # we need to interpolate the differential bkg npred
                bkg_map.quantity /= bkg_map.geom.bin_volume()
                self._response_bkg_cached = [bkg_map.interp_by_coord(coords, method='linear'), bkg_sum]
#                 print('full calculation of bkg')
                self._background_parameters_cached = self.background_model.parameters.value
                if not np.isfinite(self._response_bkg_cached[1]):
                    print(f'full rec: {self.bkg_renorm()}, {self.background_model.parameters.value[self._bkg_norm_idx]}, {self._background_parameters_cached[self._bkg_norm_idx]}')
                return self._response_bkg_cached
    
        # case of bkg but no extra model        
        elif self.background:
            bkg_map = self.background.copy()
            # interpolate and sum the bkg values
            bkg_sum = bkg_map.data[self.mask].sum()
            coords = self.events_in_mask.map_coord(self.background.geom)
            # we need to interpolate the differential bkg npred
            bkg_map.quantity /= bkg_map.geom.bin_volume()
            self._response_bkg_cached = [bkg_map.interp_by_coord(coords, method='linear'), bkg_sum]
#             print('interpolating because of no bkg model')
            return self._response_bkg_cached
    
        # case of no bkg at all
        else: 
            if self.events_in_mask is not None: 
                return (np.zeros(len(self.events_in_mask.table)), 0.0)
            else: return (np.zeros(0),0)

    @property
    def _background_parameters_changed(self):
        values = self.background_model.parameters.value
        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._background_parameters_cached == values)
        return changed

    @property
    def _background_parameter_norm_only_changed(self):
        """Only norm parameter changed"""
        norm_only_changed = False
        idx = self._bkg_norm_idx
    
        values = self.background_model.parameters.value
        if idx is not None and self._background_parameters_cached is not None:
            if self._background_parameters_cached[idx] == 0:
                # then the cache can't be used
                return False
            changed = self._background_parameters_cached != values
            norm_only_changed = np.count_nonzero(changed) == 1 and changed[idx]

        return norm_only_changed

    def bkg_renorm(self):
        value = self.background_model.parameters.value[self._bkg_norm_idx]
        value_cached = self._background_parameters_cached[self._bkg_norm_idx]
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
        return idx

            
    def info_dict(self):
        """Info dict with summary statistics

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = {}
        info["name"] = self.name

        
        signal, background = np.nan, np.nan
        
        if self.events: 
             if(self.mask):
                    signal = self.response_signal()[1]
                    background = self.response_background()[1]
             info["events"] = len(self.events.table)
        else: info["events"] = None

        info["signal"]=float(signal)
        info["background"]=float(background)
             
        exposure_min = np.nan * u.Unit("cm s")
        exposure_max = np.nan * u.Unit("cm s")
        livetime = np.nan * u.s

        if self.exposure is not None:
            mask_exposure = self.exposure.data > 0

            if self.mask_safe is not None:
                mask_spatial = self.mask_safe.reduce_over_axes(func=np.logical_or).data
                mask_exposure = mask_exposure & mask_spatial[np.newaxis, :, :]

            if not mask_exposure.any():
                mask_exposure = slice(None)

            exposure_min = np.min(self.exposure.quantity[mask_exposure])
            exposure_max = np.max(self.exposure.quantity[mask_exposure])
            livetime = self.exposure.meta.get("livetime", np.nan * u.s).copy()

        info["exposure_min"] = exposure_min.item()
        info["exposure_max"] = exposure_max.item()
        info["livetime"] = livetime
        
        info["stat_type"] = self.stat_type
        
        stat_sum = np.nan
        if self.events is not None and self._models is not None and self.mask is not None:
            stat_sum = self.stat_sum()
        info["stat_sum"] = float(stat_sum)
   
        return info

    def to_mapdataset(self, name=None, geom=None):
        kwargs={}
        if name is None:
            kwargs['name']=self.name
        else:
            kwargs['name']=name
        if geom is None:
            geom = self.geoms['geom']
            
        counts = Map.from_geom(geom)
        counts.fill_events(self.events)
        kwargs['counts'] = counts
            
        for key in [
            "edisp",
            "psf",
            "mask_safe",
            "mask_fit",
            "exposure",
            "gti",
            "meta_table",
            "models",
        ]:
            kwargs[key] = getattr(self, key)
            
        kwargs['background'] = self.background
        
        return MapDataset(**kwargs)
        
        
        
        
        
        
        
        
        
        
        
        
        