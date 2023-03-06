import logging
import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.modeling.models import PointSpatialModel, TemplateNPredModel
from utils import *
# import warnings

PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg

log = logging.getLogger(__name__)


class UnbinnedEvaluator:
    """Sky model evaluation at events' coordinates.
    Evaluates a sky model's differential flux at the events's coordinates 
    and returns an array with an entry for each contributing event.
    Convolution with IRFs is done by interpolating the PDFs at each event's coordinates
    and multiplying this with the integrated model prediction.
    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Sky model
    events : `~gammapy.data.EventList`
            EventList with the events inside the mask
    mask : `~gammapy.maps.Map`
        Mask to apply to the likelihood for fitting.
    exposure : `~gammapy.maps.Map` 
        Exposure map
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    use_cache : bool
        Use caching of the previous response. 
    energy_axis : `~gammapy.maps.MapAxis`
        True energy axis for the model integration. If None the energy axis of the exposure is used.
    spatialbs : `~astropy.units.quantity.Quantity`
        The spatial binsize for the model evaluation. If None the binsize of the exposure will be up-/downsampled
        according to the model's `evaluation_bin_size_min`.
    """

    def __init__(
        self,
        model,
        events=None,
        mask = None,
        exposure= None,
        gti = None,
        evaluation_mode="local",
        use_cache=True,
        energy_axis=None,
        spatialbs=None
    ):

        self.model = model
        self.events = events
        self.mask = mask
        self.exposure= exposure
        self.gti = gti  # TODO: Check if we really need this
        self.use_cache = use_cache
        self._init_position = None
        self.contributes = True
        self.psf_containment = None
        self.energy_axis = energy_axis
        self.spatialbs = spatialbs
        self.geom = None
        self.irf_cube = None
        self._psf_width = 0.0 * u.deg
        
        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

        # TODO: this is preliminary solution until we have further unified the model handling
        if (
            isinstance(self.model, TemplateNPredModel)
            or self.model.spatial_model is None
            or self.model.evaluation_radius is None
        ):
            self.evaluation_mode = "global"

        # define cached computations
        self._cached_parameter_values = None
        self._cached_parameter_values_spatial = None
        self._cached_position = (0, 0)
        self._computation_cache = None
        self._neval = 0  # for debugging
        self._renorm = 1
        self._spatial_oversampling_factor = 1
        self.irf_unit = u.Unit('')
        if self.exposure is not None:
            if not self.geom.is_region or self.geom.region is not None:
                self.update_spatial_oversampling_factor(self.geom)

    def reset_cache_properties(self):
        """Reset cached properties."""
        del self._compute_npred

    # just use the exposure geometry because of edisp problems otherwise.
    def _init_geom(self, exposure):
        """True energy map geometry (`~gammapy.maps.Geom`) on which the model will be integrated"""
        
        geom = exposure.geom
        
        # cutout if neccessary
        if self.evaluation_mode == "local":
            self.contributes = self.model.contributes(mask=self.mask, margin=self.psf_width)
            if self.contributes:
                radius = self.model.evaluation_radius
                if radius is not None:
                    geom=geom.cutout(self.model.position, (radius+CUTOUT_MARGIN)*2)
        
        # adjust the spatial binsize if neccessary
        res_scale = self.spatialbs or self.model.evaluation_bin_size_min
#         res_scale = self.spatialbs if self.spatialbs is not None else self.model.evaluation_bin_size_min # avoids AstropyDeprecationWarning
        if res_scale is not None:
            pixel_size=np.max(geom.pixel_scales)
            if pixel_size > res_scale:
                geom = geom.upsample(int(np.ceil(pixel_size/res_scale)))
            elif pixel_size < res_scale/2:
                geom = geom.downsample(int(np.floor(res_scale/pixel_size)))
        
        # adjust the energy axis if given
        if self.energy_axis is not None:
            geom = geom.to_image().to_cube([self.energy_axis])
        self.geom = geom

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        # TODO: simplify and clean up
        if isinstance(self.model, TemplateNPredModel):
            return False
        elif not self.contributes:
            return False
        elif self.exposure is None:
            return True
        elif self.geom.is_region:
            return False
        elif self.evaluation_mode == "global" or self.model.evaluation_radius is None:
            return False
        elif not self.parameters_spatial_changed(reset=False):
            return False
        else:
            return self.irf_position_changed

    @property
    def psf_width(self):
        """Width of the PSF"""
        return self._psf_width

    def use_psf_containment(self, geom):
        """Use psf containment for point sources and circular regions"""
        if not geom.is_region:
            return False

        is_point_model = isinstance(self.model.spatial_model, PointSpatialModel)
        is_circle_region = isinstance(geom.region, CircleSkyRegion)
        return is_point_model & is_circle_region

    @property
    def cutout_width(self):
        """Cutout width for the model component"""
        return self.psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)

    def update(self, events, exposure, psf=None, edisp=None, mask=None, use_modelpos=False):
        """Update the integration geometry, the kernel cube and the acceptance cube of the EventEvaluator, 
        based on the current position of the model component.
        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        psf : `gammapy.irf.PSFMap`
            PSF map.
        edisp : `gammapy.irf.EDispMap`
            Edisp map.
        geom : `WcsGeom`
            Counts geom
        mask : `~gammapy.maps.Map`
            Mask to apply to the likelihood for fitting.
        use_modelpos : bool
            Wether or not to evaluate the IRFs at the model position or at each skycoord of the integration geom 
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")
        self.events = events
        self.mask = mask
        self.irf_unit = u.Unit('')
        self._cached_position = self.model.position_lonlat
        if self.evaluation_mode == "local":
            self.contributes = self.model.contributes(mask=mask, margin=self.psf_width)
        if self.contributes:
            # 1. get the contributing events which are close enough to the model
            del self.event_mask
            coords = events.map_coord(mask.geom)
            events = events.select_row_subset(self.event_mask)
            
            if isinstance(self.model, TemplateNPredModel):
                # the TemplateNpredModel only needs to be interpolated at 
                # the events' coordinates. No IRF cube necessary.
                return
            
            # init the proper integration geometry
            self._init_geom(exposure)
            self.exposure = exposure.interp_to_geom(self.geom)
            ### rely on float32 precision
            self.exposure.data = self.exposure.data.astype(np.float32)
            if use_modelpos == True:
                position = self.model.position
            else: position=None
            # get the edisp kernel factors for each event
            if edisp is not None:
                edisp_factors = make_edisp_factors(edisp, self.geom, events, position=position)
                self.irf_unit /= u.TeV
            else:
                edisp_factors = 1.0

            # get the psf kernel factors for each event
            if psf is not None and self.model.spatial_model:
                # TODO: Set a width according to a containment
                self._psf_width = psf.psf_map.geom.axes['rad'].edges.max() 
                psf_factors = make_psf_factors(psf, self.geom, events, position=position)
                self.irf_unit /= u.sr
            else:
                psf_factors = 1.0

            if len(self.geom.data_shape)+1 != len(edisp_factors.shape):
                if not isinstance(edisp_factors, float):
                    edisp_factors = np.expand_dims(edisp_factors, axis=(-1,-2))
            # maybe use sparse matrix
            self.irf_cube = psf_factors * edisp_factors
            ### rely on float32 precision
            self.irf_cube = self.irf_cube.astype(np.float32)
            
            if mask is not None:
                self.acceptance = make_acceptance(self.geom, mask, edisp, psf, self.model.position)
            else: self.acceptance = 1.0
            
        self.reset_cache_properties()
        self._computation_cache = None
    
    @lazyproperty
    def event_mask(self):
        """create a mask for events too far away from the model"""
        # the spatial part: separation from the model center
        separation = self.events.radec.separation(self.model.position)
        # TODO: Define an individual width for each event dependent on its energy
        mask_spatial = separation < self.cutout_width/2 
        
        # possibility for an energy or temporal mask
        
        return mask_spatial

    

    @lazyproperty
    def _compute_npred(self):
        """Compute npred"""
        if isinstance(self.model, TemplateNPredModel):
            npred = self.model.evaluate()
            # interpolate on the events
            coords = self.events.map_coord(self.mask.geom)
            response = npred.interp_by_coord(coords)
            total = np.sum(npred.data[self.mask.data])
            
        else:
            if not self.parameter_norm_only_changed:
                npred=self.model.integrate_geom(self.geom, self.gti) 
                ### rely on float32 precision
                npred.data = npred.data.astype(np.float32)
                npred *= self.exposure.quantity
                response = self.irf_cube * npred
                total = npred * self.acceptance.data
                axis_idx = np.arange(len(response.shape)) # the indices to sum over
                axis_idx=np.delete(axis_idx, 0) # dim 0 needs to be the event axis
                response = response.to_value(self.irf_unit).sum(axis=tuple(axis_idx))
                total = total.quantity.to_value('').sum() 
                self._computation_cache = [response, total]
                self._cached_parameter_values = self.model.parameters.value
            else:
                response, total = self._computation_cache
                response *= self.renorm()
                total *= self.renorm()
                self._computation_cache = [response, total]
                self._cached_parameter_values = self.model.parameters.value
        return [response, total]

    @property
    def apply_psf_after_edisp(self):
        return (
            self.psf is not None and "energy" in self.psf.psf_kernel_map.geom.axes.names
        )

    def compute_npred(self):
        """Evaluate model predicted counts.
        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        if self.model.parameters.value[self._norm_idx] == 0:
            # just return 0, don't update cache or cached parameters
            return [np.zeros(self.irf_cube.shape[0]), 0.0]

        if self.parameters_changed or not self.use_cache:
            del self._compute_npred

        return self._compute_npred

    @property
    def parameters_changed(self):
        """Parameters changed"""
        values = self.model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values == values)
        return changed

    @property
    def parameter_norm_only_changed(self):
        """Only norm parameter changed"""
        norm_only_changed = False
        idx = self._norm_idx
        values = self.model.parameters.value
        
        if idx is not None and self._computation_cache is not None:
            if self._cached_parameter_values[idx] == 0 or not self.use_cache:
                # then the cache can't be used
                return False
            changed = self._cached_parameter_values != values
            norm_only_changed = np.count_nonzero(changed) == 1 and changed[idx]
        return norm_only_changed

    def parameters_spatial_changed(self, reset=True):
        """Parameters changed
        Parameters
        ----------
        reset : bool
            Reset cached values
        Returns
        -------
        changed : bool
            Whether spatial parameters changed.
        """
        values = self.model.spatial_model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values_spatial == values)

        if changed and reset:
            self._cached_parameter_values_spatial = values

        return changed

    @property
    def irf_position_changed(self):
        """Position for IRF changed"""

        # Here we do not use SkyCoord.separation to improve performance
        # (it avoids equivalence comparisons for frame and units)
        lon_cached, lat_cached = self._cached_position
        lon, lat = self.model.position_lonlat

        separation = angular_separation(lon, lat, lon_cached, lat_cached)
        changed = separation > CUTOUT_MARGIN.to_value(u.rad)

#         if changed:
#             self._cached_position = lon, lat

        return changed

    @lazyproperty
    def _norm_idx(self):
        """norm index"""
        names = self.model.parameters.names
        ind = [idx for idx, name in enumerate(names) if name in ["norm", "amplitude"]]
        if len(ind) == 1:
            return ind[0]
        else:
            return None

    def renorm(self):
        value = self.model.parameters.value[self._norm_idx]
        if self._cached_parameter_values is None:
            return 1.0
        else:
            value_cached = self._cached_parameter_values[self._norm_idx]
            return value / value_cached