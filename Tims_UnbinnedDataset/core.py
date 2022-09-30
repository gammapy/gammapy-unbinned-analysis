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
from utils import EdispInv
from gammapy.utils.integrate import trapz_loglog

PSF_CONTAINMENT = 0.999
CUTOUT_MARGIN = 0.1 * u.deg


class EventDataset():
    stat_type = "cash"
    tag = "EventDataset"
    
    def __init__(self, MapDataset, events, name='test-eds', tight_mask=False):
        """
        MapDataset: gammapy.datasets.MapDataset with the IRFs
        events_table: gammapy.data.EventList with the event data
        tight_mask: Flag if events in the edge pixels of the dataset's 
                    mask_safe which are beyond the bin center should be masked
        """
        self.ds = MapDataset.copy(name=MapDataset.name)
        self.ds.models = MapDataset.models.copy()
        self.events = events
        self.name = name
        self.meta_table=None
        self.tight_mask = tight_mask
        self._init_mask_safe()
        self._init_mask_linear()
        self._init_mask_nearest()
        
    @property
    def models(self):
        return self.ds.models
    
    @property
    def tab(self):
        return self.events.table#[self.mask_safe]
    
    def _init_mask_safe(self):
        coords = self.events.map_coord(self.ds.mask_safe.geom)
        if self.tight_mask:
            self.ds.mask_safe.data = self.ds.mask_safe.data.astype(int)
            mask_interp=self.ds.mask_safe.interp_by_coord(coords, method="linear")
            self.mask_safe = mask_interp > 0.999 # alow for some tolerance (==1 removes too many events)
        else:
            self.mask_safe = self.ds.mask_safe.get_by_coord(coords)==1
            
    def _init_mask_linear(self):
        coords = self.events.map_coord(self.ds.mask_safe.geom)
#         self.ds.mask_safe.data = self.ds.mask_safe.data.astype(int) # not needed, also bool interpolates to floats
        mask_interp=self.ds.mask_safe.interp_by_coord(coords, method="linear")
        self.mask_linear = mask_interp > 0.999 # alow for some tolerance (==1 removes too many events)
    def _init_mask_nearest(self):
        coords = self.events.map_coord(self.ds.mask_safe.geom)
        mask = self.ds.mask_safe.get_by_coord(coords)==1
        self.mask_nearest = np.logical_xor(mask,self.mask_linear)  # TODO calls mask_linear which can be avoided if moved to stat_sum()
    
#     def npred_sliced(self):
#         # for now just slice in energy
#         npred = self.ds.npred()
#         axis = npred.geom.axes['energy']
#         lower, upper = dataset.energy_range_safe
#         i,j = lower.data.shape
#         idx_min = axis.coord_to_idx(lower.data[i//2,j//2])
#         idx_max = axis.coord_to_idx(upper.data[i//2,j//2])
        
#         if idx_min == -1: idx_min = None
#         if idx_max == -1: idx_max = None
        
#         return npred.slice_by_idx({'energy':slice(idx_min,idx_max+1)})
        
    
    def integrate_npred(self, npred):
        """
        Compute integration of npred by 
        evaluating at the geometry edges and using trapz_loglog for energy axis
        It is something like a smoothing which intruces a bias but prevents super soft indices. And it takes long
        """
        edge_coords=npred.geom.get_coord(mode='edges', 
                                         axis_name=('lon','lat','energy'))        
        energy_edges = npred.geom.axes['energy'].edges
        mask = self.ds.mask_safe.data

        interp=npred.interp_by_coord(edge_coords)

        int_e = trapz_loglog(interp,
                             energy_edges[:,None,None], 
                             axis=0)

        #avg over all four edges for each pixel
        avg=(int_e[:,:-1,:-1] + int_e[:,1:,1:] + int_e[:,1:,:-1] + int_e[:,:-1,1:])/4
        avg/=np.diff(energy_edges)[:,None,None]

        return avg[mask].sum()
    
    
    def stat_sum(self, response_only=False):
        """
        Calculating the TS value for the unbinned dataset. 
        Essentially interpolating the npred cube at the events 
        positions (ra,dec,energy) and summing the log of those values.
        """
        npred = self.ds.npred()
        # need to mask the total npred so only regions contibute where events are
        mask = self.ds.mask_safe.data
        coords = self.events.map_coord(self.ds.mask_safe.geom)
        # set the event coords from table
        coords_linear = coords.apply_mask(self.mask_linear)
        response_linear = npred.interp_by_coord(coords_linear, method='linear')
        
        # if some interpolation values are <= 0 use nearest interpolation
        _mask_nearest = self.mask_nearest
        _mask_nearest[self.mask_linear][response_linear<=0] = True
        
        coords_nearest = coords.apply_mask(_mask_nearest)
        # interpolate the predicted counts
#         axis = npred.geom.axes['energy']
#         lower, upper = dataset.energy_range_safe
#         i,j = lower.data.shape
#         idx_min = axis.coord_to_idx(lower.data[i//2,j//2])[0]
#         idx_max = axis.coord_to_idx(upper.data[i//2,j//2])[0]
        
#         if idx_min == -1: idx_min = None
#         if idx_max == -1: idx_max = None      
#         npredS= npred.slice_by_idx({'energy':slice(idx_min,idx_max+1)})
    
        response_nearest = npred.interp_by_coord(coords_nearest, method='nearest')
        response = np.concatenate((response_linear[response_linear>0],response_nearest))   #[response_linear>0]
        if response_only:
            return np.log(response)
        # if npred = 0 at some events position the model is rouled out (TS=inf)
        if np.all(response>0):
            logL = np.sum(np.log(response)) - npred.data[mask].sum()
        else:
#             print(np.where(response==0))
            return np.inf
        return -2 * logL

#     def stat_sum(self):
#         """
#         Calculating the TS value for the unbinned dataset. 
#         Essentially interpolating the npred cube at the events 
#         positions (ra,dec,energy) and summing the log of those values.
#         """
#         npred = self.ds.npred()
#         # need to mask the total npred so only regions contibute where events are
#         mask = self.ds.mask_safe.data
#         npred.data[~mask] = -np.inf
#         coords = self.events.map_coord(self.ds.mask_safe.geom)
#         mask_ = self.ds.mask_safe.get_by_coord(coords).astype(bool)
#         coords_linear = self.events.map_coord(self.ds.mask_safe.geom).apply_mask(mask_)
#         response_linear = npred.interp_by_coord(coords_linear, method='linear')
        
#         # if some interpolation values are <= 0 use nearest interpolation
#         mask_nearest = response_linear <= 0
#         coords_nearest = coords_linear.apply_mask(mask_nearest)
    
#         response_nearest = npred.interp_by_coord(coords_nearest, method='nearest')
#         response = np.concatenate((response_linear[response_linear>0],response_nearest))
#         # if npred = 0 at some events position the model is rouled out (TS=inf)
#         if np.all(response>0):
#             logL = np.sum(np.log(response)) - npred.data[mask].sum()
#         else:
# #             print(np.where(response==0))
#             return np.inf
#         return -2 * logL

    def contribution_events(self):
        """
        compute the contribution of each event to the total likelihood
        """
        nevents = len(self.tab)
        npred = self.ds.npred()
        mask = self.ds.mask_safe.data
        npred_tot = npred.data[mask].sum()
        # set the event coords from table
        coords = MapCoord.create((self.tab["RA"].quantity.to('deg'),
                                 self.tab["DEC"].quantity.to('deg'),
                                 self.tab["ENERGY"].quantity.to('TeV')), 
                                 frame='icrs', axis_names=['energy'])
        # interpolate the predicted counts
        axis = npred.geom.axes['energy']
        lower, upper = dataset.energy_range_safe
        i,j = lower.data.shape
        idx_min = axis.coord_to_idx(lower.data[i//2,j//2])[0]
        idx_max = axis.coord_to_idx(upper.data[i//2,j//2])[0]
        
        if idx_min == -1: idx_min = None
        if idx_max == -1: idx_max = None      
        npredS= npred.slice_by_idx({'energy':slice(idx_min,idx_max+1)})
        response = npredS.interp_by_coord(coords, method='linear')
#         return response
        return (np.log(response) - npred_tot/nevents) * -2.
    
    def contribution_bins(self):
        """
        compute the contribution of each event in the binned likelihood
        """
        
        stat_array = self.ds.stat_array()
        counts = self.ds.counts
        contrib=[]
        indices=[]
        coordinates=[]
        for row in self.tab:
            c=MapCoord.create((row["RA"]*self.tab["RA"].unit,
                                 row["DEC"]*self.tab["DEC"].unit,
                                 row["ENERGY"]*self.tab["ENERGY"].unit), 
                                 frame='icrs', axis_names=['energy'])
            idx = counts.geom.coord_to_idx(c)
            nevents = counts.get_by_idx(idx)
            contrib.append(stat_array.T[idx]/nevents)
            indices.append(idx)
            coordinates.append(c)
        return np.array(contrib), np.array(indices), np.array(coordinates)



class EventDataset2():
    stat_type = "cash"
    tag = "DirectEventDataset"
    
    def __init__(self, MapDataset, observation, name='test-eds'):
        """
        MapDataset: gammapy.datasets.MapDataset for some convinience (true and reco geom need the same shape)
        observation: gammapy.data.Observation with the event data and IRFs
        name: str, a name for the dataset  
        """
        self.ds = MapDataset.copy(name=MapDataset.name)  # TODO: only use the relevant things from the ds to save memory
        self.ds.models = MapDataset.models.copy()
        self.events = observation.events
        self.obs = observation
        self.name = name
        self.meta_table=None
        self._evaluators=None
        self._background_cache=None
        
        self._init_mask_safe()
        self._init_acceptance()
        self._init_evaluators()
        
        
    @property
    def models(self):
        # use the dataset's models for now
        return self.ds.models
    
    def _init_evaluators(self):
        """Set UnbinnedEvaluator(s)"""
        self._evaluators = {}

        if self.models is not None:
            models = DatasetModels(self.models)
            models = models.select(datasets_names=self.ds.name)
            
            irfs={'psf':self.obs.psf, 'edisp':self.obs.edisp, 'exposure':self.ds.exposure}
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

        
        
    def _init_mask_safe(self):
        """
        create a mask for events which are not part of any evaluator
        and will be discarded in the analysis
        For now use the mask_safe of the datset
        """
        coords = self.events.map_coord(self.ds.mask_safe.geom)
        self.mask_safe = self.ds.mask_safe.get_by_coord(coords)==1
        
    def _init_acceptance(self):
        """
        Build the acceptance cube (dimeansions of true geom)
        Contains the fraction of each pixel that contributes to the sum of npred
        Strategy: Convolve the mask safe with the psf and edisp        
        """
        mask = Map.from_geom(geom=self.ds.mask_safe.geom.as_energy_true,
                            data=self.ds.mask_safe.data.astype(float))
        
        # compute the Kernels
        # for now use Kernels from the Map center, BUT acceptance is especially
        # important for mask_safe edges so use Kernel for larger offset???
        psf_kernel = self.ds.psf.get_psf_kernel(geom=self.ds.exposure.geom)
        if isinstance(self.ds.edisp, gammapy.irf.edisp.map.EDispMap):
            energy_axis = self.ds.mask_safe.geom.axes['energy']
            edisp_kernel = self.ds.edisp.get_edisp_kernel(energy_axis)
        else:
            edisp_kernel = self.ds.edisp.get_edisp_kernel()
        edisp_kernel.data = edisp_kernel.data.T  # the edisp is applied to reco energies
        mask=mask.apply_edisp(edisp_kernel)  # order of edisp and psf doesn't matter much
        mask_convolved=mask.convolve(psf_kernel) # but edisp does ereco --> etrue and psf is in etrue
#         mask_convolved=mask.convolve(psf_kernel)
#         mask_convolved=mask_convolved.apply_edisp(edisp_kernel)
        self.acceptance = mask_convolved
        
    def response_background(self):
        """
        compute the response of the background.
        returns: interpolated bkg value for all events, sum of bkg counts
        """
        
        if self._background_cache is not None and not self.ds._background_parameters_changed:
            return self._background_cache
        
        # get the background map, use the MapDataset caching
        background = self.ds.background
        if self.ds.background_model and background:
            if self.ds._background_parameters_changed:
                values = self.ds.background_model.evaluate_geom(geom=background.geom)
                if self.ds._background_cached is None:
                    self.ds._background_cached = background * values
                else:
                    self.ds._background_cached.quantity = (
                        background.quantity * values.value
                    )
            bkg_map = self.ds._background_cached
        else:
            bkg_map = background
       
        # interpolate and sum the bkg values
        events = self.events.select_row_subset(self.mask_safe)
        coords = events.map_coord(background.geom)
        bkg_sum = bkg_map.data[self.ds.mask_safe.data].sum()
        self._background_cache = bkg_map.interp_by_coord(coords, method='linear'), bkg_sum
        return self._background_cache
    
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

class UnbinnedEvaluator():
    def __init__(self, model, irfs, events, pointing, acceptance, max_radius=None):
        """
        model: SkyModel
        irfs: dict with "psf"--> PSF3D; "exposure"-->WcsNDMap with exosure data; "edisp"-->
        """
        self.model = model
        self.irfs = irfs
        self.events=events
        self.max_radius=max_radius
        self.pointing=pointing
        self.acceptance = acceptance.cutout(position=self.model.position,
                                           width=self.cutout_width, odd_npix=True).data
        self.irf_cube=None
        
    @property
    def psf_width(self, containment=PSF_CONTAINMENT):
        """Width of the PSF"""
        if "psf" in self.irfs.keys():
            #TODO: apply energy mask, so only use valid energies, not the whole range
            energy_axis = self.irfs['exposure'].geom.axes["energy_true"]
            offset = self.pointing.separation(self.model.position)

            radii = self.irfs["psf"].containment_radius(
                fraction=containment, offset=offset, energy_true=energy_axis.center
            )
            max_radius = np.max(radii)
        else:
            psf_width = 0 * u.deg
        return max_radius*2
    
    @property
    def cutout_width(self):
        """Cutout width for the model component"""
        return self.psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)
        
    def build_exposure(self):
        self.exposure = self.irfs['exposure'].cutout(
                    position=self.model.position, width=self.cutout_width, odd_npix=True
                )
        
    def init_mask(self):
        """
        Mask events far away from the model. This is an additional mask to the i.e. energy mask on events
        """
        geom_image=self.exposure.geom.to_image()
        coords = self.events.map_coord(geom_image)
        self.mask = geom_image.contains(coords)
    
    def build_psf(self):
        geom=self.exposure.geom
        psf3d=self.irfs['psf']
        psf3d.normalize()
        energy_axis_true = geom.axes["energy_true"]
        rad=geom.separation(self.events.radec[self.mask,None,None,None])
#         offsets=geom.separation(self.pointing)
        offsets = self.pointing.separation(self.model.position)
        psf_factors=psf3d.evaluate(energy_true=energy_axis_true.center[None,:,None,None], 
                                             rad=rad, offset=offsets)
        psf_factors /= psf_factors.sum(axis=(2,3), keepdims=True) # normalize
        return np.nan_to_num(psf_factors)
    
#         # point psf
#         coords = self.events.map_coord(geom.to_image()).apply_mask(self.mask)
#         idx_lon, idx_lat = geom.to_image().coord_to_idx(coords)
#         psf_factors = np.zeros(tuple([test.mask.sum()])+test.exposure.geom.data_shape)
#         for e,(i,j) in enumerate(zip(idx_lon,idx_lat)):
#             psf_factors[e,:,i,j]=1
#         return psf_factors
        
    def build_edisp(self):
        # maybe instead of EdispInv class, evaluate EnergyDispersion2D and renormalize similar to build_psf?
        event_e = self.events.energy[self.mask]
        offsets = self.events.radec[self.mask].separation(self.pointing) # maybe this needs to be one offset for each pixel
        energy_true = self.exposure.geom.axes['energy_true']
        # based on inverse Edisp
#         edisp = EdispInv(self.irfs['edisp'], event_e.min(), event_e.max())
#         edisp_factors = edisp.evaluate(offset = offsets[:,None], energy=event_e[:,None], energy_true=energy_true.center[None,:])

        edisp2d=self.irfs['edisp']
        edisp_factors=np.zeros((len(event_e),len(energy_true.center)))
        for i,(offset,e) in enumerate(zip(offsets,event_e)):
            kernel=edisp2d.to_edisp_kernel(offset=0.1*u.deg, energy_true=energy_true.edges)
            edisp_factors[i]=kernel.evaluate(energy=e).flatten()
        return edisp_factors
    
#         # diagonla edisp
#         factors = np.zeros((len(event_e),len(energy_true.center)))
#         for i,e in enumerate(event_e):
#             dist = np.abs(energy_true - e)
#             factors[i,dist.argmin()] = 1
#         return factors

    
    
    def init_irf_cube(self):
        """
        compute the static cube representing the IRFs
        with dimensions (Nevents, energy, lon, lat)
        For each event and model pixel: IRF corrected exposure that contributes 
        to that event
        """
        self.build_exposure()
        self.init_mask()
        psf = self.build_psf()
        edisp = self.build_edisp()
        self.irf_cube = self.exposure.quantity* psf * edisp[:,:,None,None]
        
    def compute_response(self):
        """
        compute the npred values at the event coordinates 
        and the sum of counts from the model
        
        Returns
        -------
        response: `np.array` (shape n)
            npred values at the event coordinates (only contribution events)
        mask: `np.array` (shape N with n true entries)
            the mask to map the response to the total reponse
        npred_sum: `float`
            summed npred of the model
        """
        if self.irf_cube is None:
            self.init_irf_cube()
        npred=self.model.integrate_geom(self.exposure.geom)
        response = np.multiply(npred, self.irf_cube).sum(axis=(1,2,3))
        npred_sum = npred * self.exposure.quantity * self.acceptance
        return response.to_value(''), self.mask, npred_sum.quantity.to_value('').sum()  # maybe sum quantity.to_value for npred_sum