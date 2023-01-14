import numpy as np
from gammapy.irf import EnergyDispersion2D, EDispMap, EDispKernelMap

def make_edisp_factors(edisp, geom, events, position=None, pointing=None):
    """Calculate the energy dispersion factors for the events.

        Parameters
        ----------
        edisp : The energy dispersion. Supported classes are:
            `~gammapy.irf.EDsipMap` (prefered)
            `~gammapy.irf.EDispKernelMap` (less precise)
            `~gammapy.irf.EnergyDispersion2D` (needs position and pointing)
        geom : `~gammapy.maps.WcsGeom`
            The true geometry for the numeric integration
        events : `~gammapy.data.EventList`
            EventList with the relevant events
        position : `~astropy.coordinates.SkyCoord`
            Position (centre) of the model at which the EDisp is evaluated. 
            Should be a single coordinate. If None the skycoords of the geom are used.
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinate.
            It needs to be give in case of the EnergyDispersion2D        

        Returns
        -------
        edisp : `~astropy.units.quantity.Quantity`
            The energy dispersion kernels for the events. 
            The shape is (Nevents,Nebins,(lon,lat)) with dP/dE_reco, 
            the differential probablity for each true energy bin 
            to reconstruct at the event's energy.
        """
    e_axis_true = geom.axes['energy_true']
    if position is None:
        # interpolate on the true coordinates of geom
        coords = {'skycoord': geom.get_coord(sparse=True).skycoord.squeeze()[None,None,...]}
        expand_outdim = True
        coords['energy_true'] = e_axis_true.center[:,None,None]
        coords['energy'] = events.energy[:,None,None,None] # event,e_true,lon,lat
    else:
        # interpolate on the position
        coords = {'skycoord': position}
        expand_outdim = False
        coords['energy_true'] = e_axis_true.center
        coords['energy'] = events.energy[:,None] # event,e_true
        
    if isinstance(edisp, EDispMap):
        # projected but with migra axis
        coords['migra'] = coords['energy']/coords['energy_true']
        factors = edisp.edisp_map.interp_by_coord(coords, fill_value=0.)
        factors = factors / coords['energy_true']
    
    elif isinstance(edisp, EDispKernelMap):
        # projected already with energy axis
        ## TODO: Print warning that KernelMap is not precise
        e_reco_bins = edisp.edisp_map.geom.axes['energy'].edges.diff()
        if expand_outdim: e_reco_bins = e_reco_bins[:,None,None]
        factors = edisp.edisp_map.interp_by_coord(coords, fill_value=0.)
        factors = factors / e_reco_bins
        
    elif isinstance(edisp, EnergyDispersion2D):
        coords['offset'] = coords['skycoord'].separation(pointing)
        coords['migra'] = coords['energy']/coords['energy_true']
        del coords['energy']
        del coords['skycoord']
        factors = edisp.evaluate(method='linear', **coords)
        factors = factors / coords['energy_true']
        m_min, m_max = edisp.axes['migra'].edges[[0,-1]]
        mask = (coords['migra'] < m_max) & (coords['migra'] > m_min)
        factors *= mask
        
    else:
        raise ValueError("No valid edisp class. \
Need one of the following: EdispMap, EdispKernelMap, EnergyDispersion2D")
    return factors

def make_exposure_factors(livetime, aeff, pointing, coords):
    """Get energy dispersion for a given event and true energy axis.

        Parameters
        ----------
        livetime : `~astropy.units.quantity.Quantity`
            livetime of the observation    
        aeff : `~gammapy.irf.effective_area.EffectiveAreaTable2D`
            effective area from the observaton
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinates.
        coords : `~gammapy.maps.coord.MapCoord`
            coordinates on which the model will be evaluated. Needs true energy axis and skycoord.

        Returns
        -------
        exposure : `~numpy.ndarray`
            the exposure values for the unbinned evaluator.
        """
    offsets = coords.skycoord.separation(pointing)
    exposure = aeff.evaluate(offset=offsets, energy_true=coords["energy_true"])
    return (exposure * livetime).to("m2 s")

def make_acceptance(mask_safe, edisp, psf, model_pos, pointing, geom_model):
    """Compute the acceptance cube with dimensions of `geom_model`. The cube can be multiplied to the model which is integrated on `geom_model` and the sum over the result will give the total number of model counts inside the `mask_safe` which is in reco coordinates.

        Parameters
        ----------
        mask_safe : `~gammapy.maps.WcsNDMap`
            analysis mask in reco coordinates    
        edisp : `~gammapy.irf.EnergyDispersion2D`
            Energy Dispersion from the observaton
        psf : `~gammapy.irf.PSF3D`
            Point spread function from the observaton
        model_pos : `~astropy.coordinates.SkyCoord`
            Ceter position of the model. In the future this can be extracted from the geom_model
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinates.
        geom_model : `~gammapy.maps.WcsGeom`
            Geometry with coordinates on which the model will be evaluated. Needs true energy axis and skycoord.

        Returns
        -------
        acceptance : `~gammapy.maps.WcsNDMap`
            the acceptance map for the unbinned evaluator.
    """
#     It will be model specific. 
#     offset = offset of model_pos and pointing (same as is used for binned)
#     geom_model = geom on which the model is evaluated, maybe we want to pass coords instead
#     a new mask needs to be interpolated from the mask safe on the geom
    offset = model_pos.separation(pointing)
    ## get the new mask_safe based on the geometry on which the model is evaluated
    geom_model_mask = geom_model.to_image().to_cube([mask_safe.geom.axes['energy']])
    coords_model = geom_model_mask.get_coord()  # or take the coords directly if coords are passed
    # the mask_safe in model geometry
    mask_model = Map.from_geom(geom=geom_model_mask.as_energy_true, 
                               data=mask_safe.interp_by_coord(coords_model))
    
    ## create edisp 
    # because we want to convolve a reco map we do a trick and create a transposed EdispKernel
    edisp_kernel = edisp.to_edisp_kernel(offset, 
                                         energy_true=geom_model.axes['energy_true'].edges , 
                                         energy=mask_safe.geom.axes['energy'].edges)
    # change the axes
    ax_reco = geom_model.axes['energy_true'].copy()
    ax_true = mask_safe.geom.axes['energy'].copy()
    ax_reco._name="energy"
    ax_true._name='energy_true'
    edisp_kernelT = EDispKernel(axes=[ax_true, ax_reco], data = edisp_kernel.data.T)
    
    ## create the psf
    # first a PSFMap with 3x3 pixels around the model_position
    geom_psf = mask_safe.geom.to_image().cutout(model_pos, mask_safe.geom.pixel_scales*3)
    geom_psf = geom_psf.to_cube([psf.axes['rad'], geom_model.axes['energy_true']])
    psfmap = make_psf_map(psf, pointing, geom_psf)
    # now get the kernel at the model position
    psf_kernel = psfmap.get_psf_kernel(geom_model, position=model_pos)

    acceptance=mask_model.apply_edisp(edisp_kernelT)  # order of edisp and psf doesn't matter much
    return acceptance.convolve(psf_kernel) # but edisp does ereco --> etrue and psf is in etrue