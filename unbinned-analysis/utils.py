import numpy as np
from gammapy.irf import EnergyDispersion2D

def make_edisp_factors(edisp2d, events, energy_axis_true, pointing, model_pos):
    """Get energy dispersion for a given event and true energy axis.

        Parameters
        ----------
        edisp2d : `~gammapy.irf.EnergyDispersion2D`
            Energy Dispersion from the observaton
        events : `~gammapy.data.EventList`
            EventList with the relevant events (from the observaton)
        
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis on which the model will be evaluated
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinates.
        model_pos : `~astropy.coordinates.SkyCoord`
            Position (centre) of the model. Should be a single coordinates.

        Returns
        -------
        edisp : `~numpy.ndarray`
            the energy dispersion kernel for the unbinned evaluator. The shape is (Nevents,Nebins) with dP/dE_reco, the probablity for each true energy bin to reconstruct at the event's energy.
        """
    # use event offsets for larger models + good psf, true offset is closer to event offset
    offsets = events.radec.separation(pointing)[:,None] 
    # use model offset for small models or bad psf, true offset is closer to model offset
    offset = model_pos.separation(pointing)
    
    event_migra = events.energy[:,None]/energy_axis_true.center
    
    # interpolate the edisp to the true energy axis, possibly even model offset
    mm,ee,oo = np.meshgrid(edisp2d.axes['migra'].center,
                           energy_axis_true.center,
                           edisp2d.axes['offset'].center)
    dmigra = edisp2d.axes['migra'].bin_width[:,None]
    data = edisp2d.evaluate(offset=oo, energy_true=ee, migra=mm, method='linear') 
    edisp = EnergyDispersion2D(axes=[energy_axis_true,edisp2d.axes['migra'],edisp2d.axes['offset']],
                               data=data, interp_kwargs = edisp2d.interp_kwargs)
    edisp.normalize() # not sure if we want to normalize here, if we don't normalize all the interpolation could be done at once
#     edisp.data *= dmigra.value # multiply with dmigra to get P
    edisp.data /= energy_axis_true.center.value[:,None,None]
    values = edisp.evaluate(offset=offset,energy_true=energy_axis_true.center, migra=event_migra, method='linear')
    # cut by migra axis edges since we don't want extrapolation
    m_min, m_max = edisp2d.axes['migra'].edges[[0,-1]]
    mask = (event_migra < m_max) & (event_migra > m_min)
    return values*mask

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