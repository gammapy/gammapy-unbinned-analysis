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