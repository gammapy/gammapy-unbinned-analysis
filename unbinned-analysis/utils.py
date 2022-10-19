def make_edisp_factors(edisp2d, events, energy_axis_true, pointing, model_pos):
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
    edisp.normalize() # not sure if we want to normalize here
#     edisp.data *= dmigra.value # multiply with dmigra to get P
    edisp.data /= energy_axis_true.center.value[:,None,None]
    values = edisp.evaluate(offset=offset,energy_true=energy_axis_true.center, migra=event_migra, method='linear')
    # cut by migra axis edges since we don't want extrapolation
    m_min, m_max = edisp2d.axes['migra'].edges[[0,-1]]
    mask = (event_migra < m_max) & (event_migra > m_min)
    return values*mask