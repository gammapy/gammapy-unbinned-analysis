import numpy as np
from gammapy.irf import EnergyDispersion2D, EDispMap, EDispKernelMap, EDispKernel, PSFMap, PSF3D, PSFKernel 
from gammapy.maps import Map
from astropy.nddata import block_reduce
from gammapy.maps import RegionGeom

def make_edisp_factors(edisp, geom, events, position=None, pointing=None, dtype=np.float64):
    """Calculate the energy dispersion factors for the events.

        Parameters
        ----------
        edisp : The energy dispersion. Supported classes are:
            `~gammapy.irf.EDsipMap` (prefered)
            `~gammapy.irf.EDispKernelMap` (less precise)
            `~gammapy.irf.EnergyDispersion2D` (not yet projected, needs pointing)
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
        edisp = edisp.copy() # so we don't change the original data 
        edisp.edisp_map.quantity = edisp.edisp_map.quantity/ e_reco_bins[:,None,None]
        factors = edisp.edisp_map.interp_by_coord(coords, fill_value=0.)
        unit = edisp.edisp_map.unit  # for some reason the interpolated values have no units
        factors *= unit
        
    elif isinstance(edisp, EnergyDispersion2D):
        coords['offset'] = coords['skycoord'].separation(pointing)
        coords['migra'] = coords['energy']/coords['energy_true']
        del coords['energy']
        del coords['skycoord']
        factors = edisp.evaluate(method='linear', **coords)
        factors = factors / coords['energy_true']
        # do a cut on migra min/max to avoid extrapolation on this axis
        m_min, m_max = edisp.axes['migra'].edges[[0,-1]]
        mask = (coords['migra'] < m_max) & (coords['migra'] > m_min)
        factors *= mask
        
    else:
        raise ValueError("No valid edisp class. \
Need one of the following: EdispMap, EdispKernelMap, EnergyDispersion2D")
    return factors.astype(dtype)

def make_psf_factors(psf, geom, events, position=None, pointing=None, dtype=np.float64, factor=4, min_split=10):
    """Calculate the energy dispersion factors for the events.

        Parameters
        ----------
        psf : The Point Spead Function. Supported classes are:
            `~gammapy.irf.PSFMap` (projected)
            `~gammapy.irf.PSF3D` (not yet projected, needs pointing)
        geom : `~gammapy.maps.WcsGeom`
            The true geometry for the numeric integration
        events : `~gammapy.data.EventList`
            EventList with the relevant events
        position : `~astropy.coordinates.SkyCoord`
            Position (centre) of the model at which the psf is evaluated. 
            Should be a single coordinate. If None the skycoords of the geom are used.
        pointing : `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinate.
            It needs to be give in case of the EnergyDispersion2D 
        dtype : data type of the output array
        factor : int
            oversampling factor to compute the PSF
        min_split: int
            minimum number of events for the splitting

        Returns
        -------
        psf : `~astropy.units.quantity.Quantity`
            The PSF kernels for the events. 
            The shape is (Nevents,Nebins,lon,lat) with dP/dOmega, 
            the differential probablity for each true pixel 
            to reconstruct at the event's position.
        """
   
    e_axis_true = geom.axes['energy_true']
    geom = geom.upsample(factor)
    geom_radec = geom.get_coord(sparse=True).skycoord.squeeze()
    block_size = (1,) * (1+len(geom.axes)) + (factor, factor)
    coords = {'skycoord': position or geom_radec[None,None,...]}
    ### split the events to lower peak memory
    n_events = len(events.table)
    n_split = min(factor**2+1, int(n_events/min_split))
    splits=np.linspace(0,events.radec.shape[0],factor**2+1,dtype=int)[1:-1]
    event_radec_list = np.split(events.radec, splits)
    factor_list=[]
    for event_radec in event_radec_list:
        # to avoid errors in block_reduce:
        if len(event_radec)==0:
            continue
        coords['rad'] = event_radec[:,None,None,None].separation(geom_radec)
        # shape of (event,e_true,lon,lat)

        if isinstance(psf, PSFMap):
            if 'energy' in psf.psf_map.geom.axes_names:
                coords['energy'] = events.energy[:,None,None,None]
            if 'energy_true' in psf.psf_map.geom.axes_names:
                coords['energy_true'] = e_axis_true.center[:,None,None]

            factors = psf.psf_map.interp_by_coord(coords, fill_value=0.)
            unit=psf.psf_map.unit  # for some reason the interpolated values have no units

        elif isinstance(psf, PSF3D):
            if 'energy' in psf.axes.names:
                coords['energy'] = events.energy[:,None,None,None]
            if 'energy_true' in psf.axes.names:
                coords['energy_true'] = e_axis_true.center[:,None,None]
            if 'offset' not in coords.keys():
                coords['offset'] = coords['skycoord'].separation(pointing)
                del coords['skycoord']
            factors = psf.evaluate(method='linear', **coords)
            unit=factors.unit
            factors = factors.value

        else:
            raise ValueError("No valid psf class. \
    Need one of the following: PSFMap, PSF3D")
        factor_list.append(block_reduce(factors, tuple(block_size), func=np.nanmean))
    return np.vstack(factor_list).astype(dtype)*unit

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

def make_acceptance(geom, mask, edisp, psf, model_pos, 
                    pointing=None, max_radius=None, containment=0.999, factor=4, dtype=np.float64, debug=0):
    """Compute the acceptance cube with dimensions of `geom`. 
    The cube can be multiplied to the model which is integrated on `geom` 
    and the sum over the result will give the total number of model counts inside 
    the `mask_safe` which is in reco coordinates.

        Parameters
        ----------
        geom: `~gammapy.maps.WcsGeom`
            Geometry with coordinates on which the model will be evaluated. Needs true energy axis and skycoord.
        mask: `~gammapy.maps.WcsNDMap`
            analysis mask in reco coordinates    
        edisp: `~gammapy.irf.EnergyDispersion2D` or `~gammapy.irf.EDispMap` or `EDispKernelMap`
            Energy Dispersion of the observaton
        psf: `~gammapy.irf.PSF3D` or `~gammapy.irf.PSFMap`
            Point spread function of the observaton
        model_pos: `~astropy.coordinates.SkyCoord`
            Ceter position of the model. (In the future this can be extracted from the geom_model)
        pointing: `~astropy.coordinates.SkyCoord`
            Pointing position of the observation. Should be a single coordinates. Only needed if 
            the IRFs are not projected on the sky
        max_radius : `~astropy.coordinates.Angle`
            maximum angular size of the PSF kernel map
        containment : float
            Containment fraction to use as size of the PSF kernel. The max. radius
            across all energies is used. The radius can be overwritten using
            the `max_radius` argument.
        factor : int
            oversampling factor to compute the PSF
        
        Returns
        -------
        acceptance : `~gammapy.maps.WcsNDMap`
            the acceptance map for the unbinned evaluator.
    """
    ## get the energy axes and copy them for the transposed EDispKernel
    e_reco = mask.geom.axes['energy']
    e_true = geom.axes["energy_true"]
    fake_reco = e_true.copy(name='energy')
    fake_true = e_reco.copy(name='energy_true')
    geom_model_mask = geom.to_image().to_cube([e_reco])
    
    if not isinstance(geom, RegionGeom):
        ## the PSF needs a larger map for evaluation
        if isinstance(psf, (PSFMap, PSF3D)):
            width_geom = geom.width.max()
            if max_radius is None:
                kwargs = {
                    "fraction": containment,
                    "energy_true": e_true.center,
                }
                if isinstance(psf, PSFMap):
                    kwargs['position'] = model_pos
                elif isinstance(psf, PSF3D):
                    kwargs['offset'] = model_pos.separation(pointing)
                radii = psf.containment_radius(**kwargs)
                max_radius = np.max(radii)
            geom_model_mask=geom_model_mask.to_odd_npix(max_radius + width_geom/2)

        ## interpolate the mask to the integration geometry on which the back folding takes place    
        coords = geom_model_mask.get_coord()  
        # the mask_safe in model geometry but with axis energy label energy_true
        mask_model = Map.from_geom(geom=geom_model_mask.as_energy_true, 
                                   data=mask.interp_by_coord(coords))
    
    ## create edisp 
    # because we want to convolve a reco map we do a trick and create a transposed EdispKernel
    if isinstance(edisp, EDispMap):
        # we need to adapt to possibly different e_true axis
        # so we we interpolate to different e_true before generating the kernel
        migra = e_reco.edges[:,None]/e_true.center
        coords = {
            "skycoord": model_pos,
            "energy_true": e_true.center,
            "migra": migra,
        }
        values = edisp.edisp_map.integral(axis_name="migra", coords=coords)

        axis = 0 # diff alon the migra (reco) axis which is at dim0 for coord[skycoord] beeing a single position
        data = np.clip(np.diff(values.to_value(''), axis=axis), 0, np.inf)
        edisp_kernelT = EDispKernel(axes=[fake_true, fake_reco], data = data)
        
    elif isinstance(edisp, EDispKernelMap):
        # the reco energy axis should match, but this should be the case if the mask is in reco coords of the dataset
        # different axis probably need new normalizationand this is not tested
        assert e_reco == edisp.edisp_map.geom.axes["energy"]
        coords = {
            "skycoord": model_pos,
            "energy_true": e_true.center,
            "energy": e_reco.center[:,None],
        }
        data = edisp.edisp_map.interp_by_coord(coords)  # interpolate to new e_true axis
        edisp_kernelT = EDispKernel(axes=[fake_true, fake_reco], data = data)
     
    elif isinstance(edisp, EnergyDispersion2D):
        offset = model_pos.separation(pointing)
        edisp_kernel = edisp.to_edisp_kernel(offset, 
                                             energy_true=e_true.edges , 
                                             energy=e_reco.edges)
        
        edisp_kernelT = EDispKernel(axes=[fake_true, fake_reco], data = edisp_kernel.data.T)
        
    else: edisp_kernelT = None
  
    ## create the psf
    if isinstance(psf, PSFMap):
        model_pos = psf._get_nearest_valid_position(model_pos)
        if isinstance(geom, RegionGeom):
            geom_psf = mask.geom.to_image().to_cube([e_true])
        
        else:
            # need to interpolate to the e_true of the geom
            geom_psf = geom.to_odd_npix(max_radius=max_radius)
        geom_upsampled = geom_psf.upsample(factor=factor)
        coords = geom_upsampled.get_coord(sparse=True)
        rad = coords.skycoord.separation(geom.center_skydir)

        coords = {
        "energy_true": coords["energy_true"],
        "rad": rad,
        "skycoord": model_pos,
        }

        data = psf.psf_map.interp_by_coord(
        coords=coords,
        method="linear",
        fill_value=0
        )

        kernel_map = Map.from_geom(geom=geom_upsampled, data=np.clip(data, 0, np.inf))
        kernel_map = kernel_map.downsample(factor, preserve_counts=True)
        psf_kernel = PSFKernel(kernel_map, normalize=True)
        
    elif isinstance(psf, PSF3D):
        # first a PSFMap with 3x3 pixels around the model_position
        geom_psf = mask.geom.to_image().cutout(model_pos, mask.geom.pixel_scales*3)
        geom_psf = geom_psf.to_cube([psf.axes['rad'], geom.axes['energy_true']])
        psfmap = make_psf_map(psf, pointing, geom_psf)
        # now get the kernel at the model position
        psf_kernel = psfmap.get_psf_kernel(geom, position=model_pos)
        
    else: psf_kernel = None
    
    if isinstance(geom, RegionGeom):
        mask2 = mask.copy()
        mask2._geom = mask.geom.as_energy_true
        acc_cube = psf_kernel.psf_kernel_map.quantity * mask2.apply_edisp(edisp_kernelT).data
        acceptance = acc_cube.sum(axis=(1,2), keepdims=True)
        if debug==1:
            return mask2.apply_edisp(edisp_kernelT)
        if debug==2:
            return psf_kernel
        
    else:
        acceptance=mask_model.apply_edisp(edisp_kernelT)  # First apply the edisp to go from reco energy to true energy
        if debug==1:
            return acceptance
        if debug==2:
            return psf_kernel
        if psf_kernel:
            acceptance=acceptance.convolve(psf_kernel)
            acceptance=acceptance.cutout(geom.center_skydir, width_geom)
        acceptance.data = acceptance.data.astype(dtype)
    return acceptance