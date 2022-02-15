#!/usr/bin/python3.8

import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from regions import CircleSkyRegion
from gammapy.data import GTI, EventList
from gammapy.irf import EDispKernelMap, EDispMap, PSFKernel, PSFMap
from gammapy.maps import Map, MapAxis
from gammapy.datasets import MapDataset
from gammapy.datasets.map import MapEvaluator
from gammapy.datasets.map import get_axes
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel
from gammapy.utils.fits import HDULocation, LazyFitsData
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.table import hstack_columns


__all__ = ["UnbinnedDataset"]


def unit_array_to_string(a):
    a_str = "["
    len_arr = len(a)
    if len_arr >6:
        for i in [0,1,2]:
            ia      = a[i] 
            a_str  += '%.2f, '    % ia.value
        a_str      += "..., "
        for i in [-3,-2,-1]:
            ia      = a[i]
            if i == -1:
                a_str += '%.2f'   % ia.value
            else:
                a_str += '%.2f, ' % ia.value
    else:
        for i in range(len_arr):
            ia = a[i] 
            if i == len_arr-1:
                a_str += '%.2f'   % ia.value
            else:
                a_str += '%.2f, ' % ia.value

    a_str += "] "+str(a.unit)
    return a_str


def random_energies(generator, n_events,random_state="random-seed"):
     
    names    = generator.geom.axes_names
    matching = [s for s in names if "energy" in s][0]
    unit     = generator.geom.axes[matching].unit
    
    if n_events > 0:
        en_array = generator.sample_coord(n_events=n_events, 
                          random_state=random_state)[matching] 
    else:
        en_array = u.Quantity([]) * u.TeV
    
    return en_array


class UnbinnedDataset(MapDataset):
    """Perform unbinned model likelihood fit on maps.

    Parameters
    ----------
    events : `~gammapy.data.EventList`
        Event list
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Exposure cube
    background : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Background cube
    mask_fit : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFMap` or `~gammapy.utils.fits.HDULocation`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel` or `~gammapy.irf.EDispMap` or `~gammapy.utils.fits.HDULocation`
        Energy dispersion kernel
    mask_safe : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask defining the safe data range.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing information on observations used to create the dataset.
        One line per observation for stacked datasets.


    See Also
    --------
    MapDataset, SpectrumDataset, FluxPointsDataset
    """

    stat_type = "unbinned"
    tag = "UnbinnedDataset"

    def __init__(
        self,
        events=None,
        models=None,
        counts=None,
        exposure=None,
        background=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        gti=None,
        meta_table=None,
        name=None,
    ):
        super().__init__(models,
            counts,
            exposure,
            background,
            psf,
            edisp,
            mask_safe,
            mask_fit,
            gti,
            meta_table,
            name,
            )
        
        self.events = events    
    
    def update_flux(self):
        evaluator                 = list( self.evaluators.values() )[0]
        if hasattr(self, 'geoms'): # for gammapy >= 0.19
            evaluator.update(self.exposure, self.psf, self.edisp, self.geoms['geom'], 
                         self.mask)
        # SIGNAL FLUX
        self.signal_flux_true     = evaluator.apply_exposure(evaluator.compute_flux())
        self.signal_flux_obs      = evaluator.apply_edisp(self.signal_flux_true)
        # BKG FLUX
        if hasattr(self.npred_background(), 'data'):
            self.bkg_flux         = self.npred_background()
        else: # WE SET THE BKG TO ZERO IF THE BKG IS NOT PROVIDED
            self.bkg_flux         = self.signal_flux_obs.copy()
            self.bkg_flux.data    = np.zeros_like(self.signal_flux_obs.data) 
        # TOTAL PRED. SIGNAL COUNTS
        self.tot_signal_true      = np.sum(self.signal_flux_true.data)
        self.tot_signal_obs       = np.sum(self.signal_flux_obs.data)
        # TOTAL PRED. BKG COUNTS
        self.tot_bkg              = np.sum(self.bkg_flux.data)
   
            
    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")
        str_ += "\t{:32}: {{events}} \n".format("Event list")
        str_ += "\t{:32}: {{counts:.0f}} \n".format("Total counts")
        str_ += "\t{:32}: {{background:.2f}}\n".format("Total background counts")
        str_ += "\t{:32}: {{excess:.2f}}\n\n".format("Total excess counts")

        str_ += "\t{:32}: {{npred:.2f}}\n".format("Predicted counts")
        str_ += "\t{:32}: {{npred_background:.2f}}\n".format(
            "Predicted background counts"
        )
        str_ += "\t{:32}: {{npred_signal:.2f}}\n\n".format("Predicted excess counts")

        str_ += "\t{:32}: {{exposure_min:.2e}}\n".format("Exposure min")
        str_ += "\t{:32}: {{exposure_max:.2e}}\n\n".format("Exposure max")

        str_ += "\t{:32}: {{n_bins}} \n".format("Number of total bins")
        str_ += "\t{:32}: {{n_fit_bins}} \n\n".format("Number of fit bins")

        # likelihood section
        str_ += "\t{:32}: {{stat_type}}\n".format("Fit statistic type")
        str_ += "\t{:32}: {{stat_sum:.2f}}\n\n".format(
            "Fit statistic value (-2 log(L))"
        )

        info = self.info_dict()
        if self.events is None:
            info["events"] = self.events
        else:
            energies = self.events.energy
            info["events"] = unit_array_to_string(energies)
        
        str_ = str_.format(**info)
        

        # model section
        n_models, n_pars, n_free_pars = 0, 0, 0
        if self.models is not None:
            n_models = len(self.models)
            n_pars = len(self.models.parameters)
            n_free_pars = len(self.models.parameters.free_parameters)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)
        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

        
        
    def get_coords(self,signal=True,bkg=True,obs=True):
        """Return the coordinates (edges, bin center, ..) 
            of the binned predicted differential counts

        Parameters
        ----------
        obs          : Bool
                True if dN/dE in observed  energy,
                i.e. by incluidng edisp effects
                By default is True
        signal        : Bool
                True if you want to include in the
                simulation the signal of gamma
                By default is True
        bkg          : Bool
                True if you want to include in the
                simulation the background
                By default is True

        Returns
        -------
        tuple of `~numpy.narray`
            Edges of the energy bins,
            Center of the energy bins,
            Width of the energy bins ,
            The corresponding dN/dE
        """
        if bkg and not obs:
            raise ValueError("If bkg is True then also obs must be True!")
        if not signal and not bkg:
            raise ValueError("Error, neither the bkg nor the signal is True!")

        
        # Define a counts array of zeros
        if obs or bkg:
            axes          =  self.signal_flux_obs.geom.axes
            energy_axe    =  axes['energy']
            counts        =  np.zeros_like(self.signal_flux_obs.data)
        else:
            axes          =  self.signal_flux_true.geom.axes
            energy_axe    =  axes['energy_true']
            counts        =  np.zeros_like(self.signal_flux_true.data)
        while len(counts.shape) > 1:
            counts      = counts.sum(axis=1)
        
        # Get the signal counts
        if obs:
            signal_counts =  self.signal_flux_obs.data
        else:
            signal_counts =  self.signal_flux_true.data
        while len(signal_counts.shape) > 1:
            signal_counts      = signal_counts.sum(axis=1)
            
        # bkg counts
        bkg_counts        =  self.bkg_flux.data
        while len(bkg_counts.shape) > 1:
            bkg_counts      = bkg_counts.sum(axis=1)
        
        if bkg:
            counts       += bkg_counts
        if signal:
            counts       += signal_counts

        bin_centers      = np.array(energy_axe.center)
        bin_width        = np.array(energy_axe.bin_width)
        bin_edges        = np.array(energy_axe.edges)
        
        return bin_edges,bin_centers, bin_width, counts
    

    def predicted_dnde(self,energy,obs=True,bkg=True,signal=True):
        """Return the predicted differential counts [1/TeV]
           for the energy given in input

        Parameters
        ----------
        energy       : Quantity
                must have energy dimension
        obs          : Bool
                True if dN/dE in observed  energy,
                i.e. by incluidng edisp effects
                By default is True
        signal        : Bool
                True if you want to include in the
                simulation the signal of gamma
                By default is True
        bkg          : Bool
                True if you want to include in the
                simulation the background
                By default is True

        """
        if bkg and not obs:
            raise ValueError("If bkg is True then also obs must be True!")
        if not signal and not bkg:
            raise ValueError("Error, neither the bkg nor the signal is True!")
            
        energy       = u.Quantity([energy]).flatten()
        energy       = energy.to(u.TeV)
        bin_edges,bin_centers, bin_width, counts = self.get_coords(obs=obs,bkg=bkg, signal=signal)
        xp           = bin_centers
        fp           = counts/bin_width
        cond_left    = (energy >= bin_edges[0]*u.TeV) * (energy <= bin_centers[0]*u.TeV)
        cond_right   = (energy <= bin_edges[-1]*u.TeV) * (energy >= bin_centers[-1]*u.TeV)
        energy[cond_left]  = bin_centers[0]*u.TeV
        energy[cond_right] = bin_centers[-1]*u.TeV
        dnde         = np.interp(energy.value, xp, fp, left=0, right=0)
        return dnde/ u.TeV

    def plot_predicted_dnde(self, ax=None, fig=None, obs=True,bkg=False, signal=True, line=True,**kwargs):
        """Plot the predicted differential counts dN/dE * E**2

        Parameters
        ----------
        energy       : Quantity
                must have energy dimension
        obs          : Bool
                True if dN/dE in observed  energy,
                i.e. by incluidng edisp effects
                By default is True
        signal        : Bool
                True if you want to include in the
                simulation the signal of gamma
                By default is True
        bkg          : Bool
                True if you want to include in the
                simulation the background
                By default is True
        """
        ax.grid(True,which='both',linewidth=0.8)
        ax.set_ylabel(r' $dN / dE \; E^2$   [TeV]',size=30)
        ax.set_xlabel('E [TeV]',size=30)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        bin_edges,bin_centers, bin_width, counts = self.get_coords(obs=obs,bkg=bkg, signal=signal)
        dnde = counts/bin_width*bin_centers**2
        if line:
            ax.plot(bin_centers,dnde,**kwargs)
        else:
            ax.scatter(bin_centers,dnde,**kwargs)
        return ax, fig
    
    def plot_observed_dnde(self, ax=None, fig=None, emin=None,emax=None,en_bins=10,**kwargs):
        """Plot the observed differential counts dN/dE * E**2

        Parameters
        ----------
        energy       : Quantity
                must have energy dimension
        obs          : Bool
                True if dN/dE in observed  energy,
                i.e. by incluidng edisp effects
                By default is True
        signal        : Bool
                True if you want to include in the
                simulation the signal of gamma
                By default is True
        bkg          : Bool
                True if you want to include in the
                simulation the background
                By default is True
        """
        ax.grid(True,which='both',linewidth=0.8)
        ax.set_ylabel(r' $dN / dE \; E^2$   [TeV]',size=30)
        ax.set_xlabel('E [TeV]',size=30)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        if hasattr(self.events, 'energy'):
            energies = self.events.energy
            energies = energies.to( u.TeV)
            energies = energies.value 
            if emin is None:
                emin = np.min(energies)
            if emax is None:
                emax = np.max(energies)
            bins         = np.logspace(np.log10(emin*0.99),np.log10(emax),en_bins)
            n,bin_edges  = np.histogram(energies ,bins=bins)
            binwidth     = bin_edges[1:]-bin_edges[:-1]
            bincenters   = np.sqrt(bin_edges[1:]*bin_edges[:-1])
            menStd       = np.sqrt(n)/binwidth*bincenters**2
            y            = n/binwidth*bincenters**2
            ax.errorbar(bincenters, y , menStd, fmt='o',**kwargs, elinewidth=2, markersize=5, capsize=4)

        return ax, fig
    

    
    def fake_events(self, signal=True, bkg=True, obs=True,random_state="random-seed"):
        """Simulate fake events for the current model and reduced IRFs.

        This method overwrites the events defined on the dataset object
        and counts are also updated accordingly.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
                Defines random number generator initialisation.
                Passed to `~gammapy.utils.random.get_random_state`.
        obs          : Bool
                True if dN/dE in observed  energy,
                i.e. by incluidng edisp effects
                By default is True
        signal        : Bool
                True if you want to include in the
                simulation the signal of gamma
                By default is True
        bkg          : Bool
                True if you want to include in the
                simulation the background
                By default is True

        """
        if bkg and not obs:
            raise ValueError("If bkg is True then also obs must be True!")
        if not signal and not bkg:
            raise ValueError("Error, neither the bkg nor the signal is True!")
        
        random_state      = get_random_state(random_state)
        
        # EVENTS SIMULATION
        # IN THE FUTURE: INCLUDE THE ANGLE SIMULATION
        #
        # FIRST, CHECK IF WE WANT TO SIMULATE JUST THE EVENTS IN TRUE ENERGY
        if not obs:
            sign_nevents  = random_state.poisson(self.tot_signal_true)
            en_array      = random_energies(self.signal_flux_true, sign_nevents,random_state)    
        else:
            # SIGNAL SIMULATION
            sign_nevents  = random_state.poisson(self.tot_signal_obs)
            en_array_sign = random_energies(self.signal_flux_obs, sign_nevents,random_state)
  
            # BKG SIMULATION
            bkg_nevents   = random_state.poisson(self.tot_bkg)
            en_array_bkg  = random_energies(self.bkg_flux, bkg_nevents,random_state)
    
            if signal and bkg:
                en_array  = np.append( np.array(en_array_sign.value), en_array_bkg.value) * en_array_bkg.unit
            if signal and not bkg:
                en_array  = en_array_sign
            if not signal and bkg:
                en_array  = en_array_bkg
                
        # CONVERTING THE ENERGIES IN AN EVENT LIST CLASS   
        wcs_geom          = self.geoms['geom']
        center_coord      = wcs_geom.center_coord
        astropy_table     = Table(  data=[en_array, 
                                      np.ones_like(en_array.value)*center_coord[0] ,
                                      np.ones_like(en_array.value)*center_coord[1]                 
                                        ],
                                    names=["ENERGY","RA","DEC"]
                                  )
        self.events       = EventList(astropy_table)
        
        # SAVING THE SIMULATED EVENTS IN COUNTS
        self.counts       = Map.from_geom(wcs_geom)
        self.counts.fill_events(self.events)
        
    
    
    def stat_sum(self,**kwargs):
        """Unbinned likelihood given the current model parameters."""
        self.update_flux()
        
        if self.events is None:
            stat     = -np.inf
        else:
            energies = self.events.energy
            s, b     = self.tot_signal_obs,  self.tot_bkg
            marks    = self.predicted_dnde(energies, obs=True).value
            # CHECK IF ALL MARKS ARE BIGGER THAN ZERO
            if np.sum( marks>0) == len(marks):
                logmarks = np.sum(np.log(marks))
            else:
                logmarks = -np.inf   
            logL = -s  -b + logmarks 
            stat = -2*logL
        
        return stat
    
