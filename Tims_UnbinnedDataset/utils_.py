from gammapy.utils.interpolation import (
    ScaledRegularGridInterpolator,
    interpolation_scale,
)
from astropy.utils import lazyproperty
# from gammapy.irf.core import IRF  # possibly inherite from IRF
import numpy as np
import astropy.units as u
from gammapy.maps import MapAxis, MapAxes #WcsGeom, Map, MapAxes, WcsNDMap
import matplotlib.pyplot as plt

class EdispInv():
    """
    Class to give the PDF for an event with e_reco = E to come from a true energy E'
    """
    default_interp_kwargs = dict(
        bounds_error=False,
        fill_value=0.0,
    )
    def __init__(self, edisp2d, emin, emax, e_steps=120, interp_kwargs=None):
        """
        edisp2d: gammapy.irf.EnergyDispersion2D
        energy_reco: MapAxis with reconstructed energies
        """
        self._edisp2d = edisp2d
        self.energy_reco = np.geomspace(emin, emax, e_steps) # fine sampled e_reco edges
        self.e_reco =MapAxis.from_edges(self.energy_reco, name="energy", interp="log")
        # keep original axes to loose no information through interpolation
        self.offset=self._edisp2d.axes["offset"]
        self.migra=self._edisp2d.axes["migra"]
        self.e_true=self._edisp2d.axes["energy_true"]
        self.axes = MapAxes([self.offset, self.e_true, self.e_reco])
        kernels = []
        pdfs = []
        for off in self.offset.center:
            kernel=self._edisp2d.to_edisp_kernel(offset=off, 
                                                 energy=self.energy_reco)
            data = kernel.data # probabilities normed along e_reco (axis=1)
            # renorm along e_true
            with np.errstate(invalid='ignore'):
                data /= data.sum(axis=0)
                
            data = np.nan_to_num(data)
            pdf = data / np.diff(self.e_true.edges)[:,np.newaxis]
            kernels.append(data)
            pdfs.append(pdf)
            
#         self.data = NDDataArray(axes=axes, data=np.array(kernels))
#         self.data_pdf = NDDataArray(axes=axes, data=np.array(pdfs))
        
        self.kernels=np.array(kernels)
        self.pdfs = np.array(pdfs)/self.e_true.unit
        
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs.copy()
        self.interp_kwargs = interp_kwargs
    @lazyproperty   
    def _interpolate(self):
        kwargs = self.interp_kwargs.copy()
        # Allow extrap[olation with in bins
        kwargs["fill_value"] = None
        points = [a.center for a in self.axes]
        points_scale = tuple([a.interp for a in self.axes])
        return ScaledRegularGridInterpolator(
            points,
            self.kernels,
            points_scale=points_scale,
            **kwargs,
        )
    @staticmethod
    def _mask_out_bounds(invalid):
        return np.any(invalid, axis=0)
    
    def evaluate(self, method=None, **kwargs):
        """Evaluate IRF
        Parameters
        ----------
        **kwargs : dict
            Coordinates at which to evaluate the IRF
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values
        """
        # TODO: change to coord dict?
        non_valid_axis = set(kwargs).difference(self.axes.names)
        if non_valid_axis:
            raise ValueError(
                f"Not a valid coordinate axis {non_valid_axis}"
                f" Choose from: {self.axes.names}"
            )

        coords_default = self.axes.get_coord()

        for key, value in kwargs.items():
            coord = kwargs.get(key, value)
            if coord is not None:
                coords_default[key] = u.Quantity(coord, copy=False)
        data = self._interpolate(coords_default.values(), method=method)

        if self.interp_kwargs["fill_value"] is not None:
            idxs = self.axes.coord_to_idx(coords_default, clip=False)
            invalid = np.broadcast_arrays(*[idx == -1 for idx in idxs])
            mask = self._mask_out_bounds(invalid)
            if not data.shape:
                mask = mask.squeeze()
            data[mask] = self.interp_kwargs["fill_value"]
            data[~np.isfinite(data)] = self.interp_kwargs["fill_value"]
        return data
    
    def plot_pdf_vs_Etrue(self, ax=None, e_reco=None, offset=None, **kwargs):
        
        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = [Angle(1, "deg")]
            
        if e_reco is None:
            e_reco = [1]*u.TeV

        energy_true = self.axes["energy_true"]

        
        
        for o in offset:
            for e in e_reco:
                z = self.evaluate(
                    offset=o,
                    energy_true=energy_true.center,
                    energy=e,
                )
                ax.semilogx(energy_true.center, z,label=f'{e} - {o}', **kwargs)
            
        ax.legend()
        ax.set_xlabel(f'Energy True [{energy_true.unit}]')
        ax.set_ylabel('Probability density')
        return ax