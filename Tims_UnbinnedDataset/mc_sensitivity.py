from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from gammapy.data import DataStore, Observation
from gammapy.datasets import MapDataset, MapDatasetEventSampler, Datasets
from gammapy.estimators import FluxPointsEstimator
from gammapy.maps import MapAxis, WcsGeom, Map, MapAxes, MapCoord
from gammapy.irf import load_cta_irfs
from gammapy.makers import MapDatasetMaker, FoVBackgroundMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.modeling.models import (
#     Model,
#     Models,
    SkyModel,
#     PowerLawSpectralModel,
#     PowerLawNormSpectralModel,
    PointSpatialModel,
    LogParabolaSpectralModel,
#     GaussianSpatialModel,
#     TemplateSpatialModel,
#     ExpDecayTemporalModel,
#     LightCurveTemplateTemporalModel,
    FoVBackgroundModel,
)
from gammapy.maps.geom import pix_tuple_to_idx
import warnings
from gammapy.utils.integrate import trapz_loglog
import sys
import os
os.chdir('/home/hpc/caph/mppi086h/gammapy-unbinned-analysis/EventDataset/')
print(os.getcwd())
sys.path.append('/home/hpc/caph/mppi086h/gammapy-unbinned-analysis/EventDataset/')
from EventDatasetMaker import EventDatasetMaker
from EventDataset import EventDataset

### Read inputs ###

# read in the command line arguments
spatial_binsz = float(sys.argv[1]) # in deg
n_ebins = int(sys.argv[2])
n=100  # number of simulations

### create a dataset
data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
obs_id = [23523] # just one observation 
obs1 = data_store.get_observations(obs_id)[0]
crab_pos = SkyCoord(184.557, -5.784, unit='deg', frame='galactic') 
obs_pos = obs1.pointing_radec
## choose energy binning
ebins = np.logspace(0,2,n_ebins+1)
ebins_true = np.geomspace(0.5,100,41)

energy_axis = MapAxis.from_edges(
    ebins, unit="TeV", name="energy", interp="log"  
)
energy_axis_true = MapAxis.from_edges(
    ebins_true, unit="TeV", name="energy_true", interp="log"  
)
migra_axis = MapAxis.from_bounds(
    0.2, 5, nbin=150, node_type="edges", name="migra"
)
geom = WcsGeom.create(
    skydir=obs_pos,
    binsz=spatial_binsz,
    width=(3.5, 3.5),
    frame="icrs",  # same frame as events
    proj="CAR",
    axes=[energy_axis],
)

circle = CircleSkyRegion(
    center=crab_pos, radius=0.3 * u.deg
)
data = geom.region_mask(regions=[circle], inside=False)
exclusion_mask = ~geom.region_mask(regions=[circle])
maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)
## make the dataset
maker = MapDatasetMaker(background_oversampling=2)
maker_safe_mask = SafeMaskMaker(methods=['offset-max'], offset_max='1.5 deg')
# providing the migra axis seems essential so that edisp is a EdispMap and no EdispKernelMap
reference = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, migra_axis=migra_axis)  

# the binned dataset
dataset = maker.run(reference, obs1)
dataset = maker_safe_mask.run(dataset, obs1)
dataset.mask_safe *= geom.energy_mask(energy_min=1*u.TeV)
bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
dataset.models=bkg_model

# the unbinned dataset
eds_maker = EventDatasetMaker(safe_mask_maker=maker_safe_mask, selection=None)
eds = eds_maker.run(reference, obs1)
eds.background = dataset.background.copy()
eds.mask_safe = dataset.mask_safe.copy()
bkg_model2 = FoVBackgroundModel(dataset_name=eds.name)
eds.models=bkg_model2

### set the model
model_gauss = SkyModel(
    spatial_model=PointSpatialModel(lon_0="184.557 deg", lat_0="-5.784 deg", frame = 'galactic'),
    spectral_model=LogParabolaSpectralModel(amplitude='3.5e-11 cm-2 s-1 TeV-1', 
                                          reference='1 TeV', 
                                          alpha=1.8, 
                                          beta=0.4
                                         ),
    name='crab_model'
    )

model_gauss.spectral_model.amplitude.value /= 10. # 10 times weaker signal than the Crab

model_gauss.spatial_model.parameters.freeze_all()
margin=0.05
model_gauss.spatial_model.lon_0.min = model_gauss.spatial_model.lon_0.value - margin
model_gauss.spatial_model.lon_0.max = model_gauss.spatial_model.lon_0.value + margin
model_gauss.spatial_model.lat_0.min = model_gauss.spatial_model.lat_0.value - margin
model_gauss.spatial_model.lat_0.max = model_gauss.spatial_model.lat_0.value + margin

model_gauss.spectral_model.amplitude.min = 0.
model_gauss.spectral_model.alpha.min = 0.
model_gauss.spectral_model.beta.min = 0.

dataset.models += model_gauss
eds.models += model_gauss.copy(name="crab_model")

eds.evaluators['crab_model'].spatialbs = spatial_binsz*u.deg  # spatial_binsz*u.deg

### calculate the avg events per bin
npred_sum = dataset.npred().data[dataset.mask_safe].sum()
fit_bins = dataset.mask_safe.data.sum()

### Compare analysis for n MC simulations

params=dataset.models.parameters.free_parameters.names # ['amplitude', 'alpha', 'beta', 'norm', 'tilt']
par_input=[dataset.models.parameters[par].value for par in params]

par_binned=[]
err_binned=[]
par_unbin =[]
err_unbin =[]
dTS_binned = []
dTS_unbinned = []

for ii in range(n): 
    ### draw random revents
    for i,par in enumerate(params):
        dataset.models.parameters.free_parameters[par].value = par_input[i]
        eds.models.parameters.free_parameters[par].value = par_input[i]
    sampler = MapDatasetEventSampler(random_state=ii)
    events = sampler.run(dataset, obs1)

    ### set up binned and unbinned datasets
    counts = Map.from_geom(geom)
    counts.fill_events(events)
    dataset.counts=counts

    eds.events = events
    eds.evaluators['crab_model'].exposure = None # force update
    eds._response_bkg_cached = None
    del eds.evaluators['crab_model']._compute_npred

    ### fit both datasets
    fit = Fit(optimize_opts={"print_level": 0})
    result = fit.run([dataset])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result2 = fit.run([eds])

    ### parameter comparison
    par_binned.append([result.parameters[par].value for par in params])
    err_binned.append([result.parameters[par].error for par in params])

    par_unbin.append([result2.parameters[par].value for par in params])
    err_unbin.append([result2.parameters[par].error for par in params])

    ### test of the source significance
    ts_binned = result.total_stat
    ts_unb = result2.total_stat

    with dataset.models.parameters.restore_status():
        dataset.models.parameters['amplitude'].value = 0
        dataset.models['crab_model'].parameters.freeze_all()
        res=fit.optimize(dataset)
        ts_binned0 = res.total_stat
    with eds.models.parameters.restore_status():
        eds.models.parameters['amplitude'].value = 0
        eds.models['crab_model'].parameters.freeze_all()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res2=fit.optimize([eds])
        ts_unb0 = res2.total_stat
    dTS_binned.append(ts_binned0-ts_binned)
    dTS_unbinned.append(ts_unb0-ts_unb)
    
result_dict = dict(
spatial_binsz       = spatial_binsz,
n_ebins             = n_ebins,
npred_sum           = npred_sum,
fit_bins            = fit_bins,
n_simulations       = n,
dTS_binned          = np.array(dTS_binned),
dTS_unbinned        = np.array(dTS_unbinned),
par_binned          = np.array(par_binned),
par_unbin           = np.array(par_unbin),
err_binned          = np.array(err_binned),
err_unbin           = np.array(err_unbin),
)

#write the results
path = Path('/home/hpc/caph/mppi086h/vault/unbinned_analysis/simulations_of_weak_test_source_new/')
path.mkdir(exist_ok=True)
np.save(path / 'result_{:.2f}deg_{}ebins.npy'.format(spatial_binsz, n_ebins), result_dict)