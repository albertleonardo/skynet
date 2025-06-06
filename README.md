# SKYNET
**S**eismological **K**nowledge **Y**ardstick **NET**works
A series of deep learning models for seismic phase picking at regional distances.

The manuscript has just been published! check it out here
[https://seismica.library.mcgill.ca/article/view/1431]
>Aguilar Suarez, A. L., & Beroza, G. (2025). Picking Regional Seismic Phase Arrival Times with Deep Learning. Seismica, 4(1). https://doi.org/10.26443/seismica.v4i1.1431



## Quick Installation
Clone the repository
```bash
git clone https://github.com/albertleonardo/skynet.git
cd skynet
```
Create an enviroment 
```
conda env create -f env.yaml
conda activate skynet
```

## Applying to continuous data
The model can be applied to any waveform in the form of an obspy Stream, allowing many different [formats](https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html).
See the [tutorial](https://github.com/albertleonardo/skynet/skynet_tutorial.ipynb) for details on loading a model, and applying it to getting picks.
```python
import skynet
model = skynet.load_model('regional_picker')
```

## Seisbench integration
Models are now available via [Seisbench](https://github.com/seisbench/seisbench)




## Data

We trained intial models using the CREW dataset, details here: https://github.com/albertleonardo/CREW

The paper is here: https://seismica.library.mcgill.ca/article/view/1049

The dataset is here: https://redivis.com/datasets/1z6w-e1w70hpmt

It is also integrated into Seisbench, info here: https://seisbench.readthedocs.io/en/stable/pages/benchmark_datasets.html#crew

We are actively working on bridging the local and regional scales.
