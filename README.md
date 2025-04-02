# SKYNET
**S**eismological **K**nowledge **Y**ardstick **NET**works
A series of deep learning models for seismic phase picking at regional distances.

The manuscript is currently under review and this site is under heavy development.
> Aguilar Suarez and Beroza, Picking Regional Seismic Phase Arrival Times with Deep Learning. under review in Seismica


## Installation
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





## Data

We trained intial models in the CREW dataset, details here: https://github.com/albertleonardo/CREW

The paper is here: https://seismica.library.mcgill.ca/article/view/1049

The dataset is here: https://redivis.com/datasets/1z6w-e1w70hpmt

It is also integrated into Seisbench, info here: https://seisbench.readthedocs.io/en/stable/pages/benchmark_datasets.html#crew

We are actively working on bridging the local and regional scales.
