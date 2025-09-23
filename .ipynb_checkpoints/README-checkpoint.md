# pyNOMIC
## A pipeline for reducing data from the NOMIC camera on the LBT

## Required files: 
- NOMIC master bad pixel map (`"master_badmap.fits"`)

## Arguments:

Example: `python3 rvpipe.py 'data' -c 6 -n 20 -i`

## Outputs:

`"{obj}_NOMIC_reduced.npz"`: 

## Other files:

`"pyNOMIC_reduction.ipynb"`: Jupyter notebook for running the pipeline