# UNSUPERVISED ANOMALY DETECTION FOR MULTIVARIATE TIME SERIES USING  DIFFUSION MODEL

## TIMEADDM

This repository contains code for the paper, UNSUPERVISED ANOMALY DETECTION FOR MULTIVARIATE TIME SERIES USING  DIFFUSION MODEL.
(The code is being sorted out and we will continue to update it.)

##  OVERVIEW

We make the first attempt to design a novel diffusion-based anomaly detection model (named TimeADDM) for MTS data using the effective learning mechanism of DMs.  To enhance the learning effect on MTS data, we propose to apply diffusion steps to the representations that accumulate the global time correlations through recurrent embedding. 

## HOW TO RUN

- Train a embedding model:
> python lstm_ae_emb.py  --'dataset'  your dataset

- Sampling and detecting:
> python main.py will train the whole model and will get the reconstructed data and detected score.

## DATASET

We apply our method on four datasets, the SWaT and WADI datasets, in which we did not upload data in this repository.Please refer to [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/) and send request to iTrust is you want to try the data.
