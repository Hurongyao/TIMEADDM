# UNSUPERVISED ANOMALY DETECTION FOR MULTIVARIATE TIME SERIES USING  DIFFUSION MODEL

## TIMEADDM

This repository contains code for the paper, UNSUPERVISED ANOMALY DETECTION FOR MULTIVARIATE TIME SERIES USING  DIFFUSION MODEL.
(The code is being sorted out and we will continue to update it.)

##  Overview

We make the first attempt to design a novel diffusion-based anomaly detection model (named TimeADDM) for MTS data using the effective learning mechanism of DMs.  To enhance the learning effect on MTS data, we propose to apply diffusion steps to the representations that accumulate the global time correlations through recurrent embedding. 

## Datasets

1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
2. SMAP (Soil Moisture Active Passive satellite) also is a public dataset from NASA. 
3. WADI (Water Distribution) is obtained from 127 sensors of the critical infrastructure system under continuous operations. 
4. SWAT (Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous operations. 

We apply our method on four datasets, the SWAT and WADI datasets, in which we did not upload data in this repository.Please refer to [https://itrust.sutd.edu.sg/](https://itrust.sutd.edu.sg/) and send request to iTrust is you want to try the data.

## How to run

- Train and detect:

> python main.py  --config test.yml  --doc ./{dataset}  --sequence
>
> Then you will train the whole model and will get the reconstructed data and detected score.

## How to run with your own data

- By default, datasets are placed under the "data" folder. If you need to change the dataset, you can modify the dataset path  in the main file.Then you should change the corresponding parameters of TIMEEMB.py and diffusion.py

> python main.py  --'dataset'  your dataset



## Result

We  use dataset PSM for testing demonstration, you can run main.py directly and get the corresponding result.
