# CBC_estimator

Implementation of an NPE approach to gravitational wave parameter estimation through the use of normalizing flows.
Once complete it will allow the user to train basic machine learning models to perform a regression task on a CBC signal based on its q-transform

### Usage
The process is meant to have three stages:
    -Dataset generation: Creation of data through the injection of signals in real noise. This part has extra requirements and it is therefore kept separate.
    -Model training:
    -Inference on trained models:

### Requirements
This code relies on glasflow.nflows for its implementation of normalizing flows...