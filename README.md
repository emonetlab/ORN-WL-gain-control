# ORN-WL-gain-control

This code allows exploration of coding and decoding fidelity of *Drosophila* olfactory ORNs, using a biophysical model of ORN sensing and either compressed sensing decoding or classification tasks for decoding. 

## Getting Started

### Installation

The code runs in Python 3 with just a few c scientific and plotting packages. Classification tasks require TensorFlow. The code has been run on and tested with Anaconda, and it is recommended to create an environment, using the package list via the CS-variability-adaptation.yml:​	

```
$ conda env create -f ORN-WL-gain-control.yml
```

Tensor flow installed with the yml file may not be optimized for your machine specifications, so for better performance, you may re-install it separately following their documentation.

## Usage

The biological system consists of many nonlinear sensors (the ORNs), responding to an odor stimulus which consists of several odor components. This is the coding task. Decoding takes the repertoire of ORN responses and either reconstructs or clusters the generating signal from these responses. 

There are several user-defined specifications for each such task, consisting of choice of stimulus, sensor statistics, the nature of the adaptive feedback, etc. To allow the user versatility in testing, all of these parameters are specified in a so-called "specs file", which contains all coding and decoding specifications. Every simulation or estimation requires a specs file. Writing them is straightforward, and they are allow for a good deal of versatility.

### Make a specs file. 

Specs files are format .txt. 
