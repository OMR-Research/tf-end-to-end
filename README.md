# Optical Music Recognition (img2midi) <a target="_blank" href="https://colab.research.google.com/github/FreshMag/tf-end-to-end/blob/master/omr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This repository is a fork of the [original project by Calvo-Zaragoza](https://github.com/OMR-Research/tf-end-to-end) that
was used for the experiments reported in the paper [End-to-End Neural Optical Music Recognition of Monophonic Scores](http://www.mdpi.com/2076-3417/8/4/606).
More information can be found below. Please consider taking a look at the original repository if you want to know all
the details about how the model was originally trained and created.

## What you can find on this repository

- All the original code used to *load the model trained on the PrIMuS dataset and make predictions*
- Additional code to make the original one more usable
- Additional code used to pre-process music sheets' images taken from photos
- Additional code that translates the format of the models' output into midi (or other music encoding formats)

All the additional code is part of a Computer Vision project I worked on. 

## Quick usage

It is possible to quickly test and use all the functionalities using the [notebook provided](omr.ipynb), on Google Colab for example.

Alternatively, you can run the file `ctc_predict.py` providing the required command line arguments:
- path to the input image
- path to the used models
- path to the vocabulary file

You can copy the following line and substitute the required paths with the actual ones:
```
python ctc_predict.py -image <PATH/TO/image.png> -model <PATH/TO/model.meta> -vocabulary <PATH/TO/vocabulary.txt>
```

In case you're manually running the python script, **one** of (or both) the models must be downloaded from the following links (presented also in
the original repository).

* [Agnostic model](https://grfia.dlsi.ua.es/primus/models/PrIMuS/Agnostic-Model.zip)
* [Semantic model](https://grfia.dlsi.ua.es/primus/models/PrIMuS/Semantic-Model.zip)

More details about them can be found in the original repository.


# Citations

```
@Article{Calvo-Zaragoza2018,
  AUTHOR = {Calvo-Zaragoza, Jorge and Rizo, David},
  TITLE = {End-to-End Neural Optical Music Recognition of Monophonic Scores},
  JOURNAL = {Applied Sciences},
  VOLUME = {8},
  YEAR = {2018},
  NUMBER = {4},
  ARTICLE NUMBER = {606},
  URL = {http://www.mdpi.com/2076-3417/8/4/606},
  ISSN = {2076-3417},
  DOI = {10.3390/app8040606}
}
```
