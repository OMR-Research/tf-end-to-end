# tf-deep-omr

TensorFlow code to perform end-to-end Optical Music Recognition on monophonic scores through Convolutional Recurrent Neural Networks and CTC-based training.

## Citation

This repository was used for the experiments reported in the paper:

[End-to-End Neural Optical Music Recognition of Monophonic Scores](http://www.mdpi.com/2076-3417/8/4/606)

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

## Corpora

This repository is intended for the Printed Images of Music Staves (PrIMuS) dataset.

PrIMuS can be donwloaded from https://grfia.dlsi.ua.es/primus/


## Training

Assuming that PrIMuS dataset has been downloaded, and all its samples has been placed in the same folder, the training of the models can be done with `ctc_training.py`. It is necessary to build a list of training samples and the set of symbols (vocabulary). Examples of these files are given in `Data`folder.


### Semantic  

```
python ctc_training.py -semantic -corpus <path_to_PrIMuS> -set Data/train.txt -vocabulary Data/vocabulary_semantic.txt  -save_model ./trained_semantic_model
```


### Agnostic  

```
python ctc_training.py -corpus <path_to_PrIMuS> -set Data/train.txt -vocabulary Data/vocabulary_agnostic.txt -save_model ./trained_agnostic_model
```


## Recognition

For running inference over an input image, `ctc_predict.py` can be used, along with an input image, a trained model, and the corresponding vocabulary file. 

The repository is not provided with trained models but these can be download from:

* [Agnostic model](https://grfia.dlsi.ua.es/primus/models/PrIMuS/Agnostic-Model.zip)
* [Semantic model](https://grfia.dlsi.ua.es/primus/models/PrIMuS/Semantic-Model.zip)

These models were the result of the traning process for one of the folds of the 10-fold cross-validation considered in the paper.

Let's see an example for the sample from PrIMuS provided in `Data/Example`:

![Alt text](Data/Example/000051652-1_2_1.png?raw=true "000051652-1_2_1")

This sample belongs to the test set of the aforementioned fold, so it was not seen by the networks during their training stage.


### Semantic

Running

```
python ctc_predict.py -image Data/Example/000051652-1_2_1.png -model Models/semantic_model.meta -vocabulary Data/vocabulary_semantic.txt
```

should get the following prediction

``
clef-C1 	keySignature-EbM 	timeSignature-2/4 	multirest-23 	barline 	rest-quarter 	rest-eighth 	note-Bb4_eighth 	barline 	note-Bb4_quarter. 	note-G4_eighth 	barline 	note-Eb5_quarter. 	note-D5_eighth 	barline 	note-C5_eighth 	note-C5_eighth 	rest-quarter 	barline 
``

The ground-truth of this example is given in `Data/Example/000051652-1_2_1.semantic`:

``
clef-C1	    keySignature-EbM	timeSignature-2/4	multirest-23	barline	rest-quarter	rest-eighth	        note-Bb4_eighth	    barline	    note-Bb4_quarter.	note-G4_eighth	barline	    note-Eb5_quarter.	note-D5_eighth	barline	    note-C5_eighth	note-C5_eighth	rest-quarter	barline
``

It can be observed that the staff section is perfectly recognized by the model.


### Agnostic

Running

```
python ctc_predict.py -image Data/Example/000051652-1_2_1.png -model Models/agnostic_model.meta -vocabulary Data/vocabulary_agnostic.txt
```

should get the following prediction

``
clef.C-L1 	accidental.flat-L4 	accidental.flat-L2 	accidental.flat-S3 	digit.2-L4 	digit.4-L2 	digit.2-S5 	digit.3-S5 	multirest-L3 	barline-L1 	rest.quarter-L3 	rest.eighth-L3 	note.eighth-L4 	barline-L1 	note.quarter-L4 	dot-S4 	note.eighth-L3 	barline-L1 	note.quarter-S5 	dot-S5 	note.eighth-L5 	barline-L1 	note.eighth-S4 	note.eighth-S4 	rest.quarter-L3 
``

The ground-truth of this example is given in `Data/Example/000051652-1_2_1.agnostic`:

``
clef.C-L1	accidental.flat-L4	accidental.flat-L2	accidental.flat-S3	digit.2-L4	digit.4-L2	digit.2-S5	digit.3-S5	multirest-L3	barline-L1	rest.quarter-L3	    rest.eighth-L3	note.eighth-L4	barline-L1	note.quarter-L4	    dot-S4	note.eighth-L3	barline-L1	note.quarter-S5	    dot-S5	note.eighth-L5	barline-L1	note.eighth-S4	note.eighth-S4	rest.quarter-L3	barline-L1
``

As discussed in the paper, this representation often misses the last barline.


## Contact: 

* Jorge Calvo Zaragoza (jcalvo@dlsi.ua.es)

