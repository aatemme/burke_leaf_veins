burke_leaf_veins
==============================

Segmentation of leaf veins from microscope images

Requires
--------

* Python 3.6
* [pipenv](https://pipenv.readthedocs.io/en/latest/)

Setup
------
Package management and python virtual environments are managed by
[pipenv](https://pipenv.readthedocs.io/en/latest/)

run

```bash
pipenv install
```

or

```bash
make setup_environment
```

To have pipenv install required packages and setup a virtual environment.

Build Training Data sets
-------------------------

To create the training dataset, run:

```bash
  make clean_targets
  make dataset
```

This will create the folder data/processed/veins with the training data enclosed.

The traced (target) images are slightly modified to remove non-binary
pixels and to remove small gaps between lines that should be
connected.

The real images are ready out of the box. They are just copied
from the raw data folder to the processed data folder.

Image names in the data/raw/Veins machine learning with Chris/blacklist.csv
will be ignored.


Segment Images
---------------

The models require ~12.5Gb of memory to segment an image. Segmentations were performed on the CPU due to the large memory requirements. On the cpu, one images takes ~1.5min to segment.

## FINAL
To segment a folder of images using the final trained model, see the
`src/data/segment_extract_length_folder_sapleo2.sh`. This enables segmentation
of images on UGA's sapelo2 cluster.

## v3_dilations.
v3_dilations was the last model derivation tried before training the last model.
It is kept here for posterity.  
The model weights from the noAug_dilations_epoch800 save point is used.

### To segment the images set aside for testing using the v3_dilations model:
```bash
  make models/v3/results/dilations/test/
```

### To segment all images using the v3_dilations model:
```bash
  make models/v3/results/dilations/train/
```

### To segment the vein images for other species:
```bash
  make models/v3/results/generalization/
```

The other species vein images were also run through the network at 4x scale by adding `image = rescale(image,4,anti_aliasing=True)` to segment_images.py at:
```python
...
for image_path in tqdm(images):
    image = io.imread(image_path)
    image = rescale(image,4,anti_aliasing=True)
...
```

### To extract vein lengths (in pixels) from segmentation probabilities:

```bash
make v3_results
```

This will save the measurements to a sqlite database at
reports/data/results.sqlite in the tables:
  - v3_test_vein_length: For the test image data set
  - v3_train_vein_length: For the train image data set

### To add manual measurements to the results database
```bash
make manual_measurments
```

This will save the measurements to a sqlite database at
reports/data/results.sqlite in a table named manually_measured

### To export all data in database to csv files:

```bash
make export_csv
```

Will result in csv files in the `reports/csv/` directory.

Notebooks
----------

The `notbooks` directory contains all jupyter notebooks. Jupyter notebook
can be started using:
```
pipenv run jupyter notebook
```

* `notebooks/V3/` contains the analysis of the many V3 models.
* `notebooks/CompareToData.ipynb` compares the V3 model segmentations to the manually measured vein lengths.
  * `notebooks/Correlation.png` is generated from this notebook
* `notebooks/Model Generalziation.ipynb` is an exploration of how the model
generalizes to other vein species images
* `notebooks/Segment Whole Images.ipynb` was used to explore how much padding would be needed to segment whole images using the neural network.
* `notebooks/Clean_Up_Target_Imgaes.ipynb` explores cleaning up the segmentation masks originally provided to me by the Burke lab.

Project Organization
------------
Based off of [cookiecutter data science](https://github.com/drivendata/cookiecutter-data-science):

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile            <- The requirements file for reproducing the analysis environment,
    │                         generated by pip
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
