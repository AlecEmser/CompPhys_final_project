# PHYS 5070 Final Project: Higgs Boson (H->bb) Tagger

## Dependencies
- [uproot](https://uproot.readthedocs.io/en/latest/index.html#)
  - I think that XRootD is automatically installed with this, but just in case you can install that [here](https://pypi.org/project/xrootd/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [scikit-learn](https://scikit-learn.org/stable/index.html)
- [keras](https://keras.io/)
- [tqdm](https://github.com/tqdm/tqdm)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

## How to Run

### Step 1: Generate Training and Test Data
```shell
python buildDataSets.py --ntrain 5 --ntest 2
```
Arguments:
```
--ntrain    Number of files to include in training dataset

--ntest     Number of files to include in testing dataset

```

### Step 2: Inspect Features to Improve Model

 - featureModelling.ipynb

Use Jupyter Notebook to analyze feature importance and correlation. Improve feature selection and re-run sample production before training model.

### Step 3: Train Models
```shell
python trainNNModel.py -nd 20 -gs 
```
Optional Arguments:
```
-nd --ndebug       Run miniature training with selected number of examples in data 

-gs --gridsearch   Run training with exhaustive gridsearch to optimize meta-parameters

```
### Step 4: Check Model Accuracy
TBD

## Authors
Noah Zipper and Alec Emser