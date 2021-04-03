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

## How to Run

### Step 1: Generate Training and Test Data
```bash
python buildDataSets.py
```
### Step 3: Train Models
```bash
python trainNNModel.py
```
### Step 3: Check Model Accuracy
TBD

## Authors
Noah Zipper and Alec Emser