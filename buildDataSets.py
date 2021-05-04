import argparse
import uproot
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import trange


def parse_args():
    """Parse command-line arguments for script """
    parser = argparse.ArgumentParser(
        description='Loads and formats input data for Hbb classifier (takes ~45-60s per file).')
    parser.add_argument('--ntrain',
                        type=int,
                        choices=range(1, 83),
                        metavar="[1-83]",
                        help='Number of training files to process')
    parser.add_argument('--ntest',
                        type=int,
                        choices=range(1, 10),
                        metavar="[1-9]",
                        help='Number of training files to process')
    args = parser.parse_args()
    return args


def load_remote_data(nFiles, features, spectators, labels, filetype=None):
    """Loads remote data files from CMS OpenData and formats into feature 
       vectors (X), output labels (y), and spectator variables (spect)

    Args:
        nFiles (int): Number of ROOT files to process
        features (list of str): Features chosen to use in model
        spectators (list of str): 'Spectator' variables used to assess model output
        labels (list of str): Output label used for training
        filetype (str, optional): Specification if data is for training or testing. Defaults to None.

    Raises:
        Exception: ValueError raised if filetype isn't correctly specified

    Returns:
        X_out (ndarray): 2D array of feature vectors
        y_out (ndarray): 1D array of output labels
        spects_out (ndarray): 2D array of spectator variables
    """

    # Choose from training or testing datasets
    if 'Train' in filetype:
        filerange = (9, 9+nFiles)
        file_label = 'train'
    elif 'Test' in filetype:
        filerange = (0, nFiles)
        file_label = 'test'
    else:
        valid = ['Train', 'Test']
        raise ValueError(f'load_remote_data: filetype must be one of {valid}')

    X = []
    y = []
    spects = []

    # Iterate through chosen number of samples and combine datasets
    print(f"\nIterating through {file_label}ing data files...")
    for i in trange(*filerange):
        # Read in file with uproot
        with uproot.open(f'root://eospublic.cern.ch//eos/opendata/cms/datascience/HiggsToBBNtupleProducerTool/HiggsToBBNTuple_HiggsToBB_QCD_RunII_13TeV_MC/{file_label}/ntuple_merged_{i}.root') as file:
            tree = file["deepntuplizer/tree"]

            # Create dataframes for features, labels, and spectators
            data_features = pd.DataFrame([])
            for feature in features:
                data_features[feature] = tree[feature].array(library="pd")

            data_spects = pd.DataFrame([])
            for spectator in spectators:
                data_spects[spectator] = tree[spectator].array(library="pd")

            data_labels = pd.DataFrame([])
            for label in labels:
                data_labels[label] = tree[label].array(library="pd")

            # Merge into single dataframe
            data = pd.concat([data_features, data_spects, data_labels], keys=[
                             'Features', 'Spectators', 'Labels'], axis=1)

            # Assign new output label from label branches
            data = data.assign(Label=0*data.Labels.fj_isQCD +
                               1*data.Labels.label_H_bb)
            data.drop(columns=['Labels'], inplace=True)

            # Convert to numpy arrays for model-use
            file_X = data.Features.to_numpy()
            file_y = data.Label.to_numpy()
            file_spects = data.Spectators.to_numpy()

        X.append(file_X)
        y.append(file_y)
        spects.append(file_spects)

    # Remove unecessary dimensions
    X_out = np.asarray(X).reshape(-1, np.asarray(X).shape[-1])
    y_out = np.asarray(y).flatten()
    spects_out = np.asarray(spects).reshape(-1, np.asarray(spects).shape[-1])

    return X_out, y_out, spects_out


def main(args):

    # Select features for model to process
    features = [
        'fj_jetNTracks',
        'fj_nSV',
        'fj_tau0_trackEtaRel_0',
        'fj_tau0_trackEtaRel_1',
        'fj_tau0_trackEtaRel_2',
        'fj_tau1_trackEtaRel_0',
        'fj_tau1_trackEtaRel_1',
        'fj_tau1_trackEtaRel_2',
        'fj_tau_flightDistance2dSig_0',
        'fj_tau_flightDistance2dSig_1',
        'fj_tau_vertexDeltaR_0',
        'fj_tau_vertexEnergyRatio_0',
        'fj_tau_vertexEnergyRatio_1',
        'fj_tau_vertexMass_0',
        'fj_tau_vertexMass_1',
        'fj_trackSip2dSigAboveBottom_0',
        'fj_trackSip2dSigAboveBottom_1',
        'fj_trackSip2dSigAboveCharm_0',
        'fj_trackSipdSig_0',
        'fj_trackSipdSig_0_0',
        'fj_trackSipdSig_0_1',
        'fj_trackSipdSig_1',
        'fj_trackSipdSig_1_0',
        'fj_trackSipdSig_1_1',
        'fj_trackSipdSig_2',
        'fj_trackSipdSig_3',
        'fj_z_ratio',
    ]

    # Select variables to plot performance
    spectators = [
        'fj_pt',
        'fj_sdmass',
    ]

    # Select variables to use as labels
    labels = [
        'fj_isQCD',
        'sample_isQCD',
        'label_H_bb',
    ]

    # Load, format, and save training dataset
    X_train, y_train, spects_train = load_remote_data(
        args.ntrain, features, spectators, labels, filetype='Train')

    train_data = (X_train, y_train, spects_train)
    train_filename = 'Data/train_data_v2.pickle'
    with open(train_filename, 'wb') as file:
        pkl.dump(train_data, file)
    print(f"\nTraining data saved to '{train_filename}'")

    # Load, format, and save training dataset
    X_test, y_test, spects_test = load_remote_data(
        args.ntest, features, spectators, labels, filetype='Test')

    test_data = (X_test, y_test, spects_test)
    test_filename = 'Data/test_data_v2.pickle'
    with open(test_filename, 'wb') as file:
        pkl.dump(test_data, file)
    print(f"\nTest data saved to '{test_filename}'")


if __name__ == '__main__':
    args = parse_args()
    main(args)
