import uproot
import numpy as np
import pandas as pd
import pickle as pkl

def loadRemoteData(nFiles, features, spectators, labels, filetype=None):
    """Loads remote data files from CMS OpenData and formats into 
       feature vectors (X), output labels (y), and spectator variables (spect)

    Args:
        nFiles (int): Number of root files to process
        features (list of str): Features chosen to use in model
        spectators (list of str): 'Spectator' variables to used to visualize model output
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
        filerange = [9,10+nFiles]
        file_label = 'train'
    elif 'Test' in filetype:
        filerange = [0,1+nFiles]
        file_label = 'test'
    else:
        valid = ['Train', 'Test']
        raise ValueError(f'loadRemoteData: filetype mist be one of {valid}')

    X = []
    y = []
    spects = []

    # Iterate through chosen number of samples and combine datasets
    for i in range(filerange[0],filerange[1]):
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
            data = pd.concat([data_features, data_spects,data_labels],keys=['Features', 'Spectators', 'Labels'], axis=1)
            
            # Assign new output label from label branches 
            data = data.assign(Label = 0*data.Labels.fj_isQCD+1*data.Labels.label_H_bb)
            data.drop(columns=['Labels'],inplace=True)
            
            # Convert to numpy arrays for model-use
            file_X = data.Features.to_numpy()
            file_y = data.Label.to_numpy()
            file_spects = data.Spectators.to_numpy()
    
        X.append(file_X)
        y.append(file_y)
        spects.append(file_spects)

    # Remove unecessary dimensions
    X_out = np.squeeze(np.asarray(X))
    y_out = np.squeeze(np.asarray(y))
    spects_out = np.squeeze(np.asarray(spects))
    
    return X_out, y_out, spects_out

def main():

    # Select number of files to generate training and testing data
    nTrainFiles = 1
    nTestFiles = 1

    # Select features for model to process
    features = [
                'fj_jetNTracks',
                'fj_nSV',
                'fj_trackSipdSig_0',
                'fj_trackSipdSig_1',
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

    # Load and format datasets
    X_train, y_train, spects_train = loadRemoteData(nTrainFiles, features, spectators, labels, filetype='Train')
    X_test, y_test, spects_test = loadRemoteData(nTestFiles, features, spectators, labels, filetype='Test')

    # Save to pickle files
    train_data = (X_train, y_train, spects_train)
    pickle_filename = 'Data/train_data.pickle'
    with open(pickle_filename, 'wb') as file:
        pkl.dump(train_data, file)
    print(f"\nTraining data saved to '{pickle_filename}'")

    test_data = (X_test, y_test, spects_test)
    pickle_filename = 'Data/test_data.pickle'
    with open(pickle_filename, 'wb') as file:
        pkl.dump(test_data, file)
    print(f"\nTest data saved to '{pickle_filename}'")


if __name__=='__main__':
    main()