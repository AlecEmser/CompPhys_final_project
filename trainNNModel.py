import numpy as np
import pickle as pkl
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout

def preprocessData(X_all, y_all, testing=False):
    # Sample small amount of data for testing purposes
    if testing:
        X_all = X_all[:10]
        y_all = y_all[:10]

    # Re-scale feature vectors to unit variance
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Save scaler to pickle file for use in bracket predictions
    scaler_filename = 'Models/scaler.pickle'
    with open(scaler_filename, 'wb') as file:
        pkl.dump(scaler, file)
    print(f"Scaler saved to '{scaler_filename}'")

    # # Apply principle component analysis to reduce dimensionality
    # pca = decomposition.PCA(n_components = 10)
    # pca.fit(X_all)
    # X_all= pca.transform(X_all)

    return X_all, y_all

def buildNN(input_dim = None, optimizer = 'adam', hidden_layers = 3, hidden_activation = 'relu', hidden_dropout = 0.2, hidden_width = 100):
    NN = Sequential()
    NN.add(Dense(   50, 
                    input_dim = input_dim, 
                    kernel_initializer = 'random_uniform', 
                    activation = 'sigmoid'))

    for _ in range(hidden_layers):
        NN.add(Dropout(hidden_dropout))
        NN.add(Dense(hidden_width, activation = hidden_activation))

    NN.add(Dropout(0.2))
    NN.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
    NN.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return NN

def trainNN(NN, model_search_params, X_train, y_train):
    # Optimize meta-parameters with grid search and cross-validation
    clf = GridSearchCV( NN,
                        param_grid = model_search_params,
                        scoring = 'accuracy',
                        n_jobs = -1)

    print('Starting Grid Search (This Will Take A While)...')
    clf.fit(X_train, y_train)

    # Return optimal network
    return clf.best_estimator_.model, clf.best_params_

def main():
    # Load pre-processed data
    with open('Data/train_data.pickle', 'rb') as file:
        X, y, _ = pkl.load(file)
        
    # Pre-process training data
    X_train, y_train = preprocessData(X, y, testing=False)

    # Build classifier network
    NN = KerasClassifier(build_fn = buildNN, input_dim = X_train.shape[1])

    # Select meta-parameters for grid search optimization
    model_search_params = {  
                'epochs': [2],
                'batch_size': [2],
                'hidden_layers' : [3], 
                'hidden_activation' : ['relu'], 
                'hidden_dropout' : [0.2], 
                'hidden_width' : [50],
                'optimizer': ['adam']
            }

    # Train network with grid search to optimize parameters
    NN_opt, features_opt = trainNN(NN, model_search_params, X_train, y_train)

    
    # Output model information
    print('\tOptimal Meta-Parameters: ')
    for pair in features_opt.items():
        print(f'\t{pair[0]} = {pair[1]}')

    # Save trained model
    NN_opt.save('Models/NN_model.h5')

if __name__ == "__main__":
    main()