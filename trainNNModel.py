import argparse
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout

def parse_args():
    parser = argparse.ArgumentParser(description='Trains and saves Keras NN model.')
    parser.add_argument('-nd',
                        '--ndebug',
                        type=int,
                        help='Number of examples to use for debugging purposes')
    parser.add_argument('-gs',
                        '--gridsearch',
                        action='store_true',
                        help='Perform exhaustive grid search for meta-parameters')
    args = parser.parse_args()

    return args

def preprocessData(X_all, y_all, debug=False):
    # Sample small amount of data for debug purposes
    if debug:
        X_all = X_all[:args.ndebug]
        y_all = y_all[:args.ndebug]

    # Re-scale feature vectors to unit variance
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Save scaler to pickle file for use in bracket predictions
    scaler_filename = 'Models/scaler.pickle'
    with open(scaler_filename, 'wb') as file:
        pkl.dump(scaler, file)
    print(f"Scaler saved to '{scaler_filename}'")

    return X_all, y_all

def buildNN(input_dim = None, optimizer = None, hidden_layers = None, hidden_activation = None, hidden_dropout = None, hidden_width = None):
    NN = Sequential()
    NN.add(Dense(   25, 
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

def trainNN(NN, X_train, y_train, model_search_params=None):
    if model_search_params is not None:
        # Optimize meta-parameters with grid search and cross-validation
        clf = GridSearchCV( NN,
                            param_grid = model_search_params,
                            scoring = 'accuracy',
                            n_jobs = 1)
        print('Starting Grid Search (This Will Take A While)...')
        clf.fit(X_train, y_train)

        model, score, params = clf.best_estimator_.model, clf.best_score_, clf.best_params_
    else:
        # Fit estimator to training data
        clf = NN
        clf.fit(X_train, y_train)

        # Find KFold cross-validation score for estimator
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.seed())
        results = cross_val_score(clf, X_train, y_train, cv=kfold)

        model, score, params = clf.model, results.mean(), clf.get_params()
  
    return model, score, params

def main(args):
    # Load pre-processed data
    with open('Data/train_data.pickle', 'rb') as file:
        X, y, _ = pkl.load(file)
        
    # Pre-process training data
    X_train, y_train = preprocessData(X, y, args.ndebug)

    # Build classifier 
    NN = KerasClassifier(   build_fn = buildNN, 
                            input_dim = X_train.shape[1], 
                            optimizer = 'adam', 
                            hidden_layers = 3, 
                            hidden_activation = 'relu', 
                            hidden_dropout = 0.2, 
                            hidden_width = 25, 
                            epochs=2, 
                            batch_size=5)

    # Select meta-parameters for grid search optimization
    model_search_params = {  
                'epochs': [2, 5],
                'batch_size': [2, 5],
                'hidden_layers' : [3], 
                'hidden_activation' : ['relu'], 
                'hidden_dropout' : [0.2], 
                'hidden_width' : [25],
                'optimizer': ['adam']
            }

    # Train network with grid search to optimize parameters
    NN_Model_opt, NN_score_opt, NN_features_opt = trainNN(NN, X_train, y_train, model_search_params if args.gridsearch else None)

    # Output model information
    print(f'\nOptimal NN Model Score: {NN_score_opt*100}%')
    print('Optimal Meta-Parameters: ')
    for pair in NN_features_opt.items():
        print(f'\t{pair[0]} = {pair[1]}')

    # Save trained model
    NN_Model_opt.save('Models/NN_model.h5')

if __name__ == "__main__":
    args = parse_args()
    main(args)