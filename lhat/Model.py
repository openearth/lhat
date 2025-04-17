#This modul contains all models used to create susceptibility/hazard maps
import os
import rasterio
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


class MachineLearning:
    '''
    Module contains routines required to run machine learning model and parameterise
    according to best generated accuracy scores. The model determines, based on its input
    data, the probability of a landslide occurring in a pixel.

    :param path: path to the saved model
    :param model_name: 'SVM', 'RF', 'LR'
    :param saveModel: True or False
    '''
    def __init__(self, X, y, pathToSavedModel, model_name = 'SVM', modelExist = False):

        saveModel = False
        if modelExist==False:
            saveModel = True

        self.model_name = model_name
        # path to save the file
        self.pathToSavedModel = pathToSavedModel

        if modelExist:
            self.loadMLModel()
        else:
            # predictor and target label
            self.X = X
            self.y = y

            # split X and y to train and test sets
            self.scaler = self.trainTestSplit()


            # train the selected model
            if self.model_name == 'SVM':
                self.bestModel = self.supportVectorMachine()
            elif self.model_name == 'RF':
                self.bestModel = self.randomForest()
            elif self.model_name == 'LR':
                self.bestModel = self.logisticRegression()
            else:
                'Model name is not correctly specified!'

            if saveModel:
                self.saveMLModel()


    def trainTestSplit(self):
        '''
        Splits the dataset into training and test sets. Test size is by default
        determined as 20% of the dataset.
        '''
        # Split and Scale the data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=101)

        scaler = StandardScaler()
        scaler.fit(X_train)

        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        return scaler


    def trainModel(self, baselineModel, modelParameters):

        '''
        Trains the machine learning model of choice and performs self parameterisation
        based on GridSearchCV. Parameters achieving the highest accuracy are selected.
        Parameters achieving the highest AUC are selected.

        :param baselineModel:
            This is the model that is used for training (based on choice by user, either
            Support Vector, Random Forest or Logistic Regression)

        :param modelParameters:
            This is a set of values for hyper parameters of the model that is used in cross-validation

        :return:
            The best model
        '''

        # Cross-Validation
        auc_scorer = make_scorer(roc_auc_score)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=baselineModel, param_grid=modelParameters,
                                   scoring=auc_scorer,
                                   cv=5, n_jobs=-1, verbose=1)

        # Fit the grid search to the data
        grid_search.fit(self.X_train, self.y_train)
        print ('best model parameters: \n', grid_search.best_params_)

        #Evaluate the best model
        best_model = grid_search.best_estimator_

        best_model_auc_train = self.evaluateTrainedModel(best_model, self.X_train, self.y_train, 'train')
        best_model_auc_test = self.evaluateTrainedModel(best_model, self.X_test, self.y_test, 'test')

        # best_model_accuracy_train = self.evaluateTrainedModel(best_model, self.X_train, self.y_train)
        # best_model_accuracy_test = self.evaluateTrainedModel(best_model, self.X_test, self.y_test)

        # print('best model accuracy on train set: \n', best_model_accuracy_train)
        # print('best model accuracy on test set: \n', best_model_accuracy_test)

        print('best model AUC on train set: \n', best_model_auc_train)
        print('best model AUC on test set: \n', best_model_auc_test)
        
        return best_model

    def evaluateTrainedModel(self, model, X , y, data_type):
        '''
        The trained model is evaluated for accuracy, AUC and a confusion matrix is made

        :param model:
            Machine learning model of choice paramterised by best performing in terms of accuracy

        :param X:
            Input training data to perform accuracy scoring on

        :param y:
            Actual landslide class (1 = landslide; 0 = no landslide) to compare with
            predicted class.

        :param data_type:
            str if train or test data provided, to save the ROC curve

        :return:
            Prints accuracy score, AUC score and confusion matrix
        '''
        predictions = model.predict(X) #fit to scaled object
        accuracy = accuracy_score(y, predictions)
        auc = roc_auc_score(y, predictions)
        confusion = pd.DataFrame(confusion_matrix(y, predictions))
        print('Model Performance')
        print('Accuracy Score = {}%'.format(accuracy * 100))
        print('AUC = {}%'.format(auc * 100))
        print('Confusion matrix = \n', confusion)

        # plot ROC curve from probabilities 
        y_pred_probas = model.predict_proba(X)[:,1]
        RocCurveDisplay.from_predictions(y, y_pred_probas)
        # save for training and for testing data
        plt.savefig(os.path.join(self.pathToSavedModel, self.model_name + f'_ROCcurve_{data_type}.png'))

        return auc  
    

    def logisticRegression(self):

        #Hyper-parameters
        model_parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                            'multi_class': ['auto'],
                            'C': [2, 1, 0.5]}

        #define the baseline model
        print('*********Logistic Regression********* \n')
        baselineModel = LogisticRegression(random_state=101)

        #train the model
        best_model = self.trainModel(baselineModel,model_parameters)

        return best_model

    def randomForest(self):

        # Hyper-parameters
        model_parameters = {
            'bootstrap': [True],
            'max_depth': [10, 20, 50, 100, 200],
            'max_features': [2],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [10, 50, 100, 200, 500, 1000]
        }

        # define the baseline model
        baselineModel = RandomForestClassifier(random_state=101)

        # train the model
        print('*********Random Forest Classifier********* \n')
        best_model = self.trainModel(baselineModel, model_parameters)

        return best_model

    def supportVectorMachine(self):

        from sklearn import svm

        # Hyper-parameters
        model_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                            'gamma':['auto','scale'],
                            'degree': [2, 3, 4],
                            'C': [2, 1, 0.5],
                             'probability': [True]}

        # define the baseline model
        baselineModel = svm.SVC(random_state=101)

        # train the model
        print ('*********Suport Vector Machine********* \n')
        best_model = self.trainModel(baselineModel, model_parameters)

        return best_model

    def saveMLModel(self):

        joblib.dump(self.bestModel, os.path.join(self.pathToSavedModel,self.model_name+'_bestModel.sav'))
        joblib.dump(self.scaler, os.path.join(self.pathToSavedModel, self.model_name + '_scaler.sav'))

    def loadMLModel(self):

        print ('load the saved model and scaler!')
        try:
            self.bestModel = joblib.load(os.path.join(self.pathToSavedModel,self.model_name+'_bestModel.sav'))
            self.scaler = joblib.load(os.path.join(self.pathToSavedModel, self.model_name + '_scaler.sav'))
        except:
            print('trained model or scaler do not exist! Check the output folder')

    def predict_proba(self, raster_stack, estimator, scaler, file_path, reference,
                      no_data):

        """Apply class probability prediction of a scikit learn model to a RasterStack.

        :param estimator:
            Estimator object implementing 'fit'. The object is used to fit the data.
        """

        n_classes = 2 # binary classification

        ref = rasterio.open(reference, lock = False)

        with rasterio.open(file_path, "w",
                           driver = 'GTiff',
                           height = ref.height,
                           width = ref.width,
                           count = n_classes,
                           dtype = np.float64,
                           crs = ref.crs,
                           transform = ref.transform,
                           nodata = no_data) as dst:
            # raster_stack (12, 2255, 4532)
            stack_arr = np.ma.MaskedArray(raster_stack, mask = ~np.isfinite(raster_stack), fill_value=-9999)
            result = self.probfun(stack_arr, estimator, scaler)
            result = np.ma.filled(result, fill_value = -9999)
            dst.write(result.astype(np.float64))
            dst.close()

        ref.close()
        # return
        return result

    def probfun(self, img, estimator, scaler):

        """Class probabilities function.

        :param img: A window object, and a 3d ndarray of raster data with the dimensions in
            order of (band, rows, columns).
        :type img: tuple (window, numpy.ndarray)

        :param estimator: The object to use to fit the data.
        :type estimator: Estimator object implementing the 'fit'

        :param scaler: e.g. StandardScaler
        :type scaler: scikit-learn scaler class

        :return: Multi-band raster as a 3d numpy array containing the probabilities
            associated with each class. ndarray dimensions are in the order of (class,
            row, column).
        :rtype: numpy.ndarray
        """

        # reshape each image block matrix into a 2D matrix
        # first reorder into rows, cols, bands (transpose)
        # then resample into 2D array (rows=sample_n, cols=band_values)
        # img is a masked array of shape (12, 2255, 4532)
        n_features, rows, cols = img.shape[0], img.shape[1], img.shape[2]  # columns of x, for categorical values we have multiple columns
        mask2d = img.mask.any(axis=0)  # mask of feature values
        n_samples = rows * cols
        flat_pixels = img.transpose(1, 2, 0).reshape((n_samples, n_features)) # tot pixels per feature
        flat_pixels = flat_pixels.filled(0)


        # predict probabilities, predict_proba takes shape (n_samples, n_features)
        result_proba = estimator.predict_proba(scaler.transform(flat_pixels))

        # reshape class probabilities back to 3D image [iclass, rows, cols]
        result_proba = result_proba.reshape((rows, cols, result_proba.shape[1]))

        # reshape band into rasterio format [band, row, col]
        result_proba = result_proba.transpose(2, 0, 1)

        # repeat mask for n_bands
        mask3d = np.repeat(
            a=mask2d[np.newaxis, :, :], repeats=result_proba.shape[0], axis=0
        )

        # convert proba to masked array
        result_proba = np.ma.masked_array(result_proba, mask=mask3d, fill_value=np.nan)

        return result_proba
