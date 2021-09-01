# Copyright 2018-2021 VMware, Inc.
# SPDX-License-Identifier: BSD-3-Clause

#!/usr/bin/env python3


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array


###############################################################################
###############################################################################

class RadeClassifier(BaseEstimator, ClassifierMixin):
    """
    A RADE classifier.

    An efficient classifier that augments either a Random Forest or an XGBoost classifier to
    obtain lower model memory size, lower training time and lower classification latency.

    The main building blocks of RADE are:

    Coarse-grained classifier - a small model that is trained using the entire training dataset.
    The coarse-grained classifier is sufficient to classify the majority of the classification queries correctly, such
    that a classification is valid only if its corresponding confidence level is greater than or equal to the
    classification confidence threshold.

    Fine-grained classifiers - 'expert' classifiers that are trained to succeed specifically where the coarse-grained model is not sufficiently
    confident and is more likely to make a classification mistake.


    Parameters
    ----------
    base_classifier : string, optional (default='RF')
        'RF' or 'XGB'.
        The classifier type of the coarse-grained and fine-grained classifiers.
        RADE supports Random-Forest ('RF') and XGBoost ('XGB').

    cg_params : dict or None, optional (default=None)
        If None, the classifier uses the defaults according to base_classifier.
        i.e., default_cg_params_RF = {'n_estimators': 10, 'max_depth': 5} for base_classifier='RF',
        or default_cg_params_XGB = {'n_estimators': 10, 'max_depth': 3} for base_classifier='XGB'.
        Parameters for the cg classifier.

    fg_normal_params : dict or None, optional (default=None)
        If None, the classifier uses the defaults according to base_classifier.
        i.e., default_fg_normal_params_RF = {'n_estimators': 25, 'max_depth': 20} for base_classifier='RF',
        or default_fg_normal_params_XGB = {'n_estimators': 30, 'max_depth': 3} for base_classifier='XGB'.
        Parameters for the fg normal classifier.

    fg_anomaly_params : dict or None, optional (default=None)
        If None, the classifier uses the defaults according to base_classifier.
        i.e., default_fg_anomaly_params_RF = {'n_estimators': 25, 'max_depth': 20}, for base_classifier='RF',
        or default_fg_anomaly_params_XGB = {'n_estimators': 30, 'max_depth': 3}, for base_classifier='XGB'.
        Parameters for the fg anomaly classifier.

    training_confidence_threshold : float, optional (default=None)
        If None, the classifier uses the defaults according to base_classifier.
        i.e., default_training_confidence_threshold_RF = 0.89 for base_classifier='RF',
        or default_training_confidence_threshold_XGB = 0.79 for base_classifier='XGB'.
        A value in [0,1].
        The training confidence threshold (TCT).

    classification_confidence_threshold : float, optional (default=None)
        If None, the classifier uses the defaults according to base_classifier.
        i.e., default_classification_confidence_threshold_RF = 0.79 for base_classifier='RF',
        or default_classification_confidence_threshold_XGB = 0.79 for base_classifier='XGB'.
        A value in [0,1].
        The classification confidence threshold (CCT).

    collect_telemetry : boolean, (default=False)
        If True, collect telemetry on the training_data_fraction of the fg normal and anomaly classifiers.
        See also telemetry_ attribute.

    random_seed : int, optional (default=42)
        Random seed.

    verbose : int, optional (default=0)
        If 0 prints exceptions only, if equal or bigger than 1 prints also warnings.

    Attributes
    ----------

    classes_ : array of shape (n_classes,) classes labels.

    cg_clf_ : Classifier
        The cg classifier, either Random Forest (base_classifier='RF') or XGBoost (base_classifier='XGB').

    fg_clf_normal_ : Classifier
        The fg normal classifier, either Random Forest (base_classifier='RF') or XGBoost (base_classifier='XGB').

    fg_clf_anomaly_ : Classifier
        The fg anomaly classifier, either Random Forest (base_classifier='RF') or XGBoost (base_classifier='XGB').

    cg_train_using_feature_subset : list or None, optional (default=None)
        List of columns to use for training the cg classifier (when None, all columns are used).

    cg_only_ : boolean
        True if only the cg classifier is fitted.

    telemetry_ : dict (if collect_telemetry is True)
        Contains the training_data_fraction of the fg normal and anomaly classifiers.

    Example program
    ---------------

    from rade_classifier import RadeClassifier
    from sklearn.datasets import make_classification
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000000, n_features=4,
                                n_informative=2, n_redundant=0,
                                random_state=0, shuffle=False, weights=[0.99, 0.01])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RadeClassifier()

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    print(classification_report(y_test, y_predicted, digits=5))

    Notes
    -----
    More details can be found in [1].
    See section 5.2 in order to tune RADE (e.g., by grid-search).

    References
    ----------
    [1] Shay Vargaftik, Isaac Keslassy, Ariel Orda, Yaniv Ben-Itzhak,
    "RADE: Resource-Efficient Supervised Anomaly Detection Using Decision Tree-Based Ensemble Methods"
    https://arxiv.org/abs/1909.11877

    """

    ###########################################################################
    ###########################################################################

    def __init__(self,

                 base_classifier='RF',
                 random_seed=42,

                 cg_params=None,

                 fg_normal_params=None,
                 fg_anomaly_params=None,

                 training_confidence_threshold=None,
                 classification_confidence_threshold=None,

                 # default configurations:
                 # RF:
                 default_training_confidence_threshold_RF=0.89,
                 default_classification_confidence_threshold_RF=0.79,

                 default_cg_params_RF=
                 {
                     'n_estimators': 10,
                     'max_depth': 5
                 },

                 default_fg_normal_params_RF=
                 {
                     'n_estimators': 25,
                     'max_depth': 20
                 },

                 default_fg_anomaly_params_RF=
                 {
                     'n_estimators': 25,
                     'max_depth': 20
                 },

                 # XGB:
                 default_training_confidence_threshold_XGB=0.79,
                 default_classification_confidence_threshold_XGB=0.79,

                 default_cg_params_XGB=
                 {
                     'n_estimators': 10,
                     'max_depth': 3
                 },

                 default_fg_normal_params_XGB=
                 {
                     'n_estimators': 30,
                     'max_depth': 3
                 },

                 default_fg_anomaly_params_XGB=
                 {
                     'n_estimators': 30,
                     'max_depth': 3
                 },

                 cg_train_using_feature_subset=None,

                 collect_telemetry=False,

                 verbose=0

                 ):

        self.base_classifier = base_classifier
        self.random_seed = random_seed

        self.cg_params = cg_params
        self.fg_normal_params = fg_normal_params
        self.fg_anomaly_params = fg_anomaly_params

        self.training_confidence_threshold = training_confidence_threshold
        self.classification_confidence_threshold = classification_confidence_threshold

        self.collect_telemetry = collect_telemetry

        self.cg_train_using_feature_subset = cg_train_using_feature_subset

        self.verbose = verbose

        ### RF defaults
        self.default_training_confidence_threshold_RF = default_training_confidence_threshold_RF
        self.default_classification_confidence_threshold_RF = default_classification_confidence_threshold_RF
        self.default_cg_params_RF = default_cg_params_RF
        self.default_fg_normal_params_RF = default_fg_normal_params_RF
        self.default_fg_anomaly_params_RF = default_fg_anomaly_params_RF

        ### XGBoost defaults
        self.default_training_confidence_threshold_XGB = default_training_confidence_threshold_XGB
        self.default_classification_confidence_threshold_XGB = default_classification_confidence_threshold_XGB
        self.default_cg_params_XGB = default_cg_params_XGB
        self.default_fg_normal_params_XGB = default_fg_normal_params_XGB
        self.default_fg_anomaly_params_XGB = default_fg_anomaly_params_XGB

    ###########################################################################
    ###########################################################################
    def verify_parameters(self, X, y):

        if self.classification_confidence_threshold and not self.training_confidence_threshold:
            if self.base_classifier == 'RF':
                if self.classification_confidence_threshold > self.default_training_confidence_threshold_RF:
                    if self.verbose > 0:
                        print(
                            "Warning: classification_confidence_threshold ({}) > "
                            "default_training_confidence_threshold_RF ({}).\n".
                                format(self.classification_confidence_threshold,
                                       self.default_training_confidence_threshold_RF))
            elif self.base_classifier == 'XGB':
                if self.classification_confidence_threshold > self.default_training_confidence_threshold_XGB:
                    if self.verbose > 0:
                        print(
                            "Warning: classification_confidence_threshold ({}) > "
                            "default_training_confidence_threshold_XGB ({}).\n".
                                format(self.classification_confidence_threshold,
                                       self.default_training_confidence_threshold_XGB))
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))

        elif not self.classification_confidence_threshold and self.training_confidence_threshold:
            if self.base_classifier == 'RF':
                if self.default_classification_confidence_threshold_RF > self.training_confidence_threshold:
                    if self.verbose > 0:
                        print(
                            "Warning: default_classification_confidence_threshold_RF ({}) > "
                            "training_confidence_threshold ({}).\n".
                                format(self.default_classification_confidence_threshold_RF,
                                       self.training_confidence_threshold))
            elif self.base_classifier == 'XGB':
                if self.default_classification_confidence_threshold_XGB > self.training_confidence_threshold:
                    if self.verbose > 0:
                        print(
                            "Warning: default_classification_confidence_threshold_XGB ({}) > "
                            "training_confidence_threshold ({}).\n".
                                format(self.default_classification_confidence_threshold_XGB,
                                       self.training_confidence_threshold))
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))

        elif self.classification_confidence_threshold and self.training_confidence_threshold:
            if self.classification_confidence_threshold > self.training_confidence_threshold:
                if self.verbose > 0:
                    print("Warning: classification_confidence_threshold ({}) > training_confidence_threshold ({}).\n".
                          format(self.classification_confidence_threshold, self.training_confidence_threshold))

        if self.cg_train_using_feature_subset is not None:
            ### empty is not allowed
            if not len(self.cg_train_using_feature_subset):
                raise Exception(
                    "Illegal cg_train_using_feature_subset (err1): {}\nShould be None or specify unique columns".format(
                        self.cg_train_using_feature_subset))

            ### duplicates are not allowed
            if len(self.cg_train_using_feature_subset) != len(set(self.cg_train_using_feature_subset)):
                raise Exception(
                    "Illegal cg_train_using_feature_subset (err2): {}\nShould be None or specify unique columns".format(
                        self.cg_train_using_feature_subset))

            ### translate column names (if X is a dataframe) to indices
            if isinstance(X, pd.DataFrame):
                if all(elem in X.columns for elem in self.cg_train_using_feature_subset):
                    self.cg_train_using_feature_subset = [X.columns.get_loc(i) for i in
                                                          self.cg_train_using_feature_subset]

            ### verify legal column values                          
            if not set(self.cg_train_using_feature_subset).issubset(set(range(X.shape[1]))):
                raise Exception(
                    "Illegal cg_train_using_feature_subset (err3): {}\nShould be None or specify unique columns".format(
                        self.cg_train_using_feature_subset))

    ###########################################################################
    ###########################################################################

    def fit(self, X, y):

        ### set numpy seed
        np.random.seed(self.random_seed)

        ### base classifier type options
        baseClassifierTypes = {

            'RF': RandomForestClassifier,
            'XGB': XGBClassifier

        }

        ## RADE parameters input checks
        self.verify_parameters(X, y)

        ### input verification - required by scikit         
        X, y = check_X_y(X, y)

        ### store the classes seen during fit - required by scikit
        self.classes_ = unique_labels(y)

        ### store the number of features passed to the fit method
        self.n_features_in_ = X.shape[1]

        ### binary classifier
        if len(self.classes_) >= 3:
            raise Exception("RADE is a binary classifier")

        ### collect telemetry
        if self.collect_telemetry:
            self.telemetry_ = {}
            self.telemetry_['normal_fg_training_data_fraction'] = 0
            self.telemetry_['anomaly_fg_training_data_fraction'] = 0

        ### init coarse-grained (cg) classifier
        self.cg_clf_ = baseClassifierTypes[self.base_classifier](random_state=self.random_seed)

        ### set cg params
        if self.cg_params is None:
            if self.verbose > 0:
                print("Warning: no kwards for the coarse-grained model. Use the default configuration.\n")
            if self.base_classifier == 'RF':
                self.cg_clf_.set_params(**self.default_cg_params_RF)
            elif self.base_classifier == 'XGB':
                self.cg_clf_.set_params(**self.default_cg_params_XGB)
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))
        else:
            self.cg_clf_.set_params(**self.cg_params)

        ### train cg 
        if self.cg_train_using_feature_subset == None:
            self.cg_clf_.fit(X, y)
        else:
            self.cg_clf_.fit(X[:, self.cg_train_using_feature_subset], y)

        ### tags
        try:
            self.__normal_tag_ = np.min(self.classes_)
            self.__anomaly_tag_ = np.max(self.classes_)
        except:
            self.__normal_tag_ = self.classes_[0]
            self.__anomaly_tag_ = self.classes_[1]

        ### single class            
        if self.__normal_tag_ == self.__anomaly_tag_:
            self.cg_only_ = True
            if self.verbose > 0:
                print("Warning: received only a single class for training, no fg models.\n")
            return self
        else:
            self.cg_only_ = False

        ### init fine-grained (fg) classifiers
        self.fg_clf_normal_ = baseClassifierTypes[self.base_classifier](random_state=self.random_seed)
        self.fg_clf_anomaly_ = baseClassifierTypes[self.base_classifier](random_state=self.random_seed)

        ### set fg normal params
        if self.fg_normal_params is None:
            if self.verbose > 0:
                print("Warning: no kwards for the fine-grained normal model. Use the default configuration.\n")
            if self.base_classifier == 'RF':
                self.fg_clf_normal_.set_params(**self.default_fg_normal_params_RF)
            elif self.base_classifier == 'XGB':
                self.fg_clf_normal_.set_params(**self.default_fg_normal_params_XGB)
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))
        else:
            self.fg_clf_normal_.set_params(**self.fg_normal_params)

        ### set fg anomaly params
        if self.fg_anomaly_params is None:
            if self.verbose > 0:
                print("Warning: no kwards for the fine-grained anomaly model. Use the default configuration.\n")
            if self.base_classifier == 'RF':
                self.fg_clf_anomaly_.set_params(**self.default_fg_anomaly_params_RF)
            elif self.base_classifier == 'XGB':
                self.fg_clf_anomaly_.set_params(**self.default_fg_anomaly_params_XGB)
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))
        else:
            self.fg_clf_anomaly_.set_params(**self.fg_anomaly_params)

        ### classify training data by cg to obtain metadata
        if self.cg_train_using_feature_subset == None:
            cg_classification_distribution = self.cg_clf_.predict_proba(X)
        else:
            cg_classification_distribution = self.cg_clf_.predict_proba(X[:, self.cg_train_using_feature_subset])

        cg_classification = np.take(self.classes_, np.argmax(cg_classification_distribution, axis=1))
        cg_classification_confidence = np.max(cg_classification_distribution, axis=1)

        ### prepare train data filters
        if self.training_confidence_threshold is None:
            if self.base_classifier == 'RF':
                cg_low_confidence_indeces = (cg_classification_confidence <
                                             self.default_training_confidence_threshold_RF)
            elif self.base_classifier == 'XGB':
                cg_low_confidence_indeces = (
                        cg_classification_confidence < self.default_training_confidence_threshold_XGB)
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))
        else:
            cg_low_confidence_indeces = (cg_classification_confidence < self.training_confidence_threshold)

        true_anomaly_indeces = (y == self.__anomaly_tag_)
        cg_normal_classification_indeces = (cg_classification == self.__normal_tag_)
        cg_anomaly_classifications_indeces = (cg_classification == self.__anomaly_tag_)

        ### training data for fg models 
        fg_normal_training_data_filter = cg_low_confidence_indeces & (
                true_anomaly_indeces | cg_normal_classification_indeces)
        fg_normal_training_data_X = X[fg_normal_training_data_filter]
        fg_normal_training_data_y = y[fg_normal_training_data_filter]

        fg_anomaly_training_data_filter = cg_low_confidence_indeces & (
                true_anomaly_indeces | cg_anomaly_classifications_indeces)
        fg_anomaly_training_data_X = X[fg_anomaly_training_data_filter]
        fg_anomaly_training_data_y = y[fg_anomaly_training_data_filter]

        ### train the fg models
        if len(unique_labels(fg_normal_training_data_y)) == 2 and sum(fg_normal_training_data_filter) > 1:
            ### collect telemetry
            if self.collect_telemetry:
                self.telemetry_['normal_fg_training_data_fraction'] = sum(fg_normal_training_data_filter) / len(
                    fg_normal_training_data_filter)
                ### train
            self.fg_clf_normal_.fit(fg_normal_training_data_X, fg_normal_training_data_y)
            self.fg_normal_fitted_ = True
        else:
            if self.verbose > 0:
                print("Warning: no fine-grained normal model training.\n")
            self.fg_normal_fitted_ = False

        if len(unique_labels(fg_anomaly_training_data_y)) == 2 and sum(fg_anomaly_training_data_filter) > 1:
            ### collect telemetry
            if self.collect_telemetry:
                self.telemetry_['anomaly_fg_training_data_fraction'] = sum(fg_anomaly_training_data_filter) / len(
                    fg_anomaly_training_data_filter)
            ### train
            self.fg_clf_anomaly_.fit(fg_anomaly_training_data_X, fg_anomaly_training_data_y)
            self.fg_anomaly_fitted_ = True
        else:
            if self.verbose > 0:
                print("Warning: no fine-grained anomaly model training.\n")
            self.fg_anomaly_fitted_ = False

        ### for speed       
        if not self.fg_normal_fitted_ and not self.fg_anomaly_fitted_:
            self.cg_only_ = True

        ### a call to fit should return the classifier - required by scikit
        return self

    ###########################################################################
    ###########################################################################

    def predict_basic(self, X, proba=False):

        ### set numpy seed
        np.random.seed(self.random_seed)

        ### check is that fit had been called - required by scikit
        check_is_fitted(self)

        ### input verification - required by scikit
        X = check_array(X)

        ### collect telemetry
        if self.collect_telemetry:
            self.telemetry_['normal_fg_test_data_fraction'] = 0
            self.telemetry_['anomaly_fg_test_data_fraction'] = 0

        ### no fg models?
        if self.cg_only_:
            if not proba:
                if self.cg_train_using_feature_subset == None:
                    return self.cg_clf_.predict(X)
                else:
                    return self.cg_clf_.predict(X[:, self.cg_train_using_feature_subset])
            else:
                if self.cg_train_using_feature_subset == None:
                    return self.cg_clf_.predict_proba(X)
                else:
                    return self.cg_clf_.predict_proba(X[:, self.cg_train_using_feature_subset])

        ### classify test data by cg to obtain metadata        
        if self.cg_train_using_feature_subset == None:
            cg_classification_distribution = self.cg_clf_.predict_proba(X)
        else:
            cg_classification_distribution = self.cg_clf_.predict_proba(X[:, self.cg_train_using_feature_subset])

        cg_classification = np.take(self.classes_, np.argmax(cg_classification_distribution, axis=1))
        cg_classification_confidence = np.max(cg_classification_distribution, axis=1)

        ### prepare test data filters
        if self.classification_confidence_threshold is None:
            if self.base_classifier == 'RF':
                cg_low_confidence_indeces = (cg_classification_confidence <
                                             self.default_classification_confidence_threshold_RF)
            elif self.base_classifier == 'XGB':
                cg_low_confidence_indeces = (cg_classification_confidence <
                                             self.default_classification_confidence_threshold_XGB)
            else:
                raise Exception('Unsupported base_classifier {}'.format(self.base_classifier))
        else:
            cg_low_confidence_indeces = (cg_classification_confidence < self.classification_confidence_threshold)

        normal_cg_classification_indeces = (cg_classification == self.__normal_tag_)
        anomaly_cg_classifications_indeces = (cg_classification == self.__anomaly_tag_)

        ### test data for fg models
        fg_normal_test_data_filter = cg_low_confidence_indeces & normal_cg_classification_indeces
        fg_normal_test_data = X[fg_normal_test_data_filter]

        fg_anomaly_test_data_filter = cg_low_confidence_indeces & anomaly_cg_classifications_indeces
        fg_anomaly_test_data = X[fg_anomaly_test_data_filter]

        ### predict
        if not proba:

            classification_results = cg_classification

            if self.fg_normal_fitted_ and np.any(fg_normal_test_data_filter):

                ### collect telemetry
                if self.collect_telemetry:
                    self.telemetry_['normal_fg_test_data_fraction'] = sum(fg_normal_test_data_filter) / len(
                        fg_normal_test_data_filter)

                classification_results[fg_normal_test_data_filter] = self.fg_clf_normal_.predict(fg_normal_test_data)

            if self.fg_anomaly_fitted_ and np.any(fg_anomaly_test_data_filter):

                ### collect telemetry
                if self.collect_telemetry:
                    self.telemetry_['anomaly_fg_test_data_fraction'] = sum(fg_anomaly_test_data_filter) / len(
                        fg_anomaly_test_data_filter)

                classification_results[fg_anomaly_test_data_filter] = self.fg_clf_anomaly_.predict(fg_anomaly_test_data)

            return classification_results

        ### predict proba
        else:

            classification_distribution_results = cg_classification_distribution

            if self.fg_normal_fitted_ and np.any(fg_normal_test_data_filter):

                ### collect telemetry
                if self.collect_telemetry:
                    self.telemetry_['normal_fg_test_data_fraction'] = sum(fg_normal_test_data_filter) / len(
                        fg_normal_test_data_filter)

                classification_distribution_results[fg_normal_test_data_filter] = self.fg_clf_normal_.predict_proba(
                    fg_normal_test_data)

            if self.fg_anomaly_fitted_ and np.any(fg_anomaly_test_data_filter):

                ### collect telemetry
                if self.collect_telemetry:
                    self.telemetry_['anomaly_fg_test_data_fraction'] = sum(fg_anomaly_test_data_filter) / len(
                        fg_anomaly_test_data_filter)

                classification_distribution_results[fg_anomaly_test_data_filter] = self.fg_clf_anomaly_.predict_proba(
                    fg_anomaly_test_data)

            return classification_distribution_results

            ###########################################################################

    ###########################################################################

    def predict(self, X):

        return self.predict_basic(X)

    def predict_proba(self, X):

        return self.predict_basic(X, proba=True)

    ###########################################################################
    ########################################################################### 

    ### getters

    def get_telemetry(self):

        try:
            return self.telemetry_
        except:
            print("\nError: get_telemetry was called but telemetry is disabled.\n")

    def get_sub_classifier(self, clf):

        if clf == 'cg':
            return self.cg_clf_

        elif clf == 'fg_normal':
            if self.fg_normal_fitted_:
                return self.fg_clf_normal_
            else:
                return None

        elif clf == 'fg_anomaly':
            if self.fg_anomaly_fitted_:
                return self.fg_clf_anomaly_
            else:
                return None

        else:
            raise Exception("unknown sub-classifier type, possible options are: cg / fg_normal / fg_anomaly")

    ###########################################################################
    ########################################################################### 
    def _more_tags(self):

        return {'binary_only': True}

    ###########################################################################
    ###########################################################################
