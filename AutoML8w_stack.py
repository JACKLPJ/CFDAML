#!/usr/bin/env python
# coding: utf-8
# %%
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import VotingClassifier
#from sklearn.ensemble import StackingClassifier
import time
from CFDAML import SklearnModels as sm
from CFDAML import DataModeling8w as dm
#from multiprocessing import Pool
#import multiprocessing
#from functools import partial
from sklearn.preprocessing import StandardScaler
import os 
'''
X: The attribute characteristics of the dataset 
y: The label of the dataset
k: The number of nearest neighbors, int 
N: The top-N best models of each nearest neighbor, int
time_per_model: The time(s) limit for each model to train
data_pre_processing: Whether preprocessing dataset, bool
'''


class Automl:
    #time_per_model=360

    def __init__(
            self,
            data_pre_processing=False,
            system='linux',
            N_jobs = -1,
            verbose = False,
            time_per_model=360):#,
        
           # address_databases='./DataBases/')
        self.verbose = verbose
        self.n_estimators = 200
        self.scaler = StandardScaler()
        self.data_pre_processing = data_pre_processing
        self.k = 20
        self.N = 10
        path = os.path.realpath(os.curdir)#获取当前目录的绝对路径
      #  path = os.path.join(path, "1.txt")#加上文件名
        #print(path)
        self.address=os.path.join(path, "CFDAML\\DataBases\\")
        self.address_data_feats_featurized = self.address+'data_feats_featurized.csv'#address_data_feats_featurized
        self.address_pipeline = self.address+'pipelines.json'#address_pipeline
        self.address_Top50 = self.address+'datasetTop50.csv'#address_Top50
        self.DoEnsembel = True#DoEnsembel
        self.y = []
        self.time_per_model = time_per_model
        #sm.time_per_model = time1
        self.ensemble_clf = []
        self.N_jobs = N_jobs
        # if system=='linux':
        #     #multiprocessing.set_start_method('spawn',force=True)
        #     multiprocessing.set_start_method('forkserver',force=True)

    def pre_processing_X(self, X):
        col = list(X.columns)
        for j in col:
            if X[j].dtypes == 'object' or X[j].dtypes == 'O':
                b = X[j].unique()
                for i in range(len(b)):
                    X[j].loc[X[j] == b[i]] = i
                X[j] = X[j].astype("int")
        
        return X

    

    def fit(self, Xtrain, ytrain):
        X = Xtrain.copy(deep=True)
        y = ytrain.copy(deep=True)
        self.y = ytrain.copy(deep=True)

        preprocessing_dics, model_dics = dm.data_modeling(
            X, y, self.k, self.N, self.address_data_feats_featurized,
            self.address_pipeline, self.address_Top50, self.verbose).result  #_preprocessor
        # print('#######################################')
        n = len(preprocessing_dics)
        y = y.astype('int')
        accuracy = []
        great_models = []
        Y_hat = []
        model_name = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.25,
                                                        random_state=0)

        td = time.perf_counter()
        for i in range(n):
           # t_m=time.perf_counter()
            if model_dics[i][0] == 'xgradient_boosting':
                try:
                    Str, Clf, acc,y_hat_sub = sm.XGB(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1
            elif model_dics[i][0] == 'gradient_boosting':
                try:
                    Str, Clf, acc,y_hat_sub = sm.GradientBoosting(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'lda':
                try:
                    Str, Clf, acc,y_hat_sub = sm.LDA(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'extra_trees':
                try:
                    Str, Clf, acc,y_hat_sub = sm.ExtraTrees(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'random_forest':
                try:
                    Str, Clf, acc,y_hat_sub = sm.RandomForest(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'decision_tree':
                try:
                    Str, Clf, acc,y_hat_sub = sm.DecisionTree(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'libsvm_svc':
                try:
                    Str, Clf, acc,y_hat_sub = sm.SVM(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'k_nearest_neighbors':
                try:
                    Str, Clf, acc,y_hat_sub = sm.KNN(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'bernoulli_nb':
                try:
                    Str, Clf, acc,y_hat_sub = sm.BernoulliNB(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            elif model_dics[i][0] == 'multinomial_nb':
                try:
                    Str, Clf, acc,y_hat_sub = sm.MultinomialNB(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1

            else:
                try:
                    Str, Clf, acc,y_hat_sub = sm.QDA(X_train,
                                             X_test,
                                             y_train,
                                             y_test,
                                             model_dics[i],
                                             self.data_pre_processing,
                                             preprocessing_dics[i])
                except:
                    acc = -1
            if acc > -1:
                accuracy.append(acc)
                great_models.append(Clf)
                Y_hat.append(y_hat_sub)
                model_name.append(Str)

        
        Y_hat=np.array(Y_hat)
        
        sort_id0 = sorted(range(len(accuracy)),
                          key=lambda m: accuracy[m],
                          reverse=True)
        
        mean_acc = np.mean(accuracy)#np.median(accuracy)#
        #mean_f1 = np.mean(f1_scores)
        estimators_stacking = []  #[great_models[sort_id[0]]]
        #X_val_predictions = [all_results[sort_id[0]][-1]]
        id_n = len(sort_id0)
        id_i = 0
        base_acc_s = []  #[accuracy[sort_id[0]]]
        
        pre=[]
        while accuracy[sort_id0[id_i]] > mean_acc: 
            pre.append(sort_id0[id_i])
                
            id_i += 1
        
        Y_hat=Y_hat[pre]
        n_pre=len(Y_hat)   
        #Res_=[] 
        
        td = time.perf_counter()

        res_=[] 
        Sort=[]
        #pool = Pool()  
        for i in range(n_pre):
            aa=self.Sum_diff(i,n_pre,Y_hat)
            res_.append(aa[0])
            Sort.append(aa[1])
        
        c = sorted(range(len(Sort)), key=lambda k: Sort[k])
        res_ = np.array(res_)[c]
        
        Rubbish=set()
        
        final=[]
        for i in range(n_pre):
            if i not in Rubbish:
                final.append(pre[i])
                for k in range(len(res_[i])):
                    if res_[i][k] == 0: 
                        Rubbish.add(i+k+1)
        
        #print(final)
        if len(final)==1:
            self.DoEnsembel=False
        estimators_stacking=[great_models[i] for i in final]#.append(great_models[sort_id0[id_i]])
        base_acc_s=[accuracy[i] for i in final]#.append(accuracy[sort_id0[id_i]])
        
       # print(self.imbalance)#, fa)
        if self.verbose:
            print(id_n, len(base_acc_s))
            print(base_acc_s, mean_acc)
        #print(base_f1_s, mean_f1)
        from CFDAML.mlxtend.classifier import StackingClassifier
        #from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        if self.DoEnsembel:
            te = time.perf_counter()
            meta_clf = RandomForestClassifier(n_jobs=1,
                                              n_estimators=self.n_estimators)
            

            eclf_stacking = StackingClassifier(classifiers=estimators_stacking,
                                               meta_classifier=meta_clf,
                                               use_probas=True,
                                               preprocessing=self.data_pre_processing,
                                               fit_base_estimators=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            #accuracy.append(
            eclf_stacking = eclf_stacking.fit(X_train, y_train)
            self.ensemble_clf = [estimators_stacking, eclf_stacking]
            if self.verbose:
                print('Ensemble val score:',
                  accuracy_score(y_test, eclf_stacking.predict(X_test)))
            
                print('The time of ensemble is: {}'.format(time.perf_counter() -
                                                       te))
            
        else:

            self.clf = [model_name[sort_id0[0]], great_models[sort_id0[0]]]
            if self.verbose:
                print(self.clf)
            #allresult = [great_models[sort_id[0]], accuracy[sort_id[0]]]
            return self
        
    def Sum_diff(self,i,n,Y_hat):
        res=[]
        for j in range(i+1,n):
            res.append(np.sum(Y_hat[i]!=Y_hat[j])) 
        return [res,i]

    def predict(self, Xtest):
        X_Test = Xtest.copy(deep=True)
        X_Test = self.pre_processing_X(X_Test)
        if self.DoEnsembel:

            # X_test_predictions = self.scaler.transform(X_test_predictions)
            ypre = self.ensemble_clf[1].predict(X_Test)
        else:
            if self.clf[0] == 'mnb':
                from sklearn import preprocessing
                min_max_scaler = preprocessing.MinMaxScaler()
                X_Test = min_max_scaler.fit_transform(X_Test)
            if self.data_pre_processing:
                X_Test=sm.Preprocessing(X_Test, self.clf[1][1])
           # t = time.perf_counter()
            
            ypre = self.clf[1][0].predict(X_Test)
       # print(ypre)
        #print(self.y.dtypes)
        try:
            if self.y.iloc[:,0].dtypes == 'object' or self.y.iloc[:,0].dtypes == 'O':
                b = self.y.iloc[:,0].unique()
                return [b[i] for i in ypre]
        except:
            if self.y.dtypes == 'object' or self.y.dtypes == 'O':
                b = self.y.unique()
                return [b[i] for i in ypre]
        return ypre
