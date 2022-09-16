import tensorflow as tf
import sklearn
import pandas as pd
from data_preprocessing import *
from text_preprocessing import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Flatten
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate#
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, plot_roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Train:
    
    def __init__(self):
        
        pass
    
    def print_grid_search_metrics(self,gs):
        print ("Best score: " + str(gs.best_score_))
        print ("Best parameters set:")
        best_parameters = gs.best_params_
        for param_name in sorted(best_parameters.keys()):
            print(param_name + ':' + str(best_parameters[param_name]))
    
    def plot_roc(self,fpr,tpr, threshold, model):
      plt.figure(1)
      plt.plot([0, 1], [0, 1], 'k--')
      plt.plot(fpr, tpr, label=model)
      plt.xlabel('False positive rate')
      plt.ylabel('True positive rate')
      plt.title('ROC curve - {0} Model'.format(model))
      plt.legend(loc='best')
      plt.show()
    
if __name__ == "__main__":
    train = Train()
    
    path = "C:/BA_Case_study.xlsx"
    
    category = ['Address','Scheme','Staus','Gender','Region ',' Sale of Equipment Status',
           'Equipment Warranty', 'Salary Slab', 'Mode ' ]
    
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    text_cleaning_re = "@\S+#$/-|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    embeddings_index = {}
    
    display_labels = ["Not_Churn", "Churn"]
    
    MAX_NB_WORDS = 100000
    MAX_SEQUENCE_LENGTH = 30
    GLOVE_EMB = './glove.6B.300d.txt'
    EMBEDDING_DIM = 300
    LR = 1e-3
    BATCH_SIZE = 1024
    EPOCHS = 10
    MODEL_PATH = './best_model.hdf5'
    
    data_pre = DataPreprocessing(path)
    text_pre = TextPreprocessing()
    
    df_payment = data_pre.load_paymentData("Payments Data")
    df_Pattern = data_pre.load_patternData("Use Pattern")
    df_Defaults = data_pre.load_defaultsData("Defaults data")
    df_CallCentre = data_pre.load_callcentreData("Call Centre Data")
    
    df_data = data_pre.load_mainData("Demographic Data",df_payment,
                                     df_Pattern,df_Defaults,df_CallCentre)
    
    # Spliting Data
    df_train, df_val = data_pre.splitData(df_data)
    
    # Preprocess Data
    df_train = data_pre.prepocessData(df_train,"train")
    df_val = data_pre.prepocessData(df_val,"val")
    
    # Encode Data
    df_train = data_pre.encodeData(df_train,category)
    df_val = data_pre.encodeData(df_val,category)
    
    df_train_num = df_train.drop(columns=['Account Number', 'Age', 
                                          'Full Statement of the Enquiry'], axis=1)
    df_val_num = df_val.drop(columns=['Account Number', 'Age', 
                                          'Full Statement of the Enquiry'], axis=1)
    
    # Train Data
    
    X_train = df_train_num.drop('Churn Date', axis=1)
    y_train = np.array(df_train_num['Churn Date'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                        test_size=0.2, random_state=41)
    
    # Validation Data
    X_val = df_val_num.drop('Churn Date', axis=1)
    y_val = np.array(df_val_num['Churn Date'])
    
    # Scaling Train data
    
    features = X_train.columns.values
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_train.columns = features
    
    print("The shape of X-train numerical data is ")
    print(X_train.shape)
     
    # Scaling Val data
    
    features_val = X_val.columns.values
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(X_val)
    X_val = pd.DataFrame(scaler.transform(X_val))
    X_val.columns = features_val
    
    rf_clf = RandomForestClassifier()
    lr_clf = LogisticRegression()
    knn_clf = KNeighborsClassifier()
    svc_clf = SVC()
    model_ls = [rf_clf, lr_clf, knn_clf,svc_clf]
    model_names = ['Random Forest','logistic Reg', 'KNN','SVC']
    
    for i in range(len(model_ls)):
        score = cross_val_score(model_ls[i], X_train, y_train, cv=5)
        print("Model: ",model_names[i], "Score: ", score)
        
        
    parameters = {
    'C' : [0.001,0.01, 1, 3 ,5],
    'kernel':['rbf','poly'],
    'degree': [2,3,4],  # degree of the polynominal kernel
    "probability":[True]
    }
    
    model = GridSearchCV(SVC(class_weight='balanced'),parameters, cv=2)
    model.fit(X_train, y_train)
    
    train.print_grid_search_metrics(model)
    best_SVC_model = model.best_estimator_
    
    y_proba_train = model.predict_proba(X_train)[:,1]
    y_proba_test = model.predict_proba(X_test)[:,1]
    y_proba_val = model.predict_proba(X_val)[:,1]
    
    y_pred_train = []
    y_pred_test = []
    y_pred_val = []
    
    for i in y_proba_train:
        if i>0.4:
            y_pred_train.append(1)
        else:
            y_pred_train.append(0)
    
    for i in y_proba_test:
        if i>0.4:
            y_pred_test.append(1)
        else:
            y_pred_test.append(0)
    
    
    print(" The accuracy score of Test Data is ")
    print(accuracy_score(y_test, y_pred_test))
    
    print(" The f1 score of of Test Data is ")
    f1 = f1_score(y_test, y_pred_test,average='weighted')
    print(f1)
    
    print(" The precision score of of Test Data is ")
    precision = precision_score(y_test, y_pred_test,average='weighted')
    print(precision)
    
    # evaluate model
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=5, n_jobs=-1)
    # summarize performance
    print('Mean ROC AUC: %.3f' % mean(scores))
    
    cm = confusion_matrix(y_train,y_pred_train)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels,
                cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
    cm = confusion_matrix(y_test,y_pred_test)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels,
                cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
    #run.log_image('Plot_confusion_train', plot=plt)
    fpr_RF, tpr_RF, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    print(threshold)
    ## Using self-define ROC Curve plot function
    train.plot_roc(fpr_RF, tpr_RF, threshold, 'SVC')
    print("AUC of RF",auc(fpr_RF,tpr_RF))
    
    predicted = model.predict_proba(X_test)[:,1]
    
    accuracy_ls = []
    
    for thresh in threshold:
        y_pred = np.where(predicted>thresh,1,0)
        
        accuracy_ls.append(accuracy_score(y_test,y_pred,normalize=True))
        
    accuracy_ls = pd.concat([pd.Series(threshold), pd.Series(accuracy_ls)],
                            axis=1)
    accuracy_ls.columns = ['thresholds','accuracy']
    accuracy_ls.sort_values(by='accuracy', ascending=False,inplace=True)
    print(accuracy_ls)
    
    

    
    
    
    
    
    
    