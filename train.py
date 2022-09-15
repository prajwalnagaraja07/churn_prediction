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
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

class Train:
    
    def __init__(self):
        
        pass
    
if __name__ == "__main__":
    
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
    LR = 1e-4
    BATCH_SIZE = 128
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
    df_train, df_valid = data_pre.splitData(df_data)
    
    # Preprocess Data
    df_train = data_pre.prepocessData(df_train,"train")
    df_val = data_pre.prepocessData(df_valid,"val")
    
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
    print(X_train.shape),
    
    # Scaling Val data
    
    features_val = X_val.columns.values
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(X_val)
    X_val = pd.DataFrame(scaler.transform(X_val))
    X_val.columns = features_val
    


    not_churn, churn = np.bincount(df_train_num['Churn Date'])
    total_count = len(df_train_num['Churn Date'])
    weight_no_claim = (1 / not_churn) * (total_count) / 2.0
    weight_claim = (1 / churn) * (total_count) / 2.0
    #class_weights = {0: 1, 1: 25}
    
    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(y_train),
                                            y = y_train                                                    
                                        )
    class_weights = dict(zip(np.unique(y_train), class_weights))
    
    # Model1 for numerical Data
    
    input_layer_m1 = Input(shape=(15,))
    Layer_1 = Dense(500, activation="relu")(input_layer_m1)
    Layer_2 = Dense(300, activation="relu")(Layer_1)
    Layer_3 = Dense(200, activation="relu")(Layer_2)
    flat1 = Flatten()(Layer_2)
    
    # LSTM for Text Classification
    # Training Data
    
    df_train_text = df_train.filter(['Full Statement of the Enquiry', 
                                     'Churn Date'], axis=1)
    df_train_text['Full Statement of the Enquiry'] = df_train_text['Full Statement of the Enquiry'].apply(lambda x: text_pre.preprocess(x,text_cleaning_re,stop_words,stemmer))
    
    df_train_text = df_train_text.rename(columns={'Full Statement of the Enquiry': 'text', 
                                                  'Churn Date': 'churn'})
    text = df_train_text.text
    
    X_train_text,word_index,vocab_size = text_pre.tokenize(text, MAX_SEQUENCE_LENGTH)
    
    X_train_text, X_test_text = train_test_split(X_train_text, 
                                                 test_size=0.2, random_state=41)
    
    # Validation Data
    
    df_val_text = df_val.filter(['Full Statement of the Enquiry', 
                                     'Churn Date'], axis=1)
    df_val_text['Full Statement of the Enquiry'] = df_val_text['Full Statement of the Enquiry'].apply(lambda x: text_pre.preprocess(x,text_cleaning_re,stop_words,stemmer))
    
    df_val_text = df_val_text.rename(columns={'Full Statement of the Enquiry': 'text', 
                                                  'Churn Date': 'churn'})
    val_text = df_val_text.text
    
    X_val_text, word_index_v, vocab_size_v = text_pre.tokenize(val_text, MAX_SEQUENCE_LENGTH)
    
    f = open(GLOVE_EMB, encoding="utf8")
    for line in f:
        values = line.split()
        word = value = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('Found %s word vectors.' %len(embeddings_index))
    
    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                          EMBEDDING_DIM,
                                          weights=[embedding_matrix],
                                          input_length=MAX_SEQUENCE_LENGTH,
                                          trainable=False)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Conv1D(64, 5, activation='relu')(x)
    x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    flat2 = Flatten()(x)
    
    merge = concatenate([flat1,flat2])
    hidden1 = Dense(200,activation='relu')(merge)
    hidden2 = Dense(100,activation='relu')(hidden1)
    hidden3 = Dense(50,activation='relu')(hidden2)
    output = Dense(1,activation='sigmoid')(hidden3)
    model = Model(inputs=[input_layer_m1,sequence_input],outputs=[output])
    
    model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',
              metrics=['accuracy'])
    ReduceLROnPlateau = ReduceLROnPlateau(factor=0.1,
                                         min_lr = 0.01,
                                         monitor = 'val_loss',
                                         verbose = 1)
    
    history = model.fit([X_train,X_train_text], y_train, batch_size=BATCH_SIZE, epochs=10,
                        validation_data=([X_test,X_test_text], y_test),
                        callbacks=[ReduceLROnPlateau],
                        class_weight=class_weights)
    
    # Predicting Train data
    y_pred_train = model.predict([X_train,X_train_text])
    
    # Unscaling the ypred value
    ypred_lis = []
    
    for i in y_pred_train:
        if i>0.6:
            ypred_lis.append(1)
        else:
            ypred_lis.append(0)
            
            
    #y_pred_train = np.argmax(y_pred_train,axis=1)
    
    
    # Predicting Test data
    y_pred = model.predict([X_test,X_test_text])
    
    ypredt_lis = []
    
    for i in y_pred:
        if i>0.6:
            ypredt_lis.append(1)
        else:
            ypredt_lis.append(0)

    
    keras.models.save_model(model,"churn_model.h5")
    
            
    # Accuracy of Train Dataset 
    accuracy = accuracy_score(y_train, ypred_lis)
    print(" The accuracy of of Train Data is ")
    print(accuracy)
    
    print(" The f1 score of of Train Data is ")
    f1 = f1_score(y_train, ypred_lis,average='weighted')
    print(f1)
    
    accuracy = accuracy_score(y_test, ypredt_lis)
    print(" The accuracy of of Test Data is ")
    print(accuracy)
    
    print(" The f1 score of of Test Data is ")
    f1 = f1_score(y_test, ypredt_lis,average='weighted')
    print(f1)
    
    print(" The precision score of of Test Data is ")
    precision = precision_score(y_test, ypredt_lis,average='weighted')
    print(precision)
    
    
    # PLotting acuracy curves
    fig_acc = plt.figure("Acuuracy plots")
    plt.plot(history.history["accuracy"]) 
    #plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","val"],loc ="upper left")
    
    cm = confusion_matrix(y_train,ypred_lis)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels,
                cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
    
    cm = confusion_matrix(y_test,ypredt_lis)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=display_labels, yticklabels=display_labels,
                cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)
    
    # Predicting Val data
    
    # Predicting Test data
    y_predv = model.predict([X_val,X_val_text])
    
    ypredv_lis = []
    
    for i in y_predv:
        if i>0.5:
            ypredv_lis.append(1)
        else:
            ypredv_lis.append(0)
    
    df_valid["Predicted Churn"]= ypredv_lis
    print(df_valid)
    
    for col in ['Address', 'Scheme', 'Staus', 'Gender', 'Region ',' Sale of Equipment Status','Equipment Warranty', 'Mode ',
           'Salary Slab']:
        df_valid[col] = df_valid[col].astype('category')
        
    _,axss = plt.subplots(2,2, figsize=[20,10])
    sns.countplot(x='Predicted Churn', hue='Salary Slab', data=df_valid, ax=axss[0][0])
    sns.countplot(x='Predicted Churn', hue='Address', data=df_valid, ax=axss[0][1])
    sns.countplot(x='Predicted Churn', hue='Staus', data=df_valid, ax=axss[1][0])
    sns.countplot(x='Predicted Churn', hue='Gender', data=df_valid, ax=axss[1][1])
    
    
    
    
    
    
    
    