import tensorflow as tf
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessing:
    
    def __init__(self,path):
        
        self.path = path

    
    def load_paymentData(self,sheet_name):
        data = pd.read_excel(self.path, sheet_name= sheet_name)
        data = data.groupby('Account Number', as_index=False).agg({
                            'sum':'sum', 'Mode ': lambda x: x.mode().iat[0]})
        return data
    
    def load_patternData(self,sheet_name):
        data = pd.read_excel(self.path, sheet_name= sheet_name)
        data.drop("Unnamed: 0", inplace=True, axis=1)
        data.rename(columns = {'Account Number ':'Account Number'}, inplace = True)
        data = data.groupby('Account Number', as_index=False).agg({'Used pattern in hours':'sum',
                                                                      'Usage Post Limit Utilization in hrs':'sum',
                                                                      'data used Gb':'sum'})
        return data
    
    def load_defaultsData(self,sheet_name):
        data = pd.read_excel(self.path, sheet_name= sheet_name)
        data = data.groupby('Account Number', as_index=False).agg({'Default Sum':'sum'})
    
        return data
    
    def load_callcentreData(self,sheet_name):
        data = pd.read_excel(self.path, sheet_name= sheet_name)
        data['Full Statement of the Enquiry'] = data['Full Statement of the Enquiry'].map(lambda x: x[9:])
        data.drop("Unnamed: 0", inplace=True, axis=1)
        data = data.groupby(['Account Number'], as_index = False).agg({'Full Statement of the Enquiry': ' '.join})
        return data
    
    def load_mainData(self,sheet_name,paymentData,patternData,defaultsData,callcentreData):
        data = pd.read_excel(self.path, sheet_name= sheet_name)
        data = pd.merge(data,paymentData, how = 'left', on = ['Account Number'])
        data = pd.merge(data,patternData, how = 'left', on = ['Account Number'])
        data = pd.merge(data,defaultsData, how = 'left', on = ['Account Number'])
        data = pd.merge(data,callcentreData, how = 'left', on = ['Account Number'])
        data.drop("Unnamed: 0", inplace=True, axis=1)
        return data
    
    def splitData(self,data):
        mask = data['Account Number'].str.contains("M")
        df_train = data[mask]
        df_val = data[~mask]
        return df_train,df_val
    
    def prepocessData(self,data,data_type):
        # Salary Column
        data['Salary Slab'] = pd.cut(data['Salary Slab'], bins=[1,4,7,10,14], labels=[1,2,3,4])
        data['Address'] = data['Address'].fillna('NOTKNOWN')
        data['Gender'] = data['Gender'].fillna('NOTKNOWN')
        data['Equipment Warranty'] = data['Equipment Warranty'].fillna('NOTKNOWN')
        data['Churn Date'] = data['Churn Date'].apply(lambda x: 1 if isinstance(x, pd.Timestamp) else 0)
        data['Used pattern in hours'] = data['Used pattern in hours'].fillna(0)
        data['Usage Post Limit Utilization in hrs'] = data['Usage Post Limit Utilization in hrs'].fillna(0)
        data['data used Gb'] = data['data used Gb'].fillna(0)
        data['Default Sum'] = data['Default Sum'].fillna(0)
        data['sum'] = data['sum'].fillna(0)
        data['Full Statement of the Enquiry'] = data['Full Statement of the Enquiry'].fillna('NOTKNOWN')
        data['Mode '] = data['Mode '].fillna('NOTKNOWN')
        data['Salary Slab'] = data['Salary Slab'].cat.add_categories(0).fillna(0)
        data['Age of Home'] = data['Age of Home'].fillna(0)
        data.drop(columns=['Equipment Warranty Expiry Date', 'Commence Date'], axis=1, inplace=True)
        data['Age'] = data['Age'].fillna(0)
        data.drop(columns=['Professional Info'], axis=1, inplace=True)
        
        
        return data
        
    def encodeData(self, data, category):
        # Encode Categorical Columns
        le = LabelEncoder()
        data[category] = data[category].apply(le.fit_transform)
        return data
    
    