import pandas as pd
from Visualizer import Visualizer
from sklearn.preprocessing import StandardScaler
import os

pd.set_option('display.max_columns', None)

class DataHandler():

    def __generate_missing_df__(self):
        self.df_missing = self.df.isnull().sum()
        self.df_missing = self.df_missing[self.df_missing > 0]
        self.visualizer = Visualizer(self.df)        

    def __init__(self, filepath: str):        
        self.filepath = filepath
        if(filepath != None and os.path.isfile(filepath) and filepath.endswith('.csv')):            
            self.df = pd.read_csv(filepath)            
            self.__generate_missing_df__()
            self.scaler = StandardScaler()
        else:
            raise Exception('Filepath is not valid')
        

    def get_data(self):
        return self.df
    
    def inspect_data(self):
        print(self.df.head())
        print(self.df.describe())
        print(self.df.info())
        print("")
        print("Columns with missing values: ")
        print(self.df_missing)
        self.visualizer.plot_boxplots()
        self.visualizer.plot_correlation_heatmap()
        self.visualizer.plot_histograms()

    def clean_data(self):
        self.df.drop(columns=['CUST_ID'], inplace=True)
        self.df.fillna(self.df.mean(), inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.__generate_missing_df__()
        self.visualizer.df = self.df

    def standardize_data(self):        
        cols = self.df.columns
        self.df_scaled = self.scaler.fit_transform(self.df)
        self.df_scaled = pd.DataFrame(self.df_scaled, columns=cols)
        self.visualizer.df = self.df
        
if __name__ == '__main__':
    dh = DataHandler('./data/CC GENERAL.csv')
    dh.clean_data()
    dh.inspect_data()
    dh.plot_histograms()
    
    