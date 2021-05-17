import pickle

import pandas as pd
import sqlite3
from sqlite3 import Error

import model_training
db_connection = sqlite3.connect('Datenbank/ImmoDB.db')
input_df = pd.read_sql_query('SELECT * FROM Meta_Data_upd', con=db_connection, index_col="index")
input_df.drop_duplicates(subset=['plz'], inplace=True)
input_df.to_sql(name='Meta_Data_upd2', con=db_connection, if_exists='replace')

#load_XGB_modell = pickle.load(open('XGB_Standardmodell_20210421-1620.pckl', 'rb'))
#output = int(load_XGB_modell.predict(input_df)[0])
if __name__ == "__main__":
   print(input_df)