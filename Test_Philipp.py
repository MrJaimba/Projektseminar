import pandas as pd
import sqlite3

from xgboost import plot_importance

import machine_learning as ml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from streamlit import cli as stcli
db_connection = sqlite3.connect('Datenbank/ImmoDB.db')
#immobilienart = 'Villa'
#immobilienart_string = 'SELECT immobilienart_targetenc FROM Encoding_immobilienart WHERE immobilienart=\'' + immobilienart + '\''
#immobilienart = np.float32(pd.read_sql_query(immobilienart_string, con=db_connection).iloc[0][0])

#heizung = 'Sonstige'
#heizung_string = 'SELECT heizung_targetenc FROM Encoding_heizung WHERE heizung=\'' + heizung + '\''
#heizung = pd.read_sql_query(heizung_string, con=db_connection)

#immobilienzustand = 'Sonstige'
#immobilienzustand_string = 'SELECT immobilienzustand_targetenc FROM Encoding_immobilienzustand WHERE immobilienzustand=\'' + immobilienzustand + '\''
#immobilienzustand = pd.read_sql_query(immobilienzustand_string, con=db_connection)

#energietyp = 'Sonstige'
#energietyp_string = 'SELECT energietyp_targetenc FROM Encoding_energietyp WHERE energietyp=\'' + energietyp + '\''
#energietyp = pd.read_sql_query(energietyp_string, con=db_connection)

#energie_effizienzklasse = 'A'
#energie_effizienzklasse_string = 'SELECT energie_effizienzklasse_targetenc FROM Encoding_energie_effizienzklasse WHERE energie_effizienzklasse=\'' + energie_effizienzklasse + '\''
#energie_effizienzklasse = pd.read_sql_query(energie_effizienzklasse_string, con=db_connection)

#Meta_Daten_Beispiel = pd.read_sql_query('SELECT * FROM Meta_Data WHERE plz=97070', con=db_connection, index_col="index")

#verstädterung = Meta_Daten_Beispiel['Grad_der_Verstädterung'].to_list()[0]
#soziolage = Meta_Daten_Beispiel['sozioökonomische_Lage'].to_list()[0]

#verstädterung_string = 'SELECT Grad_der_Verstädterung_targetenc FROM Encoding_Grad_der_Verstädterung WHERE Grad_der_Verstädterung=\'' + verstädterung + '\''
#verstädterung = np.float32(pd.read_sql_query(verstädterung_string, con=db_connection).iloc[0][0])
#Meta_Daten_Beispiel['Grad_der_Verstädterung'] = verstädterung

#soziolage_string = 'SELECT sozioökonomische_Lage_targetenc FROM Encoding_sozioökonmische_Lage WHERE sozioökonomische_Lage=\'' + soziolage + '\''
#soziolage = pd.read_sql_query(soziolage_string, con=db_connection)

#Meta_Daten_Beispiel= Meta_Daten_Beispiel.assign(supermarkt_im_plz_gebiet=(Meta_Daten_Beispiel['Supermarkt im PLZ Gebiet'] == 'JA').astype(int))


#Meta_Daten_Beispiel = pd.read_sql_query('SELECT * FROM Meta_Data', con=db_connection, index_col="index")
#Meta_Daten_Beispiel.columns = Meta_Daten_Beispiel.columns.str.lstrip()
#Meta_Daten_Beispiel.columns = Meta_Daten_Beispiel.columns.str.rstrip()
#Meta_Daten_Beispiel.to_sql(name='Meta_Data', con=db_connection, if_exists='replace')
#Trainingsdaten = pd.read_sql_query('SELECT * FROM ML_Trainingsdaten', db_connection, index_col='index')
#Trainingsdaten.columns = Trainingsdaten.columns.str.lstrip()
#Trainingsdaten.columns = Trainingsdaten.columns.str.rstrip()
#Trainingsdaten.to_sql(name='ML_Trainingsdaten', con=db_connection, if_exists='replace')
#Trainingsdaten = ml.outlier_drop(Trainingsdaten)
#Trainingsdaten = ml.boolean(Trainingsdaten)
#Trainingsdaten = ml.variables(Trainingsdaten)
#Trainingsdaten.drop(columns=["plz"], inplace=True)
#Trainingsdaten = Trainingsdaten.astype({"angebotspreis":int})
#x_test, x_train, y_test, y_train = ml.tr_te_spl(Trainingsdaten)
#x_train_num, x_val_num = ml.numeric(x_train, x_test)
#x_train_num, x_val_num = ml.standardization(x_train_num, x_val_num)
#x_train_cat, x_val_cat = ml.category(x_train, x_test)
#x_train_target, x_val_target = ml.target_encoding(x_train_cat, x_val_cat, y_train)
#x_train, x_test = ml.joint(x_train_num, x_train_target, x_val_num, x_val_target)
#cols_train = x_train.select_dtypes(include=[np.float64]).columns
#x_train[cols_train] = x_train[cols_train].astype(np.float32)
#cols_test = x_test.select_dtypes(include=[np.float64]).columns
#x_test[cols_test] = x_test[cols_test].astype(np.float32)
#x_train.fillna(0, inplace=True)
#x_test.fillna(0, inplace=True)
#x_train = x_train.reindex(sorted(x_train.columns), axis=1)
#x_test = x_test.reindex(sorted(x_test.columns), axis=1)
#print(np.where(x_train.values >= np.finfo(np.float64).max))

#ml.ml_tests(x_train, x_test, y_train, y_test, Trainingsdaten)
#Metadaten = pd.read_excel(r'Files/Meta_Data/Metadaten Bayern 14.04.xlsx', skiprows=[0,1,2])
#Metadaten = Metadaten.dropna(axis='columns')
#Metadaten.drop(columns=["Hilfe Gemeindeschlüssel", "Hilfe Ort"], inplace=True)
#Metadaten.to_sql(name='Meta_Data', con=db_connection, if_exists='replace')
#Metadaten = Metadaten.astype({" Arbeitslosenquote in Prozent": float, " Anteil nicht erfolgreicher beruflicher Bildungsgänge" : float })
#Daten = pd.read_excel(r'Files/Tests/imputed_data_20210404.xlsx', index_col="Unnamed: 0")
#Daten = Daten.astype({"plz":int})
#Trainingsdaten = pd.merge(Daten, Metadaten, how="inner", on="plz")





if __name__ == "__main__":
 #model = pickle.load(open('XGB_Standardmodell, 'rb'))
 #plot_importance(model, max_num_features=10)
 #fig = plt.gcf()
 #fig.set_size_inches(17.5, 8 )
 #plt.savefig('Files/Feature_Importances_Grafiken/feature_importances.jpg')
 filename = 'GUI.py'
 sys.argv = ["streamlit", "run", filename]
 sys.exit(stcli.main())
 Test_Philipp.py







 #os.system('Streamlit run C:\\Users\\Philipp Höppner\\Desktop\\Projektseminar-master\\Projektseminar\\Files\\GUI\\' + filename)
 #print(immobilienart)
 #print(heizung)
 #print(immobilienzustand)
 #print(energietyp)
 #print(energie_effizienzklasse)
 ##print(Meta_Daten_Beispiel.columns)
 #print(verstädterung)
 #print(Meta_Daten_Beispiel['Grad_der_Verstädterung'])
 #print(soziolage)
 #print(soziolage)
 #print(Meta_Daten_Beispiel[['Grad der Verstädterung', 'sozioökonmische Lage']])
 #print(Trainingsdaten.columns)
 #print(Trainingsdaten.info())
 #print(x_train.info())
 #print(x_test.info())
 #print(y_train.describe)


