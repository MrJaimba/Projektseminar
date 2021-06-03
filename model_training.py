import time
import sqlite3
from sqlite3 import Error
import numpy as np
import pandas as pd
import data_modeling as dm
import machine_learning as ml
# import webscraper as ws
# import gui

global db_conncetion


def setup_database(path):
    db_connection = None
    try:
        db_connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful!")
    except Error as e:
        print(f"The error '{e}' occurred!")

    return db_connection


def training_pipeline():
    # Datumsstring für Zwischenspeichern von Files - für bessere Übersicht an csv-Namen anhängen
    datestr = time.strftime("%Y%m%d")

    # Set up database
    print("Step 1: Set up database...")

    db_connection = setup_database(r"Datenbank/ImmoDB.db")
    db_cursor = db_connection.cursor()

    # Read input data
    print("Step 2: Read in data...")

    # Zeilen zur Erzeugung der Datenbanktabellen auskommentiert
    # Spätere Verwaltung der DB mit dem Tool "DB Browser for SQLite" - https://sqlitebrowser.org/

    immonet_data = dm.read_data_from_immonet()
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection)
    # Alternative für später:
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection, if_exists = 'append oder replace')

    immoscout_data = dm.read_data_from_immoscout()
    # immoscout_data.to_sql(name='Immoscout_data_raw', con=db_connection)

    geo_data = dm.read_geo_data()
    # geo_data.to_sql(name='Geo_data_raw', con=db_connection)

    inhabitants_data = dm.read_data_from_inhabitants()
    # inhabitants_data.to_sql(name='Inhabitants_data_raw', con=db_connection)

    # Merge input data
    print("Step 3: Merge data...")

    immonet_data_geo_inh = dm.add_geo_inhabitants_immonet(immonet_data, geo_data, inhabitants_data)
    immoscout_data_geo_inh = dm.add_geo_inhabitants_immoscout(immoscout_data, geo_data, inhabitants_data)

    merged_data = dm.merge_data(immonet_data_geo_inh, immoscout_data_geo_inh)
    # merged_data.to_csv("Files/Tests/merged_data_" + datestr + ".csv", encoding = 'utf-8-sig')

    # Preprocessing
    print("Step 4: Preprocess data...")

    preprocessed_data = dm.preprocess_data(merged_data)
    # preprocessed_data.to_csv("Files/Tests/preprocessed_data_" + datestr + ".csv", encoding= 'utf-8-sig')

    # EDA_Grafiken
    # print("Step 5: EDA-Grafiken...")

    # eda(preprocessed_data)

    # Imputation
    print("Step 6: Impute data...")

    imputed_data = dm.impute_data(preprocessed_data)
    # imputed_data.to_csv("Files/Tests/imputed_data_" + datestr + ".csv", encoding= 'utf-8-sig')
    # imputed_data.to_excel("Files/Tests/imputed_data_" + datestr + ".xlsx")

    # DB Operations
    # imputed_data.to_sql(name='Imputed_Data_RAW', con=db_connection)

    # plz_einwohner = pd.read_excel("Files/Meta_Data/PLZ_Einwohnerzahlen.xlsx")
    # plz_einwohner['plz'] = plz_einwohner['plz'].astype(str)
    # plz_einwohner.to_sql(name='Meta_Data', con=db_connection, if_exists='replace')

    # plz_ort = pd.read_excel("Files/Meta_Data/PLZ_Ort.xls")
    # plz_ort['plz'] = plz_ort['plz'].astype(str)
    # plz_ort.drop(columns=['osm_id'])
    # plz_ort.to_sql(name='Meta_Data_ort', con=db_connection, if_exists='replace')

    # Machine Learning
    print("Step 7: Machine learning tests...")
    imputed_data = pd.read_sql_query('SELECT * FROM ML_Trainingsdaten_upd', db_connection, index_col='index')
    # Ausreisser bei imputed_data entfernen (Vorbereitung ML-Test)
    imputed_data = ml.outlier_drop(imputed_data)

    # Alle Kategorien mit JA/NEIN in 1/0 umwandeln
    imputed_data = ml.boolean(imputed_data)

    # PLZ als durchschnittlicher Angebotspreis pro plz und zimmergröße als variable
    # imputed_data = ml.variables(imputed_data)

    imputed_data.drop(columns=["plz"], inplace=True)
    # train_test_split
    x_test, x_train, y_test, y_train = ml.tr_te_spl(imputed_data)

    # Sample nur mit numerischen Variablen erzeugen
    x_train_num, x_val_num = ml.numeric(x_train, x_test)

    # Normalisierung der numerischen Daten (Wichtig Standardisierung auskommentieren)
    # x_train_num, x_val_num = ml.normalisation(x_train_num, x_val_num)

    # Standardisierung der numerischen Daten (Wichtig Normalisierung auskommentieren)
    # x_train_num, x_val_num = ml.standardization(x_train_num, x_val_num)

    # Sample mit nur kategorischen Variablen erzeugen (Mehr als zwei Kategorien)
    x_train_cat, x_val_cat = ml.category(x_train, x_test)

    # Target Encoding der kategorischen Variablen
    x_train_target, x_val_target = ml.target_encoding(x_train_cat, x_val_cat, y_train)

    # Zusammenführen kategorischer und numerischer Variablen + Speicherung unter Standardnamen
    x_train, x_test = ml.joint(x_train_num, x_train_target, x_val_num, x_val_target)

    # Anpassung an Anforderung von Random Forrest und Gradient Boosting -> float64 in float32 konvertieren
    cols_train = x_train.select_dtypes(include=[np.float64]).columns
    x_train[cols_train] = x_train[cols_train].astype(np.float32)
    cols_test = x_test.select_dtypes(include=[np.float64]).columns
    x_test[cols_test] = x_test[cols_test].astype(np.float32)

    # Grundstücksfläche von Wohnungen mit 0 auffüllen
    x_train.fillna(0, inplace=True)
    x_test.fillna(0, inplace=True)

    # Spalten sortieren
    x_train = x_train.reindex(sorted(x_train.columns), axis=1)
    x_test = x_test.reindex(sorted(x_test.columns), axis=1)

    x_train.to_sql(name='X_train', con=db_connection, if_exists='replace')
    x_test.to_sql(name='X_test', con=db_connection, if_exists='replace')
    # Durchführung der ML-Test
    ml.ml_tests(x_train, x_test, y_train, y_test, imputed_data)

    # Testausgaben
    # print("Optional: Create Excel files...")

    # merged_data.to_excel(excel_writer="Files/Tests/merged_data" + datestr + ".xlsx", sheet_name="Immobilien")

    print("... done.")


if __name__ == "__main__":
    training_pipeline()
