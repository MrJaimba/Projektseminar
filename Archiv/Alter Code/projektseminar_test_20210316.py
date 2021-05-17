import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import xgboost as xgb
import sqlite3

from sqlite3 import Error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def read_data_from_immonet():
    immonet_data = pd.read_excel(r"../../Files/Input_Data/Immonet_Bayern_31032021.xlsx", sheet_name="Tabelle2")

    return immonet_data


def read_data_from_immoscout():
    immoscout_data_haeuser = pd.read_excel(r"../../Files/Input_Data/Immoscout_Bayern_Häuser_17032021.xlsx",
                                           sheet_name="Tabelle3")
    immoscout_data_wohnungen = pd.read_excel(r"../../Files/Input_Data/Immoscout_Bayern_Wohnungen_17032021.xlsx", sheet_name="Tabelle2")

    immoscout_data = pd.concat([immoscout_data_haeuser, immoscout_data_wohnungen], axis=0, ignore_index=True)

    return immoscout_data


def read_geo_data():
    # Datensatz mit Koordinaten von Timo
    geo_data = pd.read_excel(r'../../Files/Input_Data/PLZ_Geodaten.xlsx', sheet_name='PLZ')
    return geo_data


def read_data_from_inhabitants():
    # Datensatz mit Einwohnern von Yanina
    inhabitants = pd.read_excel(r'../../Files/Input_Data/PLZ_Einwohnerzahlen.xlsx', sheet_name='Tabelle2')
    return inhabitants


def add_geo_inhabitants_immonet(immonet_data, geo_data, inhabitants):
    # Koordinaten und Einwohner auf Immonet-Daten anpassen

    immonet_data = immonet_data.dropna(subset=['plz'])
    immonet_data['plz'] = immonet_data['plz'].astype(int)
    immonet_data['plz'] = immonet_data['plz'].astype(str)

    geo_data = geo_data.astype(str)
    list_plz_immonet = immonet_data['plz']

    dict_breitengrad_immonet = dict(zip(geo_data['PLZ'], geo_data['Breitengrad']))
    list_breitengrad_immonet = [dict_breitengrad_immonet.get(key) for key in list_plz_immonet]

    dict_laengengrad_immonet = dict(zip(geo_data['PLZ'], geo_data['Längengrad']))
    list_laengengrad_immonet = [dict_laengengrad_immonet.get(key) for key in list_plz_immonet]

    inhabitants = inhabitants.astype(str)
    dict_einwohner_immonet = dict(zip(inhabitants['plz'], inhabitants['einwohner']))
    list_einwohner_immonet = [dict_einwohner_immonet.get(key) for key in list_plz_immonet]

    immonet_data['breitengrad'] = list_breitengrad_immonet
    immonet_data['laengengrad'] = list_laengengrad_immonet
    immonet_data['einwohner'] = list_einwohner_immonet

    immonet_data = immonet_data.dropna(subset=['breitengrad'])
    immonet_data = immonet_data.dropna(subset=['laengengrad'])
    immonet_data = immonet_data.dropna(subset=['einwohner'])

    immonet_data_new = immonet_data

    return immonet_data_new


def add_geo_inhabitants_immoscout(immoscout_data, geo_data, inhabitants):
    # Koordinaten und Einwohner auf Immonet - Daten anpassen
    geo_data = geo_data.astype(str)
    immoscout_data["plz"] = immoscout_data["PLZ und Ort"].astype(str).apply(lambda row: row[:5])
    list_plz_immoscout = immoscout_data['plz']

    dict_breitengrad_immoscout = dict(zip(geo_data['PLZ'], geo_data['Breitengrad']))
    list_breitengrad_immoscout = [dict_breitengrad_immoscout.get(key) for key in list_plz_immoscout]

    dict_laengengrad_immoscout = dict(zip(geo_data['PLZ'], geo_data['Längengrad']))
    list_laengengrad_immoscout = [dict_laengengrad_immoscout.get(key) for key in list_plz_immoscout]

    inhabitants = inhabitants.astype(str)
    dict_einwohner_immoscout = dict(zip(inhabitants['plz'], inhabitants['einwohner']))
    list_einwohner_immoscout = [dict_einwohner_immoscout.get(key) for key in list_plz_immoscout]

    immoscout_data['breitengrad'] = list_breitengrad_immoscout
    immoscout_data['laengengrad'] = list_laengengrad_immoscout
    immoscout_data['einwohner'] = list_einwohner_immoscout

    immoscout_data = immoscout_data.dropna(subset=['breitengrad'])
    immoscout_data = immoscout_data.dropna(subset=['laengengrad'])
    immoscout_data = immoscout_data.dropna(subset=['einwohner'])

    immoscout_data_new = immoscout_data

    return immoscout_data_new


def merge_data(immonet_data_new, immoscout_data_new):

    # Immoscout Format an Immonet Format anpassen:
    immoscout_data_new.columns = immoscout_data_new.columns.str.lower()

    immoscout_data_new = immoscout_data_new.drop(columns=["plz und ort", "web-scraper-order"])

    immoscout_data_new.rename(
        columns={"anzahl badezimmer": "anzahl_badezimmer", "anzahl schlafzimmer": "anzahl_schlafzimmer",
                 "zimmer": "anzahl_zimmer", "einkaufspreis": "angebotspreis",
                 "balkon/ terrasse": "terrasse_balkon", "wohnfläche": "wohnflaeche", "etage": "geschoss",
                 "grundstück": "grundstuecksflaeche", "stufenloser zugang": "barrierefrei",
                 "objektzustand": "immobilienzustand",
                 "keller ja/nein": "unterkellert", "gäste-wc ja/nein": "gaeste_wc",
                 "energie­effizienz­klasse": "energie_effizienzklasse",
                 "wesentliche energieträger": "befeuerungsart", "end­energie­verbrauch": "energie_verbrauch",
                 "typ": "immobilienart", "heizungsart": "heizung", "vermietet ja/nein": "vermietet",
                 "garage/ stellplatz": "anzahl_parkplatz"}, inplace=True)

    # Spalteninhalte anpassen:
    # Annahme NaN ist NEIN

    immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse'] + '' + immonet_data_new['balkon']
    immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse_balkon'].apply(
        lambda row: 'JA' if 'JA' in row else 'NEIN')
    immonet_data_new = immonet_data_new.drop(columns=['terrasse', 'balkon'])

    immonet_data_new['anzahl_badezimmer'] = immonet_data_new["anzahl_badezimmer"].apply(
        lambda row: '0' if row == 'E' else row)
    immonet_data_new["anzahl_badezimmer"] = immonet_data_new["anzahl_badezimmer"].astype(int)

    immoscout_data_new["aufzug"] = immoscout_data_new["aufzug"].astype(str).apply(
        lambda row: "JA" if row == "Personenaufzug" else "NEIN")

    immoscout_data_new["terrasse_balkon"] = immoscout_data_new["terrasse_balkon"].astype(str).apply(
        lambda row: "JA" if "Balkon" in row else "NEIN")

    immoscout_data_new["unterkellert"] = immoscout_data_new["unterkellert"].apply(
        lambda row: "JA" if row == "keller" else "NEIN")

    immoscout_data_new["gaeste_wc"] = immoscout_data_new["gaeste_wc"].apply(
        lambda row: "JA" if row == "Gäste-WC" else "NEIN")

    immoscout_data_new["barrierefrei"] = immoscout_data_new["barrierefrei"].apply(
        lambda row: "JA" if row == 'Stufenloser Zugang' else "NEIN")

    immoscout_data_new["baujahr"] = immoscout_data_new["baujahr"].apply(
        lambda row: re.sub('[\\D]', '', str(row)))
    immoscout_data_new["baujahr"] = pd.to_numeric(immoscout_data_new["baujahr"])

    immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new["grundstuecksflaeche"].astype(str).apply(
        lambda row: re.sub('[.m²]', '', row))
    immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new["grundstuecksflaeche"].apply(
        lambda x: x.replace('.', ''))
    immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new["grundstuecksflaeche"].apply(
        lambda x: x.replace(',', '.'))
    immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new["grundstuecksflaeche"].astype(float)

    immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].astype(str).apply(
        lambda row: re.sub('[m²]', '', row))
    immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].apply(
        lambda x: x.replace('.', ''))
    immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].apply(
        lambda x: x.replace(',', '.'))
    immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].astype(float)

    immoscout_data_new["vermietet"] = immoscout_data_new["vermietet"].astype(str).apply(
        lambda row: "JA" if row == "Vermietet" else "NEIN")

    immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].fillna(0)
    immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].apply(
        lambda row: re.sub('[\\D]', '', str(row)))
    immoscout_data_new["anzahl_parkplatz"] = pd.to_numeric(immoscout_data_new["anzahl_parkplatz"])
    immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].fillna(1)

    immoscout_data_new["energie_verbrauch"] = immoscout_data_new["energie_verbrauch"].apply(
        lambda row: re.sub('[^0-9,]', '', str(row)))
    immoscout_data_new["energie_verbrauch"] = immoscout_data_new["energie_verbrauch"].apply(
        lambda row: re.sub(',', '.', str(row)))
    immoscout_data_new["energie_verbrauch"] = pd.to_numeric(immoscout_data_new["energie_verbrauch"])

    # Spalten alphabetisch sortieren
    immonet_data = immonet_data_new.reindex(sorted(immonet_data_new.columns), axis=1)
    immoscout_data = immoscout_data_new.reindex(sorted(immoscout_data_new.columns), axis=1)

    # Innerjoin reicht hier aus
    merged_data = pd.concat([immoscout_data_new, immonet_data_new], axis=0, ignore_index=True, join="inner")

    # Duplikate droppen
    merged_data = merged_data.drop_duplicates(subset=['wohnflaeche', 'grundstuecksflaeche', 'anzahl_zimmer'])

    return merged_data


def eda(data):

    # Differenzierung von numerischen und kategorischen Variablen
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    # numeric_data = numeric_data.drop(columns=["angebotspreis"])

    categoric_data = data.select_dtypes(include=['object', 'category'])
    categoric_data = categoric_data.drop(columns=["breitengrad", "laengengrad", "plz"])

    # Overview
    print(data.info())
    print(data.describe())

    # Null values
    count = round(data.isnull().sum(), 2)
    percent = round((data.isnull().sum() / data.shape[0]) * 100, 2)
    null_values_eda = pd.concat([count, percent], axis=1)
    null_values_eda.reset_index(inplace=True)
    null_values_eda.rename(columns={0: 'Missing Values Count', 1: 'Missing Values %'}, inplace=True)
    # data = data[data['Missing Values Count'] != 0]
    print(null_values_eda)

    # Angebotspreis Histogramm
    sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
    sns.histplot(data['angebotspreis'], stat='count', bins='auto').set(xlabel='Angebotspreis', ylabel='Anzahl')
    plt.ticklabel_format(style="plain")
    plt.title("Angebotspreis Histogramm")
    plt.savefig(r"Files/EDA_Grafiken/Angebotspreis_Histogram")
    plt.clf()

    # Korrelation numerische Variablen
    corrMatrix = numeric_data.corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.savefig(r"Files/EDA_Grafiken/Heatmap_Correlation")
    plt.clf()

    # Übersicht: Boxplots + Histogramme für numerische Variablen
    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20, 90))
    fig.subplots_adjust(hspace=.8, wspace=.3)
    i = 0
    for col in numeric_data.columns:
        # sns.distplot erzeugt Future Warnings, da die Methode ersetzt wurde - funktioniert aber noch
        sns.distplot(numeric_data[col], ax=axes[i][0]).set_title("Histogram of " + col)
        sns.boxplot(numeric_data[col], ax=axes[i][1]).set_title("Boxplot of " + col)
        i = i + 1
    plt.savefig(r"Files/EDA_Grafiken/Numerics_Boxplots_Histograms")
    plt.clf()

    # Countplots für kategorische Variablen
    CatFacetGrid = sns.FacetGrid(categoric_data.melt(), col='variable', sharex=False, dropna=True, sharey=False, height=4,
                                 col_wrap=4)
    CatFacetGrid.set_xticklabels(rotation=90)
    CatFacetGrid.map(sns.countplot, 'value')
    plt.savefig(r"Files/EDA_Grafiken/Countplots_Categories")
    plt.clf()

    # Angebotspreis und kategorische Variablen
    fig, ax = plt.subplots(10, 1, figsize=(20, 200))
    for var, subplot in zip(categoric_data, ax.flatten()):
        ax = sns.boxplot(x=var, y='angebotspreis', data=data, ax=subplot)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.savefig(r"Files/EDA_Grafiken/Relations_Angebotspreis_Categories")
    plt.clf()

    print("...eda finished!")


def preprocess_data(merged_data):
    # Tausender Stellen - Scraper Fehler -> abgeschnittene Nullen korrigieren
    merged_data.loc[merged_data["angebotspreis"] <= 10000, "angebotspreis"] = merged_data["angebotspreis"] * 1000

    # Umbenennungen
    merged_data.rename(columns={"befeuerungsart": "energietyp"}, inplace=True)

    # Zeilen ohne Angebotspreis droppen
    merged_data = merged_data.dropna(subset=["angebotspreis"])

    # Nicht verwendbare Spalten droppen
    merged_data = merged_data.drop(
        columns=['anzahl_schlafzimmer', 'energie_verbrauch', 'geschoss'])

    # Spalten-Datentypen bearbeiten
    merged_data["einwohner"] = pd.to_numeric(merged_data["einwohner"])
    merged_data["terrasse_balkon"] = merged_data["terrasse_balkon"].astype("category")
    merged_data["barrierefrei"] = merged_data["barrierefrei"].astype("category")
    merged_data["energietyp"] = merged_data["energietyp"].astype("category")
    merged_data["energie_effizienzklasse"] = merged_data["energie_effizienzklasse"].astype("category")
    merged_data["gaeste_wc"] = merged_data["gaeste_wc"].astype("category")
    merged_data["heizung"] = merged_data["heizung"].astype("category")
    merged_data["immobilienart"] = merged_data["immobilienart"].astype("category")
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].astype("category")
    merged_data["plz"] = merged_data["plz"].astype("category")
    merged_data["unterkellert"] = merged_data["unterkellert"].astype("category")
    merged_data["vermietet"] = merged_data["vermietet"].astype("category")
    merged_data["aufzug"] = merged_data["aufzug"].astype("category")


    # Kategorische Spalten anpassen (Kategorien zusammenfassen, kleine Kategorien in Sammler "Sonstige" zusammenfassen)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: str(row).split(",")[0])
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Pellets' if row == "Holzpellets" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Gas' if row == "Flüssiggas" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Fernwärme' if row == "Erdwärme" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Öl", "Gas", "Fernwärme",
                                              "Luft-/Wasserwärme", "Holz", "Pellets", "Solar", "Strom"] else row)
    merged_data["heizung"] = merged_data["heizung"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Zentralheizung", "Etagenheizung", "Ofenheizung",
                                              "Fußbodenheizung"] else row)

    merged_data["immobilienart"] = merged_data["immobilienart"].apply(
        lambda row: 'Einfamilienhaus' if row == "Einfamilienhaus (freistehend)" else row)

    merged_data["immobilienart"] = merged_data["immobilienart"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Einfamilienhaus", "Wohngrundstück", "Wohnung",
                                              "Etagenwohnung",
                                              "Sonstiges", "Mehrfamilienhaus", "Erdgeschosswohnung",
                                              "Dachgeschosswohnung",
                                              "Zweifamilienhaus", "Doppelhaushälfte", "Villa", "Reihenmittelhaus",
                                              "Reihenendhaus", "Bungalow",
                                              "Maisonette", "Apartment", "Stadthaus", "Schloss", "Bauernhaus",
                                              "Herrenhaus", "Reiheneckhaus", "Penthouse"] else row)

    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenovierungsbedürftig' if row == "Renovierungsbedürftig" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollsaniert' if row == "Saniert" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenoviert' if row == "Renovierungsbedürftig" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Unbekannt", "Erstbezug", "Projektiert", "Neubau",
                                              "Teil- oder vollrenovierungsbedürftig", "Neuwertig",
                                              "Teil- oder vollsaniert", "Teil- oder vollrenoviert", "Gepflegt",
                                              "Altbau", "Modernisiert"] else row)

    preprocessed_data = merged_data

    return preprocessed_data


def impute_data(preprocessed_data):
    # Zufällig mit vorhandenen Werten auffüllen
    preprocessed_data.loc[preprocessed_data["anzahl_badezimmer"] == 0, "anzahl_badezimmer"] = np.nan
    preprocessed_data["anzahl_badezimmer"] = preprocessed_data["anzahl_badezimmer"].apply(
        lambda x: np.random.choice(range(1, 4), p=[0.65, 0.30, 0.05]) if np.isnan(x) else x)
    preprocessed_data["anzahl_zimmer"] = preprocessed_data["anzahl_zimmer"].apply(
        lambda x: np.random.choice(preprocessed_data["anzahl_zimmer"].dropna().values) if np.isnan(x) else x)
    preprocessed_data["baujahr"] = preprocessed_data["baujahr"].apply(
        lambda x: np.random.choice(preprocessed_data["baujahr"].dropna().values) if np.isnan(x) else x)

    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].astype("category")
    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].fillna("Unbekannt")

    preprocessed_data["energie_effizienzklasse"] = preprocessed_data["energie_effizienzklasse"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["energie_effizienzklasse"] = preprocessed_data["energie_effizienzklasse"].fillna("Unbekannt")

    preprocessed_data["heizung"] = preprocessed_data["heizung"].astype("category")
    preprocessed_data["heizung"] = preprocessed_data["heizung"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["heizung"] = preprocessed_data["heizung"].fillna("Unbekannt")

    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].astype("category")
    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].fillna("Unbekannt")

    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].astype("category")
    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].fillna("Unbekannt")

    # Aufzug: Annahme, wenn nicht explizit angegeben, dann existiert kein Aufzug
    preprocessed_data.loc[preprocessed_data["aufzug"].isna(), "aufzug"] = "NEIN"

    imputed_data = preprocessed_data

    return imputed_data


def print_feature_importances(model, data):
    importances = pd.Series(data=model.feature_importances_,
                            index=data.columns)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Features Importances')
    plt.show()


def ml_tests(imputed_data):
    # ScikitLearn Anforderung: Nur numerische Werte - Transformation der kategorischen Spalten
    categorical_mask = (imputed_data.dtypes == "category")
    categorical_columns = imputed_data.columns[categorical_mask].tolist()
    category_enc = pd.get_dummies(imputed_data[categorical_columns])
    imputed_data = pd.concat([imputed_data, category_enc], axis=1)
    imputed_data = imputed_data.drop(columns=categorical_columns)

    imputed_data = imputed_data.reset_index()

    # Ausgabe
    print(imputed_data.info())
    imputed_data.to_excel(excel_writer="Files/Tests/imputed_data.xlsx", sheet_name="Immobilien")

    # XGBoost Standardmodell
    print("XGBoost Standardmodell:")
    x = imputed_data.drop(columns=["angebotspreis"]).values
    y = imputed_data["angebotspreis"].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=20, seed=123)
    xg_reg.fit(x_train, y_train)
    preds = xg_reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % rmse)
    print()

    print_feature_importances(model=xg_reg, data=imputed_data.drop(columns=["angebotspreis"]))

    # Grid Search parameter Tuning
    print("Grid Search Parameter Tuning:")
    gbm_param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'n_estimators': [50],
        'max_depth': [2, 5]
    }
    gbm = xgb.XGBRegressor(objective="reg:squarederror")
    grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring="neg_mean_squared_error", cv=4, verbose=1)
    grid_mse.fit(x_train, y_train)
    print("Best parameters found: ", grid_mse.best_params_)
    print("Lowest RMSE Grid Search found: ", np.sqrt(np.abs(grid_mse.best_score_)))
    print()

    # Randomized Search parameter tuning
    print("Randomized Search Parameter Tuning:")
    gbm_param_grid2 = {
        'n_estimators': [25],
        'max_depth': range(2, 12)
    }

    gbm2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    randomized_mse = RandomizedSearchCV(estimator=gbm2, param_distributions=gbm_param_grid2,
                                        scoring="neg_mean_squared_error", n_iter=5, cv=4, verbose=1)
    randomized_mse.fit(x_train, y_train)
    print("Best parameters found: ", randomized_mse.best_params_)
    print("Lowest RMSE Randomized Search found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

    dm_train = xgb.DMatrix(data=x_train, label=y_train)
    dm_test = xgb.DMatrix(data=x_test, label=y_test)
    params = {"booster": "gblinear", "objective": "reg:squarederror"}
    xg_reg2 = xgb.train(dtrain=dm_train, params=params, num_boost_round=15)
    preds2 = xg_reg2.predict(dm_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds2))
    print("RMSE: %f" % rmse)

    reg_params = [0.1, 0.3, 0.7, 1, 10, 100]
    params1 = {"objective": "reg:squarederror", "max_depth": 3}
    rmses_l2 = []
    for reg in reg_params:
        params1["lambda"] = reg
        cv_results_rmse = xgb.cv(dtrain=dm_train, params=params1, nfold=3, num_boost_round=15, metrics="rmse",
                                 as_pandas=True)
        rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

    print("Best rmse as a function of l2:")
    print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))
    print()

    print_feature_importances(model=xg_reg2, data=imputed_data.drop(columns=["angebotspreis"]))

    # Stochastic Gradient Boosting
    print("Stochastic Gradient Boosting:")
    sgbr = GradientBoostingRegressor(max_depth=4,
                                     subsample=0.9,
                                     max_features=0.75,
                                     n_estimators=200,
                                     random_state=2)

    sgbr.fit(x_train, y_train)
    y_pred = sgbr.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % rmse)
    print()

    print_feature_importances(model=sgbr, data=imputed_data.drop(columns=["angebotspreis"]))

    # Random Forrest
    print("Random Forrest:")
    rf = RandomForestRegressor(n_estimators=25,
                               random_state=2)
    rf.fit(x_train, y_train)
    y_pred2 = rf.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
    print("RMSE: %f" % rmse)
    print()

    print_feature_importances(model=rf, data=imputed_data.drop(columns=["angebotspreis"]))


def setup_database(path):
    db_connection = None
    try:
        db_connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful!")
    except Error as e:
        print(f"The error '{e}' occurred!")

    return db_connection


def main():
    # Set up database
    print("Step 1: Set up database...")

    db_connection = setup_database(r"../../Datenbank/ImmoDB.db")
    db_cursor = db_connection.cursor()

    # Read input data
    print("Step 2: Read in data...")

    immonet_data = read_data_from_immonet()
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection)
    # Alternative für später:
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection, if_exists = 'append oder replace oder fail')

    immoscout_data = read_data_from_immoscout()
    # immoscout_data.to_sql(name='Immoscout_data_raw', con=db_connection)

    geo_data = read_geo_data()
    # geo_data.to_sql(name='Geo_data_raw', con=db_connection)

    inhabitants_data = read_data_from_inhabitants()
    # inhabitants_data.to_sql(name='Inhabitants_data_raw', con=db_connection)

    # Merge input data
    print("Step 3: Merge data...")

    immonet_data_geo_inh = add_geo_inhabitants_immonet(immonet_data, geo_data, inhabitants_data)
    immoscout_data_geo_inh = add_geo_inhabitants_immoscout(immoscout_data, geo_data, inhabitants_data)

    merged_data = merge_data(immonet_data_geo_inh, immoscout_data_geo_inh)
    merged_data.to_csv("../../Files/Tests/merged_data.csv")

    # Preprocessing
    print("Step 4: Preprocess data...")

    preprocessed_data = preprocess_data(merged_data)
    preprocessed_data.to_csv("../../Files/Tests/preprocessed_data.csv")

    # EDA_Grafiken
    # print("Step 5: EDA_Grafiken...")

    # eda(preprocessed_data)

    # Imputation
    print("Step 6: Impute data...")

    imputed_data = impute_data(preprocessed_data)
    imputed_data.to_csv("../../Files/Tests/imputed_data.csv")
    imputed_data.to_excel("../../Files/Tests/imputed_data.xlsx")


    # Machine Learning
    # print("Step 7: Machine learning tests...")

    # ml_tests(imputed_data)

    # Testausgaben
    # print("Optional: Create Excel files...")

    # immonet_data.to_excel(excel_writer="Files/Tests/ImmoscoutDataTest.xlsx", sheet_name="Immobilien")
    # immoscout_data.to_excel(excel_writer="Files/Tests/ImmoscoutDataTest.xlsx", sheet_name="Immobilien")
    # geo_data.to_excel(excel_writer="Files/Tests/GeoDataTest.xlsx", sheet_name="Geodaten")
    # inhabitants_data.to_excel(excel_writer="Files/Tests/InhabitantsDataTest.xlsx", sheet_name="Einwohner")
    #
    # immonet_data_geo_inh.to_excel(excel_writer="Files/Tests/ImmoscoutDataGeoInhTest.xlsx", sheet_name="Immobilien")
    # immoscout_data_geo_inh.to_excel(excel_writer="Files/Tests/ImmoscoutDataGeoInhTest.xlsx", sheet_name="Immobilien")
    #
    # merged_data.to_excel(excel_writer="Files/Tests/merged_data.xlsx", sheet_name="Immobilien")

    print("... done.")


if __name__ == "__main__":
    main()
