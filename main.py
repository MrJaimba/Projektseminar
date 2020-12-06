import pandas as pd
import numpy as np
import re

if __name__ == "__main__":

    #Exceldateien importieren
    Immoscout24Base = pd.read_excel(r"Files/20201124_Immoscout24.xlsx", sheet_name="Häuser Wü und Landkreis")
    Immoscout24Update = pd.read_excel(r"Files/20201129_Immoscout24_update.xlsx", sheet_name="Häuser neu")
    ImmonetBase = pd.read_excel(r"Files/20201124_Immonet.xlsx", sheet_name="Tabelle2")

    #Yaninas Datensätze an Lennart anpassen
    Immoscout24Base.columns = Immoscout24Base.columns.str.lower()
    Immoscout24Base["plz"] = Immoscout24Base["plz und ort"].apply(lambda row: row[:5])
    Immoscout24Base["ort"] = Immoscout24Base["plz und ort"].apply(lambda row: row[5:])
    Immoscout24Base = Immoscout24Base.drop(columns="plz und ort")

    Immoscout24Update.columns = Immoscout24Update.columns.str.lower()
    Immoscout24Update["plz"] = Immoscout24Update["plz und ort"].apply(lambda row: row[:5])
    Immoscout24Update["ort"] = Immoscout24Update["plz und ort"].apply(lambda row: row[5:])
    Immoscout24Update = Immoscout24Update.drop(columns=["plz und ort", "web-scraper-order"])

    Immoscout24Base = Immoscout24Base.reindex(sorted(Immoscout24Base.columns), axis=1)
    Immoscout24Update = Immoscout24Update.reindex(sorted(Immoscout24Update.columns), axis=1)
    ImmonetBase = ImmonetBase.reindex(sorted(ImmonetBase.columns), axis=1)

    #Yaninas Datensätze zusammenführen und Spalten umbenennen
    Immoscout24AllBase = pd.concat([Immoscout24Base, Immoscout24Update], axis=0, ignore_index=True)
    Immoscout24AllBase.rename(columns={"anzahl badezimmer": "anzahl_badezimmer", "anzahl schlafzimmer": "anzahl_schlafzimmer",
                                "zimmer": "anzahl_zimmer", "einkaufspreis": "angebotspreis",
                                "balkon/ terrasse": "balkon", "wohnfläche": "wohnflaeche", "etage": "geschoss",
                                "grundstück": "grundstuecksflaeche", "stufenloser zugang": "barrierefrei",
                                "aufzug": "fahrstuhl", "objektzustand": "immobilienzustand",
                                "keller ja/nein": "unterkellert", "gäste-wc ja/nein": "gaeste_wc",
                                "energie­effizienz­klasse": "energie_effizienzklasse",
                                "wesentliche energieträger" : "befeuerungsart", "end-energie-verbrauch" : "energie_verbrauch",
                                "typ" : "immobilienart", "heizungsart" : "heizung", "vermietet ja/nein" : "vermietet"}, inplace=True)
    Immoscout24AllBase = Immoscout24AllBase.reindex(sorted(Immoscout24AllBase.columns), axis=1)

    #Spalteninhalte anpassen: Annahme NaN ist NEIN
    Immoscout24AllBase["unterkellert"] = Immoscout24AllBase["unterkellert"].apply(lambda row: "JA" if row == "keller" else "NEIN")
    Immoscout24AllBase["gaeste_wc"] = Immoscout24AllBase["gaeste_wc"].apply(lambda row: "JA" if row == "Gäste-WC" else "NEIN")
    #Immoscout24AllBase["balkon"] = Immoscout24AllBase["balkon"].apply(lambda row: "NEIN" if row.isnull() else "JA")
    #Immoscout24AllBase["barrierefrei"] = Immoscout24AllBase["barrierefrei"].apply(lambda row: "NEIN" if row.isnull() else "JA")
    Immoscout24AllBase["baujahr"] = Immoscout24AllBase["baujahr"].apply(lambda row: np.NaN if row == "unbekannt" else None)
    Immoscout24AllBase["grundstuecksflaeche"] = Immoscout24AllBase["grundstuecksflaeche"].apply(lambda row: re.sub('[.m²]', '', row))
    Immoscout24AllBase["wohnflaeche"] = Immoscout24AllBase["wohnflaeche"].apply(lambda row: re.sub('[m²]', '', row))

    ImmobilienAll = pd.concat([Immoscout24AllBase, ImmonetBase], axis=0, ignore_index=True, join="inner")

    ImmobilienAll2 = pd.concat([Immoscout24AllBase, ImmonetBase], axis=0, ignore_index=True, join="outer")

    #ImmobilienAll2.to_excel(excel_writer="Files/ImmobilienAll2v2.xlsx", sheet_name="ImmobilienAll")


    with pd.option_context('display.max_rows', 5, 'display.max_columns', 17):
        print(ImmobilienAll2)
    print("und zurück")
