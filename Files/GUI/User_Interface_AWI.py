import sqlite3
import sys
from streamlit import cli as stcli

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit.script_runner import RerunException, StopException
from streamlit_pandas_profiling import st_profile_report


# Änderungen an csv:
# Datensatz von Lennart (Excel-Datei) kopieren, in Excel und Spalten zusammen (csv Format) dann folgende Änderungen:
# Alle ohne Grundstücksfläche und Wohnfläche raus
# Spalten raus: Längengrad, Breitengrad, PLZ
from xgboost import plot_importance

db_connection = sqlite3.connect('Datenbank/ImmoDB.db')
# allgemeine Streamlit Einstellungen (Tab Name; Icon; Seitenlayout; Menü)
st.set_page_config('AWI', 'Projektseminar/Files/GUI/Logo AWI klein.jpg', 'centered', 'expanded')


#Logo einfügen
st.image('Files/GUI/Logo AWI.jpg')
st.image('Files/GUI/AbstandshalterAWI.jpg')


#Infotext
st.subheader('Starte jetzt deine Immobilienbewertung mit AWI')
st.write('Du möchtest den Wert deiner Immobilie exakt berechnen und benötigst weiterführende Analysemöglichkeiten rund um deine Immobilie? Dann bist du bei AWI genau richtig!')


#Button und Infotext
if st.button('mehr Informationen'):
    st.write('Was ist AWI?')
    st.write('AWI ist ein Analysetool für die Immobilienwertermittlung und bietet dir umfangreiche Analysemöglichkeiten rund um die Bewertung deiner Immobilie.')
    st.write('---')
    st.write('Wofür steht AWI?')
    st.write('AWI ist ein Akronym für: Analysetool für die Wertermittlung von Immobilien')
    st.write('---')
    st.write('Was ist an AWI besonders?')
    st.write('Immobilienbewertungstools gibt es wie Sand am Meer :palm_tree: Doch AWI ist keine gewohnliche App zur reinen Bewertung von Immobilien. Um eine fundierte Entscheidungen in Hinblick auf den Kauf bzw. Verkauf einer Immobilie zu treffen, sind tiefgreifende Informationen zur eigenen Immobilie notwendig. Im Gegensatz zu herkömmlichen Tools, bietet AWI diese Informationen. Neben einer aussagekräftigen Immobilienbewertung, die AWI mithilfe eines intelligenten Machine Learning Ansatzes ermittelt, bietet AWI weiterführende Analysen, um eine möglichst umfangreiche Immobilienbetrachtung zu ermöglichen. AWI bietet beispielsweise eine exakte Beschreibung welchen Einfluss die von dir hinterlegten Faktoren auf den Immobilienpreis nehmen. So kannst du zum Beispiel zukünftige Investitionen in deine Immobilie besser planen.')        
st.write('---')


#Überschrift und Abstandshalter
st.image('Files/GUI/AbstandshalterAWI.jpg')
st.subheader('Beschreibe deine Immobilie:')


#Definition der UI Eingabemaske (Features)
def user_input_features():
    
        #Eingabefeld 1 (Wohnfläche, Grundstücksfläche, Baujahr, PLZ)
        wohnflaeche = st.slider('Wohnfläche', 0, 1000, 145)
        grundstuecksflaeche = st.slider('Grundstücksfläche', 0, 1000, 430)
        
        col1, col2 = st.beta_columns(2)
        with col1:
            baujahr = st.number_input('Baujahr', max_value=2023, value=2007)
        with col2:
            plz = st.number_input('Wie lautet deine PLZ?', min_value=63739, max_value=97909, value=97070)
            if plz not in pd.read_sql_query('SELECT plz FROM Meta_Data_upd', con=db_connection)['plz'].to_list():
                st.error('Diese Postleitzahl befindet sich nicht in der Datenbank')
                raise StopException()
        
        #Eingabefeld 2 in Expander mit zwei Spalten (Anzahl Zimmer, Anzahl Parkplatz, Immoart, Anzahl Badezimmer,...)
        weitereDetails = st.beta_expander('weitere Details')
        with weitereDetails:
            col1, col2 = st.beta_columns(2)
            
            with col1:
                anzahl_zimmer = st.slider('Anzahl Zimmer', 0, 10, 5)
                anzahl_parkplatz = st.slider('Anzahl Parkplätze', 0, 5, 1)
                immobilienart = st.selectbox('Immobilienart', (
                    'Doppelhaushälfte',
                    'Einfamilienhaus',
                    'Etagenwohnung',
                    'Sonstige',
                    'Mehrfamilienhaus',
                    'Erdgeschosswohnung',
                    'Dachgeschosswohnung',
                    'Zweifamilienhaus',
                    'Wohnung',
                    'Villa',
                    'Reihenmittelhaus',
                    'Reihenendhaus',
                    'Bungalow',
                    'Maisonette',
                    'Apartment',
                    'Stadthaus',
                    'Schloss',
                    'Bauernhaus',
                    'Herrenhaus',
                    'Reiheneckhaus',
                    'Penthouse',
                    'Unbekannt'))
                heizung = st.selectbox('Art der Heizung',(
                    'Zentralheizung',
                    'Etagenheizung',
                    'Fußbodenheizung',
                    'Ofenheizung',
                    'Sonstige',                    
                    'Unbekannt'))
                
            with col2:
                anzahl_badezimmer = st.slider('Anzahl Badezimmer', 0, 5, 2)
                terrasse_balkon = st.selectbox('Terrasse/ Balkon',('NEIN', 'JA'))
                immobilienzustand = st.selectbox('Immobilienzustand', (
                    'Neuwertig',
                    'Altbau',
                    'Erstbezug',
                    'Gepflegt',
                    'Modernisiert',
                    'Neubau',                    
                    'Projektiert',
                    'Sonstige',
                    'Teil- oder vollrenovierungsbedürftig',
                    'Teil- oder vollsaniert',
                    'Teil- oder vollrenoviert',
                    'Unbekannt'))                
                energietyp = st.selectbox('Energietyp',(
                    'Gas',
                    'Fernwärme',
                    'Holz',
                    'Luft-/Wasserwärme',
                    'Öl',
                    'Pellets',
                    'Solar',
                    'Sonstige',
                    'Strom',
                    'Unbekannt'))
        
        #Eingabefeld 3 in Expander (Gäste_WC, barrierefrei, Aufzug,...)
        weitereKonfigurationen = st.beta_expander('spezifische Angaben')
        with weitereKonfigurationen:            
            gaeste_wc = st.selectbox('Gäste WC?',('NEIN', 'JA'))
            barrierefrei = st.selectbox('barrierefrei?',('NEIN', 'JA'))
            aufzug = st.selectbox('Aufzug?',('NEIN', 'JA'))
            unterkellert = st.selectbox('unterkellert?',('NEIN', 'JA'))
            vermietet = st.selectbox('aktuell vermietet?',('NEIN', 'JA'))
            energie_effizienzklasse = st.selectbox('Energieeffizienzklasse',(
                'A',
                'A+',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',                
                'Unbekannt'))

        #Kategorische Variablen codieren
        immobilienart_string = 'SELECT immobilienart_targetenc FROM Encoding_immobilienart WHERE immobilienart=\'' + immobilienart + '\''
        immobilienart = np.float32(pd.read_sql_query(immobilienart_string, con=db_connection).iloc[0][0])

        heizung_string = 'SELECT heizung_targetenc FROM Encoding_heizung WHERE heizung=\'' + heizung + '\''
        heizung = np.float32(pd.read_sql_query(heizung_string, con=db_connection).iloc[0][0])

        immobilienzustand_string = 'SELECT immobilienzustand_targetenc FROM Encoding_immobilienzustand WHERE immobilienzustand=\'' + immobilienzustand + '\''
        immobilienzustand = np.float32(pd.read_sql_query(immobilienzustand_string, con=db_connection).iloc[0][0])

        energietyp_string = 'SELECT energietyp_targetenc FROM Encoding_energietyp WHERE energietyp=\'' + energietyp + '\''
        energietyp = np.float32(pd.read_sql_query(energietyp_string, con=db_connection).iloc[0][0])

        energie_effizienzklasse_string = 'SELECT energie_effizienzklasse_targetenc FROM Encoding_energie_effizienzklasse WHERE energie_effizienzklasse=\'' + energie_effizienzklasse + '\''
        energie_effizienzklasse = np.float32(pd.read_sql_query(energie_effizienzklasse_string, con=db_connection).iloc[0][0])


        #Zuordnung der Eingabe-Features
        data = {'plz': plz,
                'immobilienart': immobilienart,
                'immobilienzustand': immobilienzustand,
                'barrierefrei': barrierefrei,
                'terrasse_balkon': terrasse_balkon,
                'unterkellert': unterkellert,
                'vermietet': vermietet,
                'energietyp': energietyp,
                'heizung': heizung,
                'gaeste_wc': gaeste_wc,
                'energie_effizienzklasse': energie_effizienzklasse,
                'aufzug': aufzug,
                'anzahl_badezimmer': anzahl_badezimmer,
                'anzahl_zimmer': anzahl_zimmer,
                'anzahl_parkplatz': anzahl_parkplatz,
                'baujahr': baujahr,
                'grundstuecksflaeche': grundstuecksflaeche,
                'wohnflaeche': wohnflaeche}
        features = pd.DataFrame(data, index=[0])

        features = features.assign(aufzug=(features['aufzug'] == 'JA').astype(int))
        features = features.assign(barrierefrei=(features['barrierefrei'] == 'JA').astype(int))
        features = features.assign(gaeste_wc=(features['gaeste_wc'] == 'JA').astype(int))
        features = features.assign(terrasse_balkon=(features['terrasse_balkon'] == 'JA').astype(int))
        features = features.assign(unterkellert=(features['unterkellert'] == 'JA').astype(int))
        features = features.assign(vermietet=(features['vermietet'] == 'JA').astype(int))

        #Metadaten aus Datenbank auslesen
        Metadaten = pd.read_sql_query('SELECT * FROM Meta_Data_upd2 WHERE plz=plz', con=db_connection, index_col="index")
        Metadaten = Metadaten.assign(
            supermarkt_im_plz_gebiet=(Metadaten['Supermarkt im PLZ Gebiet'] == 'JA').astype(int))
        Metadaten.drop(columns=['Supermarkt im PLZ Gebiet'], inplace=True)

        features = features.merge(Metadaten, how="inner", on="plz")
        features.drop(columns=['plz'], inplace=True)

        verstädterung = features['Grad_der_Verstädterung'].to_list()[0]
        verstädterung_string = 'SELECT Grad_der_Verstädterung_targetenc FROM Encoding_Grad_der_Verstädterung WHERE Grad_der_Verstädterung=\'' + verstädterung + '\''
        verstädterung = np.float32(pd.read_sql_query(verstädterung_string, con=db_connection).iloc[0][0])
        features['Grad_der_Verstädterung'] = verstädterung

        soziolage = features['sozioökonomische_Lage'].to_list()[0]
        soziolage_string = 'SELECT sozioökonomische_Lage_targetenc FROM Encoding_sozioökonmische_Lage WHERE sozioökonomische_Lage=\'' + soziolage + '\''
        soziolage = np.float32(pd.read_sql_query(soziolage_string, con=db_connection).iloc[0][0])
        features['sozioökonomische_Lage'] = soziolage

        #num_scaler = pickle.load(open('Projektseminar/num_scaler.pckl', 'rb'))

        #cat_features = features[['energietyp', 'energie_effizienzklasse',
         #                               'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verstädterung',
         #                               'sozioökonomische_Lage']]
        #features.drop(columns=['energietyp', 'energie_effizienzklasse',
                                     #   'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verstädterung',
                                      #  'sozioökonomische_Lage'], inplace=True)

        #features = pd.DataFrame(num_scaler.transform(features),
                             #  columns=features.columns, index=features.index)

        #features.to_sql(name='Features_scaler', con=db_connection, if_exists='replace')

        #features = pd.concat([features, cat_features], axis=1)

        features.rename(columns={'immobilienart': 'immobilienart_targetenc', 'immobilienzustand': 'immobilienzustand_targetenc',
                                 'energietyp': 'energietyp_targetenc', 'energie_effizienzklasse': 'energie_effizienzklasse_targetenc',
                                 'heizung': 'heizung_targetenc', 'Grad_der_Verstädterung': 'Grad_der_Verstädterung_targetenc',
                                 'sozioökonomische_Lage': 'sozioökonomische_Lage_targetenc'}, inplace=True)


        features = features.reindex(sorted(features.columns), axis=1)
        features.to_sql(name='Features', con=db_connection, if_exists='replace')
        return features

input_df = user_input_features()

# Abstandshalter
st.write('')
st.image('Files/GUI/AbstandshalterAWI.jpg')

modell = st.selectbox('Wähle Vorhersagemodell', ('XG Boost', 'Gradient Boosting', 'Random Forrest',
                                                 'Voting Regressor', 'Stacking Regressor'))
# Einlesen des Models aus der Pickle-Datei
if modell == 'XG Boost':
    load_modell = pickle.load(open('XGB_Standardmodell.pckl', 'rb'))
elif modell == 'Gradient Boosting':
    load_modell = pickle.load(open('sgbr_Standardmodell.pckl', 'rb'))
elif modell == 'Random Forrest':
    load_modell = pickle.load(open('rf_Standardmodell.pckl', 'rb'))
elif modell == 'Voting Regressor':
    load_modell = pickle.load(open('Voting_Regressor.pckl', 'rb'))
elif modell == 'Stacking Regressor':
    load_modell = pickle.load(open('Stacking_Regressor.pckl', 'rb'))

rmse = st.beta_expander('Übersicht der RMSE-Werte der Modelle')
with rmse:
    st.image('Files/Feature_Importances_Grafiken/RMSE.jpg')

# Abstandshalter
st.write(' ')
st.image('Files/GUI/AbstandshalterAWI.jpg')

# Definition des Outputs
output = ''
if st.button('Wertanalyse starten'):
    output = int(load_modell.predict(input_df))
    output = str(output) + '€'
    st.success('Der Wert Ihrer Immobilie liegt bei {}'.format(output))
    

# Abstandshalter
st.write('')
st.image('Files/GUI/AbstandshalterAWI.jpg')
    
# Überschrift 3: Datenanalyse
st.subheader('Du benötigst weitere Informationen?')
st.write(
    'Neben einer genauen Bewertung deines Immobilienwertes bieten wir dir tiefgreifendere Informationen für eine fundierte Bewertung. Neben informativen Hinweisen zu deiner geographischen Umgebung, stellen wir die im folgenden interaktive Datenvisualisierungen bereit. Am Ende der Seite findest du sogar eine explorative Datenanalyse, die dir einen Einblick in unsere Daten bieten soll.')

# weitere Informationen zur Umgebung
Metadaten_plz = st.beta_expander('weitere Informationen zur Umgebung')
with Metadaten_plz:
    st.write('Wähle eine Postleitzahl, zu der du weitere Informationen erhalten möchtest:')

    plz = st.number_input('', min_value=63739, max_value=97909, value=97070)
    st.write('---')

    Meta = pd.read_sql_query('SELECT * FROM Meta_Data_upd2 WHERE plz=plz', con=db_connection, index_col="index")
    #Meta_ort = Meta[Meta['plz'] == plz]['Hilfe Ort'].to_list()[0]
    Meta_einwohner = Meta[Meta['plz'] == plz]['Einwohner je PLZ'].to_list()[0]
    Meta_einkommen = Meta[Meta['plz'] == plz]['Durschnittseinkommen'].to_list()[0]
    Meta_arbeit = Meta[Meta['plz'] == plz]['Arbeitslosenquote in Prozent'].to_list()[0]
    Meta_sozio = Meta[Meta['plz'] == plz]['sozioökonomische_Lage'].to_list()[0]
    Meta_miete = Meta[Meta['plz'] == plz]['Kaltmiete / qm'].to_list()[0]
    Meta_abschluss = Meta[Meta['plz'] == plz]['Anteil nicht erfolgreicher beruflicher Bildungsgänge'].to_list()[0]
    Meta_schulabbrecher = Meta[Meta['plz'] == plz]['Anteil Schulabbrecher'].to_list()[0]
    Meta_allghoch = Meta[Meta['plz'] == plz]['Anteil Absolventen mit allgemeiner Hochschulreife'].to_list()[0]
    Meta_schulen = Meta[Meta['plz'] == plz]['Statistik der allgemein bildenden Schulen'].to_list()[0]
    Meta_landbetriebe = Meta[Meta['plz'] == plz]['Anzahl landwirtschaftlicher Betriebe'].to_list()[0]
    Meta_betriebe = Meta[Meta['plz'] == plz]['Anzahl Betriebe'].to_list()[0]
    Meta_wohnsiedlung = Meta[Meta['plz'] == plz]['Anteil Wohnfläche an gesamter Siedlungsfläche'].to_list()[0]
    Meta_siedlunggesamt = Meta[Meta['plz'] == plz]['Anteil Siedlungsfläche an Gesamtfläche'].to_list()[0]
    Meta_grüngesamt = Meta[Meta['plz'] == plz]['Anteil Grünflächen an Gesamtfläche'].to_list()[0]
    Meta_erholunggesamt = Meta[Meta['plz'] == plz]['Anteil Erholungsflächen an Gesamtfläche'].to_list()[0]
    Meta_supermarkt = Meta[Meta['plz'] == plz]['Supermarkt im PLZ Gebiet'].to_list()[0]
    Meta_zahnarzt = Meta[Meta['plz'] == plz]['Erreichbarkeit von Zahnärzten'].to_list()[0]
    Meta_apotheken = Meta[Meta['plz'] == plz]['Erreichbarkeit von Apotheken'].to_list()[0]
    Meta_lebensmittel = Meta[Meta['plz'] == plz]['Erreichbarkeit von Lebensmittelgeschäften'].to_list()[0]
    Meta_durchschnittsalter = Meta[Meta['plz'] == plz]['Durschnittsalter'].to_list()[0]
    Meta_lte = Meta[Meta['plz'] == plz]['LTE Abdeckung'].to_list()[0]
    Meta_breitband = Meta[Meta['plz'] == plz]['Breitbandversorgung'].to_list()[0]
    Meta_verschuldung = Meta[Meta['plz'] == plz]['Verschuldung pro Einwohner in 1000'].to_list()[0]
    Meta_verstädterung = Meta[Meta['plz'] == plz]['Grad_der_Verstädterung'].to_list()[0]
    Meta_übernachtungen = Meta[Meta['plz'] == plz]['Anzahl Gästeübernachtungen in 2019'].to_list()[0]

    col1, col2 = st.beta_columns(2)
    with col1:
        #st.write('Ort deiner Postleitzahl:')
        #st.info(Meta_ort)

        st.write('Durchschnittseinkommen:')
        st.info(Meta_einkommen)

        st.write('Sozioökonomische Lage:')
        st.info(Meta_sozio)

    with col2:
        st.write('Einwohner:')
        st.info(Meta_einwohner)

        st.write('Arbeitslosenquote in Prozent:')
        st.info(Meta_arbeit)

        st.write('Kaltmiete pro qm:')
        st.info(Meta_miete)

    st.write('---')
    st.write('Durch Klicken auf die jeweiligen Button, erhältst du nähere Informationen:')

    if st.button('Informationen zum Bildungsniveau'):
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write('Anteil Schulabbrecher:')
            st.info(Meta_schulabbrecher)

            st.write(' Anteil nicht erfolgreicher beruflicher Bildungsgänge:')
            st.info(Meta_abschluss)

        with col2:
            st.write('Anzahl allgemein bildender Schulen:')
            st.info(Meta_schulen)

            st.write('Anteil Absolventen mit allgemeiner Hochschulreife:')
            st.info(Meta_allghoch)

    if st.button('finanzielle und soziale Indikatoren'):
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write('Durchschnittsalter:')
            st.info(Meta_durchschnittsalter)

            st.write('LTE Abdeckung:')
            st.info(Meta_lte)

            st.write('Breitbandversorgung:')
            st.info(Meta_breitband)

        with col2:
            st.write('Verschuldung pro Einwohner in 1000:')
            st.info(Meta_verschuldung)

            st.write('Grad der Verstädterung:')
            st.info(Meta_verstädterung)

            st.write('Anzahl Gästeübernachtungen:')
            st.info(Meta_übernachtungen)

    if st.button('Anteil der Flächennutzung'):
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write('Anzahl landwirtschaftlicher Betriebe:')
            st.info(Meta_landbetriebe)

            st.write('Anzahl Betriebe:')
            st.info(Meta_betriebe)

            st.write('Anteil Wohnfläche an der Siedlungsfläche:')
            st.info(Meta_wohnsiedlung)

        with col2:
            st.write('Anteil Grünflächen an Gesamtfläche:')
            st.info(Meta_grüngesamt)

            st.write('Anteil Siedlungsfläche an Gesamtfläche:')
            st.info(Meta_siedlunggesamt)

            st.write('Anteil Erholungsflächen an Gesamtfläche:')
            st.info(Meta_erholunggesamt)

    if st.button('Erreichbarkeiten wesentlicher Einrichtungen'):
        col1, col2 = st.beta_columns(2)
        with col1:
            st.write('Supermarkt im PLZ Gebiet?:')
            st.info(Meta_supermarkt)

            st.write('Erreichbarkeit von Zahnärzten:')
            st.info(Meta_zahnarzt)

        with col2:
            st.write('Erreichbarkeit von Apotheken:')
            st.info(Meta_apotheken)

            st.write('Erreichbarkeit von Lebensmittelgeschäften:')
            st.info(Meta_lebensmittel)

# Abstandshalter
st.write('')
st.image('Files/GUI/AbstandshalterAWI.jpg')


feature_importances = st.beta_expander('Anzeige der wichtigsten Features')
with feature_importances:
    if modell == 'XG Boost':
        st.image('Files/Feature_Importances_Grafiken/xgb_feature_importances.jpg')
    elif modell == 'Gradient Boosting':
        st.image('Files/Feature_Importances_Grafiken/sgbr_feature_importances.jpg')
    elif modell == 'Random Forrest':
        st.image('Files/Feature_Importances_Grafiken/rf_feature_importances.jpg')
    else:
        st.write('Voting und Stacking Regressor sind Kombinationen der drei vorherigen Modelle. Um Informationen zur Feature Importance zu erhalten, sollten diese ausgewählt werden.')

# Abstandshalter
st.write('')
st.image('Files/GUI/AbstandshalterAWI.jpg')

# weitere graphische Darstellungen
if st.button('Graphische Datenanalyse'):
    data = pd.read_csv('Files/GUI/imputed_data_original.csv')
    st.map(data)
    

    # EDA
if st.button('Explorative Datenanalyse'):
    load_pr = pickle.load(open('pr.pkl', 'rb'))
    st_profile_report(load_pr)

if __name__ == "__main__":
    #filename = 'User_Interface_AWI.py'
    #sys.argv = ["streamlit", "run", filename]
    #sys.exit(stcli.main())
    print('Hello')
