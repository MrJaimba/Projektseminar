import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Änderungen an csv:
# Datensatz von Lennart (Excel-Datei) kopieren, in Excel und Spalten zusammen (csv Format) dann folgende Änderungen:
# Alle ohne Grundstücksfläche und Wohnfläche raus
# Spalten raus: Längengrad, Breitengrad, PLZ


# allgemeine Streamlit Einstellungen (Tab Name; Icon; Seitenlayout; Menü)
st.set_page_config('Immobilienbewertung', 'UniWueLogo.png', 'centered', 'expanded')


#Logo einfügen
st.image('Logo AWI.jpg')
st.image('AbstandshalterAWI.jpg')


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
st.image('AbstandshalterAWI.jpg')
st.subheader('Beschreibe deine Immobilie:')


#Definition der UI Eingabemaske (Features)
def user_input_features():
    
        #Eingabefeld 1 (Wohnfläche, Grundstücksfläche, Baujahr, PLZ)
        wohnflaeche = st.slider('Wohnfläche', 0, 1000, 200)
        grundstuecksflaeche = st.slider('Grundstücksfläche', 0, 1000, 200)
        baujahr = st.slider('Baujahr', 1980, 2023, 2000)
        PLZ = st.number_input('Wie lautet deine PLZ?', min_value=63739, max_value=97909, value=97070)
        
        #Eingabefeld 2 in Expander mit zwei Spalten (Anzahl Zimmer, Anzahl Parkplatz, Immoart, Anzahl Badezimmer,...)
        weitereDetails = st.beta_expander('weitere Details')
        with weitereDetails:
            col1, col2 = st.beta_columns(2)
            
            with col1:
                anzahl_zimmer = st.slider('Anzahl Zimmer', 0, 10, 2)
                anzahl_parkplatz = st.slider('Anzahl Parkplätze', 0, 5, 0)
                immobilienart = st.selectbox('Immobilienart', (
                    'Wohnung',
                    'Einfamilienhaus',
                    'Etagenwohnung',
                    'Sonstige',
                    'Mehrfamilienhaus',
                    'Erdgeschosswohnung',
                    'Erdgeschosswohnung',
                    'Dachgeschosswohnung',
                    'Zweifamilienhaus',
                    'Doppelhaushälfte',
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
                    'Etagenheizung',
                    'Fußbodenheizung',
                    'Ofenheizung',
                    'Sonstige',
                    'Zentralheizung',
                    'Unbekannt'))
                
            with col2:
                anzahl_badezimmer = st.slider('Anzahl Badezimmer', 0, 5, 1)
                terrasse_balkon = st.selectbox('Terrasse/ Balkon',('NEIN', 'JA'))
                immobilienzustand = st.selectbox('Immobilienzustand', (
                    'Altbau',
                    'Erstbezug',
                    'Gepflegt',
                    'Modernisiert',
                    'Neubau',
                    'Neuwertig',
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
            einwohner = st.slider('Einwohner', 0, 10000, 2000)
        
        
        #Zuordnung der Eingabe-Features
        data = {'immobilienart': immobilienart,
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
                'einwohner': einwohner,
                'grundstuecksflaeche': grundstuecksflaeche,
                'wohnflaeche': wohnflaeche}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()

    
# Kombination der Input_Features mit dem Datensatz
immo_data_raw = pd.read_csv('imputed_data.csv')
immo_data = immo_data_raw.drop(columns=['angebotspreis'])
df = pd.concat([input_df,immo_data],axis=0)


# Encoding der Object-Variablen
encode = [
    'immobilienart',
    'immobilienzustand',
    'barrierefrei',
    'terrasse_balkon',
    'unterkellert',
    'vermietet',
    'energietyp',
    'heizung',
    'gaeste_wc',
    'energie_effizienzklasse',
    'aufzug']
         
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]


# Einlesen des Models aus der Pickle-Datei
load_random_forrest_reg = pickle.load(open('random_forrest_reg.pkl', 'rb'))


# Abstandshalter
st.write('')
st.image('AbstandshalterAWI.jpg')


# Definition der Prediction
x_new = df.values


# Definition des Outputs
output = ''
if st.button('Wertanalyse starten'):
    output = load_random_forrest_reg.predict(x_new)
    output = str(output) + '€'
    st.success('Der Wert Ihrer Immobilie liegt bei {}'.format(output))
    

# Abstandshalter
st.write('')
st.image('AbstandshalterAWI.jpg')


# weitere graphische Darstellungen
if st.button('weiterführende Analysen'):
    data = pd.read_csv('imputed_data_original.csv')
    st.map(data)
