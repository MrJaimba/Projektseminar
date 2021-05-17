import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

#Änderungen an csv
#alle ohne Grundstücksfläche und Wohnfläche raus

immo_data = pd.read_csv('imputed_data_UI.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = immo_data.copy()
target = 'immobilienart'
encode = ['barrierefrei', 'aufzug', 'energietyp', 'energie_effizienzklasse', 'gaeste_wc', 'heizung', 'terrasse_balkon', 'unterkellert', 'vermietet']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Einfamilienhaus':0, 
                 'Wohnung':1, 
                 'Etagenwohnung':2, 
                 'Sonstige':3, 
                 'Mehrfamilienhaus':4, 
                 'Erdgeschosswohnung':5, 
                 'Erdgeschosswohnung':6, 
                 'Dachgeschosswohnung':7, 
                 'Zweifamilienhaus':8, 
                 'Doppelhaushälfte':9, 
                 'Villa':10, 
                 'Reihenmittelhaus':11, 
                 'Reihenendhaus':12, 
                 'Bungalow':13, 
                 'Maisonette':14, 
                 'Apartment':15, 
                 'Stadthaus':16, 
                 'Schloss':17, 
                 'Bauernhaus':18, 
                 'Herrenhaus':19, 
                 'Reiheneckhaus':20, 
                 'Penthouse':21, 
                 'Unbekannt':22,
                 'Sonstiges':23}

def target_encode(val):
    return target_mapper[val]

df['immobilienart'] = df['immobilienart'].apply(target_encode)

# Separating X and y
X = df.drop('immobilienart', axis=1)
Y = df['immobilienart']

# Build random forest training_pipeline
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the training_pipeline
import pickle
pickle.dump(clf, open('immo_data_UI_clf.pkl', 'wb'))

# allgemeine Einstellungen (Tab Name; Icon; Seitenlayout; Menü)
st.beta_set_page_config('Immobilienbewertung', ' ', 'centered', 'expanded')

# Bild einfügen
st.image('UniWueLogo.png', width=250)

# Infotext groß
st.markdown("<h1 style='text-align: center; color: #004188;'>Willkommen bei WueRate</h1>", unsafe_allow_html=True)

# Trennlinie einfügen
st.write('---')

st.markdown("<h3 style='text-align: center; color: #000000;'>Schnelle Bewertung Ihrer Immobilie in Bayern</h3>", unsafe_allow_html=True)

#Infotext Ergänzung klein
st.write('Sie möchten eine Immobilie kaufen oder verkaufen und benötigen Hilfe bei der Bewertung?')
st.write('Kein Problem, WueRate bietet Ihnen eine fundierte Immobilienbewertung mithilfe maschinellen Lernens. Auf Basis einer umfangreichen Datenanalyse stellt Ihnen WueRate den optimalen Angebotspreis bereit.')

# Trennlinie einfügen
st.write('---')

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        angebotspreis = st.sidebar.slider('angebotspreis', 0, 3000000, 500000)
        barrierefrei = st.sidebar.selectbox('barrierefrei',('JA','NEIN'))
        aufzug = st.sidebar.selectbox('aufzug',('JA','NEIN'))
        gaeste_wc = st.sidebar.selectbox('gaeste_wc',('JA','NEIN'))
        terrasse_balkon = st.sidebar.selectbox('terrasse_balkon',('JA','NEIN'))
        unterkellert = st.sidebar.selectbox('unterkellert',('JA','NEIN'))
        vermietet = st.sidebar.selectbox('vermietet',('JA','NEIN'))
        energietyp = st.sidebar.selectbox('energietyp',('Fernwärme', 'Gas', 'Holz', 'Luft-/Wasserwärme', 'Öl', 'Pellets', 'Solar', 'Sonstige', 'Strom', 'Unbekannt'))
        anzahl_parkplatz = st.sidebar.slider('anzahl_parkplatz', 0, 5, 2)
        energie_effizienzklasse = st.sidebar.selectbox('energie_effizienzklasse',('A', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Unbekannt'))
        heizung = st.sidebar.selectbox('heizung',('Etagenheizung', 'Fußbodenheizung', 'Ofenheizung', 'Sonstige', 'Zentralheizung', 'Unbekannt'))
        anzahl_badezimmer = st.sidebar.slider('anzahl_badezimmer', 0, 10, 2)
        anzahl_zimmer = st.sidebar.slider('anzahl_zimmer', 0, 10, 2)
        baujahr = st.sidebar.slider('baujahr', 1950, 2000, 2023)
        einwohner = st.sidebar.slider('einwohner', 0, 2000, 100000)
        grundstuecksflaeche = st.sidebar.slider('grundstuecksflaeche', 0, 200, 1000)
        wohnflaeche = st.sidebar.slider('wohnflaeche', 0, 200, 1000)
        data = {'angebotspreis': angebotspreis, 
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
    input_df.drop(columns=["angebotspreis"])

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
immo_data_raw = pd.read_csv('imputed_data_UI.csv')
immo_data = immo_data_raw.drop(columns=['immobilienart'])
df = pd.concat([input_df,immo_data],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['barrierefrei', 'terrasse_balkon', 'unterkellert', 'vermietet', 'energietyp', 'heizung', 'gaeste_wc', 'energie_effizienzklasse', 'aufzug']
         
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification training_pipeline
load_clf = pickle.load(open('immo_data_UI_clf.pkl', 'rb'))

# Apply training_pipeline to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
immo_data_immoart = np.array(['Einfamilienhaus',
                             'Wohnung',
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
                             'Unbekannt'])                       
st.write(immo_data_immoart[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

#EDA hinten dran