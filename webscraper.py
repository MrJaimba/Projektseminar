from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import re
import csv


class ImmoFetcher:

    def fetch(self):
        # Start URL: Erste Seite der Suchergebnisse
        url = 'https://www.immonet.de/immobiliensuche/sel.do?pageoffset=1&listsize=26&objecttype=1&locationname=W%C3' \
              '%BCrzburg&acid=&actype=&city=153145&ajaxIsRadiusActive=true&sortby=16&suchart=2&radius=50&pcatmtypes' \
              '=-1_1&pCatMTypeStoragefield=-1_1&parentcat=-1&marketingtype=1&fromprice=&toprice=&fromarea=&toarea' \
              '=&fromplotarea=&toplotarea=&fromrooms=&torooms=&wbs=-1&fromyear=&toyear=&fulltext=&absenden=Ergebnisse' \
              '+anzeigen '

        # Initialisiert leere Liste für alle Immobilien in den Suchergebnissen
        immo_list = []

        # While-Schleife für alle URLs der Suchergebnisseiten
        while url != '':
            # print(url)

            # Request Source URL der Suchergebnisseite
            source_searchsite = requests.get(url)
            soup_searchsite = BeautifulSoup(source_searchsite.text, 'html.parser')

            # Alle Immobilienkacheln (div-Container) auf der aktuellen Suchergebnisseite
            containers = soup_searchsite.find_all('div', class_='flex-grow-1 display-flex flex-direction-column box-25 '
                                                                'overflow-hidden cursor-hand')

            # Innere Schleife über alle gefundenen Immobilienkacheln (div-Container) der aktuellen Suchergebnisseite
            for container in containers:

                # Absprung auf Subseite / eigentliche Immobilienseite
                next_immo_link = container.find('a', class_='block ellipsis text-225 text-default')
                immo_url = next_immo_link.attrs['href']
                immo_url = urljoin(url, immo_url)

                # print('immo_url: ' + immo_url)

                # Request Source URL der Immobilienseite
                source_immosite = requests.get(immo_url)
                soup_immosite = BeautifulSoup(source_immosite.text, 'html.parser')

                # Auslesen der eigentlichen Immobilienseite:

                # 1. Basic Features:

                # Postleitzahl (Integer) auslesen - Gleicher p-Tag wie Ort, Trennung der Informationen
                plz = soup_immosite.find('p', class_='text-100 pull-left')
                if plz is not None:
                    plz = soup_immosite.find('p', class_='text-100 pull-left').text.strip()
                    plz = re.findall(r"\d{5}", plz)[0]
                    plz = int(plz) if plz else 1
                else:
                    # Dummy
                    plz = int(0)
                # print('postleitzahl: ' + str(plz))

                # Ort (String) auslesen - Gleicher p-Tag wie plz, Trennung der Informationen
                ort = soup_immosite.find('p', class_='text-100 pull-left')
                if ort is not None:
                    ort = soup_immosite.find('p', class_='text-100 pull-left').text.strip()
                    # PLZ entfernen
                    ort = ''.join(i for i in ort if not i.isdigit())
                    ort = ort.strip()
                    ort = str(ort)
                else:
                    ort = str('')
                # print('ort: ' + ort)

                # Angebotspreis (Float) auslesen
                angebotspreis = soup_immosite.find('div', id='priceid_1')
                if angebotspreis is not None:
                    angebotspreis = soup_immosite.find('div', id='priceid_1').text.strip()
                    angebotspreis = re.sub('[^0-9,]', '', angebotspreis)
                    angebotspreis = angebotspreis.replace(',', '.')
                    angebotspreis = float(angebotspreis) if angebotspreis else 1.01
                else:
                    # Dummy
                    angebotspreis = float(1.00)
                # print('angebotspreis in €: ' + str(angebotspreis))

                # Baujahr (Integer) auslesen
                baujahr = soup_immosite.find('div', id='yearbuild')
                if baujahr is not None:
                    baujahr = soup_immosite.find('div', id='yearbuild').text.strip()
                    baujahr = int(baujahr) if baujahr else 1
                else:
                    # Dummy
                    baujahr = int(0)
                    # print('baujahr: ' + str(baujahr))

                # 2. Flächenangaben:

                # Grundstücksfläche (Float) auslesen
                grundstuecksflaeche = soup_immosite.find('div', id='areaid_3')
                if grundstuecksflaeche is not None:
                    grundstuecksflaeche = soup_immosite.find('div', id='areaid_3').text.strip()
                    grundstuecksflaeche = re.sub('[^0-9,]', '', grundstuecksflaeche)
                    grundstuecksflaeche = grundstuecksflaeche.replace(',', '.')
                    grundstuecksflaeche = float(grundstuecksflaeche) if grundstuecksflaeche else 1.01
                else:
                    # Dummy
                    grundstuecksflaeche = float(1.00)
                # print('grundstücksfläche in m²: ' + str(grundstuecksflaeche))

                # Wohnfläche (Float) auslesen
                wohnflaeche = soup_immosite.find('div', id='areaid_1')
                if wohnflaeche is not None:
                    wohnflaeche = soup_immosite.find('div', id='areaid_1').text.strip()
                    wohnflaeche = re.sub('[^0-9,]', '', wohnflaeche)
                    wohnflaeche = wohnflaeche.replace(',', '.')
                    wohnflaeche = float(wohnflaeche) if wohnflaeche else 1.01
                else:
                    # Dummy
                    wohnflaeche = float(1.00)
                # print('wohnflaeche in m²: ' + str(wohnflaeche))

                # 3. Anzahl Zimmer, Parkplätze etc.

                # Anzahl der Zimmer (Integer) auslesen
                anzahl_zimmer = soup_immosite.find('div', id='equipmentid_1')
                if anzahl_zimmer is not None:
                    anzahl_zimmer = soup_immosite.find('div', id='equipmentid_1').text.strip()
                    anzahl_zimmer = re.sub('[^0-9]', '', anzahl_zimmer)
                    anzahl_zimmer = int(anzahl_zimmer) if anzahl_zimmer else 999
                else:
                    # Dummy
                    anzahl_zimmer = int(99)
                # print('zimmer_anzahl: ' + str(anzahl_zimmer))

                # Anzahl der Schlafzimmer (Integer) auslesen
                anzahl_schlafzimmer = soup_immosite.find('p', id='otherDescription')
                if anzahl_schlafzimmer is not None:
                    anzahl_schlafzimmer = soup_immosite.find('p', id='otherDescription').text.strip().split()
                    if 'Schlafzimmer:' in anzahl_schlafzimmer:
                        index_schlafzimmer = anzahl_schlafzimmer.index('Schlafzimmer:')
                        anzahl_schlafzimmer = anzahl_schlafzimmer[index_schlafzimmer + 1]
                    else:
                        # Dummy
                        anzahl_schlafzimmer = int(999)
                else:
                    # Dummy
                    anzahl_schlafzimmer = int(99)
                # print('anzahlschlafzimmer: ' + str(anzahl_schlafzimmer))

                # Anzahl der Badezimmer (Integer) auslesen
                anzahl_badezimmer = soup_immosite.find('p', id='otherDescription')
                if anzahl_badezimmer is not None:
                    anzahl_badezimmer = soup_immosite.find('p', id='otherDescription').text.strip().split()
                    if 'Badezimmer:' in anzahl_badezimmer:
                        index_beginn = anzahl_badezimmer.index('Badezimmer:')
                        anzahl_badezimmer = soup_immosite.find('p', id='otherDescription').text.strip().split()[
                                            index_beginn:]
                        anzahl_badezimmer = anzahl_badezimmer[1][0]
                    else:
                        anzahl_badezimmer = '0'

                elif soup_immosite.find('li', id='featureId_35') is not None:
                    anzahl_badezimmer = '1'

                else:
                    anzahl_badezimmer = '0'
                # print('anzahl_badezimmer: ' + anzahl_badezimmer)

                # Anzahl Parkplätze (Integer) auslesen
                anzahl_parkplatz = soup_immosite.find('div', id='equipmentid_13')
                if anzahl_parkplatz is not None:
                    anzahl_parkplatz = soup_immosite.find('div', id='equipmentid_13').text.strip()
                    anzahl_parkplatz = int(anzahl_parkplatz) if anzahl_parkplatz else 999
                else:
                    anzahl_parkplatz = int(0)
                # print('anzahl_parkplatz: ' + str(anzahl_parkplatz))

                # Anzahl Etagen (Integer) auslesen
                anzahl_etagen = soup_immosite.find('li', id='featureId_135')
                if anzahl_etagen is not None:
                    anzahl_etagen = soup_immosite.find('li', id='featureId_135').text.strip()
                    anzahl_etagen = re.sub('[^0-9]', '', anzahl_etagen)
                    anzahl_etagen = int(anzahl_etagen) if anzahl_etagen else 999
                else:
                    anzahl_etagen = int(99)
                # print('etagen: ' + str(anzahl_etagen))

                # 4. Binäre Features

                # Gäste-WC (Boolean) auslesen
                gaeste_wc = soup_immosite.find('li', id='featureId_70')
                if gaeste_wc is not None:
                    gaeste_wc = True
                else:
                    gaeste_wc = False
                # print('gaeste_wc: ' + str(gaeste_wc))

                # Balkon (Boolean) auslesen
                balkon = soup_immosite.find('li', id='featureId_57')
                if balkon is not None:
                    balkon = True
                else:
                    balkon = False
                # print('balkon: ' + str(balkon))

                # Terrasse (Boolean) auslesen
                terrasse = soup_immosite.find('li', id='featureId_67')
                if terrasse is not None:
                    terrasse = True
                else:
                    terrasse = False
                # print('terasse: ' + str(terrasse))

                # Keller (Boolean) auslesen
                unterkellert = soup_immosite.find('li', id='featureId_69')
                if unterkellert is not None:
                    unterkellert = True
                else:
                    unterkellert = False
                # print('unterkellert: ' + str(unterkellert))

                # Vermietet (Boolean) auslesen
                vermietet = soup_immosite.find('li', id='featureId_318')
                if vermietet is not None:
                    vermietet = True
                else:
                    vermietet = False
                # print('vermietet: ' + str(vermietet))

                # Aufzug (Boolean) auslesen
                aufzug = soup_immosite.find('li', id='featureId_68')
                aufz_sonstiges = soup_immosite.find('p', id='otherDescription')
                aufz_objektbeschreibung = soup_immosite.find('p', id='objectDescription')

                if aufzug is not None:
                    aufzug = 'JA'
                elif aufz_sonstiges is not None:
                    aufz_sonstiges = soup_immosite.find('p', id='otherDescription').text.strip()
                    aufz_sonstiges = re.findall(r"[\s\W\w]([Aa]ufzug[\w\W])", aufz_sonstiges)
                    if aufz_sonstiges:
                        aufzug = 'JA'
                    elif aufz_objektbeschreibung is not None:
                        aufz_objektbeschreibung = soup_immosite.find('p', id='objectDescription').text.strip()
                        aufz_objektbeschreibung = re.findall(r"[\s\W\w]([Aa]ufzug[\w\W])", aufz_objektbeschreibung)
                        if aufz_objektbeschreibung:
                            aufzug = 'JA'
                        else:
                            aufzug = 'NEIN'
                    else:
                        aufzug = 'NEIN'
                else:
                    aufzug = 'NEIN'
                # print('aufzug: ' + aufzug)

                # Barrierefrei (Boolean) auslesen
                barrierefrei = soup_immosite.find('li', id='featureId_167')
                barrierefrei2 = soup_immosite.find('li', id='featureId_285')
                ba_ausstattung = soup_immosite.find('div', id='ausstattung')
                ba_objektbeschreibung = soup_immosite.find('p', id='objectDescription')
                ba_sonstiges = soup_immosite.find('p', id='otherDescription')

                if barrierefrei is not None:
                    barrierefrei = 'JA'
                elif barrierefrei2 is not None:
                    barrierefrei = 'JA'
                elif ba_ausstattung is not None:
                    ba_ausstattung = soup_immosite.find('div', id='ausstattung').text.strip()
                    ba_ausstattung = re.findall(r"[\s\W\w]([Bb]arrierefrei[\w\W])", ba_ausstattung)
                    if ba_ausstattung:
                        barrierefrei = 'JA'
                    elif ba_objektbeschreibung is not None:
                        ba_objektbeschreibung = soup_immosite.find('p', id='objectDescription').text.strip()
                        ba_objektbeschreibung = re.findall(r"[\s\W\w]([Bb]arrierefrei[\w\W])", ba_objektbeschreibung)
                        if ba_objektbeschreibung:
                            barrierefrei = 'JA'
                        elif ba_sonstiges is not None:
                            ba_sonstiges = soup_immosite.find('p', id='otherDescription').text.strip()
                            ba_sonstiges = re.findall(r"[\s\W\w]([Bb]arrierefrei[\w\W])", ba_sonstiges)
                            if ba_sonstiges:
                                barrierefrei = 'JA'
                            else:
                                barrierefrei = 'NEIN'
                        else:
                            barrierefrei = 'NEIN'
                    else:
                        barrierefrei = 'NEIN'
                else:
                    barrierefrei = 'NEIN'
                # print('barrierefrei: ' + barrierefrei)

                # Denkmalschutz (Boolean) auslesen
                denkmalschutz = soup_immosite.find('p', id='objectDescription')
                denkm_objektbeschreibung = soup_immosite.find('p', id='objectDescription')
                denkm_sonstiges = soup_immosite.find('p', id='otherDescription')
                if denkm_objektbeschreibung is not None:
                    denkm_objektbeschreibung = soup_immosite.find('p', id='objectDescription').text.strip()
                    denkm_objektbeschreibung = re.findall(r"\w*\s*\w*(?=\s*\W*[Dd]enkmal\w*)", denkm_objektbeschreibung)
                    if denkm_objektbeschreibung:
                        denkm_kein = re.findall(r"\s*\W*\w*([Kk]ein)", denkm_objektbeschreibung[0].strip())
                        denkma_nichtunter = re.findall(r"\s*(nicht\sunter)", denkm_objektbeschreibung[0].strip())
                        if denkm_kein:
                            denkmalschutz = 'NEIN'
                        elif denkma_nichtunter:
                            denkmalschutz = 'NEIN'
                        else:
                            denkmalschutz = 'JA'
                    else:
                        denkmalschutz = 'NEIN'
                elif denkm_sonstiges is not None:
                    denkm_sonstiges = soup_immosite.find('p', id='otherDescription').text.strip()
                    denkm_sonstiges = re.findall(r"\w*\s*\w*(?=\s*\W*[Dd]enkmal\w*)", denkm_sonstiges)
                    if denkm_sonstiges:
                        denkm_kein = re.findall(r"\s*\W*\w*([Kk]ein)", denkm_sonstiges[0].strip())
                        denkma_nichtunter = re.findall(r"\s*(nicht\sunter)", denkm_sonstiges[0].strip())
                        if denkm_kein:
                            denkmalschutz = 'NEIN'
                        elif denkma_nichtunter:
                            denkmalschutz = 'NEIN'
                        else:
                            denkmalschutz = 'JA'
                    else:
                        denkmalschutz = 'NEIN'
                else:
                    denkmalschutz = 'NEIN'
                # print('denkmalschutz: ' + denkmalschutz)

                # 5. Kategorische Features

                # Immobilienart (String) auslesen
                immobilienart = soup_immosite.find('h2', id='sub-headline-expose')
                if immobilienart is not None:
                    immobilienart = soup_immosite.find('h2', id='sub-headline-expose').text.strip()
                else:
                    # Dummy
                    immobilienart = 'Unbekannt'
                # print('immobilienart ' + immobilienart)

                # Immobilienzustand (String) auslesen
                immobilienzustand = soup_immosite.find('div', id='objectstatecategoryValue')
                if immobilienzustand is not None:
                    immobilienzustand = soup_immosite.find('div', id='objectstatecategoryValue').text.strip()
                else:
                    immobilienzustand = 'Unbekannt'
                # print('immobilienzustand: ' + immobilienzustand)

                # Heizungsart (String) auslesen
                heizung = soup_immosite.find('div', id='heatTypeValue')
                if heizung is not None:
                    heizung = soup_immosite.find('div', id='heatTypeValue').text.strip()
                else:
                    heizung = 'Unbekannt'
                # print('heizung: ' + heizung)

                # Befeuerungsart (String) auslesen
                befeuerungsart = soup_immosite.find('div', id='heaterSupplierValue')
                if befeuerungsart is not None:
                    befeuerungsart = soup_immosite.find('div', id='heaterSupplierValue').text.strip()
                else:
                    befeuerungsart = 'Unbekannt'
                # print('befeuerungsart: ' + befeuerungsart)

                # Geschoss (String) auslesen
                geschoss = soup_immosite.find('li', id='featureId_123')
                if geschoss is not None:
                    geschoss = soup_immosite.find('li', id='featureId_123').text.strip().split()[-1]
                elif immobilienart == 'Erdgeschosswohnung':
                    geschoss = '0'
                else:
                    geschoss = ''
                # print('geschoss: ' + geschoss)

                # Energieverbrauch (String) auslesen
                energie_verbrauch = soup_immosite.find('div', id='energyValue')
                if energie_verbrauch is not None:
                    energie_verbrauch = soup_immosite.find('div', id='energyValue').text.strip().split()[0]
                else:
                    energie_verbrauch = ''
                # print('energie_verbrauch: ' + energie_verbrauch)

                # Energieeffizienzklasse (String) auslesen
                energie_effizienzklasse = soup_immosite.find('div', id='efficiencyValue')
                if energie_effizienzklasse is not None:
                    energie_effizienzklasse = soup_immosite.find('div', id='efficiencyValue').text.strip().split()[-1]
                else:
                    energie_effizienzklasse = ''
                # print('energie_effizienzklasse: ' + energie_effizienzklasse)

                # Immobilienobjekt erzeugen
                scraped = ScrapedRealEstate(immo_url, ort, plz, angebotspreis, grundstuecksflaeche, wohnflaeche,
                                            anzahl_zimmer, anzahl_schlafzimmer, anzahl_badezimmer, gaeste_wc, baujahr,
                                            anzahl_parkplatz, immobilienart, immobilienzustand, balkon, terrasse,
                                            heizung, befeuerungsart, anzahl_etagen, geschoss, unterkellert, vermietet,
                                            energie_verbrauch, energie_effizienzklasse, aufzug, barrierefrei,
                                            denkmalschutz)

                # Immobilie hinzufügen
                immo_list.append(scraped)

            # Absprung zur nächsten Seite der Suchergebnisse
            next_site_button = soup_searchsite.find('a', class_='col-sm-3 col-xs-1 pull-right text-right')
            if next_site_button:
                next_href = next_site_button.attrs["href"]
                next_href = urljoin(url, next_href)
                url = next_href
            else:
                url = ''
        return immo_list


class ScrapedRealEstate:
    def __init__(self, immo_url, ort, plz, angebotspreis, grundstuecksflaeche, wohnflaeche, anzahl_zimmer,
                 anzahl_schlafzimmer, anzahl_badezimmer, gaeste_wc, baujahr, anzahl_parkplatz, immobilienart,
                 immobilienzustand, balkon, terrasse, heizung, befeuerungsart, etagen, geschoss, unterkellert,
                 vermietet, energie_verbrauch, energie_effizienzklasse, aufzug, barrierefrei, denkmalschutz):
        self.immo_url = immo_url
        self.ort = ort
        self.plz = plz
        self.angebotspreis = angebotspreis
        self.grundstuecksflaeche = grundstuecksflaeche
        self.wohnflaeche = wohnflaeche
        self.anzahl_zimmer = anzahl_zimmer
        self.anzahl_schlafzimmer = anzahl_schlafzimmer
        self.anzahl_badezimmer = anzahl_badezimmer
        self.gaeste_wc = gaeste_wc
        self.baujahr = baujahr
        self.anzahl_parkplatz = anzahl_parkplatz
        self.immobilienart = immobilienart
        self.immobilienzustand = immobilienzustand
        self.balkon = balkon
        self.terrasse = terrasse
        self.heizung = heizung
        self.befeuerungsart = befeuerungsart
        self.etagen = etagen
        self.geschoss = geschoss
        self.unterkellert = unterkellert
        self.vermietet = vermietet
        self.energie_verbrauch = energie_verbrauch
        self.energie_effizienzklasse = energie_effizienzklasse
        self.aufzug = aufzug
        self.barrierefrei = barrierefrei
        self.denkmalschutz = denkmalschutz


# Scraper erzeugen/initialisieren
fetcher = ImmoFetcher()

# CSV Erstellung
with open('Archiv/Alte Daten/immonet_komplett.csv', 'w', newline='', encoding='utf-8') as csvfile:
    realestatewriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    realestatewriter.writerow(
        ['immo_url', 'ort', 'plz', 'angebotspreis', 'grundstuecksflaeche', 'wohnflaeche', 'anzahl_zimmer',
         'anzahl_schlafzimmer', 'anzahl_badezimmer', 'gaeste_wc', 'baujahr', 'anzahl_parkplatz', 'immobilienart',
         'immobilienzustand', 'balkon', 'terrasse', 'heizung', 'befeuerungsart', 'etagen', 'geschoss', 'unterkellert',
         'vermietet', 'energie_verbrauch', 'energie_effizienzklasse', 'aufzug', 'barrierefrei', 'denkmalschutz'])

    for x in fetcher.fetch():
        realestatewriter.writerow(
            [x.immo_url, x.ort, x.plz, x.angebotspreis, x.grundstuecksflaeche, x.wohnflaeche, x.anzahl_zimmer,
             x.anzahl_schlafzimmer, x.anzahl_badezimmer, x.gaeste_wc, x.baujahr, x.anzahl_parkplatz, x.immobilienart,
             x.immobilienzustand, x.balkon, x.terrasse, x.heizung, x.befeuerungsart, x.etagen, x.geschoss,
             x.unterkellert, x.vermietet, x.energie_verbrauch, x.energie_effizienzklasse, x.aufzug, x.barrierefrei,
             x.denkmalschutz])
