import pandas
import datetime as dt

import pandas
import datetime as dt

import csv
import urllib3
import requests


EMILIA_ROMAGNA = 8
LOMBARDIA = 3
csv_url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"

req = requests.get(csv_url)
url_content = req.content
csv_file = open('./region.csv', 'wb')
csv_file.write(url_content)

"""Importare il file"""
df = pandas.read_csv('./region.csv', index_col=None);

"""Selezionare le colonne"""
columns = ['data', 'totale_positivi', 'dimessi_guariti', 'deceduti', 'tamponi', 'totale_casi']
df_regione = pandas.DataFrame(columns=columns);
for index, row in df.iterrows():

    """Scegliere la regione da voler analizzare"""
    if (row['codice_regione'] == EMILIA_ROMAGNA):
        df_regione = df_regione.append(row[columns]);


data = df_regione.loc[:, ['totale_positivi', 'dimessi_guariti', 'deceduti']].astype('float').values

dates = df_regione.loc[:, ['data']].values
dateForPlot = [dt.datetime.strptime(d[0], '%Y-%m-%dT%H:%M:%S').date() for d in dates]



