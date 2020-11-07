import pandas
import datetime as dt


"""Importare il file"""
df = pandas.read_csv('./regioni0611.csv', index_col=None);

"""Selezionare le colonne"""
columns = ['data', 'totale_positivi', 'dimessi_guariti', 'deceduti']
df_regione = pandas.DataFrame(columns=columns);
for index, row in df.iterrows():

    """Scegliere la regione da voler analizzare"""
    if (row['codice_regione'] == 3 ):
        df_regione = df_regione.append(row[columns]);


data = df_regione.loc[:, ['totale_positivi', 'dimessi_guariti', 'deceduti']].astype('float').values

dates = df_regione.loc[:, ['data']].values
dateForPlot = [dt.datetime.strptime(d[0], '%Y-%m-%dT%H:%M:%S').date() for d in dates]
