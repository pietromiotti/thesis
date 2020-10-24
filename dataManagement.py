import pandas
import datetime as dt


data_emilia = pandas.read_csv('./dpc-emilia-new-2.csv', index_col=None);
data = data_emilia.loc[0:239, ['totale_positivi', 'dimessi_guariti', 'deceduti']].values
dates = data_emilia.loc[0:239, ['data']].values

dateForPlot = [dt.datetime.strptime(d[0], '%Y-%m-%dT%H:%M:%S').date() for d in dates]
