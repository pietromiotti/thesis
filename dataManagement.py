import pandas
data_emilia = pandas.read_csv('./dpc-emilia-new.csv', index_col=None);
data = data_emilia.loc[0:229, ['totale_positivi', 'dimessi_guariti', 'deceduti']].values