from dataManagement import data

class constants :


    """
    Parametri Iniziali
    """
    initRO = 0.9
    initT = 20
    initAlpha = 0.52
    initEpsilon = 0.16
    initBeta = 0.1


    firstDay = 0

    #Popolazione dell'Emilia Romagna
    N = 4459000

    """
    In questa fase decidiamo come procedere con il modello: 
        - totalDays: sono i giorni totali di cui possediamo i dati osservati
    A questo punto suddivido i giorni totali in due parti per dividere le stime: 
        - daysIteration sono i giorni che decidiamo di prendere in considerazione per il processo di ottimizzazione discretizzato
        - daysFirstIteration sono i giorni che predisponiamo per la prima fase del processo di ottimizzazione in cui facciamo una prima stima dei parametri
    """
    totalDays = len(data) - 10

    daysFirstIteration = 30

    daysIteration = totalDays - firstDay - daysFirstIteration

    #daysFirstIteration = firstDay + totalDays - daysIteration

    """
       Definisco il deltaT iniziale su cui fare la mia prima calibrazione con la prima iterazione

    """
    initDeltaT = 10

    """
        Definitisco la lunghezza dei miei intervalli per le calibrazioni

    """
    MIN_INTERVAL = 7
    MEDIUM_INTERVAL = 10
    MAX_INTERVAL = 15

    """
        Definisco le mie tolleranze di errore su cui poi modificare l'ampiezza degli intervalli

    """
    ERROR_RANGE_MIN_INTERVAL = 600
    ERROR_RANGE_MAX_INTERVAL = 300


    daysPrevision = 60

