import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import matplotlib.dates as mdates
import datetime as dt

from dataManagement import data, dateForPlot

from lmfit import Parameters, minimize

'''
Abbiamo implementato il modello SEIRD, una variazione del classico modello SIR.
dS/dt = - (beta/N) * S *I
dE/dT =  (beta/N) * S *I - alpha * E
dI/dt = alpha * E - 1/T * I 
dR/dT =  (1 - f)/T * I
dD/dt = f/T * I 

beta = infection rate
alpha = incubation rate
T = average infectious period
gamma = 1/T
epsilon = fraction of all removed individuals who die
'''

"Popolazione dell'Emilia Romagna"
N = 4400000

"""
In questa fase decidiamo come procedere con il modello: 
    - totalDays: sono i giorni totali di cui possediamo i dati osservati
A questo punto suddivido i giorni totali in due parti per dividere le stime: 
    - daysIteration sono i giorni che decidiamo di prendere in considerazione per il processo di ottimizzazione discretizzato
    - daysFirstIteration sono i giorni che predisponiamo per la prima fase del processo di ottimizzazione in cui facciamo una prima stima dei parametri
"""

totalDays = 240
daysIteration = 220
daysFirstIteration = totalDays - daysIteration

"""
   Definisco il deltaT per la discretizzazione del tempo: l'ampiezza degli intervalli su cui vado a ricalcolare la minimizzazione.
   Da questa derivo il numero totale degli intervalli (numberOfIntervals) fornito dalla divisione troncata del numero di giorni adibiti a questa analisi 
   (daysIteration) e l'ampiezza dell'intervallo (deltaT)

"""
deltaT = 10
mindelta = 0
numberOfIntervals = daysIteration // deltaT

"""
Inizializzo i vettori in cui andrò a memorizzare tutti i parametri nei vari intervalli di tempo deltaT
"""

betaEstimated = []
alphaEstimated = []
epsilonEstimated = []
gammaEstimated = []
roEstimated = []

betaNewEstimated = []


def betaFunction(t, ro):
    betaFirstInterval = 0
    position = int(((t - daysFirstIteration) // deltaT))
    if (t - daysFirstIteration) % deltaT == 0:
        betaFirstInterval = 1
        realPosition = position - 1
    else:
        realPosition = position

    tk = daysFirstIteration + deltaT * (realPosition)
    result = betaEstimated[realPosition] * (1 - ro * (t - tk) / t)
    if(betaFirstInterval):
        betaEstimated.append(result)
    betaNewEstimated.append(result)
    return result


# definisco il modello SEIRD
def odeModel(z, t, beta, alpha, gamma, epsilon):
    S, E, I, R, D = z

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I * (1 - epsilon)
    dDdt = gamma * I * epsilon

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def fodeModel(z, t, ro, alpha, gamma, epsilon):
    S, E, I, R, D = z

    dSdt = -betaFunction(t, ro) * S * I / N
    dEdt = betaFunction(t, ro) * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I * (1 - epsilon)
    dDdt = gamma * I * epsilon

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


# deifnisco il solver di EQ differeziali: utilizzo la funzione odeint che si basa su algotitmo LSODA (a passo variabile)
def odeSolver(t, initial_conditions, params):
    initE, initI, initR, initD = initial_conditions
    initS = N - initE - initI - initR - initD
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon = params['epsilon']

    res = odeint(odeModel, [initS, initE, initI, initR, initD], t, args=(beta, alpha, gamma, epsilon))
    return res


def fodeSolver(t, initial_conditions, params):
    initE, initI, initR, initD = initial_conditions
    initS = N - initE - initI - initR - initD
    ro = params['ro']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon = params['epsilon']

    res = odeint(fodeModel, [initS, initE, initI, initR, initD], t, args=(ro, alpha, gamma, epsilon))
    return res


# Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore valocato dal modello ed i dati effettivi
def error(params, initial_conditions, tspan, data, timek, timek_1):
    sol = odeSolver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()


def errorRO(params, initial_conditions, tspan, data, timek, timek_1):
    sol = fodeSolver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()




def prevision(daysPrevision):
    tspan = np.arange(totalDays, totalDays + daysPrevision, 1)

    exposed = data[totalDays-1, 0] * 10
    initial_conditions = [exposed, data[totalDays-1, 0], data[totalDays-1, 1], data[totalDays-1, 2]]

    previsionParameters = Parameters()

    previsionParameters.add('ro', roEstimated[roEstimated.__len__() - 1], min=0, max=1)
    previsionParameters.add('alpha', alphaEstimated[alphaEstimated.__len__() - 1])
    previsionParameters.add('gamma', gammaEstimated[gammaEstimated.__len__() - 1], min=0.04, max=0.05)
    previsionParameters.add('epsilon', epsilonEstimated[epsilonEstimated.__len__() - 1])

    sol = fodeSolver(tspan, initial_conditions, previsionParameters)
    return sol

if __name__ == "__main__":
    # Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile
    "Setting dei valori iniziali, data è la matrice contenente i valori osservati, per approfondimenti si veda in dataManagement.py"
    initI = data[0, 0]
    initE = initI * 10
    initR = data[0, 1]
    initD = data[0, 2]

    "Setting dei parametri iniziali"
    T = 20
    gamma = 1 / T
    alpha = 0.52
    epsilon = 0.16
    beta = 0.1
    R0 = beta * T

    initial_conditions = [initE, initI, initR, initD]

    # Creo un vettore da 0 a daysFirstIteration (time discretization)
    tspan = np.arange(0, daysFirstIteration, 1)

    parametersToOptimize = Parameters()
    parametersToOptimize.add('beta', beta, min=0)
    parametersToOptimize.add('alpha', alpha)
    parametersToOptimize.add('gamma', gamma, min=0.04, max=0.05)
    parametersToOptimize.add('epsilon', epsilon)

    "Avvio la prima stima di parametri sul primo range del tempo (da 0 a daysFirstIteration)"
    result = minimize(error, parametersToOptimize, args=(initial_conditions, tspan, data, 0, daysFirstIteration))

    beta0 = result.params['beta'].value
    alpha0 = result.params['alpha'].value
    epsilon0 = result.params['epsilon'].value
    gamma0 = result.params['gamma'].value
    ro0 = 0.9

    "Salvo i parametri nelle mie liste"
    betaEstimated.append(beta0)
    alphaEstimated.append(alpha0)
    epsilonEstimated.append(epsilon0)
    gammaEstimated.append(gamma0)
    roEstimated.append(ro0)

    parametersOptimized = Parameters();
    parametersOptimized.add('beta', betaEstimated[0], min=0)
    parametersOptimized.add('alpha', alphaEstimated[0])
    parametersOptimized.add('gamma', gammaEstimated[0], min=0.04, max=0.05)
    parametersOptimized.add('epsilon', epsilonEstimated[0])

    "Calcolo le soluzioni del sistema con i parametri stimati nel primo intervallo da 0 a daysFirstIteration"
    model_init = odeSolver(tspan, initial_conditions, parametersOptimized)

    indexInit = totalDays - daysIteration - 1

    "Vettori in cui salvo le soluzioni (Infetti, Guariti e Morti, Esposti)"
    totalModelInfected = []
    totalModelRecovered = []
    totalModelDeath = []
    totalModelExposed = []

    "Memorizzo i primi valori (da 0 a daysFirstIteration) delle soluzioni di Infetti, Guariti, Morti, Esposti"
    totalModelInfected[0:daysFirstIteration] = model_init[:, 2]
    totalModelRecovered[0:daysFirstIteration] = model_init[:, 3]
    totalModelDeath[0:daysFirstIteration] = model_init[:, 4]
    totalModelExposed[0:daysFirstIteration] = model_init[:, 1]

    "Definisco la mia k-esima iterata"
    for k in range(0, numberOfIntervals):
        """
        Definisco gli estremi del mio intervallo
              <-deltaT->
        -----|-----------|----------------
            timek        timek_1
        """
        timek = totalDays - daysIteration + deltaT * k
        timek_1 = totalDays - daysIteration + deltaT * (k + 1)

        """
        Allargo (volendo) la finestra di analisi per la stima dei parametri considerando elementi più "vecchi"
            <-midelta->  <-deltaT->
        --|------------|-----------|----------------
                    timek        timek_1
        timek_analysis           timek_1_analysis  
        """
        timek_analysis = timek - mindelta
        timek_1_analysis = timek_1

        tspank = np.arange(timek_analysis, timek_1_analysis, 1)
        tspank_model = np.arange(timek, timek_1, 1)

        "Aggiorno gli esposti alla k_esima iterazione"
        exposed_k = data[timek_analysis, 0]*10

        "Aggiorno le condizioni iniziali considerando i veri dati osservati"
        initial_conditions_k = [exposed_k, data[timek_analysis, 0], data[timek_analysis, 1], data[timek_analysis, 2]]

        parametersToOptimize.add('ro', roEstimated[k], min=0, max=1)
        parametersToOptimize.add('alpha', alphaEstimated[k])
        parametersToOptimize.add('gamma', gammaEstimated[k], min=0.04, max=0.05)
        parametersToOptimize.add('epsilon', epsilonEstimated[k])

        "Stimo i parametri alla k_esima iterazione con le condizioni iniziali aggiornate"
        resultForcedIteration = minimize(errorRO, parametersToOptimize,
                                         args=(initial_conditions_k, tspank, data, timek_analysis, timek_1_analysis))

        rok = resultForcedIteration.params['ro'].value
        alphak = resultForcedIteration.params['alpha'].value
        epsilonk = resultForcedIteration.params['epsilon'].value
        gammak = resultForcedIteration.params['gamma'].value

        roEstimated.append(rok)
        alphaEstimated.append(alphak)
        epsilonEstimated.append(epsilonk)
        gammaEstimated.append(gammak)

        parametersOptimized.add('ro', roEstimated[k + 1], min=0, max=1)
        parametersOptimized.add('alpha', alphaEstimated[k + 1])
        parametersOptimized.add('gamma', gammaEstimated[k + 1], min=0.04, max=0.05)
        parametersOptimized.add('epsilon', epsilonEstimated[k + 1])

        "Calcolo il modello con i parametri stimati"
        modelfk = fodeSolver(tspank_model, initial_conditions_k, parametersOptimized)

        "Salvaggio dei dati relativi alla finestra temporale pari a deltaT (da timeK a timeK_1"
        totalModelInfected[timek:timek_1] = modelfk[:, 2]
        totalModelRecovered[timek:timek_1] = modelfk[:, 3]
        totalModelDeath[timek:timek_1] = modelfk[:, 4]
        totalModelExposed[timek:timek_1] \
            = modelfk[:, 1]




    #Perform my Prevision

    daysPrevision = 10
    myprevision = prevision(daysPrevision)


    totalModelInfected[totalDays:totalDays+daysPrevision] = myprevision[:, 2]


    datapoints = daysFirstIteration + deltaT * numberOfIntervals + daysPrevision


    #Convert DataTime to String in order to Plot the data
    lastDay = dateForPlot[dateForPlot.__len__()-1]
    dayPrevision = lastDay + dt.timedelta(days=daysPrevision+1)
    days = mdates.drange(dateForPlot[0], dayPrevision, dt.timedelta(days=1))
    daysObserved = mdates.drange(dateForPlot[0], dayPrevision, dt.timedelta(days=1))

    daysArangeToPrint = mdates.drange(dateForPlot[dateForPlot.__len__()-1], dayPrevision, dt.timedelta(days=1))

    #print(daysArangeToPrint)

    for i in range(0, daysPrevision):
        print(str(mdates.num2date(daysArangeToPrint[i])) + " " + str(totalModelInfected[totalDays+i]) + "\n")

    #plt.plot(betaNewEstimated)
    # plt.plot(epsilonEstimated)
    # plt.plot(alphaEstimated)
    #plt.plot(betaEstimated)


    "Plot dei valori calcolati con il modello"
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=8))
    plt.plot(days, totalModelInfected[:], label="Infected (Model and Predicted)")


    # plt.plot(tspanfinal, totalModelExposed[:], label="Esposti (Model)")
    #plt.plot(tspanfinal, totalModelRecovered[:], label="Recovered (Model)")
    #plt.plot(tspanfinal, totalModelDeath[:], label="Death(Model)")

    "Plot dei valori osservati"
    plt.plot(dateForPlot[:], data[0:datapoints, 0], label="Infected(Observed)")
    #plt.plot(tspanfinal, data[0:datapoints, 1], label="Recovered (Observed)")
    #plt.plot(tspanfinal, data[0:datapoints, 2], label="Death (Observed)")

    # plt.plot(betaEstimated)

    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.show()
