import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import matplotlib.dates as mdates
import datetime as dt

from dataManagement import data, dateForPlot
from constants import constants

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
N = constants.N

"""
In questa fase decidiamo come procedere con il modello: 
    - totalDays: sono i giorni totali di cui possediamo i dati osservati
A questo punto suddivido i giorni totali in due parti per dividere le stime: 
    - daysIteration sono i giorni che decidiamo di prendere in considerazione per il processo di ottimizzazione discretizzato
    - daysFirstIteration sono i giorni che predisponiamo per la prima fase del processo di ottimizzazione in cui facciamo una prima stima dei parametri
"""

totalDays = constants.totalDays
daysIteration = constants.daysIteration
daysFirstIteration = constants.daysFirstIteration

"""
   Definisco il deltaT iniziale

"""
initDeltaT = constants.initDeltaT
numberOfIteration = 0


"""
Inizializzo i vettori in cui andrò a memorizzare tutti i parametri nei vari intervalli di tempo deltaT
"""


betaEstimated = []
alphaEstimated = []
epsilonEstimated = []
gammaEstimated = []
roEstimated = []

betaNewEstimated = []

firstDayIteration = []
lastDayIteration = []


def betaFunction(t, ro, iteration):
    boundarybeta = 0

    if ((t >= (lastDayIteration[iteration]-1))):
        boundarybeta= 1

    tk = firstDayIteration[iteration]
    """
    Scegliere se usare beta razionale od esponenziale
       Se si cambia, bisogna cambiare anche i vincoli di RO
    """
    #razionale
    #result = betaEstimated[iteration] * (1 - ro * (t - tk) / t)
    #esponenziale
    result = betaEstimated[iteration] * np.e**(-ro * (t - tk))

    if(boundarybeta):
        if (len(betaEstimated)-1==iteration):
            betaEstimated.append(result)
        else:
            betaEstimated[iteration+1] = result
    else:
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


def fodeModel(z, t, ro, alpha, gamma, epsilon, iteration):
    S, E, I, R, D = z

    dSdt = -betaFunction(t, ro, iteration) * S * I / N
    dEdt = betaFunction(t, ro, iteration) * S * I / N - alpha * E
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


def fodeSolver(t, initial_conditions, params, iteration):
    initE, initI, initR, initD = initial_conditions
    initS = N - initE - initI - initR - initD
    ro = params['ro']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon = params['epsilon']
    res = odeint(fodeModel, [initS, initE, initI, initR, initD], t, args=(ro, alpha, gamma, epsilon, iteration))
    return res


# Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore valocato dal modello ed i dati effettivi
def error(params, initial_conditions, tspan, data, timek, timek_1):
    sol = odeSolver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()


def errorRO(params, initial_conditions, tspan, data, iteration):
    sol = fodeSolver(tspan, initial_conditions, params, iteration)
    return (sol[:, 2:5] - data[firstDayIteration[iteration]:lastDayIteration[iteration]]).ravel()



def deltaTCalibration(infected, iteration):
    firstDay = firstDayIteration[iteration]
    lastDay = lastDayIteration[iteration]

    totalInfectedObserved = sum(data[firstDay:lastDay+1,0])
    totalInfectedModel = sum(infected[firstDay:lastDay+1])

    intervalTime = lastDay - firstDay

    rateInfectedModel = totalInfectedModel/intervalTime
    rateInfectedObserved = totalInfectedObserved/intervalTime

    if (abs(rateInfectedModel - rateInfectedObserved) >= constants.ERROR_RANGE_MIN_INTERVAL):
        deltaT = constants.MIN_INTERVAL

    elif(abs(rateInfectedModel - rateInfectedObserved) <= constants.ERROR_RANGE_MAX_INTERVAL):
        deltaT = constants.MAX_INTERVAL

    else:
        deltaT = constants.MEDIUM_INTERVAL


    firstDayIteration.append(lastDay)
    lastDayIteration.append(lastDay + deltaT)



def prevision(daysPrevision):
    tspan = np.arange(totalDays, totalDays + daysPrevision, 1)

    exposed = data[lastDayIteration[numberOfIteration]-1, 0] * 10
    initial_conditions = [exposed, data[lastDayIteration[numberOfIteration]-1, 0], data[lastDayIteration[numberOfIteration]-1, 1], data[lastDayIteration[numberOfIteration]-1, 2]]

    previsionParameters = Parameters()

    previsionParameters.add('ro', roEstimated[roEstimated.__len__() - 1])
    #previsionParameters.add('beta', betaNewEstimated[betaNewEstimated.__len__() - 1])
    previsionParameters.add('alpha', alphaEstimated[alphaEstimated.__len__() - 1])
    previsionParameters.add('gamma', gammaEstimated[gammaEstimated.__len__() - 1], min=0.04, max=0.05)
    previsionParameters.add('epsilon', epsilonEstimated[epsilonEstimated.__len__() - 1])

    sol = fodeSolver(tspan, initial_conditions, previsionParameters, numberOfIteration)
    return sol

if __name__ == "__main__":
    # Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile
    "Setting dei valori iniziali, data è la matrice contenente i valori osservati, per approfondimenti si veda in dataManagement.py"
    initI = data[0, 0]
    initE = initI * 10
    initR = data[0, 1]
    initD = data[0, 2]

    "Setting dei parametri iniziali"
    T = constants.initT
    gamma = 1 / T
    alpha = constants.initAlpha
    epsilon = constants.initEpsilon
    beta = constants.initBeta


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
    ro0 = constants.initRO

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

    firstDayIteration.append(daysFirstIteration)
    lastDayIteration.append(daysFirstIteration+initDeltaT)

    finishedIteration = 1
    k=0
    "Definisco la mia k-esima iterata"
    while(finishedIteration):
        """
        Definisco gli estremi del mio intervallo
              <-deltaT->
        -----|-----------|----------------
            timek        timek_1
        """
        timek = firstDayIteration[k]
        timek_1 = lastDayIteration[k]


        tspank = np.arange(timek, timek_1, 1)

        tspank_model = np.arange(timek, timek_1, 1)

        "Aggiorno gli esposti alla k_esima iterazione"
        exposed_k = data[timek, 0]*10

        "Aggiorno le condizioni iniziali considerando i veri dati osservati"
        initial_conditions_k = [exposed_k, data[timek, 0], data[timek, 1], data[timek, 2]]

        parametersToOptimize.add('ro', roEstimated[k], min=0)
        parametersToOptimize.add('alpha', alphaEstimated[k])
        parametersToOptimize.add('gamma', gammaEstimated[k], min=0.04, max=0.05)
        parametersToOptimize.add('epsilon', epsilonEstimated[k])

        "Stimo i parametri alla k_esima iterazione con le condizioni iniziali aggiornate"
        resultForcedIteration = minimize(errorRO, parametersToOptimize,
                                         args=(initial_conditions_k, tspank, data, k))

        rok = resultForcedIteration.params['ro'].value
        alphak = resultForcedIteration.params['alpha'].value
        epsilonk = resultForcedIteration.params['epsilon'].value
        gammak = resultForcedIteration.params['gamma'].value

        roEstimated.append(rok)
        alphaEstimated.append(alphak)
        epsilonEstimated.append(epsilonk)
        gammaEstimated.append(gammak)

        parametersOptimized.add('ro', roEstimated[k + 1], min=0)
        parametersOptimized.add('alpha', alphaEstimated[k + 1])
        parametersOptimized.add('gamma', gammaEstimated[k + 1], min=0.04, max=0.05)
        parametersOptimized.add('epsilon', epsilonEstimated[k + 1])

        "Calcolo il modello con i parametri stimati"
        modelfk = fodeSolver(tspank_model, initial_conditions_k, parametersOptimized, k)

        "Salvaggio dei dati relativi alla finestra temporale pari a deltaT (da timeK a timeK_1"
        totalModelInfected[timek:timek_1] = modelfk[:, 2]
        totalModelRecovered[timek:timek_1] = modelfk[:, 3]
        totalModelDeath[timek:timek_1] = modelfk[:, 4]
        totalModelExposed[timek:timek_1] \
            = modelfk[:, 1]

        deltaTCalibration(totalModelInfected, k)

        if(lastDayIteration[k+1]>totalDays):
            finishedIteration = 0
            numberOfIteration = k
        else:
            k=k+1



    #Perform my Prevision

    daysPrevision = constants.daysPrevision
    myprevision = prevision(daysPrevision)

    totalModelInfected[lastDayIteration[k]:lastDayIteration[k]+daysPrevision] = myprevision[:, 2]



    datapoints = lastDayIteration[k] + daysPrevision


    #Convert DataTime to String in order to Plot the data
    lastDay = dateForPlot[lastDayIteration[numberOfIteration]-1]

    dayPrevision = lastDay + dt.timedelta(days=daysPrevision+1)
    days = mdates.drange(dateForPlot[0], dayPrevision, dt.timedelta(days=1))

    daysArangeToPrint = mdates.drange(dateForPlot[lastDayIteration[numberOfIteration]] , dayPrevision, dt.timedelta(days=1))

    # print(daysArangeToPrint)

    for i in range(0, daysPrevision):
        print(str(mdates.num2date(daysArangeToPrint[i])) + " " + str(totalModelInfected[lastDayIteration[numberOfIteration] + i]) + "\n")
    #R0 = betaNewEstimated * T

    #plt.plot(epsilonEstimated)
    #plt.plot(alphaEstimated)
    #plt.plot(betaEstimated)
    #plt.plot(betaNewEstimated)
    #plt.plot(R0)


    "Plot dei valori calcolati con il modello"

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=8))
    plt.plot(days, totalModelInfected[:], label="Infected (Model and Predicted)")

    plt.plot(dateForPlot[0:lastDayIteration[k]], data[0:lastDayIteration[k], 0], label="Infected(Observed)")


    #plt.plot(dateForPlot[0:lastDayIteration[numberOfIteration]], totalModelExposed[:], label="Esposti (Model)")
    #plt.plot(dateForPlot[0:lastDayIteration[numberOfIteration]], totalModelRecovered[:], label="Recovered (Model)")
    #plt.plot(dateForPlot[0:lastDayIteration[numberOfIteration]], totalModelDeath[:], label="Death(Model)")

    "Plot dei valori osservati"
    #print(totalModelExposed)

    #plt.plot(dateForPlot[0:lastDayIteration[k]], data[0:lastDayIteration[k], 1], label="Recovered (Observed)")
    #plt.plot(dateForPlot[0:lastDayIteration[k]], data[0:lastDayIteration[k], 2], label="Death (Observed)")

    # plt.plot(betaEstimated)

    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.show()
