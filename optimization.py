import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from dataManagement import data

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

totalDays = 230
daysIteration = 200
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


#definisco il modello SEIRD
def odeModel(z, t, beta, alpha, gamma, epsilon):
    S, E, I, R, D = z

    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - alpha*E
    dIdt = alpha*E - gamma*I
    dRdt = gamma*I * (1 - epsilon)
    dDdt = gamma * I * epsilon

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


#deifnisco il solver di EQ differeziali: utilizzo la funzione odeint che si basa su algotitmo LSODA (a passo variabile)
def odeSolver(t, initial_conditions, params):
    initE, initI, initR, initD = initial_conditions
    initS = N - initE - initI - initR - initD
    beta = params['beta']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon = params['epsilon']

    res = odeint(odeModel, [initS, initE, initI, initR, initD], t, args=(beta, alpha, gamma, epsilon))
    return res


#Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore valocato dal modello ed i dati effettivi
def error(params, initial_conditions, tspan, data, timek, timek_1):
    sol = odeSolver(tspan, initial_conditions, params)
    # vengono paragonate le colonne (infetti, guariti, e deceduti), vengono ignorati i Suscettibili e Esposti
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()


if __name__ == "__main__":
#Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile

    "Setting dei valori iniziali, data è la matrice contenente i valori osservati, per approfondimenti si veda in dataManagement.py"
    initI = data[0, 0]
    initE = initI*10
    initR = data[0, 1]
    initD = data[0, 2]

    "Setting dei parametri iniziali"
    T = 20
    gamma = 1 / T
    alpha = 0.52
    epsilon = 0.16
    beta = 0.1
    R0 = beta*T


    initial_conditions = [initE, initI, initR, initD]

    #Creo un vettore da 0 a daysFirstIteration (time discretization)
    tspan = np.arange(0, daysFirstIteration, 1)

    parametersToOptimize = Parameters()
    parametersToOptimize.add('beta', beta, min=0.1, max=0.3)
    parametersToOptimize.add('alpha', alpha)
    parametersToOptimize.add('gamma', gamma,  min=0.04, max=0.05)
    parametersToOptimize.add('epsilon', epsilon)

    "Avvio la prima stima di parametri sul primo range del tempo (da 0 a daysFirstIteration)"
    result = minimize(error, parametersToOptimize, args=(initial_conditions, tspan, data, 0, daysFirstIteration))
    beta0 = result.params['beta'].value
    alpha0 = result.params['alpha'].value
    epsilon0 = result.params['epsilon'].value
    gamma0 = result.params['gamma'].value

    "Salvo i parametri nelle mie liste"
    betaEstimated.append(beta0)
    alphaEstimated.append(alpha0)
    epsilonEstimated.append(epsilon0)
    gammaEstimated.append(gamma0)

    parametersOptimized = Parameters();
    parametersOptimized.add('beta', betaEstimated[0], min=0.1, max=0.3)
    parametersOptimized.add('alpha', alphaEstimated[0])
    parametersOptimized.add('gamma', gammaEstimated[0],  min=0.04, max=0.05)
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
        timek = totalDays - daysIteration + deltaT*k
        timek_1 = totalDays - daysIteration + deltaT*(k+1)


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
        esposti_k = data[timek_analysis, 0]*10

        "Aggiorno le condizioni iniziali considerando i veri dati osservati"
        initial_conditions_k = [esposti_k, data[timek_analysis, 0], data[timek_analysis, 1], data[timek_analysis, 2]]

        parametersToOptimize.add('beta', betaEstimated[k], min=0.1, max=0.3)
        parametersToOptimize.add('alpha', alphaEstimated[k])
        parametersToOptimize.add('gamma', gammaEstimated[k],  min=0.04, max=0.05)
        parametersToOptimize.add('epsilon', epsilonEstimated[k])

        "Stimo i parametri alla k_esima iterazione con le condizioni iniziali aggiornate"
        resultIteration = minimize(error, parametersToOptimize, args=(initial_conditions_k, tspank, data, timek_analysis, timek_1_analysis))

        betak = resultIteration.params['beta'].value
        alphak = resultIteration.params['alpha'].value
        epsilonk = resultIteration.params['epsilon'].value
        gammak = resultIteration.params['gamma'].value

        betaEstimated.append(betak)
        alphaEstimated.append(alphak)
        epsilonEstimated.append(epsilonk)
        gammaEstimated.append(gammak)

        parametersOptimized.add('beta', betaEstimated[k+1], min=0.1, max=0.3)
        parametersOptimized.add('alpha', alphaEstimated[k+1])
        parametersOptimized.add('gamma', gammaEstimated[k+1], min=0.04, max=0.05)
        parametersOptimized.add('epsilon', epsilonEstimated[k+1])

        "Calcolo il modello con i parametri stimati"
        modelk = odeSolver(tspank_model, initial_conditions_k, parametersOptimized)

        "Salvaggio dei dati relativi alla finestra temporale pari a deltaT (da timeK a timeK_1"
        totalModelInfected[timek:timek_1] = modelk[:, 2]
        totalModelRecovered[timek:timek_1] = modelk[:, 3]
        totalModelDeath[timek:timek_1] = modelk[:, 4]
        totalModelExposed[timek:timek_1] = modelk[:, 1]



    datapoints = daysFirstIteration + deltaT*numberOfIntervals
    tspanfinal = np.arange(0, datapoints, 1)
    tspanparemeter = np.arange(totalDays-daysIteration, 200, deltaT)

    plt.plot()

    "Plot dei valori calcolati con il modello"
    plt.plot(tspanfinal, totalModelInfected[:], label="Infected (Model)")
    #plt.plot(tspanfinal, totalModelExposed[:], label="Esposti (Model)")
    #plt.plot(tspanfinal, totalModelRecovered[:], label="Recovered (Model)")
    #plt.plot(tspanfinal, totalModelDeath[:], label="Death(Model)")


    "Plot dei valori osservati"
    plt.plot(tspanfinal, data[0:datapoints, 0], label="Infected(Observed)")
    #plt.plot(tspanfinal, data[0:datapoints, 1], label="Recovered (Observed)")
    #plt.plot(tspanfinal, data[0:datapoints, 2], label="Death (Observed)")

    #plt.plot(betaEstimated)
    plt.legend()
    plt.show()
