import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.integrate import odeint, solve_ivp

from dataManagement import data

from numpy import exp, linspace, pi, random, sign, sin
from lmfit import Parameters, fit_report, minimize

'''
We have implemented the SEIRD model, a variation of classical SIR model.
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

#total population of Emilia Romagna
N = 4400000

"""
    In questa fase decidiamo come procedere con il modello: 
        - totalDays: sono i giorni totali di cui si posseggono dati osservati
    A questo punto suddivido i giorni totali in due parti per dividere le stime: 
        - daysIteration sono i giorni che prenderò in considerazione per iterazioni successive
        - daysFirstIteration sono i giorni che predispongo per la prima iterazione, necessaria per una stima iniziale
"""

totalDays = 200
daysIteration = 170
daysFirstIteration = totalDays - daysIteration

"""
   Definisco il deltaT per la time discretization, l'ampiezza degli intervalli su cui vado a ricalcolare la minimizzazione.
   Da questa derivo il numero totale degli intervalli fornito dalla divisione troncata del numero di giorni adibiti a questa analisi 
   (daysIteration) e l'ampiezza dell'intervallo
   """
deltaT = 10
mindelta = 0
numbersOfInterval = daysIteration // deltaT

"""
Inizializzo i vettori in cui andrò a memorizzare tutti i parametri nei singoli intervalli 
\\TODO: REFACTORING HERE IS REQUIRED
"""

betaEstimated = []
alphaEstimated = []
epsilonEstimated = []
gammaEstimated = []
roEstimated = []


def betaFunction(t,ro):
    if(t < daysFirstIteration):
        position = 0
    else:
        position = int(((t - daysFirstIteration)//deltaT))
    tk = daysFirstIteration + deltaT*(position -1)
    result= betaEstimated[position]*(1 - ro*(t - tk)/t)
    return result


#definisco il modello SEIRD
def odeModel(z, t, beta, alpha, gamma, epsilon):
    S, E, I, R, D = z

    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - alpha*E
    dIdt = alpha*E - gamma*I
    dRdt = gamma*I * (1 - epsilon)
    dDdt = gamma * I * epsilon

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def fodeModel(z, t, ro, alpha, gamma, epsilon):
    S, E, I, R, D = z

    dSdt = -betaFunction(t,ro)*S*I/N
    dEdt = betaFunction(t,ro)*S*I/N - alpha*E
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



def fodeSolver(t, initial_conditions, params):
    initE, initI, initR, initD = initial_conditions
    initS = N - initE - initI - initR - initD
    ro = params['ro']
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon = params['epsilon']

    res = odeint(fodeModel, [initS, initE, initI, initR, initD], t, args=(ro, alpha, gamma, epsilon))
    return res

#Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore valocato dal modello ed i dati effettivi
def error(params, initial_conditions, tspan, data, timek, timek_1):
    sol = odeSolver(tspan, initial_conditions, params)
    # vengono paragonate le colonne (infetti, guariti, e deceduti), vengono ignorati i Suscettibili e Esposti
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()

def errorRO(ro, iteration, initial_conditions, tspan, data, timek, timek_1):
    sol = fodeSolver(tspan, initial_conditions, [ro, alphaEstimated[iteration], gammaEstimated[iteration], epsilonEstimated[iteration]])
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()

if __name__ == "__main__":
#Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile


    initI = data[0, 0]
    initE = initI*10
    #initE = initI*10
    initR = data[0, 1]
    initD = data[0, 2]

    #Setto i parametri iniziali
    T = 20
    gamma = 1 / T
    #gamma = 0.44
    #alpha = 1 / 3.2
    alpha = 0.52
    #epsilon = 0.7
    epsilon = 0.16
    #beta = 0.077
    beta = 0.1
    R0 = beta*T

    #I giorni totali sono in realtà 208 (in dpc-emilia), prendiamo i primi 200

    #daysFirstIteration = 200

    initial_conditions = [initE, initI, initR, initD]

    #Creo un vettore da 0 a daysFirstIteration (time discretization)
    tspan = np.arange(0, daysFirstIteration, 1)

    parametersToOptimize = Parameters()
    parametersToOptimize.add('beta', beta, min=0.1, max=0.3)
    parametersToOptimize.add('alpha', alpha)
    parametersToOptimize.add('gamma', gamma,  min=0.04, max=0.05)
    parametersToOptimize.add('epsilon', epsilon)

    """ 
    Avvio la prima stima di parametri sulla primo range di parametri (da 0 a daysFirstIteration)
    """
    result = minimize(error, parametersToOptimize, args=(initial_conditions, tspan, data, 0, daysFirstIteration))
    beta0 = result.params['beta'].value
    alpha0 = result.params['alpha'].value
    epsilon0 = result.params['epsilon'].value
    gamma0 = result.params['gamma'].value

    """
    Salvo i parametri nelle mie liste
    """

    ro0 = 0.9
    #beta0, alpha0, gamma0, epsilon0 = result[""]
    betaEstimated.append(beta0)
    alphaEstimated.append(alpha0)
    epsilonEstimated.append(epsilon0)
    gammaEstimated.append(gamma0)
    roEstimated.append(ro0)

    parametersToOptimize.add('beta', betaEstimated[0], min=0.1, max=0.3)
    parametersToOptimize.add('alpha', alphaEstimated[0])
    parametersToOptimize.add('gamma', gammaEstimated[0],  min=0.04, max=0.05)
    parametersToOptimize.add('epsilon', epsilonEstimated[0])
   
    model_init = odeSolver(tspan, initial_conditions, parametersToOptimize)

   
    indexInit = totalDays - daysIteration - 1

    totalModelInfected = []
    totalModelRecovered = []
    totalModelDeath = []

    totalModelInfected[0:daysFirstIteration] = model_init[:, 2]
    totalModelRecovered[0:daysFirstIteration] = model_init[:, 3]
    totalModelDeath[0:daysFirstIteration] = model_init[:, 4]


    for i in range(0, numbersOfInterval):

        timek = totalDays - daysIteration + deltaT*i
        timek_analysis = timek - mindelta

        timek_1 = totalDays - daysIteration + deltaT*(i+1)
        timek_1_analysis = timek_1



        tspank = np.arange(timek_analysis, timek_1_analysis, 1)
        tspank_model = np.arange(timek, timek_1, 1)


        esposti_k = data[timek_analysis, 0]*10

        initial_conditions_k = [esposti_k, data[timek_analysis, 0], data[timek_analysis, 1], data[timek_analysis, 2]]

        parametersToOptimize.add('beta', betaEstimated[i], min=0.1, max=0.3)
        parametersToOptimize.add('alpha', alphaEstimated[i])
        parametersToOptimize.add('gamma', gammaEstimated[i],  min=0.04, max=0.05)
        parametersToOptimize.add('epsilon', epsilonEstimated[i])
        #resultIteration = leastsq(error, np.asarray([beta, alpha, gamma, epsilon]), args=(initial_conditions_k, tspank, data, timek_analysis, timek_1_analysis))
        resultIteration = minimize(error, parametersToOptimize, args=(initial_conditions_k, tspank, data, timek_analysis, timek_1_analysis))

        betak = resultIteration.params['beta'].value
        alphak = resultIteration.params['alpha'].value
        epsilonk = resultIteration.params['epsilon'].value
        gammak = resultIteration.params['gamma'].value


        betaEstimated.append(betak)
        alphaEstimated.append(alphak)
        epsilonEstimated.append(epsilonk)
        gammaEstimated.append(gammak)

        parametersToOptimize.add('beta', betaEstimated[i+1], min=0.1, max=0.3)
        parametersToOptimize.add('alpha', alphaEstimated[i+1])
        parametersToOptimize.add('gamma', gammaEstimated[i+1], min=0.04, max=0.05)
        parametersToOptimize.add('epsilon', epsilonEstimated[i+1])
        #Calcolo del modello di ampiezza della k_esima iterazione
        modelk = odeSolver(tspank_model, initial_conditions_k, parametersToOptimize)

        #Salvaggio dei dati relativi alla finestra temporale pari a deltaT
        totalModelInfected[timek:timek_1] = modelk[:, 2]
        totalModelRecovered[timek:timek_1] = modelk[:, 3]
        totalModelDeath[timek:timek_1] = modelk[:, 4]

    datapoints = daysFirstIteration + deltaT*numbersOfInterval
    tspanfinal = np.arange(0, datapoints, 1)
    tspanparemeter = np.arange(totalDays-daysIteration, 200,deltaT)

    plt.plot()
    #ro0 = 0.9


    #Plot initial Valued Model
    #plt.plot(tspan, I, label="Infected (Model)")
    #plt.plot(tspan, R, label="Recovered (Model)")
    #plt.plot(tspan, D, label="Death (Model)")

    #Plot Model with estimated parameters
    #print(totalModel)
    plt.plot(tspanfinal, totalModelInfected[:], label="Infected (Model 2)")
    plt.plot(tspanfinal, totalModelRecovered[:], label="Recovered (Model 2)")
    plt.plot(tspanfinal, totalModelDeath[:], label="Death(Model 2)")

    #plt.plot(tspan, E1, label="Exposed (Model 2 )")
    #plt.plot(tspan, I1, label="Infected (Model)")
    #plt.plot(tspan, R1, label="Recovered (Model)")
    #plt.plot(tspan, D1, label="Death (Model)")

    #Plot Obeserved Value

    plt.plot(tspanfinal, data[0:datapoints, 0], label="Infected(Observed)")
    plt.plot(tspanfinal, data[0:datapoints, 1], label="Recovered (Observed)")
    plt.plot(tspanfinal, data[0:datapoints, 2], label="Death (Observed)")


    plt.legend()
    plt.show()
