import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.integrate import odeint, solve_ivp

from dataManagement import data

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

#definisco il modello SEIRD
def odeModel(z, t, beta, alpha, gamma, epsilon):
    S, E, I, R, D = z
    N = S + E + I + R + D

    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - alpha*E
    dIdt = alpha*E - gamma*I
    dRdt = gamma*I * (1 - epsilon)
    dDdt = gamma * I * epsilon

    return [dSdt, dEdt, dIdt, dRdt, dDdt]



#deifnisco il solver di EQ differeziali: utilizzo la funzione odeint che si basa su algotitmo LSODA (a passo variabile)
def odeSolver(t, initial_conditions, params):
    initS, initE, initI, initR, initD = initial_conditions
    beta = params[0]
    alpha = params[1]
    gamma = params[2]
    epsilon = params[3]

    res = odeint(odeModel, [initS, initE, initI, initR, initD], t, args=(beta, alpha, gamma, epsilon))
    return res

#Definisco la funzione "error" deve essere minimizzata. Questa funzione contiene la differenza tra il valore valocato dal modello ed i dati effettivi
#vengono paragonate le colonne (infetti, recovered, e deceduti), vengono ignorati i Suscettibili e Esposti
def error(params, initial_conditions, tspan, data, timek, timek_1):
    sol = odeSolver(tspan, initial_conditions, params)
    return (sol[:, 2:5] - data[timek:timek_1]).ravel()


if __name__ == "__main__":
#Setto i parametri iniziali
    initS = 4400000

#Prendo i valori iniziali degli infetti, dei recovered e dei deceduti direttamente dal database della protezione civile
    initI = data[0, 0]
    initE = initI*10
    #initE = initI*10
    initR = data[0, 1]
    initD = data[0, 2]

    #Setto i parametri iniziali
    T = 14
    #gamma = 1 / T
    gamma = 0.030
    #alpha = 1 / 3.2
    alpha = 0.0052
    #epsilon = 0.7
    epsilon = 0.16
    #beta = 0.077
    beta = 4/100
    R0 = beta*T

    #I giorni totali sono in realtà 208 (in dpc-emilia), prendiamo i primi 200
    totalDays = 200
    daysIteration = 150
    daysFirstIteration = totalDays - daysIteration
    #daysFirstIteration = 200

    initial_conditions = [initS, initE, initI, initR, initD]

    #Creo un vettore da 0 a 200 (time discretization)
    tspan = np.arange(0, daysFirstIteration, 1)

    #Risolvo il sistema di equazioni differenziali con le condizioni iniziali (in particolare con le condizioni iniziali di beta, apha, gamma and epsilon)
    #sol = odeSolver(tspan, initial_conditions, np.asarray([beta, alpha, gamma, epsilon]))
    #S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

    #define the delta
    deltaT = 10
    numbersOfInterval = daysIteration//deltaT

    betaEstimated = []
    alphaEstimated = []
    epsilonEstimated = []
    gammaEstimated = []
    S_est = []
    E_est = []
    I_est = []
    R_est = []
    D_est = []

    result = leastsq(error, np.asarray([beta, alpha, gamma, epsilon]), args=(initial_conditions, tspan, data, 0, daysFirstIteration))

    print(result[0])
    beta0, alpha0, gamma0, epsilon0 = result[0]
    betaEstimated.append(beta0)
    alphaEstimated.append(alpha0)
    epsilonEstimated.append(epsilon0)

    gammaEstimated.append(gamma0)


    model_init = odeSolver(tspan, initial_conditions, [betaEstimated[0], alphaEstimated[0], gammaEstimated[0], epsilonEstimated[0]])

    indexInit = totalDays - daysIteration - 1
    S_init, E_init, I_init, R_init, D_init = model_init[indexInit, 0], model_init[indexInit, 1], model_init[indexInit, 2], model_init[indexInit, 3],model_init[indexInit, 4]

    totalModel= []
    totalModel[0:daysFirstIteration] = model_init[:, 2]
    #totalModel[0:50,1], totalModel[0:50,2], totalModel[0:50,3], totalModel[0:50,4] = model_init

    S_est.append(S_init)
    E_est.append(E_init)
    I_est.append(I_init)
    R_est.append(R_init)
    D_est.append(D_init)

    mindelta = 10

    for i in range(0, numbersOfInterval):
        timek = totalDays - daysIteration + deltaT*i
        timek_analysis = timek - mindelta
        timek_1 = totalDays - daysIteration + deltaT*(i+1)
        if not (timek_1 + mindelta > totalDays):
            timek_1_analysis = timek_1 + mindelta
        else:
            timek_1_analysis = timek_1
        tspank = np.arange(timek_analysis, timek_1_analysis, 1)


        initial_conditions_k = [S_est[i], E_est[i], I_est[i], R_est[i], D_est[i]]
        #print("initial condition", initial_conditions_k)


        resultIteration = leastsq(error, np.asarray([betaEstimated[i], alphaEstimated[i], gammaEstimated[i], epsilonEstimated[i]]), args=(initial_conditions_k, tspank, data, timek_analysis, timek_1_analysis))
        betak, alphak, gammak, epsilonk = resultIteration[0]


        betaEstimated.append(betak)
        alphaEstimated.append(alphak)
        epsilonEstimated.append(epsilonk)
        gammaEstimated.append(gammak)

        modelk = odeSolver(tspank, initial_conditions_k,
                           [betaEstimated[i+1], alphaEstimated[i+1], epsilonEstimated[i+1],
                            gammaEstimated[i+1]])

        totalModel[timek:timek_1] = modelk[mindelta:mindelta+deltaT,2]
        #totalModel[timek:timek_1,0], totalModel[timek:timek_1,1],totalModel[timek:timek_1,2],totalModel[timek:timek_1,3],totalModel[timek:timek_1,4] = modelk
        #print("modelk", modelk)


        Sk, Ek, Ik, Rk, Dk = modelk[deltaT -1, 0], modelk[deltaT -1, 1], modelk[deltaT -1, 2], modelk[deltaT -1, 3], modelk[deltaT -1, 4]

        S_est.append(Sk)
        E_est.append(Ek)
        I_est.append(Ik)
        R_est.append(Rk)
        D_est.append(Dk)


    # Attuo la minimizzazione per la stima dei parametri [beta, alpha, gamma, epsilon] che verranno memorizzato in result

    # result[0] contiene i parametri stimati[beta, alpha, gamma, epsilon], ricalcolo il modello con i parametri stimati
    #print("lenght" , alphaEstimated.__len__(), betaEstimated.__len__(), epsilonEstimated.__len__(), gammaEstimated.__len__())
    #print("numbersOfInterval", numbersOfInterval)

    tspanfinal = np.arange(0, 200, 1)

    ro0 = 0.9

    # Ottengo le singole soluzioni valutate
    #S1, E1, I1, R1, D1 = modelfinal[:, 0], modelfinal[:, 1], modelfinal[:, 2], modelfinal[:, 3], modelfinal[:, 4]


    #Plot initial Valued Model
    #plt.plot(tspan, I, label="Infected (Model)")
    #plt.plot(tspan, R, label="Recovered (Model)")
    #plt.plot(tspan, D, label="Death (Model)")

    #Plot Model with estimated parameters
    #print(totalModel)
    plt.plot(tspanfinal, totalModel[:], label="Infected (Model 2)")

    #plt.plot(tspan, E1, label="Exposed (Model 2 )")
    #plt.plot(tspan, I1, label="Infected (Model)")
    #plt.plot(tspan, R1, label="Recovered (Model)")
    #plt.plot(tspan, D1, label="Death (Model)")

    #Plot Obeserved Value
    plt.plot(tspanfinal, data[0:200, 0], label="Infected(Observed)")
    #plt.plot(tspanfinal, data[0:200, 1], label="Recovered (Observed)")
    #plt.plot(tspanfinal, data[0:200, 2], label="Death (Observed)")

    plt.legend()
    plt.show()
