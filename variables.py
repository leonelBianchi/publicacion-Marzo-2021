import math
import numpy as np
import matplotlib.pyplot as plt



Gamma = 2400 # Kg/m^3
E = 20000000000 # Modulo de elasticidad en Pa
Vp = math.sqrt(E/Gamma) # Velocidad de ondas de corte s^(-1)*m
Vs = 500 # Velocidad de propagación de las ondas de corte en el terreno(m/s)
L = 6 # Longitud del pilote (m)
k = 40*9.81*(100**3) #Rigidez a la fricción del suelo(N/m^3)
c = 0.01*9.81*(100**3) #Coeficiente de amortiguamiento del suelo (N*sec/m^3)
K = (50*E*math.pi*(0.4**2)*Vs)/(4*L*Vp) # Rigidez de la punta del pilote(N/m)
C = K*0.026*L/Vs # Coeficiente de amortiguamiento de la punta del pilote(N/(m*s^(-1)))


n_L = 100 # numero de segnmentos del pilote
c_sigma = math.sqrt(E/Gamma) # velocidad de onda de tension en m/s
Delta_x = L/n_L # longitud del segmento
Delta_t = Delta_x/c_sigma
n_p = 3*(L/c_sigma)*(1/Delta_t)





# -----------------Función excitación-----------------------------------------

def p(t):
    global res
    if t<0.0001:
        res = 500000*t/0.0001
    elif 0.0001<t<=0.0006:
        res = 500000
    elif 0.0006<t<=0.0007:
        res = 500000*(1-((t-0.0006)/0.0001))
    elif t>0.0007:
        res = 0
    return res 

import random


def diametro():

    diam = []
    pos = random.randint(0,10)
    estriccion = random.randint(40,100)
    perturbacion = random.randint(3,8)
    lim1 = pos-perturbacion
    lim2 = pos+perturbacion
    start = 10*pos
    end = 10*pos + 10

    for i in range(0, 100+1):
             
        if i in range(start, end):
            diam.append(0.4*estriccion/100)
        else:
            diam.append(0.4)
    
    return diam


def eventos(cantidad_eventos):
    
    eventos = []
    
    for j in range(0,cantidad_eventos):
        daño = diametro()
        eventos.append(daño)
    
    return eventos


def funcion_1(phi):
    xi = [round(i*Delta_x, 5) for i in range(0,n_L+1)]
    fii = phi
    ki = [k for i in range(0,n_L+1)]
    ci = [c for i in range(0,n_L+1)]
    ai = [math.sqrt(Gamma*E)*(fii[i]**2) for i in range(0,n_L+1)]
    bi = [Gamma*(fii[i]**2) for i in range(0,n_L+1)]
    fi_prima = [(fii[i+1]-fii[i])/(xi[i+1]-xi[i]) for i in range(0,n_L)]
    fi_prima.insert(n_L+1,0)
    
    return xi, fii, ki, ci, ai, bi, fi_prima

from numpy.linalg import det, inv

def funcion_2(phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)
    
    I = [0]
    
    for i in range(1, n_L):
        a11 = ai[i-1]+ai[i]+2*E*fii[i]*fi_prima[i]*Delta_t
        a12 = -(bi[i-1]+bi[i])-4*ci[i]*fii[i]*Delta_t
        a13 = -4*ki[i]*fii[i]*Delta_t
        a21 = -(ai[i] + ai[i+1]) + 2*E*fii[i]*fi_prima[i]*Delta_t
        a22 = -(bi[i] + bi[i+1])-4*ci[i]*fii[i]*Delta_t
        a23 = a13
        a31 = 0
        a32 = -Delta_t/2
        a33 = 1

        matrix = inv([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
        I.append(matrix)
    
    return I

def funcion_3(phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)
    
    a210 = -(ai[0]+ai[1])
    a220 = -(bi[0]+bi[1])

    I0 = inv([[(1-(4*(ci[0]*fii[0]/a220)*Delta_t)), -(4*(ki[0]*fii[0]/a220)*Delta_t)],
              [-Delta_t/2, 1]])
    
    return a210, a220, I0


def funcion_4(phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)

    a11 = ai[n_L-1]+ai[n_L]+2*E*fii[n_L]*fi_prima[n_L]*Delta_t
    a12 = -(bi[n_L-1]+bi[n_L])-4*ci[n_L]*fii[n_L]*Delta_t
    a13 = -4*ki[n_L]*fii[n_L]*Delta_t
    a21 = ((math.pi)/4)*((fii[n_L])**2)*E
    a22 = C
    a23 = K
    a31 = -Delta_x/2
    a32 = -Delta_t/2
    a33 = 1

    In = inv([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    return In

def e(x, u, u_prima, up, phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)
    
    e = (2*E*fii[x]*fi_prima[x]*u_prima)-(4*ci[x]*fii[x]*up)-(4*ki[x]*fii[x]*u) 
    return e

def funcion_5(phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)
    
    u = np.zeros((n_L+1, int(n_p)+1))
    up = np.zeros((n_L+1, int(n_p)+1))
    u_prima = np.zeros((n_L+1, int(n_p)+1))
    
    for j in range(0, int(n_p)+1):

        u_prima[0][j] = -4*(p((j*Delta_t)))/(math.pi*E*(fii[0]**2))

        for i in range(0, n_L+1):
            u[i][j] = 0
            up[i][j] = 0

            if i != 0:
                u_prima[i][j] = 0
                
    return u_prima, u, up

def funcion_6(phi):
    
    xi, fii, ki, ci, ai, bi, fi_prima = funcion_1(phi)
    I = funcion_2(phi)
    a210, a220, I0 = funcion_3(phi)    
    In = funcion_4(phi)
    u_prima, u, up = funcion_5(phi)
    
    matriz_e = []
    
    for j in range(1, int(n_p)+1):
        for i in np.arange(int((1+((-1)**(j+1)))/2),
                        min(j,n_L+1),
                        2):

            if i != 0:
                eA = (2*E*fii[i-1]*fi_prima[i-1]*u_prima[i-1][j-1])-(4*ci[i-1]*fii[i-1]*up[i-1][j-1])-(4*ki[i-1]*fii[i-1]*u[i-1][j-1]) 
                #eA = e((i), u[i-1][j-1], u_prima[i-1][j-1], up[i-1][j-1], phi)
                #matriz_e.append(eA)

            if i != n_L:
                eB = (2*E*fii[i+1]*fi_prima[i+1]*u_prima[i+1][j-1])-(4*ci[i+1]*fii[i+1]*up[i+1][j-1])-(4*ki[i+1]*fii[i+1]*u[i+1][j-1]) 

                #eB = e(i, u[i+1][j-1], u_prima[i+1][j-1], up[i+1][j-1], phi)
                #matriz_e.append(eB)
                
            if i == 0:
                q4 = up[1][j-1]-(1/a220)*(a210*(u_prima[0][j]-u_prima[1][j-1])+eB*Delta_t)-((2*E*fii[0]*fi_prima[0]*u_prima[0][j]*Delta_t)/a220)
                q5 = u[1][j-1]-(Delta_x/2)*(u_prima[0][j]+u_prima[1][j-1])+(Delta_t/2)*up[1][j-1]
                up[0][j] = I0[0][0]*q4+I0[0][1]*q5
                u[0][j] = I0[1][0]*q4+I0[1][1]*q5

            elif i != 0 and i != n_L:
                q1 = (ai[i-1]+ai[i])*u_prima[i-1][j-1]-(bi[i-1]+bi[i])*up[i-1][j-1]-eA*Delta_t
                q2 = -(ai[i]+ai[i+1])*u_prima[i+1][j-1]-(bi[i]+bi[i+1])*up[i+1][j-1]-eB*Delta_t
                q3 = ((u[i-1][j-1]+u[i+1][j-1])/2)+(Delta_x/4)*(u_prima[i-1][j-1]-u_prima[i+1][j-1])+(Delta_t/4)*(up[i-1][j-1]+up[i+1][j-1])
                u_prima[i][j] = I[i][0][0]*q1+I[i][0][1]*q2+I[i][0][2]*q3
                up[i][j] = I[i][1][0]*q1+I[i][1][1]*q2+I[i][1][2]*q3
                u[i][j] = I[i][2][0]*q1+I[i][2][1]*q2+I[i][2][2]*q3
            
            elif i == n_L:
                q1 = (ai[i-1]+ai[i])*u_prima[i-1][j-1]-(bi[i-1]+bi[i])*up[i-1][j-1]-eA*Delta_t
                q2 = 0
                q3 = u[i-1][j-1]+(Delta_x/2)*u_prima[i-1][j-1]+(Delta_t/2)*up[i-1][j-1]
                u_prima[i][j] = In[0][0]*q1+In[0][1]*q2+In[0][2]*q3
                up[i][j] = In[1][0]*q1+In[1][1]*q2+In[1][2]*q3
                u[i][j] = In[2][0]*q1+In[2][1]*q2+In[2][2]*q3

            
                
    return up

def generate_up(vector):
    
    up = funcion_6(vector)[0]
    
    return up

def plot_up(señal):
    vec = señal
    vec.reshape(301,1)
    up = vec
    up_modificado = np.zeros((150,1))

    for j in range(0, int(300), 2):
        up_modificado[int(j/2)] = up[j]

    t = np.arange(0., 150., 1)



    plt.figure(figsize=(15,5))
    plt.plot(t,up_modificado , 'r--', c="r", label="$up_{0,j}$")
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")


    plt.legend( prop={'size': 12})
    plt.title("Velocity")



    plt.show()


import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks    
    
def transformada_fourier(up):
    M = 10  # Potencia de elevacion
    n = pow(2,M) # Numero de puntos
    x = np.zeros(n)
    x[0:301] = up


    fhat = np.fft.fft(x,n)                        
    PSD = fhat * np.conj(fhat)                        
    freq = ((2*math.pi)/(Delta_t*n))*np.arange(n)              
    L = np.arange(1, np.floor(n/2), dtype='int')      


    x = PSD[L][:250]
    peaks, _ = find_peaks(x, distance=20)
    peaks2, _ = find_peaks(x, prominence=1)      
    peaks3, _ = find_peaks(x, width=20)
    peaks4, _ = find_peaks(x, threshold=0.4)    

    plt.plot(peaks2, x[peaks2], "ob"); plt.plot(x); plt.legend(['picos'])
    plt.xlim(0, 250)

    plt.show()
    
    return peaks2, PSD[L]

from scipy.interpolate import make_interp_spline

def plot_fourier(up):
    M = 10  # Potencia de elevacion
    n = pow(2,M) # Numero de puntos
    x = np.zeros(n)
    x[0:301] = up


    fhat = np.fft.fft(x,n)                        
    PSD = fhat * np.conj(fhat)                        
    freq = ((2*math.pi)/(Delta_t*n))*np.arange(n)              
    L = np.arange(1, np.floor(n/2), dtype='int')      


    x = PSD[L][:250]
    peaks, _ = find_peaks(x, distance=20)
    peaks2, _ = find_peaks(x, prominence=2000000)      
    peaks3, _ = find_peaks(x, width=20)
    peaks4, _ = find_peaks(x, threshold=0.4)    

    
    xp=np.array([i for i in range(0,len(x))])
    y=np.array(x)

    model=make_interp_spline(xp,y)

    xs=np.linspace(0,len(y),10000)
    ys=model(xs)

    plt.figure(figsize=(15,5))
    plt.plot(xs, ys)
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.xlim(0, 75)
    
       

    plt.show()
    
