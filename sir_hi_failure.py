# -*- coding: utf-8 -*-
"""

This code simulates the dynamics of the SIR model with heterogeneous recovery rates (gamma). 
By running the provided examples, it demonstrates two counter-examples showing that herd immunity 
cannot be achieved when the mean recovery time (the mean of 1/gamma) is not finite. 
 The results highlight the critical role of recovery time heterogeneity in epidemic modeling.

See arxiv reference https://arxiv.org/abs/2411.13130 for theoretical details

@author: (c) Gabriel Turinici, 2024,2025,...
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#%matplotlib inline
#%matplotlib auto

#parameters
T=600 # final time
N=150 # number of time steps
h = T/N # 
I0=10./1.e+6
S0 = 1-I0
R0=0.

y0 = [S0,I0,R0]



# define the time grid    
trange = np.linspace(0,T,num=N+1,endpoint=True)   

betasir=1./4.
gammasir=1./6.
print('reproduction number=',betasir/gammasir)

def sir_list(y,t,betaSIR,gammaSIR):
    """
    define the SIR function 
    Parameters
    ----------
    t : time
    x : list of components of dimension d

    Returns
    -------
    a list, value of the function 

    """
    S,I,R=y
    ntotal=S+I+R
    return [-betaSIR*S*I/ntotal,betaSIR*S*I/ntotal-gammaSIR*I,gammaSIR*I]

def sir_array(y,t,betaSIR,gammaSIR):
    """ like sir_list but return an array """
    return np.array(sir(y,t,betaSIR,gammaSIR))

#sir=sir_array
sir=sir_list


solution = odeint(sir, y0, trange, args=(betasir,gammasir))

Ssol=solution[:,0]
Isol=solution[:,1]
Rsol=solution[:,2]


plt.figure("sir_standard",figsize=(3,2.5),dpi=300)

for data,style,symb in zip([Ssol,Isol,Rsol],
                               ["solid","dotted","dotted"],['o','d','.']):
    #print(style)
    plt.plot(trange,data,linestyle=style,linewidth=6,alpha=0.9,marker=symb)

#plt.plot(trange,Ssol,trange,Isol,trange,Rsol,linewidth=6)
plt.legend(['S','I','R'])
plt.xlabel('time')
plt.tight_layout()
plt.savefig("sir_standard.pdf")
plt.show()
plt.close()



#second part : two groups 
def sir2gr(y,t,betaSIR,gammaSIR1,gammaSIR2):
    """
    define the SIR function 
    Parameters
    ----------
    t : time
    x : list of components of dimension d

    Returns
    -------
    a list, value of the function 

    """
    S1,I1,R1,S2,I2,R2=y
    I=I1+I2
    ntotal=S1+I1+R1+S2+I2+R2
    return [-betaSIR*S1*I/ntotal,betaSIR*S1*I/ntotal-gammaSIR1*I1,gammaSIR1*I1,
            -betaSIR*S2*I/ntotal,betaSIR*S2*I/ntotal-gammaSIR2*I2,gammaSIR2*I2,
            ]

y02 = [S0/2,I0/2,R0/2,S0/2,I0/2,R0/2]

solution2 = odeint(sir2gr, y02, trange, args=(betasir,0.0,gammasir*2))

S1sol=solution2[:,0]
I1sol=solution2[:,1]
R1sol=solution2[:,2]
S2sol=solution2[:,3]
I2sol=solution2[:,4]
R2sol=solution2[:,5]


plt.figure("sir_2groups",figsize=(3,2.5),dpi=300)

for data,style,symb in zip([S1sol+S2sol,I1sol+I2sol,R1sol+R2sol],
                               ["solid","dotted","dotted"],['o','d','.']):
    #print(style)
    plt.plot(trange,data,linestyle=style,linewidth=6,alpha=0.9,marker=symb)

#plt.plot(trange,S1sol+S2sol,trange,I1sol+I2sol,trange,R1sol+R2sol,linewidth=6)
plt.legend(['S','I','R'])
plt.xlabel('time')
plt.tight_layout()
plt.savefig("sir_2groups.pdf")
plt.show()
plt.close()


#%%  We consider now the case gamma(x) = cst * x on [0,1], see paper; 
# we discretize the Omega into several groups

#parameters
T=400 # final time
N=150 # number of time steps
I0=10./1.e+6
S0 = 1-I0
R0=0.

y0 = [S0,I0,R0]

# define the time grid for plotting
trange = np.linspace(0,T,num=N+1,endpoint=True)   

betasir=1./4.
gammasir=1./6.
print('reproduction number=',betasir/gammasir)

def sir_many_groups(y,t,betaSIR,gamma_vec):
    """
    define the SIR function 
    Parameters
    ----------
    t : time
    y : list of components of shaoe (3*n,);
    attention here the structure is S1, .. Sn, I1, ... In, R1, .... Rn
    gamma_vec is a (n,) size vector


    Returns
    -------
    a list, value of the function 

    """
    n=gamma_vec.shape[0]
    assert 3*n == y.shape[0],"wrong size in sir many groups function"
    Sv=y[:n]
    Iv=y[n:2*n]
    I=np.sum(Iv)
    ntotal=np.sum(y)
    return list(-betaSIR*Sv*I/ntotal)+list(+betaSIR*Sv*I/ntotal-gamma_vec*Iv)+list(gamma_vec*Iv)


def plot_results_manygroups(gamma_vec,test_label="sir_manygroups"):
    """
    Parameters
    ----------
    gamma_vec : 1d numy array, contains the averages of gamma over each group 
    test_label : a name for this test, plot is saved as test_label+'.pdf'

    Returns : None, plots the result
    """
    

    nrgroups=gamma_vector.shape[0]
    y0g = [S0/nrgroups]*nrgroups+ [I0/nrgroups]*nrgroups+ [R0/nrgroups]*nrgroups
        
    solution_g= odeint(sir_many_groups, y0g, trange, 
                       args=(betasir,gamma_vec))
    
    Ssol=np.sum(solution_g[:,:nrgroups],axis=-1)
    Isol=np.sum(solution_g[:,nrgroups:2*nrgroups],axis=-1)
    Rsol=np.sum(solution_g[:,2*nrgroups:],axis=-1)
    
    print('S infinity=',Ssol[-1])
    
    plt.figure(test_label,figsize=(3,2.5),dpi=300)
    for data,style,symb in zip([Ssol,Isol,Rsol],
                                   ["solid","dotted","dotted"],['o','d','.']):
        #print(style)
        plt.plot(trange,data,linestyle=style,linewidth=6,alpha=0.9,marker=symb)
    
    #plt.plot(trange,Ssol,trange,Isol,trange,Rsol,linewidth=6)
    plt.legend(['S','I','R'])
    plt.xlabel('time')
    plt.tight_layout()
    plt.savefig(test_label+".pdf")
    plt.show()
    plt.close()

nr_groups=250#number of groups
indices=np.array(range(nr_groups))

# we test the case gamma(x) = gamma_average * 2x, x in [0,1]. 
# To do the simulation we
# divide the total interval in nrgroups sub-intervals and on each take 
# the average value of gamma. For Python numbering, index k corresponds
# to sub-interval [k/nr_groups, (k+1)/nr_groups] with average (2k+1)/nr_groups
gamma_vector=gammasir*(2*indices+1)/nr_groups
plot_results_manygroups(gamma_vector,test_label="sir_manygroups")


# another test = gamma(x) = gammasir * sqrt(x)* 3/2, has gammasir average
gamma_vector_sqrt=gammasir* ( (1.+indices)**1.5-indices**1.5) /np.sqrt(nr_groups)
plot_results_manygroups(gamma_vector_sqrt,test_label="sir_manygroups_sqrt")
