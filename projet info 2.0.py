"Modeliser une particule dans un champ"

import numpy as np
from math import atan,cos,sin
import matplotlib.pyplot as plt

def convertisseur_matrice(Matrice,position,unite):
    n=len(Matrice)
    #on definit le plan
    O=[(n//2),(n//2)]
    """
    I=[(n//2)+1,n//2]
    J=[n//2,(n//2)+1]
    plan=[O,I,J]
    """
    
    #coordonnee sur l'axe des absisses
    x=0
    if position[1]>O[0]:
        while position[1]!=O[0]:
            position[1]=position[1]-1
            x=x+1
            
    elif position[1]<O[0]:
        while position[1]!=O[0]:
            position[1]=position[1]+1
            x=x+1
        x=x*(-1)
        
    #coordonnee sur l'axe des ordonnÃ©s       
    y=0
    if position[0]>O[1]:
        while position[0]!=O[1]:
            position[0]=position[0]-1
            y=y+1
            
    elif position[0]<O[1]:
        while position[0]!=O[1]:
            position[0]=position[0]+1
            y=y+1
           
        y=y*(-1)    
        
    x,y=unite*x,unite*y
    position=[x,y]
    """
    return(plan,position,unite)
    """
    return(x,y)
    
def convertisseur_plan(plan,position,unite):
    n=plan[0][0]*2
    i,j=(position[0]//unite)+n//2,(position[1]//unite)+n//2
    position=[i,j]
    return(i,j)

def matrice_des_champs(C,n):
    M=np.zeros((n+1,n+1,2))
    for i in range(0,n//2):
        for j in range(0,n//2):
            x,y=convertisseur_matrice(M,[i,j],1)#le champ dans les 4 cadrant
            R=(x**2+y**2)**0.5
            alpha=atan(y/x)
            
            M[i][j][0]=(C/(R**2))*cos(alpha)
            M[i][j][1]=(C/(R**2))*sin(alpha)
            
            M[n-i][j][0]=M[i][j][0]
            M[n-i][j][1]=-M[i][j][1]
            
            M[i][n-j][0]=-M[i][j][0]
            M[i][n-j][1]=-M[i][j][1]
            
            M[n-i][n-j][0]=-M[i][j][0]
            M[n-i][n-j][1]=-M[i][j][1]
    
    for j in range(n//2):#â¦le champ sur les lignes du plan
        x,y=convertisseur_matrice(M,[i,0],1)
        
        M[n//2][j][0]=C/x
        M[n//2][n-j][0]=-C/x
         
    for i in range(50):
        x,y=convertisseur_matrice(M,[0,j],1)
        
        M[i][n//2][1]=C/x
        M[n-i][n//2][1]=-C/x
    return(M)
    
"Matrice des champs usuels"

V=matrice_des_champs(2000,500)
"""
W=matrice_des_champs(3,750)
Z=matrice_des_champs(4,1000)
"""
def liste_temps(duree,dt):
    T=[]
    while duree>0:
        T.append(duree)
        duree=duree-dt  
    T.append(0)
    return(T[-1::-1])

def vitesse(ax,ay,dt,Vo):
    Vx=ax*dt+Vo[0]
    Vy=ay*dt+Vo[1]
    V=[Vx,Vy]
    return(V)
    
def position(vx,vy,dt,Po):
    Px=int(vx*dt+Po[0])
    Py=int(vy*dt+Po[1])
    P=[Px,Py]
    return(P)

#on réuni l'ensemble des programmes pour creer la liste de position
def modele(C,n,X,Y,Vx,Vy,duree,dt):
    L=[[X,Y]] #liste de position 
    Vitesse=[[Vx,Vy]]
    M=matrice_des_champs(C,n)
    T=liste_temps(duree,dt)
    
    P=[X,Y]
    V=[Vx,Vy]
    A=[M[X][Y][0],M[X][Y][1]]
    
    for i in range(len(T)):
        if -n<=P[0]<=n and -n<=P[1]<=n:#si la particule sort du plan le pg s'arrête
            
            A=[M[P[0]][P[1]][0],M[P[0]][P[1]][1]]
         
            V=vitesse(A[0],A[1],dt,V)
         
            P=position(V[0],V[1],dt,P)
        
            L.append(P)
            Vitesse.append(V)
        else:
            return L
    
    return L



def plan(C,n,x,y,Vx,Vy,duree,dt):
    P=modele(C,n,x,y,Vx,Vy,duree,dt)
    
    X=[]
    Y=[]
    for i in range(len(P)):
        X.append(P[i][0])
        Y.append(P[i][1])
        
    X=np.array(X)
    Y=np.array(Y)
    plt.figure("f")
    plt.plot(X,Y)
    plt.show()

import PIL.Image as pili

"""
def image(P,n):
"""    
def image(C,n,x,y,Vx,Vy,duree,dt):
    
    P=modele(C,n,x,y,Vx,Vy,duree,dt)
     
    M=np.zeros((n+1,n+1,3),dtype='uint8')
    M[n//2][n//2]=[255,255,0]
    
    for i in range(len(P)):
        if -n<=P[i][0]<=n and -n<=P[i][1]<=n:
             M[P[i][0]][P[i][1]]=255
        else:
             I=pili.fromarray(M)
             M[P[0][0]][P[0][1]]=[255,60,80]
             
             return I.show()
         
    M[P[0][0]][P[0][1]]=[255,60,80]
    I=pili.fromarray(M)
    return I.show()

import random as rd

def generer(n,duree,dt):
    C=rd.randint(0,20)
    x=rd.randint(-n//2,n//2)
    y=rd.randint(-n//2,n//2)
    Vx=rd.randint(-20,20)
    Vy=rd.randint(-20,20)
    P=modele(C,n,x,y,Vx,Vy,duree,dt)
    print(x,y,Vx,Vy,C)
    return P
    
    
    
    
    
    


    