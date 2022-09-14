
"""
Created on Fri Apr 10 09:39:32 2020
@author: Benyamin                 
"""
import time
#%%
#import required Electrical Forward Modeling module 
import numpy as np
import pandas as pd
import math as mth
import statistics
from resipy import Project
import warnings
warnings.filterwarnings('ignore')
testdir='C:/Drive E/..../ERT/'   #ERT measurements data directory 
from scipy.interpolate import NearestNDInterpolator
import statistics as st
import scipy
import requests
import psutil
import pyvista
import matplotlib.pyplot as plt
from matplotlib import pyplot
import math
import numpy.linalg as nla
import random
#%%
#import FEFLOW IFM module
import  sys
sys.path.append('C:\Program Files\DHI\2021\FEFLOW 7.4\bin64') #FEFLOW directory 
import ifm
from ifm import Enum
import random

#%%Plot
from matplotlib import interactive
from matplotlib.pyplot import cm
import imageio
#%%parallel computing 
from joblib import Parallel, delayed
#%%
class Hydro_Geophysic():
    def __init__(self):
        """
        Input
            Ensemble         :The Ensemble of the hydraulic conductivity realizations
        """
        

    def Hydraulic_Model(self,N,Top_Bottom_Layers,Hydr_Range_Layer,Final_Simulation_Time,fem_file,Time_step,Ensemble_State_Variable):
        """
        Input
            Ensemble         :The Ensemble of the hydraulic conductivity realizations
        """
        self.N=N
        self.Final_Simulation_Time=Final_Simulation_Time
        self.Ensemble_State_Variable=Ensemble_State_Variable
        doc=ifm.loadDocument(fem_file+'.fem')
        
        self.Number_node=doc.getNumberOfNodes()
        Node_Z=np.zeros([self.Number_node,1]); Node_X=np.zeros([self.Number_node,1]); Node_Y=np.zeros([self.Number_node,1]);
        for i in range(self.Number_node): 
            Node_X[i]=doc.getX(i); Node_Y[i]=doc.getY(i); Node_Z[i]=doc.getZ(i)     
        self.Node_coordinate=np.hstack((Node_X,Node_Y,Node_Z))
        
        self.Number_Element=doc.getNumberOfElements()
        Center_Element_XYZ=np.loadtxt('Element_Center.txt')
        Element_Z=Center_Element_XYZ[:,2]
        
        depth_interval_Assimilation=np.where((Element_Z>=-32)&(Element_Z<=-22)); #The interval in which the assimilation is performed 
        
        self.Id_Obs_Point=list(np.arange(0,doc.getNumberOfValidObsPoints(),1))#number of observation point
        
        self.Temp_All_Node=np.zeros([self.Number_node,self.N]); 
        if Time_step==0:            
            self.Ensemble_State_Variable=np.zeros([self.Number_Element,self.N]);
            
            for Layer in range(0,np.size(Top_Bottom_Layers,0)):
                K_model=np.loadtxt('layer'+str(Layer+1)+'_kmodel.dat')
                depth_interval=np.where((Element_Z>=Top_Bottom_Layers[Layer,1])&(Element_Z<=Top_Bottom_Layers[Layer,0]));  
                
                for j in range(0,self.N):
                    myInterpolator = NearestNDInterpolator(K_model[:,0:3], K_model[:,j+3])
                    Id=Center_Element_XYZ[depth_interval,:];Id=Id[0]
                    self.Ensemble_State_Variable[depth_interval,j]=myInterpolator(Id)
                    print('Hydraulic condoctivity values: '+str(np.power(10,np.mean(self.Ensemble_State_Variable[depth_interval,j]))*86400)+' m/d '+'Layer #'+str(Layer))
            # K_initial=self.Ensemble_State_Variable
        
        return self.Ensemble_State_Variable,depth_interval_Assimilation     
    
#%%    
    def Plot_Temperature_Monitring(self,dac_file):
        Time_All_Obs_Point=[]
        Temp_mon_file={}
        for j in range(0,self.N):
            Result=ifm.loadDocument('Aquifroid'+str(j+1)+'.dac') #load saved results as a Dac file, to investigate the variation of heat, temp.
            dacTimes = Result.getTimeSteps() #number of time-steps
        
            Temp_All_Obs_Point=[] #to save the temperature change at each obseravation point
            for t in range(0,len(dacTimes)):
                Result.loadTimeStep(dacTimes[t][0])
                Temp_log=[]
                for Id in self.Id_Obs_Point:
                    Temp_log.append(Result.getHeatValueOfObsIdAtCurrentTime(Id))
                Temp_All_Obs_Point.append(Temp_log)
            Time_All_Obs_Point.append([sub[1] for sub in dacTimes])
            Temp_mon_file["Aqui" +str(j+1)]=Temp_All_Obs_Point
            
        for sensor in self.Id_Obs_Point:
            color=iter(cm.rainbow(np.linspace(0,1,self.N))) 
            plt.figure(sensor+1)
            for realization in range(self.N):
                a=Temp_mon_file["Aqui" +str(realization+1)]
                # X=np.array([sub[sensor] for sub in a])
                # Y=np.array(Time_All_Obs_Point[realization])
                interactive(True)
                c=next(color)
                plt.plot(np.array(Time_All_Obs_Point[realization]),
                         np.array([sub[sensor] for sub in a]),
                         c=c)
            plt.title('Sensor#'+str(sensor+1))
            plt.xlabel('Time (day)')
            plt.ylabel('Temperature (C)') 
     
    #%%
    def Geometric_Factor(self):
        elec=pd.read_csv(r'elec2.csv')
        Electrod=elec.values[:,:]
        seq=pd.read_csv(r'protocol.csv')
        self.Sequence=seq.values[:,:]
        
        Elec_Position={}
        Elec_name=['a','b','m','n']
        
        dipole_dipole_index=np.arange(np.size(self.Sequence,0))  
        
        for j in range(np.size(self.Sequence,1)):
            Coordinatex=[];Coordinatey=[];Coordinatez=[]
            
            for i in range(np.size(self.Sequence,0)):
                Cor=np.where(Electrod[:,0]==self.Sequence[i,j])[0]
                Coordinatex.append(Electrod[Cor,1])
                Coordinatey.append(Electrod[Cor,2])
                Coordinatez.append(Electrod[Cor,3])
            CoordinateXYZ=np.hstack((Coordinatex,Coordinatey,Coordinatez))
            Elec_Position[Elec_name[j]]=np.copy(CoordinateXYZ) 
            
        #Claculate the Geometric factor for dipole-dipole array
        C1=Elec_Position['a'];C2=Elec_Position['b']
        P1=Elec_Position['m'];P2=Elec_Position['n']
        c1=C1[dipole_dipole_index,:];c2=C2[dipole_dipole_index,:]
        p1=P1[dipole_dipole_index,:];p2=P2[dipole_dipole_index,:]
        
        r1=np.reciprocal(np.sqrt(pow(c1[:,0]-p1[:,0],2)+pow(c1[:,1]-p1[:,1],2)+pow(c1[:,2]-p1[:,2],2)))
        r2=np.reciprocal(np.sqrt(pow(c2[:,0]-p1[:,0],2)+pow(c2[:,1]-p1[:,1],2)+pow(c2[:,2]-p1[:,2],2)))
        r3=np.reciprocal(np.sqrt(pow(c1[:,0]-p2[:,0],2)+pow(c1[:,1]-p2[:,1],2)+pow(c1[:,2]-p2[:,2],2)))
        r4=np.reciprocal(np.sqrt(pow(c2[:,0]-p2[:,0],2)+pow(c2[:,1]-p2[:,1],2)+pow(c2[:,2]-p2[:,2],2)))
        r11=np.reciprocal(np.sqrt(pow(c1[:,0]-p1[:,0],2)+pow(c1[:,1]-p1[:,1],2)+pow(-c1[:,2]-p1[:,2],2)))
        r21=np.reciprocal(np.sqrt(pow(c2[:,0]-p1[:,0],2)+pow(c2[:,1]-p1[:,1],2)+pow(-c2[:,2]-p1[:,2],2)))
        r31=np.reciprocal(np.sqrt(pow(c1[:,0]-p2[:,0],2)+pow(c1[:,1]-p2[:,1],2)+pow(-c1[:,2]-p2[:,2],2)))
        r41=np.reciprocal(np.sqrt(pow(c2[:,0]-p2[:,0],2)+pow(c2[:,1]-p2[:,1],2)+pow(-c2[:,2]-p2[:,2],2)))
        
        Geometric_Factor=4*mth.pi*np.reciprocal(r1+r11-r2-r21-r3-r31+r4+r41)
        
        return Geometric_Factor
    
    #%%
    def Forward_Modeling(self,Rf1,Rb1,Geometric_Factor,Temp_All_Node,T):
        k = Project(typ='R3t')
        k.setNcores(7)
        k.importElec(testdir+'elec.csv')
        x, y = np.meshgrid(k.elec['x'],k.elec['y'])
        z = np.zeros_like(x)-4
        topo = np.c_[x.flatten(),y.flatten(),z.flatten()] 
        k.createMesh(cl=0.3,surface=topo,fmd=30)
        Mesh_CenterXYZ=k.mesh.elmCentre
        # k.showMesh()
        k.importSequence(testdir+'sequencev2.csv')
        
       #Fine Mesh Extension coordinate    
        X=[37, 45];Y=[35, 50];Z=[-22,-32]
        list1=np.where((Mesh_CenterXYZ[:,0]>=X[0])&(Mesh_CenterXYZ[:,0]<=X[1])); 
        list2=np.where((Mesh_CenterXYZ[:,1]>=Y[0])&(Mesh_CenterXYZ[:,1]<=Y[1]))
        list3=np.where((Mesh_CenterXYZ[:,2]>=Z[1])&(Mesh_CenterXYZ[:,2]<=Z[0])); list4=np.intersect1d(np.array(list1),np.array(list2))
        inner_position=np.intersect1d(list4,np.array(list3))
        
        self.FW_Ensemble=np.zeros((np.size(self.Sequence,0),self.N))
        # Rhoa_average=[]
        
        for i in range(0,self.N):
           # print(np.amin(Temp_All_Node[:,i]))
           # print(np.amax(Temp_All_Node[:,i]))
           mf=0.0032
           Conductivity=Rb1*(mf*(Temp_All_Node[:,i]-10)+1) # Convert temperature to Conductivity (S/M)
           Resistivity=np.reciprocal(Conductivity) #Convert Conductivity to Resitivity (Ohm.m)
           
           myInterpolator = NearestNDInterpolator(self.Node_coordinate, Resistivity)
           res0=np.ones((np.size(Mesh_CenterXYZ,0),1))*np.reciprocal(Rb1)
           res0[inner_position,0]=myInterpolator(Mesh_CenterXYZ[inner_position,:])
           
           k.setRefModel(res0)
           print('Forward model starts')
           
           k.forward(noise=0.01)
           
           fwddata = k.surveys[0].df
           
           Rhoa=fwddata.values[:,1]*Geometric_Factor
           # Rhoa[index]=np.median(Rhoa)
           self.FW_Ensemble[:,i]=np.copy(abs(Rhoa))
        return Temp_All_Node,len(x),len(z)
       
#%%    
    def filter_data(self,measurement,Geometric_Factor,Time_step):
        
        measurement=np.loadtxt(measurement+'.dat')
        sequence=[]
        sequence=np.zeros((np.size(measurement,0),6))
        sequence[:,0:4]=np.copy(self.Sequence[:,0:4])
        sequence[:,4:7]=np.copy(measurement[:,Time_step:Time_step+2]) #contain the value of potential and current
            
        #find reciprocal measurements
        nb_quadrupole=len(sequence)
        loop=0
        reci_error=np.zeros((nb_quadrupole,1))
        reci_order=np.zeros((nb_quadrupole,1))
        for i in sequence:
            a=np.arange(nb_quadrupole)
            for j in range(4):
                if (j==0 or j==1):
                    k=np.where((sequence[a,j]==i[2])|(sequence[a,j]==i[3]))
                    a=a[k]
                elif (j==2 or j==3):   
                      k=np.where((sequence[a,j]==i[0])|(sequence[a,j]==i[1]))
                      a=a[k]
            if len(a)==1:
                reci_order[loop]=a
            else:
                reci_order[loop]=np.nan
            loop+=1  
            
        # Apparent resistivity calculation
        sequence[:,4]=abs((sequence[:,4]/sequence[:,5])*Geometric_Factor)
        
        #calculate the reciprocal error 
        for i in range(nb_quadrupole):
            b=int(reci_order[i])
            reci_error[i]=abs(abs(sequence[i,4])-abs(sequence[b,4]))
            
        #withdraw the reciprocal measurement from sequence
        a=np.arange(nb_quadrupole)
        i=0
        while len(a)>nb_quadrupole/2:
            k=np.where(a==reci_order[i])
            a=np.delete(a,k,0)
            i+=1
        sequence=sequence[a,:]
        reci_error=reci_error[a,:]
        reci_order=reci_order[a,:]
        self.FW_Ensemble=self.FW_Ensemble[a,:]
        #remove outliers in term of reciprocal error and also mean and std
        index=[];
        for i in range(len(sequence)):
            if reci_error[i]>0.1*abs(sequence[i,4]):
                index.append(i)
        sequence=np.delete(sequence,index,0)
        reci_error=np.delete(reci_error,index,0)
        reci_order=np.delete(reci_order,index,0)
        self.FW_Ensemble=np.delete(self.FW_Ensemble,index,0)
        index=[];
        q75,q25 = np.percentile(abs(sequence[:,4]),[75,25])
        intr_qr = q75-q25
        max_qt = q75+(1.5*intr_qr)
        min_qt = q25-(1.5*intr_qr)
        for i in range(len(sequence)):
            if (abs(sequence[i,4])<=min_qt or abs(sequence[i,4])>=max_qt):
                index.append(i)
        sequence=np.delete(sequence,index,0)
        reci_error=np.delete(reci_error,index,0)
        reci_order=np.delete(reci_order,index,0)
        self.FW_Ensemble=np.delete(self.FW_Ensemble,index,0)
        
        index=np.where(reci_error>=0.5)
        reci_error=np.delete(reci_error,index,0)
        reci_order=np.delete(reci_order,index,0)
        sequence=np.delete(sequence,index,0)
        self.FW_Ensemble=np.delete(self.FW_Ensemble,index,0)
        Field_Measurement_Mean=np.mean(sequence[:,4])
        FW_Ensemble_Mean=np.mean(self.FW_Ensemble,0)
        return sequence[:,4], self.FW_Ensemble, FW_Ensemble_Mean, Field_Measurement_Mean, reci_error
        
        #%%
class Data_assimilation :
      def __inti__(self):
          
        """
        The Ensemble Kalman Filter algorithm
        The algorithm used in this code is referenced from the following:
            P. N. Raanes, A. Carrassi, and M. Bocquet, “Improvements to Ensemble 
            Methods for Data Assimilation in the Geosciences,” Doctoral Thesis, University of Oxford, 2015.
            """"""
      notation:
      A: anomaly of state
      Aa: Analysed anomaly 
      Y: anomaly of observations
      R: observation perturbations
      HA: the co-located stimated data wirh obsrvations
      N: number of relizations
      m: number of measurements
        Input variables:
        data: data file consist of state and measurements variables data
        seld data:
            self.En_s: The Ensemble of the state variable, each column contains one relization
            self.En_m: The matrix of measurement
            self.stat_m: mean (m) and standard deviation (std) of the measurement's error
            self.N: Number of realizations
        """
        
      def original_EnKF(self,measurement,measurement_error,En_s,En_m,Assimilation_interval):
#%% observation perturbation
        N=np.size(En_s,1)
        std=np.std(np.log(measurement_error))
        E_M=np.zeros_like(En_m)            #creat a matrix of zeros 
        
        for i in range(N):         #a loop to add observation prturbation to each column of observation matrix   
            error=np.random.normal(0,std,np.size(En_m,0))
            E_M[:,i]=np.log(measurement)+error
        
        En_m=np.log(En_m);
        
        Assimilation_interval=np.asarray(Assimilation_interval);Assimilation_interval=Assimilation_interval.T
        En_s_Assimilation=En_s[Assimilation_interval[:,0],:]        
        A=En_s_Assimilation-np.mean(En_s_Assimilation,1,keepdims=True)           
        Y=En_m-np.mean(En_m,1,keepdims=True)
        
        meas_error=E_M-np.mean(E_M,1,keepdims=True)
        meas_error_cov=meas_error.T @ meas_error
        obs_error_cov=Y.T @ Y
        di=np.where(~np.eye(obs_error_cov.shape[0],dtype=bool))
        meas_error_cov[di]=0; 
                
        C=(obs_error_cov + (N-1)*meas_error_cov) *1.01
        YC=nla.solve(C.T,Y.T).T       
        KG=A @ YC.T
           
    #%% the analysis step
    
        dE=(KG@ (E_M-En_m))*100
        E_AS=En_s_Assimilation+dE
        
        En_s[Assimilation_interval[:,0],:]=E_AS
    
        return En_s,E_AS

#%% 
if __name__=='__main__':
#%%    
    doc=ifm.loadDocument('....fem') #Load FEFLOW project
    Number_node=doc.getNumberOfNodes()
    Number_Element=doc.getNumberOfElements()
    Id_Obs_Point=list(np.arange(0,doc.getNumberOfValidObsPoints(),1))
    def Parallel_FEFLOW(realization,T,Ensemble_State,Plot_Temperature_Monitring,Temp_log,Head_log):
        En_s=np.ones_like(Ensemble_State)*10
        Ensemble_State_Feflow=np.power(En_s,Ensemble_State)*86400 #Convert m/s to m/day
        Hydrolic_Conductivity=np.zeros([Number_Element,1]);
        Hydrolic_Conductivity[:,0]=Ensemble_State_Feflow[:,realization]
        K=[];Kz=[]
        K=Hydrolic_Conductivity.tolist() 
        Kz=(Hydrolic_Conductivity/10).tolist() 
        K = [item for sublist in K for item in sublist]    
        Kz=[item for sublist in Kz for item in sublist]
        doc.setParamValues(Enum.P_CONDX,K) #the unite of hydraulic conductivity is m/d.
        doc.setParamValues(Enum.P_CONDY,K)
        doc.setParamValues(Enum.P_CONDZ,Kz)
            
        doc.setFinalSimulationTime(T)
        
        if Plot_Temperature==1:
            doc.startSimulator()
            for Id in Id_Obs_Point:
                Temp_log[Id,realization]=doc.getHeatValueOfObsIdAtCurrentTime(Id)
                Head_log[Id,realization]=doc.getResultsFlowHeadValue(Id) 
                
        else:
            doc.startSimulator()
          
        #Save the simulated Temperature values at current time for all nodes (#Note: Tempreature is a Nodal variable in FEFLOW)
        for Node in range(Number_node): 
            Temp_All_Node[Node,realization]=doc.getResultsTransportHeatValue(Node)  
            
        doc.stopSimulator()
        
        print('end of '+str(realization+1)+' simulation'+' '+'Time step: '+str(T)+' day')
        
        return Temp_All_Node,Temp_log,Head_log
    
#%%
    Plot_Temperature_Monitring = False
    Plot_Temperature=1
    Top_Bottom_Layers = np.array([[-19,-21],[-21,-22],[-22,-32],[-32,-33]]) #Depth of top and bottom of each geological unit
    Hydr_Range_Layer = np.array([[-6,-0.33],[-7,-0.33],[-4,-0.5],[-9,-0.33]]) #The Hydraulic conductiviy range associated with each geological unit.
                                                                              # The value in the scale of log10                                                                            
    N = 2 #The size of ensemble 
    
    fem_file= '...' #Name of Feflow project file

    measurement_file= '...' #Contain the value of potential and current
    protocol_file= '...' #The sequence of measurement contains the number of electrode for each quadrupole 
    
    Time_Steps = np.array([...]) #Insert ERT observation Tiem-steps in day
    
    Rf1, Rb1 = 0.056, 0.0235 #The pore fluid and bulk electrical resistivity at baseline temperature 
    
    Asssimilation_Time_Step=5.72 #Last Assimilation time-step
    #Fluid and Bulk conductivity values at background temperature
    HG = Hydro_Geophysic()
    DA = Data_assimilation()
    K_Factor = HG.Geometric_Factor()
    
    Ensemble_State_initial = []; FW_Ensem_Mean = []; Field_Measurment_Mean = [];Res_cross_section=[]
    Temperature_Monitoring_log=[];Head_Monitoring_log=[];En_s_Assimilation=[];Field_measurement_En=[];fw_Ensemble_En=[]
    Running_Time=0
    i=0
    
#%%    
    for T in Time_Steps:
        
        start = time.time()
        
        Ensemble_State,Assimilation_interval = HG.Hydraulic_Model(N,Top_Bottom_Layers,Hydr_Range_Layer,T
                                                                ,fem_file,i,Ensemble_State_initial)
        # Ensemble_State_initial=Ensemble_State
        if Plot_Temperature_Monitring:
            HG.Plot_Temperature_Monitring(fem_file)
        
        Temp_All_Node= np.zeros([Number_node,N])
        
       
        Temp_log=np.zeros((len(Id_Obs_Point),N));Head_log=np.zeros((len(Id_Obs_Point),N))
        
        # n_jobs is the number of simulations which are executed simultaneously 
        Parallel(n_jobs=10, prefer="threads")(delayed(Parallel_FEFLOW)(realization,T,Ensemble_State,Plot_Temperature_Monitring,Temp_log,Head_log) 
                                             for realization in range(0,N))
        Temperature_Monitoring_log.append(Temp_log);Head_Monitoring_log.append(Head_log)

        Res,row,column=HG.Forward_Modeling(Rf1, Rb1, K_Factor,Temp_All_Node,T)   
        Res_cross_section.append(Res)
                                                    
        Field_measurement,fw_Ensemble,FW_Mean,Field_measurement_mean,measurement_error = HG.filter_data(measurement_file,K_Factor,i)
        Field_Measurment_Mean.append(Field_measurement_mean); FW_Ensem_Mean.append(FW_Mean)
        Field_measurement_En.append(Field_measurement);fw_Ensemble_En.append(fw_Ensemble)
        
        if T<=Asssimilation_Time_Step:
            Ensemble_State_initial,en_s=DA.original_EnKF(Field_measurement,measurement_error,Ensemble_State,fw_Ensemble,Assimilation_interval)
            En_s_Assimilation.append(en_s)
            
        end = time.time()
        print('{:.4f} s'.format(end-start))
        Running_Time+=(end-start)
        # if Running_Time >=30000:
        #     Running_Time=0
        #     time.sleep(3600)
        i+=2
    #%% Plot results  
    
    color=iter(cm.rainbow(np.linspace(0,1,np.size(FW_Ensem_Mean,1))))
    
    plt.figure()
    
    for j in range(N):
        Y=[]
        for i in range(len(Time_Steps)):
            Y.append(FW_Ensem_Mean[i][j])
        interactive(True)
        c=next(color)
        plt.plot(Time_Steps,np.array(Y),Time_Steps,np.array(Y),'ko',c=c)
    plt.plot(Time_Steps,np.array(Field_Measurment_Mean),'--',Time_Steps,np.array(Field_Measurment_Mean),'ko',label='Field measurment',linewidth=2)
    plt.title('Measured vs.Simulated Apparent Resistivity')
    plt.xlabel('Time (day)')
    plt.ylabel('Mean Apparent Resistivity (ohm.m)') 
    plt.grid()
