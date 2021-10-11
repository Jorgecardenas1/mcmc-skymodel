"""
Author: Jorge H. Cárdenas
University of Antioquia

"""


import numpy as np
import math
from scipy.stats import lognorm
from tqdm import tqdm
import pandas as pd

class MCMC:
    
    output_data = 0
    
    def __init__(self):
        pass
    

        
    def gaussian_sample(self,params,N):

        theta_sampled=[]
        for key, value in params.items():

            selected_samples = np.random.normal(value["nu"], value["sigma"], N)
            theta_sampled.append(selected_samples[0])

        return theta_sampled

    def mirror_gaussian_sample(self,params,N):

        theta_sampled=[]
        intermediate_sample=[]
        for key, value in params.items():

            selected_samples = np.random.uniform(low=value["min"], high=value["max"], size=1)
            #selected_samples = np.random.normal(value["nu"], value["sigma"], 1*N)
            intermediate_sample.append(selected_samples)

        return np.mean(np.array(intermediate_sample).T,axis=0)

#Assign values based on current state
#Get an array with the evaluation for an specific parameter value in the whole range of X
    
   

    def prior(self,theta):
        #in this case priors are just the required check of parameter conditions
        #it is unknown.
        #it must return an array with prior evaluation of every theta
        #i evaluate a SINGLE THETA, A SINGLE PARAMETER EVERY TIME
        #depending on which conditional probability i am evaluating

        #m, b, log_f = theta
        #if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        #    return 0.0
        #return -np.inf

        return [0.0] 
    #just assuming everything is in range

    # Metropolis-Hastings

    def set_dataframe(self,parameters):
        columns=['iteration','walker','accepted','likelihood']

        global_simulation_data = pd.DataFrame(columns=columns)  
        new_dtypes = {"iteration": int,"likelihood":np.float64,"walker":int,"accepted":"bool"}
        global_simulation_data[columns] = global_simulation_data[columns].astype(new_dtypes)


        for parameter in parameters:
            global_simulation_data.insert(len(global_simulation_data.columns), parameter,'' )
            new_dtypes = {parameter:np.float64}
            global_simulation_data[parameter] = global_simulation_data[parameter].astype(np.float64)

        return global_simulation_data

    def acceptance(self,new_loglik,old_log_lik):
        if (new_loglik > old_log_lik):

            return True
        else:
            u = np.random.uniform(0.0,1.0)
            # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
            # less likely x_new are less likely to be accepted
            return (u < (np.exp(new_loglik - old_log_lik)))

    def thining(self,dataframe,num_samples ):
        stack = pd.DataFrame()
        walkers = dataframe.walker.unique()
        
        for walker in walkers:
            
            selected = dataframe.loc[dataframe['walker'].isin([walker])]
            
            
            walker_DF = selected.tail(selected.shape[0] - int(num_samples*0.05)) #Dropping
            
            walker_DF = walker_DF[walker_DF.index % 50 == 0] 
            
            #walker_DF = walker_DF.nsmallest(int(num_samples*0.75),['likelihood']) #Thining
            
            stack = pd.concat([stack,walker_DF],ignore_index=True)
        
    
        return stack.sort_values(by=['iteration'])
    
    def MH(self,sky_model,parameters,t_sky_data,sigma_parameter,evaluateLogLikelihood,initial_point,num_walkers,num_samples,burn_sample):
        accepted  = 0.0
        row=0
        iteration=0
        thetas_samples=[]

        thetas_samples.append(initial_point)

        num_of_params=len(initial_point)
        walkers_result=[]
        dataframe = self.set_dataframe(parameters)


        with tqdm(total=(num_samples*num_walkers)) as pbar:
            for walker in range(num_walkers):
                
                initial_point = self.gaussian_sample(parameters,1)

                for n in range(num_samples):

                    pbar.update(1)        

                    old_theta = np.array(thetas_samples[len(thetas_samples)-1], copy=True) 
                    old_log_lik = evaluateLogLikelihood(old_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    params = sky_model.update_parameters(old_theta) #this has impact when using gaussian proposed distribution

                    new_theta = self.mirror_gaussian_sample(params,1)
                    new_loglik = evaluateLogLikelihood(new_theta,t_sky_data.Freq,t_sky_data.t_sky,sigma_parameter)

                    # Accept new candidate in Monte-Carlo.
                    if n>burn_sample:
                        if self.acceptance(new_loglik,old_log_lik):
                            thetas_samples.append(new_theta)
                            accepted = accepted + 1.0  # monitor acceptance

                            data = np.concatenate(([iteration, walker,1, new_loglik],new_theta),axis=0)
                            dataframe.loc[iteration] = data

                        else:
                            thetas_samples.append(old_theta)
                            data = np.concatenate(([iteration, walker, 0, old_log_lik],old_theta),axis=0)

                            dataframe.loc[iteration] = data

                    iteration += 1
                walkers_result.append(thetas_samples)



        print("accepted"+str(accepted))                
        return walkers_result, dataframe
    

