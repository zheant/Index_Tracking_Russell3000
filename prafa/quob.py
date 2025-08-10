import dcor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import subprocess



class QUOB:
    def __init__(self, stocks_returns, index_returns, K, simple_corr=False):
        #matrice et vecteur numpy
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K #cardinalité!!
        self.idx = None #liste d'indice des stonks choisit 
        
        #construire ma matrice de distance
        if simple_corr:
            self.matrix_simplecor()
        else:
            self.matrix_dcor()
        


    def matrix_dcor(self):
        
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        dcor_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                dcor_val = dcor.distance_correlation(self.stocks_returns[:, i], self.stocks_returns[:, j])
                dist = 1 - dcor_val
                dcor_mat[i, j] = dcor_mat[j, i] = Welsch_function(dist) #Welsch_function(dist)
    
        np.savetxt("dist_matrix.d", dcor_mat)
        


    def matrix_simplecor(self):
        distance_func = lambda di : np.sqrt(0.5*(1 - di))
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)
        
        distance_matrix = distance_func(corr_matrix)
        np.savetxt("dist_matrix.d", Welsch_function(distance_matrix))


    def stock_picking(self, n):
        #résolution du probleme d'optimisation 
        #retourne une liste d'indice des stonks sélectionné
        param = f"""num_vars {n} #INT number of variables/nodes
                num_k {self.K} #INT number of medoids/exemplars
                B_scale_factor {0.0333} 0.5*(self.K+1)/n#FLOAT32 scaling factor for model bias, set to 0.5*(num_k +1)/num_vars
                D_scale_factor 1.0 #FLOAT32 scaling factor for model distances, leave at 1 
                problem_path /home/ubuntu/Index_Tracking/prafa/dist_matrix/
                problem_name dist_matrix
                cost_answer -1000000 #FLOAT32 target cost to allow program to exit early if found, set to large neg value if you don't want an early exit
                T_max 0.01 #FLOAT32 parallel tempering max temperature
                T_min 0.00001 #FLOAT32 parallel tempering min temperature
                time_limit 300.0 #FLOAT64 time limit for search in seconds
                round_limit 100000000 #INT round/iteration limit for search. Search ends if no cost improvement found within a 10000 round window 
                num_replicas_per_controller 32 #INT (POW2 only) number of replicas per parallel tempering controller
                num_controllers 1 #INT (POW2 only) number of parallel tempering controllers
                num_cores_per_controller 1 #INT (POW2 only) number of cores/threads to dedicate to each controller
                ladder_init_mode 2 #INT (0,1,2) parallel tempering ladder init mode. 0->linear spacing b/w t_min & t_max. 1->linear spacing between beta_max and beta_min, then translated to T. 2->exponential spacing between T_min and T_max
                """
        
        with open('/home/ubuntu/Index_Tracking/prafa/dist_matrix/dist_matrix.params', "w") as f:
            f.write(param)

        subprocess.run(['/home/ubuntu/or_tool/cmake-build/ReplicaTOR' , '/home/ubuntu/Index_Tracking/prafa/dist_matrix/dist_matrix.params'])
        

        #lire le résultat et le mettre en liste
        with open("/home/ubuntu/Index_Tracking/prafa/dist_matrix/dist_matrix.soln.txt", "r") as f:
            ligne = f.read()

        return [int(x) for x in ligne.strip().split()]


    def calc_weights(self):
        self.idx = self.stock_picking(self.stocks_returns.shape[1])
        subset_returns = self.stocks_returns[:, self.idx]
        
        initial_weight = np.ones(len(self.idx))
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(len(self.idx))]

        # Define Constraints    
        constraint = {'type': 'eq', 'fun':lambda weight : np.sum(weight) - 1}
        objective_function = lambda weight : np.sum((subset_returns @ weight - self.index_returns)**2)
        
        # Optimization
        result = minimize(objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        return result.x

    
    def get_weights(self):
        #retourne numpy array sparse des poids
        
        weight_global = np.zeros(self.stocks_returns.shape[1])

        micro_weight = self.calc_weights()
        for i in range(len(micro_weight)):
            weight_global[self.idx[i]] = micro_weight[i]

        return weight_global
