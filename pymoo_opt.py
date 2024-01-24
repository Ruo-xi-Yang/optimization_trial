import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os           
import glob                        
import time
import subprocess
from subprocess import call
import math

import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback

# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.model.problem import Problem
# from pymoo.model.callback import Callback

from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination

# from pymoo.factory import get_sampling, get_selection, get_crossover, get_mutation, get_termination
# from pymoo.model.repair import Repair
from pymoo.core.repair import Repair

# myfiles

import FileGen

# OPTIMISATION PARAMETERS -----------------------------------------------------

output_opt_file = "1st_2D_output.txt"
parent_dir = '/home/ry619/optimization/optimization_trial'
optimisation = "2D"                  #"2D", "3D"

DVs = 2                              # number of Design Variables
OFs = 2                              # number of Objective Functions
CTs = 0                              # number of constraintis

xmin = 0.1                           # lower x bound
xmax = 0.9                           # higher x bound
ymin = 0.02                          # lower y bound
ymax = 0.4                           # higher y bound
gamma = 45                           # max. angle of triangle's LE and TE
gamma_rad = gamma*(math.pi/180)

prob_m = 0.2                         # mutation probability
prob_c = 0.2                         # crossover probability
eta_m = 15                           # mutation eta
eta_c = 20                           # crossover eta

population = 2                      # population size
offspring = 1                       # number of offspring
n_max_gen = 5                        # number of generations

global ngen
restart = "No"                       # restart optimisation from checkpoint
ngen = 0                             # initial generation
test = "No"                          # test reduced running time
rep_test = "No"                      # test repeating generation

initialseeding = "LHS"           # "DVs", "LHS", "evalPOP"
seeding_path = "../../2Dopt/12deg/pymoo_v2/gen_20/Pareto_gen20.csv"
evaluated_gen = 3
evaluated_path = "gen_%s/optimum_gen%s.csv" %(evaluated_gen,evaluated_gen)

# PROBLEM DEFINITION -----------------------------------------------------

order = 1                            # polynomial order
AoA = 12                             # Angle of attack
c = 1                                # chord length

Re = 3000                            # Reynolds number
Ma = 0.15                            # Mach number
u_inf = 1                            # free-stream velocity
u_inf_2 = 2
rho_inf = 1                          # free-stream density

gamma = 1.4                          # Heat capacity ratio
Pr = 0.71                            # Prandtl number

p_inf=(rho_inf/gamma)*pow(u_inf/Ma,2) # free-stream static pressure p_inf = 31.746
mu = c*u_inf*rho_inf/Re               # dynamic viscosity mu = 0.00033333333

# 2D OPTIMISATION -----------------

if optimisation == "2D":
    b = 1                            # span length
    dt = 0.00014                     # time step
    tstart = 0                       # simulation start time
    tperturb = 5                     # sinusoidal perturbation time
    avg_from = 5                  # extract averages from
    tend = 10                       # total convective times
    GPUs = 1                         # number of GPUs for parallelisation
    wctime = '0-03:00:00'            # wall clock time for each individual to run
    waitingtime1 = 32*60             # waiting time while all cases are running
    waitingtime2 = 32*60             # waiting time for checking if cases are running
    timeout1 = 400*waitingtime1      # exit the waiting loop for running cases
    timeout2 = 200*waitingtime2      # exit the waiting loop for queueing cases

# 3D OPTIMISATION -----------------

if optimisation == "3D":
    b = 0.6                          # span length
    dt = 0.00006                     # time step
    tstart = 0                       # simulation start time
    tperturb = 5                     # sinusoidal perturbation time
    avg_from = 90                    # extract averages from
    tend = 150                       # total convective times
    GPUs = 34                        # number of GPUs for parallelisation
    wctime = '0-24:00:00'            # wall clock time for each individual to run
    waitingtime1 = 2*3600            # time between checks if cases are finished
    waitingtime2 = 3600              # time between checks if cases are running
    timeout1 = 50*waitingtime1       # exit the waiting loop for running cases
    timeout2 = 50*waitingtime2       # exit the waiting loop for queueing cases
    run_p_sep = True                 # run perturbation in separated job
    if run_p_sep:                       
        wctimep = '0-1:00:00'           # wall clock time for perturbation of each idv
        p_waitingtime1 = 3600           # time between checks if cases are finished
        p_waitingtime2 = 3600           # time between checks if cases are running
        p_timeout1 = 50*waitingtime1    # exit the waiting loop for running cases
        p_timeout2 = 25*waitingtime2    # exit the waiting loop for queueing cases
    if test == "Yes":
        tperturb = 1; tend = 2; avg_from = 1
        waitingtime1 = 60; waitingtime2 = 60
        timeout1 = 3600; timeout2 = timeout1*2
        wctime = '0-00:40:00'
    if rep_test == "Yes":
        timeout1 = 1; timeout2 = timeout1*2

main_ini_file = "Re3000M015.ini"        # PyFR initial file name
second_ini_file = "Re6000M03.ini"       # Second initial file with doubled v
pert_ini_file = "Perturbation.ini"      # PyFR perturbation file name

# DESIGN SPACE BOUNDS ------------------------------------------------------

class CheckDesignSpaceBounds(Repair):
    
    def _do(self, problem, pop, **kwargs):
        
        # Z = pop.get("X")
        Z = pop
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n\n------------------------", file=f)
        print("    Checking bounds", file=f)
        print("------------------------\n", file=f)
        for i in range(len(Z)):
            x_a = Z[i,0]
            y_a = Z[i,1]
            if x_a<0.5:
                if y_a>x_a*math.tan(gamma_rad):
                    h = abs((y_a-x_a*math.tan(gamma_rad))*math.cos(gamma_rad))
                    y_p = abs(h*math.sin(math.pi/2-gamma_rad))
                    y_a_modif = y_a - y_p
                    x_p = abs(h*math.cos(math.pi/2-gamma_rad))
                    x_a_modif = x_a + x_p
                    Z[i,0] = x_a_modif
                    Z[i,1] = y_a_modif
                    print("y_a="+str("%.3f" %y_a)+" --> y_a="+str("%.3f" %y_a_modif)
                         , file=f)
                    print("x_a="+str("%.3f" %x_a)+" --> x_a="+str("%.3f" %x_a_modif)
                          +"\n", file=f)
            if x_a>0.5:
                if y_a>(1-x_a*math.tan(gamma_rad)):
                    h = abs((y_a-(1-x_a*math.tan(gamma_rad)))*math.cos(gamma_rad))
                    y_p = abs(h*math.sin(math.pi/2-gamma_rad))
                    y_a_modif = y_a - y_p
                    x_p = abs(h*math.cos(math.pi/2-gamma_rad))
                    x_a_modif = x_a - x_p
                    Z[i,0] = x_a_modif
                    Z[i,1] = y_a_modif
                    print("y_a="+str("%.3f" %y_a)+" --> y_a="+str("%.3f" %y_a_modif)
                          , file=f)
                    print("x_a="+str("%.3f" %x_a)+" --> x_a="+str("%.3f" %x_a_modif)
                          +"\n", file=f)
        # pop.set("X", Z)
        pop = Z
    
        f.close()
        return pop

# ALGORITHM FOR EACH GEN ---------------------------------------------------

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=DVs,				
                         n_obj=OFs,				
                         n_constr=CTs,				
                         xl=np.array([xmin, ymin]),		
                         xu=np.array([xmax, ymax]))		
                         # elementwise_evaluation=True)

    def _evaluate(self, X, out, *args, **kwargs):        
        global ngen   
        ngen = ngen + 1
        if (ngen == evaluated_gen and initialseeding == "evalPOP"):
            out["F"] = eval_pop(evaluated_path,"OF")
        elif rep_test == "Yes":    
            self.generation_begins(ngen)
            sorted_population = self.sort_population(X)
        else:
            self.generation_begins(ngen)
            sorted_population = self.sort_population(X)
            sleep_for(10)
            # PRE-PROCESS --------------------------------------------------
            i = 0
            for row in sorted_population:                                   
                idv_path = get_idv_dir(i,row[0],row[1],ngen)
                self.generate_essential_files(idv_path,ngen,i,row[0],row[1])
                sleep_for(5)
                self.write_DVs_file(idv_path,row[0],row[1])
                self.run_evaluation(i,idv_path)
                i += 1
            # RUNNING --------------------------------------------------
            (running,finished,while1broken,while2broken,timebreak1,timebreak2) = self.reset_running(X)
            if optimisation == "2D":
                sleep_for(60)
            if (optimisation == "3D" and run_p_sep):
                #running = np.full((len(X), 1), False, dtype=bool)
                #finished = np.full((len(X), 1), False, dtype=bool)
                #while1broken = False
                #while2broken = False
                timebreak1 = time.time() + p_timeout1
                timebreak2 = time.time() + p_timeout2
                self.while_loop_check(finished,p_waitingtime1,running,p_waitingtime2,sorted_population,timebreak1,timebreak2,p_timeout1,p_timeout2,tperturb,while1broken,while2broken)
                i=0
                for row in sorted_population:
                    idv_path = get_idv_dir(i,row[0],row[1],ngen)
                    os.chmod("%s/u_job.slurm" %idv_path, 509)
                    f=open('%s/%s' %(parent_dir,output_opt_file),'a')
                    subprocess.call("mv %s/*.out %s/p_output" %(idv_path,idv_path), shell=True, stdout=f)
                    subprocess.call("cd %s && sbatch u_job.slurm" %(idv_path), shell=True, stdout=f)
                    f.close()
                    sleep_for(5)
                    i += 1
            i = 0
            for row in sorted_population:
                idv_path = get_idv_dir(i,row[0],row[1],ngen)
                last_sol_file = "gen-%s-idv-%s_%s.00.pyfrs" %(ngen,i,tend-5)
                sol_path = os.path.join(idv_path, last_sol_file)
                outputfile = self.check_for_files(idv_path)
                if os.path.isfile(sol_path):
                    running[i] = True
                    finished[i] = True
                    self.print_state(running[i],finished[i],i)
                elif outputfile:
                    running[i] = True
                    self.print_state(running[i],finished[i],i)
                else:
                    self.print_state(running[i],finished[i],i)
                i = i+1        
            # (running,finished,while1broken,while2broken,timebreak1,timebreak2) = self.reset_running(X)
            self.while_loop_check(finished,waitingtime1,running,waitingtime2,sorted_population,timebreak1,timebreak2,timeout1,timeout2,tend,while1broken,while2broken)
        if not (ngen == evaluated_gen and initialseeding == "evalPOP"):
            # POST-PROCESSING --------------------------------------------------
            sleep_for(60)
            first_line(ngen,"offspring")
            i = 0
            for row in sorted_population:  
                [row[3],row[4]] = self.data_extract(i,row[0],row[1])
                self.offspring_to_file(i,ngen,row[0],row[1],row[3],row[4])  
                i = i+1
            population = sorted_population[np.argsort(sorted_population[:, 2])]
            self.finish_evaluation(ngen)
            out["F"] = population[:,[3,4]]
                                 
    def generation_begins(self,ngen):
        path_gen = "%s/gen_%s" %(parent_dir,ngen)
        if os.path.isdir(path_gen):
            print("!! gen_%s exists" %ngen)
        else:
            os.mkdir(path_gen)
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n.....................................................", file=f)
        print(".........                                   .........", file=f)
        print(".........        GENERATION "+str(ngen)+" BEGINS        .........", file=f)
        print(".........                                   .........", file=f)
        print(".....................................................", file=f)
        f.close() 

    def sort_population(self,X):
        F = np.empty([len(X), OFs])
        initial_numbering = np.array(np.arange(len(X))).T
        numbered_X = np.column_stack((X,initial_numbering))
        sorted_X = numbered_X[np.argsort(numbered_X[:, 0])]
        sorted_population = np.concatenate((sorted_X,F),axis=1)
        return sorted_population    
    
    def generate_essential_files(self,path,ngen,i,dv1,dv2):
        if os.path.isdir(path):
            print("!! gen_%s/idv_%s exists" %(ngen,i))
        else:
            os.mkdir(path)
        last_sol_file = "gen-%s-idv-%s_%s.00.pyfrs" %(ngen,i,tend)
        pert_sol_file = "gen-%s-idv-%s_%s.00.pyfrs" %(ngen,i,tperturb)
        # job file
        if optimisation == "3D":
            FileGen.cirrus_jobfile(ngen,path,GPUs,wctimep,i,dv1,dv2,AoA,pert_ini_file,
                               main_ini_file,pert_sol_file,"p",optimisation)
            
        FileGen.cirrus_jobfile(ngen,path,GPUs,wctime,i,dv1,dv2,AoA,pert_ini_file,
                               main_ini_file,pert_sol_file,"u",optimisation)
        # main ini file
        FileGen.ini_file(path,main_ini_file,gamma,mu,Pr,order,tstart,tend,dt)
        FileGen.plugin_airfoilforces(path,main_ini_file)
        FileGen.plugin_soln_writer(path,main_ini_file,(tend-tperturb)/2,ngen,i)
        FileGen.plugin_soln_avg(path,optimisation,main_ini_file,avg_from,tend,ngen,i)
        FileGen.plugin_sampler(path,optimisation,main_ini_file)
        FileGen.boundary_conditions(path,optimisation,main_ini_file,0,u_inf,rho_inf,p_inf)
        # second ini file
        FileGen.ini_file(path,second_ini_file,gamma,mu,Pr,order,tstart,tend,dt)
        FileGen.plugin_airfoilforces_second(path,second_ini_file)
        FileGen.plugin_soln_writer(path,second_ini_file,(tend-tperturb)/2,ngen,i)
        FileGen.plugin_soln_avg(path,optimisation,second_ini_file,avg_from,tend,ngen,i)
        FileGen.plugin_sampler(path,optimisation,second_ini_file)
        FileGen.boundary_conditions(path,optimisation,second_ini_file,0,u_inf_2,rho_inf,p_inf)
        # perturbation ini file
        FileGen.ini_file(path,pert_ini_file,gamma,mu,Pr,order,tstart,tperturb,dt)
        FileGen.plugin_airfoilforces(path,pert_ini_file)
        FileGen.plugin_soln_writer(path,pert_ini_file,tperturb,ngen,i)
        FileGen.plugin_sampler(path,optimisation,pert_ini_file)
        FileGen.boundary_conditions(path,optimisation,pert_ini_file,1,u_inf,rho_inf,p_inf)
        # gmsh geo file
        FileGen.gmsh_file(ngen,i,path,dv1,dv2,AoA,optimisation)
        # eval.sh file
        FileGen.eval_file(ngen,i,path,optimisation,AoA,GPUs)

    def write_DVs_file(self,path,dv1,dv2):
        f = open("%s/DVs.csv" %path, "w")
        f.write(str(dv1)+','+str(dv2))
        f.close()
        
    def run_evaluation(self,i,path):
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n--------------------", file=f)
        print("    INDIVIDUAL "+str(i), file=f)
        print("--------------------", file=f)
        f.close()
        os.chmod("%s/eval.sh" %path, 509)
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        subprocess.call("cd %s && ./eval.sh" %(path), shell=True, stdout=f)
        f.close()
        
    def reset_running(self,X):
        running = np.full((len(X), 1), False, dtype=bool)
        finished = np.full((len(X), 1), False, dtype=bool)
        while1broken = False
        while2broken = False
        timebreak1 = time.time() + timeout1
        timebreak2 = time.time() + timeout2
        return (running,finished,while1broken,while2broken,timebreak1,timebreak2)
    
    def while_loop_check(self,finished,waitingtime1,running,waitingtime2,sorted_population,timebreak1,timebreak2,timeout1,timeout2,tfinish,while1broken,while2broken):
        while False in finished:
            sleep_for(60)
            while False in running:
                sleep_for(60)
                i = 0
                for row in sorted_population:
                    idv_path = get_idv_dir(i,row[0],row[1],ngen)
                    last_sol_file = "gen-%s-idv-%s_%s.00.pyfrs" %(ngen,i,tfinish)
                    sol_path = os.path.join(idv_path, last_sol_file)
                    outputfile = self.check_for_files(idv_path)
                    if (running[i]==True and finished[i]==False): 
                        if os.path.isfile(sol_path):
                            finished[i] = True
                            self.print_state(running[i],finished[i],i)
                        else:
                            self.print_output(idv_path,i)
                            finished[i] = False
                    elif running[i]==False:
                        if outputfile and os.path.isfile(sol_path):
                            running[i] = True
                            finished[i] = True
                            self.print_state(running[i],finished[i],i)
                        elif outputfile:
                            running[i] = True
                            self.print_output(idv_path,i)
                        else:
                            self.print_state(running[i],finished[i],i)
                    elif finished[i]==True:
                        self.print_state(running[i],finished[i],i)
                    i = i+1
                # finished for of individuals
                if time.time() > timebreak2:
                     while2broken = True
                     f=open('%s/%s' %(parent_dir,output_opt_file),'a')
                     print("Exiting while loop for queue due to timeout:"+
                           str(timeout2/60)+" minutes", file=f)
                     f=close()
                     break
            self.print_while2(while2broken)
            i = 0
            for row in sorted_population:
                idv_path = get_idv_dir(i,row[0],row[1],ngen)
                last_sol_file = "gen-%s-idv-%s_%s.00.pyfrs" %(ngen,i,tfinish)
                sol_path = os.path.join(idv_path, last_sol_file) 
                if finished[i] == False: 
                    if os.path.exists(sol_path):
                        finished[i] = True
                        self.print_state(running[i],finished[i],i)
                    else:
                        self.print_output(idv_path,i)
                else:
                    self.print_state(running[i],finished[i],i)
                i = i+1
            if time.time() > timebreak1:
                     while1broken=True
                     f=open('%s/%s' %(parent_dir,output_opt_file),'a')
                     print("Exiting while loop for queue due to timeout:"+
                           str(timeout1/60)+" minutes", file=f)
                     f=close()
                     break
        self.print_while1(while1broken)

    def print_state(self,running,finished,i):
        if running==True:
            if finished==True:
                f=open('%s/%s' %(parent_dir,output_opt_file),'a')
                print("- Individual "+str(i)+" has finished", file=f)
                f.close()
            else:
                f=open('%s/%s' %(parent_dir,output_opt_file),'a')
                print("- Individual "+str(i)+" is running", file=f)
                f.close()
        else:
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("- Individual "+str(i)+" is waiting in the queue", file=f)
            f.close()

    def check_for_files(self,p):
        filepath = os.path.join(p,'*.out')
        for filepath_object in glob.glob(filepath):
            if os.path.isfile(filepath_object):
                return True
        return False	
        
    def print_output(self,p,i):
        filepath = os.path.join(p,'*.out')
        for filepath_object in glob.glob(filepath):
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("- Individual "+str(i)+" is running", file=f)
            f.close()
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            subprocess.call("\ntail --lines=1 %s" %filepath_object, shell=True, stdout=f)
            f.write("\n"); f.close()

    def print_while1(self,while1broken):
        if while1broken==False:
            f=open('%s/%s' %(parent_dir,output_opt_file),'a') 
            print("\n-----------------------------------------", file=f)
            print("All individuals have finished", file=f)
            print("-----------------------------------------", file=f)
            f.close()
        else:
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("\n-----------------------------------------", file=f)
            print("Waiting loop was broken due to timeout", file=f)
            print("Some individuals have not run correctly", file=f)
            print("Values of Cl=100 and Cd=100 are assigned", file=f)
            print("-----------------------------------------", file=f)
            f.close()

    def print_while2(self,while2broken):
        if while2broken==False:
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("\n-----------------------------------------", file=f)
            print("All individuals are out of the queue", file=f)
            print("-----------------------------------------\n", file=f)
            f.close()
        else:
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("\n-----------------------------------------", file=f)
            print("Queuing while loop was broken due to timeout", file=f)
            print("-----------------------------------------\n", file=f)
            f.close()
            
    def data_extract(self,i,dv1,dv2):        
        output_forces = 10
        # not_wanted_lines = avg_from/(dt*output_forces)
        not_wanted_lines = 7000
        q_inf = 0.5*rho_inf*u_inf**2
        S = c*b
        clvalues = []
        cdvalues = []
        idv_path = get_idv_dir(i,dv1,dv2,ngen)
        if os.path.isfile('%s/airfoil-forces.csv' %idv_path):
            with open('%s/airfoil-forces.csv' %idv_path, 'r') as OFV:      
                for line_number, line in enumerate(OFV, 1):
                    if line_number <= not_wanted_lines:
                        continue
                    linesplit = line.strip().split(",")  
                    if not linesplit:  # empty
                        continue    
                    if optimisation == "2D":
                        clvalues.append((float(linesplit[2])+float(linesplit[4]))
                                         /(q_inf*S))
                        cdvalues.append((float(linesplit[1])+float(linesplit[3]))
                                         /(q_inf*S))
                    if optimisation == "3D":
                        clvalues.append((float(linesplit[2])+float(linesplit[5]))
                                         /(q_inf*S))
                        cdvalues.append((float(linesplit[1])+float(linesplit[4]))
                                         /(q_inf*S))
                if len(clvalues)>0:
                    cl = sum(clvalues)/float(len(clvalues))
                    cd = sum(cdvalues)/float(len(cdvalues))
                else:
                    cl = 100.0; cd = 100.0
            OFV.close
        else:
            cl = 100.0; cd = 100.0
        with open('%s/cl_cd_idv.csv' %idv_path, 'w') as DV:
            DV.write(str(cl)+','+str(cd))
        DV.close
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n- Individual "+str(i), file=f)
        print("Cl = "+str("%.3f" %cl)+", Cd = "+str("%.3f" %cd), file=f)
        f.close()
        forces = [-cl,cd]
        return forces

    def offspring_to_file(self,i,ngen,dv1,dv2,of1,of2):
        idv_dir = "idv_%s_x%.3f_y%.3f" %(str(i),dv1,dv2)
        offspringfile = open('gen_%s/offspring_gen%s.csv' %(ngen,ngen),'a')
        offspringfile.write('\n'+str(i)+','+str(idv_dir)+','+str(dv1)+','+
                            str(dv2)+','+str(of1)+','+str(of2))
        offspringfile.close
            
    def finish_evaluation(self,ngen):
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n-----------------------------------------", file=f)
        print("           Evaluation finished", file=f)
        print("-----------------------------------------\n", file=f)
        print("\nGeneration "+str(ngen)+", head node computing time: "
                +str(t.get()), file=f)
        f.close()

# CALLBACK --------------------------------------------------------------------
class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.clopt = []
        self.cdopt = []
        self.paretoXF = []
        
    def notify(self, algorithm):
        global ngen
        if (ngen == evaluated_gen and initialseeding == "evalPOP"):
            paretoXF = np.concatenate((algorithm.pop.get("X"),
                                      algorithm.pop.get("F")),axis=1)
            sorted_paretoXF = paretoXF[np.argsort(paretoXF[:, 0])]
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("\nCurrent Pareto:\n Xa,        Ya,       -Cl,        Cd", file=f)
            print(sorted_paretoXF, file=f)
            print("\n\n", file=f)
            f.close()
        else:
            path_gen = "%s/gen_%s" %(parent_dir,ngen)
            paretoXF = np.concatenate((algorithm.pop.get("X"),
                                       algorithm.pop.get("F")),axis=1)
            sorted_paretoXF = paretoXF[np.argsort(paretoXF[:, 0])]
            path_found = ["NOT FOUND" for x in range(len(sorted_paretoXF))]
            found_idv = np.full((len(sorted_paretoXF), 1), False, dtype=bool)
            f=open('%s/%s' %(parent_dir,output_opt_file),'a')
            print("\nCurrent Pareto:\n Xa,        Ya,       -Cl,        Cd", file=f)
            print(sorted_paretoXF, file=f)
            print("\n\n", file=f)
            f.close()
            first_line(ngen,"optimum")
            i = 0
            for row in sorted_paretoXF:
                for n in range(ngen,1,-1):
                    for j in range(offspring):
                        idv_dir="gen_%s/idv_%s_x%.3f_y%.3f" %(str(n),str(j),
                                row[0],row[1])
                        if os.path.isdir("%s/%s" %(parent_dir,idv_dir)):
                            path_found[i] = idv_dir
                            found_idv[i] = True
                            self.print_state(found_idv[i],i,n,path_found[i])
                            break # idv loop
                    if found_idv[i] == True:
                        break # gen loop
                if found_idv[i] == False:
                    n = 1
                    for j in range(population):
                        idv_dir="gen_%s/idv_%s_x%.3f_y%.3f" %(str(n),str(j),
                                row[0],row[1])
                        if os.path.isdir("%s/%s" %(parent_dir,idv_dir)):
                            path_found[i] = idv_dir
                            found_idv[i] = True
                            self.print_state(found_idv[i],i,n,path_found[i])
                            break # idv loop
                    if found_idv[i] == False:
                        self.print_state(found_idv[i],i,n,idv_dir)
                self.optimum_to_file(path_found[i],i,ngen,row[0],row[1],row[2],row[3])
                i = i+1
        sleep_for(10)
        self.write_pareto_to_file(ngen,sorted_paretoXF)
        self.convergence_of_min(ngen)  

    def optimum_to_file(self,path_found,i,ngen,dv1,dv2,of1,of2):
        optimumfile = open('gen_%s/optimum_gen%s.csv' %(ngen,ngen),'a')
        optimumfile.write('\n'+str(i)+','+str(path_found)+','+str(dv1)+
                          ','+str(dv2)+','+str(of1)+','+str(of2))
        optimumfile.close
        
    def print_state(self,found_idv,i,n,path_found):
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        if found_idv==True:
            print("- Idv "+str(i)+" of Pareto of gen "
                  +str(n)+" found in: "+path_found, file=f)
        else:
            print("- Idv "+str(i)+" of Pareto of gen "
                  +str(n)+"NOT FOUND! Last searched: "+path_found, file=f)
        f.close()
            
    def write_pareto_to_file(self,ngen,sorted_paretoXF):
        with open('Pareto_evolution.csv','a') as P:
            P.write('\n#Pareto front generation '+str(ngen)+'\n')
            P.write('# gen,    OF1:Cl,    OF2:Cd,    DV1:x_a,    DV2:y_a \n')
        P.close
        i = 0
        for row in sorted_paretoXF:
            with open('Pareto_evolution.csv','a') as P:
                P.write(str(ngen)+','+str(row[0])+','+str(row[1]))
                P.write(','+str(row[2])+','+str(row[3])+'\n')
            P.close
            i = i+1
    
    def convergence_of_min(self,ngen):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.clopt.append(algorithm.pop.get("F")[:,0].min())
        self.cdopt.append(algorithm.pop.get("F")[:,1].min())
        with open('Convergence.csv','a') as C:
            C.write('\n# Optimum values generation '+str(ngen)+'\n')
            C.write('# nevals, opt_cl, opt_cd\n')
            C.write(str(algorithm.evaluator.n_eval)+',')
            C.write(str(algorithm.pop.get("F")[:,0].min())+',')
            C.write(str(algorithm.pop.get("F")[:,1].min())+'\n')
        C.close                                          

# SHARED FUNCTIONS ---------------------------------------------------------
    
def get_idv_dir(i,dv1,dv2,ngen):
    idv_dir = "gen_%s/idv_%s_x%.3f_y%.3f" %(str(ngen),str(i),dv1,dv2)
    idv_path = os.path.join(parent_dir, idv_dir)
    return idv_path
    
def first_line(ngen,idv_type):
    genfile = open('gen_%s/%s_gen%s.csv' %(ngen,idv_type,ngen),'a')
    genfile.write('# idv,    idv_folder,    DV1:x_a,    DV2:y_a,    '+
                  'OF1:Cl,    OF2:Cd')
    genfile.close
    
def sleep_for(seconds):
    t.pause()
    if not rep_test == "Yes":
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("\n*Sleeping for %s seconds" %seconds, file=f)
        f.close()
        time.sleep(seconds)
    t.resume()

# TIMER --------------------------------------------------------------------

from datetime import datetime
import time

class MyTimer():
    def __init__(self):
        now = datetime.now()
        current_date = now.strftime("%b-%d-%Y %H:%M:%S")
        f=open('%s/%s' %(parent_dir,output_opt_file),'w')
        print('Initializing timer '+current_date, file=f)
        f.close()
        global started_date
        started_date = current_date
        self.timestarted = None
        self.timepaused = None
        self.paused = False

    def start(self):
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        print("Starting timer",file=f)
        f.close()
        self.timestarted = datetime.now()

    def pause(self):
        if self.timestarted is None:
             raise ValueError("Timer not started")
        if self.paused:
            raise ValueError("Timer is already paused")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_date = now.strftime("%b-%d-%Y %H:%M:%S")
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        f.write("\n*Pausing timer at %s" %current_date)
        f.close()        
        self.timepaused = datetime.now()
        self.paused = True

    def resume(self):
        if self.timestarted is None:
            raise ValueError("Timer not started")
        if not self.paused:
            raise ValueError("Timer is not paused")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f=open('%s/%s' %(parent_dir,output_opt_file),'a')
        f.write("*Resuming timer at %s\n\n" %current_time)
        f.close()
        pausetime = datetime.now() - self.timepaused
        self.timestarted = self.timestarted + pausetime
        self.paused = False

    def get(self):
        if self.timestarted is None:
            raise ValueError("Timer not started")
        if self.paused:
            return self.timepaused - self.timestarted
        else:
            return datetime.now() - self.timestarted

# SEEDING ------------------------------------------------------------------

def seeding(seeding_path):
    xseeding = []
    yseeding = []
    with open('%s' %seeding_path, 'r') as SD:      
        for line_number, line in enumerate(SD, 1):
            if line.strip().startswith("#"):
                continue
            linesplit = line.strip().split(",")  
            if not linesplit:  # empty
                continue    
            xseeding.append(float(linesplit[2]))
            yseeding.append(float(linesplit[3]))
    SD.close
    X = np.column_stack((xseeding,yseeding))
    return X
    
def eval_pop(evaluated_path,data):
    xevaluated = []
    yevaluated = []
    with open('%s' %evaluated_path, 'r') as SD:      
        for line_number, line in enumerate(SD, 1):
            modifline = line.replace("     0.","000 0.")
            modifline = modifline.replace("    0.","00 0.")
            modifline = modifline.replace("   0.","0 0.")
            modifline = modifline.replace("  0.","0 0.")
            modifline = modifline.replace("   ]","000]")
            modifline = modifline.replace("  ]","00]")
            modifline = modifline.replace(" ]","0]")
            modifline = modifline.replace("    1","00 1")
            modifline = modifline.replace("   1","0 1")
            modifline = modifline.replace("  ]","00]")
            modifline = modifline.replace(" ]","0]")
            modifline = modifline.replace(". 1",". 1")
            linestrip = modifline.strip().lstrip("[").strip("]")
            if not linestrip:  # empty
                continue
            if line.strip().startswith("#"):
                continue   
            #linesplit = linestrip.strip().split(" ")
            linesplit = linestrip.strip().split(",")
            if not linesplit:  # empty
                continue    
            if data == "DV":
                xevaluated.append(float(linesplit[2]))
                yevaluated.append(float(linesplit[3]))
            if data == "OF":
                xevaluated.append(float(linesplit[4]))
                yevaluated.append(float(linesplit[5]))
    SD.close
    X = np.column_stack((xevaluated,yevaluated))
    return X

# ALGORITHM DEFINITION -----------------------------------------------------

if initialseeding == "DVs":
    f=open('%s/%s' %(parent_dir,output_opt_file),'a')
    print("\n> Initialising from pre-defined design variables\n",file=f)
    f.close()
    sampling = seeding(seeding_path)                          # specific initial sampling
if initialseeding =="LHS":
    f=open('%s/%s' %(parent_dir,output_opt_file),'a')
    print("\n> Initialising from Latin Hypercube Sampling\n",file=f)
    f.close()
    # sampling=get_sampling('real_lhs')                         # latin hypercube sampling
    sampling = LHS()
if initialseeding =="evalPOP":
    f=open('%s/%s' %(parent_dir,output_opt_file),'a')
    print("\n> Initialising from pre-evaluated population\n",file=f)
    f.close()
    sampling = eval_pop(evaluated_path,"DV")                       # already evaluated population
# crossover=get_crossover("real_sbx", prob=prob_c, eta=eta_c)   # Simulated Binary Crossover
crossover=SBX(prob=prob_c, eta=eta_c)
# mutation=get_mutation("real_pm", prob=prob_m, eta=eta_m)      # Polynomial Mutation
mutation = PolynomialMutation(prob=prob_m, eta=eta_m)
termination = get_termination("n_gen", n_max_gen)             # Termination ngen
#selection=get_selection('tournament', {'pressure':2,'func_comp':binary_tournament})
#selection = TournamentSelection(pressure=2, func_comp=binary_tournament)

# ALGORITHM --------------------------------------------------------------------

t = MyTimer()
t.start()

problem = MyProblem()

time.sleep(10)

if restart == "No":
    algorithm = NSGA2(pop_size=population,
                      n_offsprings=offspring,
                      sampling=sampling,
                      #selection=selection,
                      crossover=crossover,
                      mutation=mutation,
                      repair=CheckDesignSpaceBounds(),
                      eliminate_duplicates=True
                      )
    algorithm.setup(problem,
                    #algorithm,
                   callback=MyCallback(),
                   termination = termination,
                   verbose=True,
                   save_history=True,
                   copy_algorithm=False,
                   seed=1)       
elif restart == "Yes":
    algorithm, = np.load(f"checkpoint_%s.npy" %ngen, allow_pickle=True).flatten()    


# CHECKPOINT ---------------------------------------------------------------------

while algorithm.has_next():
    algorithm.next()
    np.save(f"checkpoint_{algorithm.n_gen}", algorithm)
    f=open('%s/%s' %(parent_dir,output_opt_file),'a')
    print("\nSaving checkpoint checkpoint_%s" %algorithm.n_gen, file=f)
    f.close()


# RUNNING TIME --------------------------------------------------------------------

global started_date
now = datetime.now()
current_date = now.strftime("%b-%d-%Y %H:%M:%S")
f=open('%s/%s' %(parent_dir,output_opt_file),'a')
print("\n*********************************************************************\n", file=f)    
print("Total optimisation running time (without sleep time): "+str(t.get()), file=f)
print("\nStarted optimisation on "+started_date, file=f)
print("\nFinished optimisation on "+current_date, file=f)
print("\n*********************************************************************\n", file=f)
f.close()
