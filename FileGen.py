#!/usr/bin/env python

import math 
import os

# SUBMISSION FILE -----------------------------------------------------------------------------------

def cirrus_jobfile(n,idvpath,GPUs,wctime,i,x,y,AoA,p_ini_file,main_ini_file,p_sol_file,t,opt):
    s=open('%s/job.sub' %(idvpath),'w')
    s.write('#!/bin/bash'+'\n')
    s.write('#$ -N testcase'+'\n')
    s.write('#$ -wd %s\n' % idvpath)
    s.write('#$ -j y'+'\n')
    s.write('#$ -pe mpi 1'+'\n')
    s.write('#$ -l v100'+'\n')

    s.write('source /home/ruoxi/.bashrc'+'\n')
    #s.write('export LD_LIBRARY_PATH='/usr/local/cuda-11.7/lib64':$LD_LIBRARY_PATH'+'\n')
    s.write("export LD_LIBRARY_PATH='/usr/local/cuda-11.7/lib64':$LD_LIBRARY_PATH\n")
    s.write("export LIBRARY_PATH='/usr/local/cuda-11.7/include':$LIBRARY_PATH\n")
    s.write("export PATH='/usr/local/cuda-11.7/bin':$PATH\n")
    s.write('module load rocks-openmpi'+'\n')

    s.write('source /home/ruoxi/PyFR-DEVELOP/pyfr-develop1/bin/activate'+'\n')

    s.write('mpiexec -n 1 pyfr -p run -b cuda '+str(AoA)+ 'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm '+main_ini_file+'\n')
        
        

        
    # s.write('#'+'\n')
    # s.write('#SBATCH --job-name=2D'+str(t)+'_idv_'+str(i)+'_x_'+str("%.3f" %x)+
    #         '_y_'+str("%.3f" %y)+'\n')
    # s.write('#SBATCH --time='+wctime+'\n')
    # s.write('#SBATCH --nodes='+str(GPUs)+'\n')
    # s.write('#SBATCH --ntasks-per-node=1'+'\n')
    # s.write('#SBATCH --partition=normal'+'\n')
    # s.write('#SBATCH --constraint=gpu'+'\n')
    # s.write('#SBATCH --account=s1075'+'\n')
    # s.write(''+'\n')
    # s.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK'+'\n')
    # s.write('export CRAY_CUDA_MPS=1'+'\n')
    # s.write('export HDF5_USE_FILE_LOCKING=\'FALSE\''+'\n')
    # s.write(''+'\n')
    # s.write('source /users/lcarosro/.bashrc'+'\n')
    # s.write('source /users/lcarosro/v2pyfrv12/bin/activate'+'\n')
    # s.write(''+'\n')
    # if opt == "3D":
    #     if t == "p":
    #         s.write('srun --unbuffered --ntasks='+str(GPUs)+
    #             ' --tasks-per-node=1 pyfr run -b cuda -p '+str(AoA)+
    #             'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm '+p_ini_file+'\n')
    #     elif t == "u":
    #         s.write('srun --unbuffered --ntasks='+str(GPUs)+
    #             ' --tasks-per-node=1 pyfr restart -b cuda -p '+str(AoA)+'AoA-gen-'
    #             +str(n)+'-idv-'+str(i)+'.pyfrm '+p_sol_file+' '+main_ini_file+'\n')
    # elif opt == "2D":
    #     s.write('srun --unbuffered --ntasks='+str(GPUs)+
    #         ' --tasks-per-node=1 pyfr run -b cuda -p '+str(AoA)+
    #         'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm '+p_ini_file+'\n')
    #     s.write('srun --unbuffered --ntasks='+str(GPUs)+
    #         ' --tasks-per-node=1 pyfr restart -b cuda -p '+str(AoA)+'AoA-gen-'
    #         +str(n)+'-idv-'+str(i)+'.pyfrm '+p_sol_file+' '+main_ini_file+'\n')
    s.close()

# PYFR FILES -----------------------------------------------------------------------------------    

def ini_file(idvpath,ini_file,gamma,mu,Pr,order,tstart,tend,dt):
    f=open('%s/%s' %(idvpath,ini_file),'w')     
    f.write('[backend]'+'\n') 
    f.write('precision = double'+'\n') 
    f.write('rank-allocator = linear'+'\n') 
    f.write(''+'\n') 
    f.write('[backend-cuda]'+'\n') 
    f.write('device-id = local-rank'+'\n') 
    f.write('mpi-type = standard'+'\n') 
    f.write(''+'\n') 
    f.write('[backend-opencl]'+'\n') 
    f.write('platform-id = 0'+'\n') 
    f.write('device-type = gpu'+'\n') 
    f.write('device-id = local-rank'+'\n') 
    f.write(''+'\n') 
    f.write('[backend-openmp]'+'\n') 
    f.write('cc = gcc'+'\n') 
    f.write(''+'\n') 
    f.write('[constants]'+'\n') 
    f.write('gamma = '+str(gamma)+'\n') 
    f.write('mu = '+str(mu)+'\n') 
    f.write('Pr = '+str(Pr)+'\n') 
    f.write(''+'\n') 
    f.write('[solver]'+'\n') 
    f.write('system = navier-stokes'+'\n') 
    f.write('order = '+str(order)+'\n') 
    f.write(''+'\n') 
    f.write('[solver-time-integrator]'+'\n') 
    f.write('scheme = rk4'+'\n') 
    f.write('controller = none'+'\n') 
    f.write('tstart = '+str(tstart)+'\n') 
    f.write('tend = '+str(tend)+'.01 \n') 
    f.write('dt = '+str(dt)+'\n') 
    f.write(''+'\n') 
    f.write('[solver-interfaces]'+'\n') 
    f.write('riemann-solver = rusanov'+'\n') 
    f.write('ldg-beta = 0.5'+'\n') 
    f.write('ldg-tau = 0.1'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-interfaces-line]'+'\n') 
    f.write('flux-pts = gauss-legendre'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-interfaces-quad]'+'\n') 
    f.write('flux-pts = gauss-legendre'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-interfaces-tri]'+'\n') 
    f.write('flux-pts = williams-shunn'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-elements-hex]'+'\n') 
    f.write('soln-pts = gauss-legendre'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-elements-pri]'+'\n') 
    f.write('soln-pts = williams-shunn~gauss-legendre'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-elements-quad]'+'\n') 
    f.write('soln-pts = gauss-legendre'+'\n') 
    f.write(''+'\n') 
    f.write('[solver-elements-tri]'+'\n') 
    f.write('soln-pts = williams-shunn'+'\n') 
    f.write(''+'\n') 
    f.write('[soln-plugin-nancheck]'+'\n') 
    f.write('nsteps = 50'+'\n') 
    f.write(''+'\n') 
    f.close()
    
def plugin_airfoilforces(idvpath,ini_file):
    f=open('%s/%s' %(idvpath,ini_file),'a')  
    f.write('[soln-plugin-fluidforce-airfoil]'+'\n') 
    f.write('nsteps = 10'+'\n') 
    f.write('file = airfoil-forces.csv'+'\n') 
    f.write('header = true'+'\n') 
    f.write('quad-deg = 1'+'\n') 
    f.write(''+'\n') 
    f.close()
    
def plugin_airfoilforces_second(idvpath,ini_file):
    f=open('%s/%s' %(idvpath,ini_file),'a')  
    f.write('[soln-plugin-fluidforce-airfoil]'+'\n') 
    f.write('nsteps = 10'+'\n') 
    f.write('file = airfoil-forces-2.csv'+'\n') 
    f.write('header = true'+'\n') 
    f.write('quad-deg = 1'+'\n') 
    f.write(''+'\n') 
    f.close()
    
def plugin_soln_writer(idvpath,ini_file,tout,ngen,idvnumber):
    f=open('%s/%s' %(idvpath,ini_file),'a')  
    f.write('[soln-plugin-writer]'+'\n') 
    f.write('dt-out = '+str(tout)+'\n') 
    f.write('basedir = .'+'\n') 
    f.write('basename = gen-'+str(ngen)+'-idv-'+str(idvnumber)+'_{t:.2f}'+'\n') 
    f.write(''+'\n') 
    f.close()

def plugin_soln_avg(idvpath,optimisation,ini_file,avg_from,tend,ngen,idvnumber):
    tout = tend - avg_from
    f=open('%s/%s' %(idvpath,ini_file),'a')   
    f.write('[soln-plugin-tavg]'+'\n') 
    f.write('nsteps = 10'+'\n') 
    f.write('tstart = '+str(avg_from)+'\n') 
    f.write('dt-out = '+str(tout)+'\n') 
    f.write('basedir = .'+'\n') 
    f.write('basename = Averages-gen-'+str(ngen)+'-idv-'+str(idvnumber)+
            '_{t:.2f}'+'\n') 
    f.write('avg-p = p'+'\n') 
    if optimisation == "2D":
        f.write('avg-vel = sqrt(u*u + v*v)'+'\n') 
    if optimisation == "3D":
        f.write('avg-vel = sqrt(u*u + v*v + w*w)'+'\n') 
    f.write('avg-rho = rho'+'\n') 
    f.write('avg-u = u'+'\n') 
    f.write('avg-v = v'+'\n') 
    if optimisation == "3D":
        f.write('avg-w = w'+'\n') 
    f.write(''+'\n') 
    f.close()
    
def plugin_sampler(idvpath,optimisation,ini_file):
    f=open('%s/%s' %(idvpath,ini_file),'a')   
    f.write('[soln-plugin-sampler]'+'\n') 
    f.write('nsteps = 10'+'\n') 
    if optimisation == "3D":
        f.write('samp-pts = [(1.05, 0.03, 0.3), (1.05, 0.06, 0.3), (1.05, 0.09, 0.3), '+
            '(1.05, 0.12, 0.3), (1.05, 0.15, 0.3), (1.05, 0.18, 0.3), (1.05, 0.21,'+
            ' 0.3), (1.05, 0.24, 0.3), (1.05, 0.27, 0.3), (1.05, 0.3, 0.3), '+
            ' (1.05, 0.33, 0.3), (1.05, 0.36, 0.3), (1.05, 0.39, 0.3),'+
            ' (1.05, 0.42, 0.3), (1.05, 0.45, 0.3), (1.05, 0.48, 0.3), (1.05, 0.51,'+
            ' 0.3), (1.05, 0.54, 0.3), (1.05, 0.57, 0.3), (1.05, 0.6, 0.3)]'+'\n') 
    if optimisation == "2D":
        f.write('samp-pts = [(1.05, 0.03), (1.05, 0.06), (1.05, 0.09), '+
            '(1.05, 0.12), (1.05, 0.15), (1.05, 0.18), (1.05, 0.21'+
            '), (1.05, 0.24), (1.05, 0.27), (1.05, 0.3), '+
            ' (1.05, 0.33), (1.05, 0.36), (1.05, 0.39),'+
            ' (1.05, 0.42), (1.05, 0.45), (1.05, 0.48), (1.05, 0.51'+
            '), (1.05, 0.54), (1.05, 0.57), (1.05, 0.6)]'+'\n')         
    f.write('file = inst_TE_data.csv'+'\n') 
    f.write('header = true'+'\n') 
    f.write(''+'\n') 
    f.close()
    
def boundary_conditions(idvpath,optimisation,ini_file,perturb,u_inf,rho_inf,p_inf):
    f=open('%s/%s' %(idvpath,ini_file),'a')  
    f.write('[soln-bcs-inlet]'+'\n') 
    f.write('type = char-riem-inv'+'\n') 
    f.write('rho = '+str(rho_inf)+'\n') 
    if perturb == 0:
        f.write('u = '+str(u_inf)+'\n') 
        f.write('v = 0'+'\n')
        if optimisation == "3D":
            f.write('w = 0'+'\n')
    if perturb == 1:
        f.write('u = '+str(u_inf)+'+0.2*sin(100*t)\n') 
        f.write('v = 0.2*sin(100*t)'+'\n')
        if optimisation == "3D":
            f.write('w = 0.2*sin(100*t)'+'\n')
    f.write('p = '+str(p_inf)+'\n') 
    f.write(''+'\n') 
    f.write('[soln-bcs-outlet]'+'\n') 
    f.write('type = char-riem-inv'+'\n')  
    f.write('rho = '+str(rho_inf)+'\n') 
    f.write('u = '+str(u_inf)+'\n') 
    f.write('v = 0'+'\n')
    if optimisation == "3D":
        f.write('w = 0'+'\n')
    f.write('p = '+str(p_inf)+'\n') 
    f.write(''+'\n') 
    f.write('[soln-bcs-horizontal]'+'\n') 
    f.write('type = char-riem-inv'+'\n')  
    f.write('rho = '+str(rho_inf)+'\n') 
    f.write('u = '+str(u_inf)+'\n') 
    f.write('v = 0'+'\n')
    if optimisation == "3D":
        f.write('w = 0'+'\n')
    f.write('p = '+str(p_inf)+'\n') 
    f.write(''+'\n') 
    f.write('[soln-bcs-airfoil]'+'\n') 
    f.write('type = no-slp-adia-wall'+'\n') 
    f.write(''+'\n') 
    f.write('[soln-ics]'+'\n')  
    f.write('rho = '+str(rho_inf)+'\n') 
    f.write('u = '+str(u_inf)+'\n') 
    f.write('v = 0'+'\n')
    if optimisation == "3D":
        f.write('w = 0'+'\n')
    f.write('p = '+str(p_inf)+'\n') 
    f.close()

# EVALUATION SH FILE -----------------------------------------------------------------------------------

def eval_file(n,i,idvpath,optimisation,AoA,GPUs):
    f=open('%s/eval.sh' %idvpath,'w')
    f.write('#!/bin/bash'+'\n') 
    f.write(''+'\n') 
    f.write('source /home/ruoxi/.bashrc'+'\n') 
    f.write('source /home/ruoxi/PyFR-develop/pyfr-develop/bin/activate'+'\n')
    f.write(''+'\n') 
    f.write('printf "\\n- GMSH"'+'\n') 
    f.write('printf "\\nGenerating mesh \\n"'+'\n') 
    if optimisation == "2D":

        f.write('/share/data/ruoxi/Dependencies/GMSH/GMSH/bin/gmsh -2 -o '+str(AoA)+
                'AoA-gen-'+str(n)+'-idv-'+str(i)+'.msh '+str(AoA)+
                'AoA-gen-'+str(n)+'-idv-'+str(i)+'.geo '+'&> gmsh.log'+'\n') 
        
        f.write('sed -n \'45p\' gmsh.log'+'\n') 
        f.write('sed -n \'46p\' gmsh.log'+'\n')
    f.write('printf "\\n- PyFR"'+'\n') 
    f.write('printf "\\nImporting gmsh mesh to pyfrm\\n"'+'\n') 
    f.write('/home/ruoxi/PyFR-DEVELOP/pyfr-develop1/bin/pyfr import '+str(AoA)+'AoA-gen-'+str(n)+'-idv-'+str(i)+'.msh '
            +str(AoA)+'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm'+'\n') 
    if optimisation == "3D":
        f.write('pyfr partition '+str(GPUs)+' '+str(AoA)+
        'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm .'+'\n') 
    f.write('printf "\\n- Job submission\\n"'+'\n') 
    if optimisation == "2D":
        # f.write('pyfr run -b openmp -p '+str(AoA)+
        #     'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm Perturbation_Re3000.ini')
        # # f.write('pyfr restart -b openmp -p '+str(AoA)+'AoA-gen-'
        # #     +str(n)+'-idv-'+str(i)+'.pyfrm '+p_sol_file+' Re3000M015.ini')
        f.write('qsub job.sub'+'\n')
    if optimisation == "3D":
        f.write('sbatch p_job.slurm'+'\n')
    f.close()

# Second Evaluation
def eval_file_second(n,i,idvpath,optimisation,AoA,GPUs):
    f=open('%s/eval2.sh' %idvpath,'w')
    f.write('#!/bin/bash'+'\n') 
    f.write(''+'\n') 
    f.write('source /home/ry619/.bashrc'+'\n') 
    f.write(''+'\n') 
    
    if optimisation == "3D":
        f.write('pyfr partition '+str(GPUs)+' '+str(AoA)+
        'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm .'+'\n') 
    f.write('printf "\\n- Job submission\\n"'+'\n') 

    if optimisation == "2D":
        f.write('pyfr run -b openmp -p '+str(AoA)+
                'AoA-gen-'+str(n)+'-idv-'+str(i)+'.pyfrm Re6000M03.ini')

    
# MESH FILE -----------------------------------------------------------------------------------

def gmsh_file(n,i,idvpath,x,y,AoA,optimisation):   
    n1 = int(math.ceil(math.sqrt(pow(x,2)+pow(y,2))/0.034)+1)  # 10;
    n2 = int(math.ceil(math.sqrt(pow(1-x,2)+pow(y,2))/0.041)+1)  # 18;  
    f=open('%s/%sAoA-gen-%s-idv-%s.geo' %(idvpath,AoA,n,i),'w')
    f.write('// Gmsh project created  by Lidia Caros Roca 2020'+'\n')
    f.write('SetFactory("OpenCASCADE");'+'\n')
    f.write(''+'\n')
    f.write('//////////////////////////////////////// GEOMETRY'+'\n')
    f.write(''+'\n')
    f.write('//// POINTS DEFINING THE TRIANGULAR AIRFOIL'+'\n')
    f.write('//+'+'\n')
    f.write('Point(1) = {0, 0, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(2) = {1, 0, 0, 0.1};'+'\n')    
    f.write('//+ ------------------------------------------------- APEX'+'\n')
    f.write('Point(3) = {'+str(x)+', '+str(y)+', 0, 0.1};   // thickness'+'\n')
    f.write('//+ ------------------------------------------------- '+'\n')     
    f.write(''+'\n')
    f.write('//// POINTS DEFINING THE BOUNDARY LAYER AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Point(5) = {0, 0.2, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(6) = {0, -0.1, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(7) = {1, -0.1, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(8) = {1, 0.2, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(9) = {'+str(x)+', '+str(y)+'+0.2, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(19) = {-0.2, 0.2, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(20) = {-0.2, -0.1, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(21) = {-0.2, 0, 0, 0.1};'+'\n')
    f.write(''+'\n')
    f.write('//// POINTS DEFINING THE SECOND BOUNDARY LAYER AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Point(22) = {'+str(x)+', '+str(y)+'+0.45, 0, 0.1};   // thickness dependant'+'\n')
    f.write('//+'+'\n')
    f.write('Point(23) = {1, 0.6, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(24) = {6, 0.6, 0, 1};'+'\n')
    f.write(''+'\n')
    f.write('//// LINES DEFINING THE TRIANGULAR AIRFOIL'+'\n')
    f.write('//+'+'\n')
    f.write('Line(1) = {1, 2};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(2) = {2, 3};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(3) = {3, 1};'+'\n')
    f.write('//// LINES DEFINING THE BOUNDARY LAYER AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Line(4) = {1, 5};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(5) = {5, 9};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(6) = {3, 9};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(7) = {9, 8};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(8) = {2, 8};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(9) = {2, 7};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(10) = {7, 6};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(11) = {1, 6};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(26) = {6, 20};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(27) = {20, 21};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(28) = {1, 21};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(29) = {19, 21};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(30) = {19, 5};'+'\n')
    f.write(''+'\n')
    f.write('//// LINES DEFINING THE SECOND BOUNDARY LAYER AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Line(31) = {9, 22};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(32) = {22, 23};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(33) = {8, 23};'+'\n')
    f.write(''+'\n')
    f.write('//// POINTS DEFINING THE WAKE AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Point(10) = {6, 0, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(11) = {6, 0.2, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(12) = {6, -0.1, 0, 1};'+'\n')
    f.write(''+'\n')
    f.write('//// POINTS DEFINING THE OUTER DOMAIN '+'\n')
    f.write('//+'+'\n')
    f.write('Point(13) = {21, 0, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(14) = {21, 10, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(15) = {21, -10, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(16) = {0, -10, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(17) = {0, 10, 0, 1};'+'\n')
    f.write('//+'+'\n')
    f.write('Point(18) = {-10, 0, 0, 1};'+'\n')
    f.write(''+'\n')
    f.write('//// ROTATE AIRFOIL ANGLE OF ATTACK 6 deg'+'\n')
    f.write(''+'\n')       
    f.write('//+           z     DX'+'\n')
    f.write('Rotate {{0, 0, 1}, {1, 0, 0}, -Pi/'+str(180/AoA)+'} {'+'\n')
    f.write('   Line{1}; Line{2}; Line{3}; Line{4}; Line{5}; Line{6}; Line{7}; Line{8}; Line{9}; Line{10}; Line{11}; Line{26}; Line{27}; Line{28}; Line{29}; Line{30}; Line{31}; Line{32}; Line{33}; }'+'\n')
    f.write(''+'\n')
    f.write('//// LINES DEFINING THE WAKE AREA'+'\n')
    f.write('//+'+'\n')
    f.write('Line(13) = {30, 11};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(14) = {10, 11};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(15) = {26, 10};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(16) = {31, 12};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(17) = {10, 12};'+'\n')
    f.write(''+'\n')
    f.write('Line(34) = {37, 24};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(35) = {11, 24};'+'\n')
    f.write(''+'\n')
    f.write('//// LINES AND CIRCLES DEFINING THE OUTER DOMAIN '+'\n')
    f.write('//+'+'\n')
    f.write('Point(1) = {0, 0, 0, 0.1};'+'\n')
    f.write('//+'+'\n')
    f.write('Circle(20) = {17, 1, 18};'+'\n')
    f.write('//+'+'\n')
    f.write('Circle(21) = {18, 1, 16};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(22) = {17, 14};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(23) = {13, 14};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(24) = {13, 15};'+'\n')
    f.write('//+'+'\n')
    f.write('Line(25) = {15, 16};'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('//// LINE LOOPS + SURFACES FOR MESHING'+'\n')
    f.write(''+'\n')
    f.write('// Boundary layer'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(1) = {28, -27, -26, -11};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(1) = {1};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(2) = {6, -5, -4, -3};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(2) = {2};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(3) = {8, -7, -6, -2};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(3) = {3};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(4) = {29, -28, 4, -30};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(4) = {4};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(5) = {11, -10, -9, -1};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(5) = {5};'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('// Wake '+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(6) = {14, -13, -8, 15};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(6) = {6};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(7) = {9, 16, 17, 15};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(7) = {7};'+'\n')
    f.write('//+'+'\n')
    f.write(''+'\n')
    f.write('// Outer domain'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(9) = {20, 21, -25, -24, -23, -22};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(10) = {29, -27, -26, -10, 16, 17, -14, 35, -34, -32, -31, -5, -30};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(8) = {9, 10};'+'\n')
    f.write(''+'\n')
    f.write('// Wake + outer BL'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(11) = {33, -32, -31, 7};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(9) = {11};'+'\n')
    f.write('//+'+'\n')
    f.write('Line Loop(12) = {35, -34, -33, 13};'+'\n')
    f.write('//+'+'\n')
    f.write('Plane Surface(10) = {12};'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    if optimisation == "2D":
        f.write('////////////////////// BOUNDARY CONDITIONS'+'\n')
        f.write(''+'\n')
        f.write('//+'+'\n')
        f.write('Physical Line("inlet") = {20, 21};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Line("outlet") = {23, 24};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Line("horizontal") = {22, 25};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Line("airfoil") = {3, 2, 1};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("fluid") = {1, 2, 3, 6, 7, 5, 4, 8, 9, 10};'+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
    f.write('////////////////// MESH'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('///// BOUNDARY LAYER TRANSFINITE LINE MESH'+'\n')
    f.write(''+'\n')
    f.write('///// Horizontal lines'+'\n')
    f.write('// Up-front'+'\n')
    f.write('//+//+ ------------------------------------------------- '+'\n')
    f.write('Transfinite Line {3} = '+str(n1)+' Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {5} = '+str(n1)+' Using Progression 1;'+'\n')
    f.write('// Up-rear'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {2} = '+str(n2)+' Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {7} = '+str(n2)+' Using Progression 1;'+'\n')
    f.write('//+//+ ------------------------------------------------- '+'\n')
    f.write('// Upper layer'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {32} = '+str(n2)+' Using Progression 1;'+'\n')
    f.write(''+'\n')
    f.write('// Down'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {1} = 30 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {10} = 30 Using Progression 1;'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('//// Vertical lines'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {6} = 7 Using Progression 1.05;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {4} = 7 Using Progression 1.05;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {12} = 7 Using Progression 1.1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {11} = 4 Using Progression 1.1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {9} = 4 Using Progression 1.1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {8} = 7 Using Progression 1.05;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {31} = 10 Using Progression 1.01;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {33} = 10 Using Progression 1.01;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {35} = 10 Using Progression 1.05;'+'\n')
    f.write(''+'\n')
    f.write('//// Front lines'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {28} = 7 Using Progression 1.01;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {30} = 7 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {29} = 7 Using Progression 0.95;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {27} = 4 Using Progression 0.95;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {26} = 7 Using Progression 1;'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('///// WAKE TRANSFINITE LINE MESH'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {13} = 70 Using Progression 1.005;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {15} = 70 Using Progression 1.005;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {16} = 70 Using Progression 1.005;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {34} = 70 Using Progression 1.005;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {14} = 7 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {17} = 4 Using Progression 1;'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('///// OUTER DOMAIN TRANSFINITE LINE MESH'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {20} = 12 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {21} = 12 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {22} = 12 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {25} = 12 Using Progression 1;'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Line {24, 23} = 15 Using Progression 1.2;'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('//// TRANSFINITE AND RECOMBINE SURFACES'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {4} = {35, 28, 25, 34};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {2} = {28, 29, 25, 27};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {3} = {29, 30, 26, 27};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {1} = {34, 25, 32, 33};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {5} = {25, 31, 26, 32};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {6} = {30, 11, 10, 26};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {7} = {26, 10, 12, 31};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {9} = {36, 37, 30, 29};'+'\n')
    f.write('//+'+'\n')
    f.write('Transfinite Surface {10} = {37, 24, 11, 30};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {4};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {2};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {3};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {1};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {5};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {7};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {6};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {8};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {9};'+'\n')
    f.write('//+'+'\n')
    f.write('Recombine Surface {10};'+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write(''+'\n')
    f.write('//// DEFINE SIZE FIELDS'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1] = Box;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].VIn = 0.2;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].VOut = 10;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].XMax = 9;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].XMin = -2;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].YMax = 2;'+'\n')
    f.write('//+'+'\n')
    f.write('Field[1].YMin = -2;'+'\n')
    if optimisation == "3D":
        f.write('//+'+'\n')
        f.write('Field[1].ZMax = 2;'+'\n')
        f.write('//+'+'\n')
        f.write('Field[1].ZMin = -2;'+'\n')
    f.write('//+'+'\n')
    f.write('Background Field = 1;'+'\n')
    f.write(''+'\n')
    if optimisation == "3D":
        f.write('NewEntities[]= Extrude{0,0,0.6}'+'\n')
        f.write('{'+'\n')
        f.write('	Surface {1};Surface {2};Surface {3};Surface {4};Surface {5};Surface {6};Surface {7};Surface {9};Surface {10}; Surface {8}; Layers{18}; Recombine;'+'\n')
        f.write('};'+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
        f.write(''+'\n')
        f.write('/////////////////////////////////////////////////////////////////// BOUNDARY CONDITIONS'+'\n')
        f.write(''+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("inlet") = {NewEntities[56],NewEntities[57]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("outlet") = {NewEntities[59],NewEntities[60]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("horizontal") = {NewEntities[58],NewEntities[61]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("airfoil") = {NewEntities[11],NewEntities[17],NewEntities[29]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Volume("fluid") = {NewEntities[1],NewEntities[7],NewEntities[13],NewEntities[19],NewEntities[25],NewEntities[31],NewEntities[37],NewEntities[43],NewEntities[49],NewEntities[55]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-0-r") = {1};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-1-r") = {2};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-2-r") = {3};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-3-r") = {4};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-4-r") = {5};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-5-r") = {6};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-6-r") = {7};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-7-r") = {9};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-8-r") = {10};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-9-r") = {8};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-0-l") = {NewEntities[0]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-1-l") = {NewEntities[6]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-2-l") = {NewEntities[12]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-3-l") = {NewEntities[18]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-4-l") = {NewEntities[24]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-5-l") = {NewEntities[30]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-6-l") = {NewEntities[36]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-7-l") = {NewEntities[42]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-8-l") = {NewEntities[48]};'+'\n')
        f.write('//+'+'\n')
        f.write('Physical Surface("periodic-9-l") = {NewEntities[54]};'+'\n')
        f.write(''+'\n')    
    f.close()            
    
