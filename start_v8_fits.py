# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 14:06:17 2021

@author: Alex

This file builds Fortran 90 programs that complete nested sampling fits to find the masses of dwarf galaxies.

Important variables are:
name=name of galaxy
model=mass profile to test
tag=run name
has_feh=data has FeH metallicity measurements. there are a different number of parameters for runs with no FeH which require different F90 files
core_str=list of supercomputer cores to run on.
"""



import os
import numpy as np
import sys
import sqlite3


#the information for each galaxy is stored in an sql database, these functions retreive information from it given a galaxy name or number key
def grab_name(name,key):
    conn=create_connection('altgrav.db')
    cur=conn.cursor()
    data=cur.execute("select "+key+" FROM info WHERE shortname='"+name+"'")
    result=(data.fetchall()[0][0])
#    data = sqlutil.get("select "+key+"FROM info WHERE shortname="+name+";",db='altgrav.db')
    return result
def grab_num(number,key):
    conn=create_connection('altgrav.db')
    cur=conn.cursor()
    data=cur.execute("select "+key+" FROM info WHERE number='"+str(number)+"'")
    result=(data.fetchall()[0][0])
#    data = sqlutil.get("select "+key+"FROM info WHERE shortname="+name+";",db='altgrav.db')
    return result
def create_connection(db_file):
    conn = None
    conn = sqlite3.connect(db_file) 
    return conn


def make_build_files(names,models,tag):# this function builds the fortran program depend
    for name in names:
        for model in models:
            if len(tag)>0:
                tag='-'+tag
            folder_base="/verafs/scratch/phy200028p/amoskowi/altgrav_posts/"
            gal_num=grab_name(name,'number')
            cat_name='altgrav_final_cats/'+str(grab_name(name,'spec_cat_filename'))
            has_feh=grab_name(name,'has_feh')
            if has_feh=='Yes':
                like_1=open('example_'+model+'_2vg/'+model+'_2vg_like_1.txt','r').read()
                like_2=open('example_'+model+'_2vg/'+model+'_2vg_like_2.txt','r').read()
                like_text=like_1+str(gal_num)+"\ncall run_info(gal_num,rfield,nstars,ML_conversion)\nOPEN(UNIT=1,FILE= &\n'"+cat_name+like_2
                fL=open('example_'+model+'_2vg/like.f90','w')
                fL.write(like_text)
                fL.close()
        
                mean_vlos=float(grab_num(gal_num,'mean_vlos'))
                upper_b1=str(np.log10(float(grab_num(gal_num,'radius_pc'))/2.0))#str(np.log10(10.0*float(grab_num(gal_num,'rh_pc'))))
                vlos_up=str(mean_vlos+30.0)
                vlos_down=str(mean_vlos-30.0)
                main_1=open('example_'+model+'_2vg/'+model+'_2vg_main_1.txt','r').read()
                main_2=open('example_'+model+'_2vg/'+model+'_2vg_main_2.txt','r').read()
                main_3=open('example_'+model+'_2vg/'+model+'_2vg_main_3.txt','r').read()
                main_text=main_1+vlos_down+'\nspriorran(7:sdim,2)='+vlos_up+main_2+upper_b1+main_3
                fM=open('example_'+model+'_2vg/main.f90','w')
                fM.write(main_text)
                fM.close()
        
                params_1=open('example_'+model+'_2vg/'+model+'_2vg_params_1.txt','r').read()
                params_2=open('example_'+model+'_2vg/'+model+'_2vg_params_2.txt','r').read()
                params_text=params_1+folder_base+model+'/v8/'+name+'-'+model+'-v8-'+tag+params_2
        
                fP=open('example_'+model+'_2vg/params.f90','w')
                fP.write(params_text)
                fP.close()
        
            if has_feh=='No':
        
                like_1=open('example_'+model+'_2vg/'+model+'_2vg_like_1_nofeh.txt','r').read()
                like_2=open('example_'+model+'_2vg/'+model+'_2vg_like_2_nofeh.txt','r').read()
                like_text=like_1+str(gal_num)+"\ncall run_info(gal_num,rfield,nstars,ML_conversion)\nOPEN(UNIT=1,FILE= &\n'"+cat_name+like_2
                fL=open('example_'+model+'_2vg/like.f90','w')
                fL.write(like_text)
                fL.close()
        
        
        
                mean_vlos=float(grab_num(gal_num,'mean_vlos'))
                upper_b1=str(np.log10(float(grab_num(gal_num,'radius_pc'))/2.0))#str(np.log10(10.0*float(grab_num(gal_num,'rh_pc'))))
                vlos_up=str(mean_vlos+30.0)
                vlos_down=str(mean_vlos-30.0)
                main_1=open('example_'+model+'_2vg/'+model+'_2vg_main_1_nofeh.txt','r').read()
                main_2=open('example_'+model+'_2vg/'+model+'_2vg_main_2_nofeh.txt','r').read()
                main_3=open('example_'+model+'_2vg/'+model+'_2vg_main_3_nofeh.txt','r').read()
                main_text=main_1+vlos_down+'\nspriorran(3:sdim,2)='+vlos_up+main_2+upper_b1+main_3
                fM=open('example_'+model+'_2vg/main.f90','w')
                fM.write(main_text)
                fM.close()
        
                params_1=open('example_'+model+'_2vg/'+model+'_2vg_params_1_nofeh.txt','r').read()
                params_2=open('example_'+model+'_2vg/'+model+'_2vg_params_2_nofeh.txt','r').read()
                params_text=params_1+folder_base+model+'/v8/'+name+'-'+model+'-v8-nf-'+tag+params_2
        
                fP=open('example_'+model+'_2vg/params.f90','w')
                fP.write(params_text)
                fP.close()
            print('\nbuilt '+name+' '+model)
        
        
def make_and_run_fit(names,models,core_list,ncores_per_fit,tag):
    #takes the input parameters and list of cores and builds the Fortran 90 files and the shell script to run the fortran 90 code in parallel
    all_cores=['space','r001','r002','r003','r004','r005','r006','r007','r008','r009','r010','r011','r012','r013','r014','r015','r016','r017','r018','r019','r020','r021','r022','r023','r024','r025','r026','r027','r028','r029','r030','r031','r032']#used to convert core list integers to core names
    fnames=''
    for name in names:
        for model in models:
            make_build_files([name],[model],tag)
            if core_list[0]=='default':
                nline=''
            else:
                core_text=''
                use_cores_list=core_list[0:ncores_per_fit]
                core_list=core_list[ncores_per_fit:]
                for core in use_cores_list:
                    core_text+=all_cores[core]+','
                core_text=core_text[0:len(core_text)-1]

                nline='#SBATCH --nodelist='+core_text+'\n'

            print('\n making '+str(name)+'   '+str(model)+' on '+core_text+'\n\n')
#these lines used to work to compile and run the fortran code, but they don't work on the department's new system
#mcompile            os.system('make clean_'+model+'_2vg')
#mcompile            os.system('make '+model+'_2vg'  )
            exename=model+'_'+name
#mcompile            os.system('cp eg_2vg '+exename)
            base='#!/bin/bash -l\n#SBATCH --nodes='+str(ncores_per_fit)+'\n#SBATCH --mem=120000\n#SBATCH  --tasks-per-node=16\n#SBATCH --time=152:00:00\n#SBATCH --job-name='+name+'_'+str(model)+'\n' 
            oline='#SBATCH --output='+name+'_'+model+'_out.txt\n'
            command='module load mpi/gcc_openmpi\nmpiexec -n '+str(int(16*int(ncores_per_fit)))+' ./'+exename
            stext=base+oline+nline+command
            script_name=name+'_'+model+tag+'_script.sh'
            fS=open(script_name,'w')
            fS.write(stext)
            fS.close()
            fnames+=exename+'   '+script_name+'\n'
#mcompile            os.system('sbatch '+script_name)
#            print('\nsubmitted '+name+' '+model+'\n')
            print('make clean_'+model+'_2vg')
            print('make '+model+'_2vg')
            print('cp '+model+'_2vg '+exename)            
            print('sbatch '+script_name)
            print(' ')

#the department transitioned to a new supercomputer and the transition...did not go smoothly. this code prints the commands instead of running them itself due to issues running parallel code from inside python environments



#read in parameters from command line
name=str(sys.argv[1])
model=str(sys.argv[2])
core_str=str(sys.argv[3])
tag=str(sys.argv[4])

if tag=='-':
    tag=''

#covert core list integers to names of cores
core_list=[]
core_str_list=core_str.split(',')
for c in core_str_list:
    core_list.append(int(c))
num_cores=len(core_list)

#for each run, build the galaxy


make_and_run_fit([name],[model],core_list,num_cores,tag)
