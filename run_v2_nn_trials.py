# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:24:30 2021

@author: Alex

This code imports an architecture for a neural network, then trains and runs predictions for it several times in paralell
the network takes in simulated measurements of the velocity of a star over time and classifies the star either as a binary system of stars or a solo star
some useful varibale explanations"
error_str=the maximum uncerainty on the error measurements
timetext=the label for these sets
n_epoch=the number of measurements per star

the NN takes in data of (v1,v2,...vn,e1,e2,...,en,t1,t2,...,tn) for measured velocities vn, systematic errors en and observation times tn. It returns a number between 0 and 1. Data is labelled 0 for a solo star and 1 for a binary.
Simulated data has been pre-normalized so all values are between 0 and 1.
"""



from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import time
import os
import sys
import multiprocessing as mp


def build_model(train_sample_size,train_sample_array_length,neurons_per_layer,learning_rate,dropout_rate,num_layers):#creates the neural network object given the hyperparameters passed to it
#    global 
    
    model_list=[keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu, input_shape=(train_sample_array_length,))]    
    i=0
    while i<num_layers:
        model_list.append(keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu))
        model_list.append(keras.layers.BatchNormalization())
        model_list.append(keras.layers.Dropout(dropout_rate))
        i+=1
    model_list.append(keras.layers.Dense(1))
    
#    model = keras.Sequential([keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu, input_shape=(train_sample_array_length,)),
#                        keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu),keras.layers.BatchNormalization(),keras.layers.Dropout(dropout_rate),
#                        keras.layers.Dense(neurons_per_layer, activation=tf.nn.relu),keras.layers.BatchNormalization(),keras.layers.Dropout(dropout_rate),
  



#                            keras.layers.Dense(1)
#                              ])
    print(len(model_list))
    model = keras.Sequential(model_list)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
            
    model.compile(loss='mean_squared_error',
                          optimizer=optimizer,
                          metrics=['mean_absolute_error', 'mean_squared_error','accuracy'])
    return model


def in_list(num,numlist):# see if an item is in a list because I forgot python has a good natural way to do this
    is_in_list=False
    for entry in numlist:
        if entry==num:
            is_in_list=True
    return is_in_list


def make_data_lists(num_for_training):#decides which blocks of training data should be used so that there is no overlap between training and test data
    training_list=[]
    validation_num=random.randint(0,19)
    while len(training_list)<num_for_training:
        sample_num=random.randint(0,19)
        if in_list(sample_num,[validation_num])==False and in_list(sample_num,training_list)==False:
            training_list.append(sample_num)
    return [validation_num,training_list]

def make_training_data(nums,n_epoch):#imports the data and splits it into data and label lists
    global error_str,timetext
    train_data_raw=[]
    train_keys_raw=[]
    for num in nums:
        raw_data=np.array(np.genfromtxt('blocks/v2/v2_block_'+str(num)+'_'+error_str+'_norm_'+str(n_epoch)+''+timetext+'.dat',dtype='float'))
        
        for line in raw_data:
            i=0
            temp_data=[]
            while i<len(line):
                if line[i]!=0.0 and line[i]!=1.0:
                    temp_data.append(line[i])
                i+=1
            train_data_raw.append(temp_data)     
            train_keys_raw.append(line[-1])
    train_data=np.array(train_data_raw)
    train_keys=np.array(train_keys_raw)
    print(train_data[-1])
    return [train_data,train_keys]

def make_predict_data(num,lines,n_epoch):#imports the prediction data
    global error_str
    train_data_raw=[]
    train_keys_raw=[]
    raw_data=np.array(np.genfromtxt('blocks/v2/v2_block_'+str(num)+'_'+error_str+'_norm_'+str(n_epoch)+''+timetext+'.dat',dtype='float'))
    
    for line in lines:
        i=0
        temp_data=[]
        while i<len(raw_data[line]):
            if raw_data[line][i]!=0.0 and raw_data[line][i]!=1.0:
                temp_data.append(raw_data[line][i])
            i+=1
        train_data_raw.append(temp_data)
        train_keys_raw.append(raw_data[line][-1])
    train_data=np.array(train_data_raw)
    train_keys=np.array(train_keys_raw)
    return [train_data,train_keys]

def write_info(num,text):#handles writing to files
    global n_epoch,error_str
    f=open('nn_text/'+str(n_epoch)+'_'+error_str+'_'+str(num)+'_status.txt','w')
    f.write(text)
    f.close()


def run_trial(grid_num):#encapsulates the training and prediction phases for each trial. grid_num is the trial of numbers. multiple trials are run to build up uncertainty estimates
    global error_str,timetext,n_epoch
    fname='bml_out/v2_hp_searches/use'+'_n'+str(n_epoch)+'_params_'+error_str+'.txt'
    #a grid search was used to select hyperparameters, these lines read in the hyperparameters from a file
    if not os.path.isfile('bml_out/v2_nn/'+str(grid_num)+'_n'+str(n_epoch)+'_params_'+error_str+'.txt'):
        raw=np.array(np.genfromtxt(fname,dtype='float'))
        orig_grid,num_training_epochsf,num_training_setsf,neurons_per_layerf,learning_rate,dropout_rate,num_layersf,loss_val,val_acc,decision_acc,num_training_epochsf,num_training_setsf,total_time=raw
        num_training_epochs=int(num_training_epochsf)
        num_training_sets=int(num_training_setsf)
        neurons_per_layer=int(neurons_per_layerf)
        num_training_epochs=int(num_training_epochsf)
        num_training_sets=int(num_training_setsf)
        num_layers=int(num_layersf)
        st='start '+str(grid_num)+'\n'#status text
        ts=time.time()#time start
        write_info(grid_num,st)
        outtext=str(grid_num)+'   '#the string that contains the output
 
        
       #OTHER RUN PARAMETERS

        outtext+=str(num_training_epochs)+'   '+str(num_training_sets)+'   '
            
            min_model=False#for testing
        if min_model==True:
            neurons_per_layer=8
            learning_rate=random.uniform(.01,.5)
            dropout_rate=0.2
            num_layers=2
            num_training_epochs=4
            num_training_sets=2
        
        
        outtext+=str(neurons_per_layer)+'   '+str(learning_rate)+'   '+str(dropout_rate)+'   '+str(num_layers)+'   '
        
#        start_time=time.time()
        
        
     
#        print(outtext)
#setup data
        validation_num,training_nums=make_data_lists(num_training_sets)

        train_data,train_keys=make_training_data(training_nums,n_epoch)#np.linspace(0,98,98,dtype=np.int16)
        st+='training data imported '+str(time.time()-ts)+'\n'
        write_info(grid_num,st)
        train_sample_size=len(train_data)
        train_sample_array_length=(3*n_epoch)-2#not a changeable paramater, buddy


#build and fit model
        model=build_model(train_sample_size,train_sample_array_length,neurons_per_layer,learning_rate,dropout_rate,num_layers)
        st+='model built '+str(time.time()-ts)+outtext+'\n'
        write_info(grid_num,st)
        #print(tf.__version__)
        
        one_e_time=time.time()
        model.fit(train_data, train_keys, epochs=num_training_epochs-1, batch_size=500)
        st+='1 epoch fit in '+str(time.time()-one_e_time)+'\n'
        write_info(grid_num,st)
        model.fit(train_data, train_keys, epochs=num_training_epochs-1, batch_size=500)
        st+='model fit '+str(time.time()-ts)+'\n'
        write_info(grid_num,st)
        
        #validate nn
        val_data,val_keys=make_training_data([validation_num],n_epoch)
        loss_val,mean_err,mean_sq_err,val_acc=model.evaluate(val_data, val_keys)
        loss_label,mean_err_label,mean_sq_err_label,acc_label=model.metrics_names
        st+='model validated '+str(time.time()-ts)+'\n'
        write_info(grid_num,st)

        print(loss_val)
        print(loss_label)
        
        print(val_acc)
        print(acc_label)
        #run the nn on simulated data to see how accurate the classifier is
        
        predict_data,predict_result=make_predict_data(validation_num,np.linspace(0,999,999,dtype=np.int16),n_epoch)
        predictions = model.predict(predict_data)
        st+='model run on test data '+str(time.time()-ts)+'\n'
        write_info(grid_num,st)

        p=0
        right=0
        wrong=0
        while p<len(predict_result):
            prediction=float(predictions[p][0])
            answer=float(predict_result[p])
            result_text=''
            if answer<0.5 and prediction<0.5:
                result_text='right'
                right+=1
            if answer>0.5 and prediction<0.5:
                result_text='wrong'
                wrong+=1
        
            if answer>0.5 and prediction>0.5:
                result_text='right'
                right+=1
            if answer<0.5 and prediction>0.5:
                result_text='wrong'
                wrong+=1
                
        #    print(str(predict_result[p])+'   '+str(predictions[p][0])+'   '+result_text)
            p+=1
        start_time=ts
        decision_acc=float(right)/float(right+wrong)
        end_time=time.time()
        print('accuracy='+str(decision_acc))
        total_time=(end_time-start_time)/60.0
        print('done in '+str(total_time))
        outtext+=str(loss_val)+'   '+str(val_acc)+'   '+str(decision_acc)+'   '+str(num_training_epochs)+'   '+str(num_training_sets)+'   '+str(total_time)
        
        out=open('bml_out/v2_nn/'+str(grid_num)+'_n'+str(n_epoch)+'_params_'+error_str+'.txt','w')
        out.write(outtext)
        out.close()
#        grid_num+=1
        print('\n\n\n'+str(grid_num)+'\n\n\n')
        st+='done '+str(time.time()-ts)+'\n'
        write_info(grid_num,st)




#set up the run

timetext='_5_years'
nlist=[int(sys.argv[1])]
error_str=str(sys.argv[2])

start=0
start_time=time.time()
#run each n_epoch in paralell
for n_epoch in nlist:
    if __name__ == '__main__':
            indices=range(11)
            n_proc = 16
    #    #    
            out = mp.Pool(n_proc).map(run_trial, indices)
    run_trial(start)
end_time=time.time()
total_time=end_time-start_time
ft=open('bml_out/v2_nn/'+nstring+'_'+error_str+'_total_time.txt','w')
ft.write(str(total_time))
ft.close()
