
# coding: utf-8

from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Activation, Dropout 
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.optimizers import SGD, adadelta, Adam
from keras.regularizers import l1_l2
from keras import backend as Kback
from keras.utils import multi_gpu_model

import tensorflow as tf
import time

import numpy as np

import pickle, os
import numpy as np
import random, math
import pandas as pd
import entropytriangle as et

import multiprocessing


from warnings import warn as warning
import sys, getopt

import gc

class Discretizer:

    nbins  = 0
    labels = []
    bins   = None

    def __init__(self,nbins):
        self.nbins = nbins
        self.labels  = range(nbins)
        low     = np.arange(0,1,1/nbins)
        high    = low+1/nbins
        low[0]  = -0.0001
        self.bins    = pd.IntervalIndex.from_arrays(low,high)    




    def indexed_discretization (self,df):

        """
        Function created for the discretization of a dataframe in #nbins equalwidth bins
        > comprobation = discretization(df,nbins)
        Parameters
        ----------
        df : raw DataFrame 
        nbins : number of bins 
        Returns
        ----------
        df : Discretized Dataframe
        """

        if(not isinstance(df,pd.DataFrame)):
            exit("Can only work with Data Frames!")
        #' Rows or columns equal to zero?
        if (len(df.columns) == 0 or len(df.index) == 0): 
            exit("Can only work with non-empty Data Frames!")
        
        indexes=df['idx']
        df = df.drop('idx',axis=1)
        df.update(df.apply(lambda x: pd.Categorical(pd.cut(x, bins = self.bins)).rename_categories(self.labels)))
        disc=df
        disc['idx']=indexes
        return disc


class ParallelExecutor():
    

    cores   = 1


    def __init__(self,cores=0):
        if (cores == 0):
            self.cores = multiprocessing.cpu_count()
        else:
            self.cores=cores


    def parallelize(self, data, func):
        partitions=self.cores
        data_split = np.array_split(data, partitions)
        pool = multiprocessing.Pool(self.cores)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data 

class ParallelDiscretizer(Discretizer):
    pe = None

    def __init__(self,nbins,ncores):
        super().__init__(nbins)
        self.pe=ParallelExecutor(cores=ncores)

    def parallelize_discretization(self,df):
        indexed_df=df
        indexed_df['idx'] = pd.Series(df.index)
        data = self.pe.parallelize(indexed_df,self.indexed_discretization)
        data = data.drop('idx',axis=1)
        final=data.astype('category')
        return final


def jentropies_aux(data):
    params=data[0]
    return et.jentropies(params[0],params[1])

def training(K,basedir,n_epochs=10,batch_length=250,numbins=32,database='MNIST',start_epoch=0,num_gpus=1):

    # tracemalloc.start()

    dim_layers=[784,1000,500,250,K]
    
    # the data, split between train and test sets
    if ( database== 'MNIST'):
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        num_classes=10
    elif ( database=='FMNIST'):
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        num_classes=10
    else:
        print("No Implementado.\n")
        print("Bases de datos validas: MNIST y FMNIST\n")
        exit(-1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    num_train_samples=x_train.shape[0]
    # num_test_samples=x_test.shape[0]
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #Configuracion de TensorFlow para el uso de dispositivos

    config = tf.ConfigProto( device_count = {'GPU': num_gpus, 'CPU': 20} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    model_compiled=False
    #Esto instancia el modelo original en la CPU
    if num_gpus>1:
        BASE_DEVICE='/cpu:0'
    elif num_gpus==1:
        BASE_DEVICE='/gpu:0'
    else:
        BASE_DEVICE='/cpu:0'
    with tf.device(BASE_DEVICE):
        if ( start_epoch > 0 ):
            try:
                ModelFile="{basedir}/model_{database}_entropy_K_{K}_iteracion_{epoch}.h5".format(
                        basedir=basedir,
                        K=K,
                        epoch=start_epoch-1,
                        database=database
                        )
                DataFile="{basedir}/{database}_entropy_K_{K}_iteracion_{epoch}.h5".format(
                        basedir=basedir,
                        K=K,
                        epoch=start_epoch-1,
                        database=database
                        )
                if ( os.path.isfile(ModelFile) and os.path.isfile(DataFile) ):
                    try:
                        model = load_model(ModelFile)
                        final_data= pd.read_parquet(DataFile)
                        buildModel=False
                        model_compiled=True
                    except:
                        buildModel = True
            except:
                buildModel=True
        else:
            buildModel=True

        # Encoder Definition
        if ( buildModel ):
            
            columns_df=['Type','H_U','H_P','DeltaH_P','M_P','VI_P','idx','epoch','batch','Batch Time','loss','input','output']
            final_data=pd.DataFrame(columns=columns_df,dtype="object").set_index('idx')
            start_epoch=0

            input_image=Input(shape=(dim_layers[0],))

            enc_layer1=Dense(dim_layers[1],activation='sigmoid')(input_image)
            enc_layer2=Dense(dim_layers[2],activation='sigmoid')(enc_layer1)
            enc_layer3=Dense(dim_layers[3],activation='sigmoid')(enc_layer2)
            enc_layer4=Dense(dim_layers[4],activation='sigmoid')(enc_layer3)


            # Decoder definition


            dec_layer3=Dense(dim_layers[3],activation='sigmoid')(enc_layer4)
            dec_layer2=Dense(dim_layers[2],activation='sigmoid')(dec_layer3)
            dec_layer1=Dense(dim_layers[1],activation='sigmoid')(dec_layer2)
            output=Dense(dim_layers[0],activation='sigmoid')(dec_layer1)

            # Model compilation

            model = Model(input_image,output)


        intermediate_layer_model_1 = Model(inputs=model.input, 
                                outputs=model.layers[1].output)
        intermediate_layer_model_2 = Model(inputs=model.input, 
                                outputs=model.layers[2].output)
        intermediate_layer_model_3 = Model(inputs=model.input, 
                                outputs=model.layers[3].output)
        intermediate_layer_model_Z = Model(inputs=model.input, 
                                outputs=model.layers[4].output)
        intermediate_layer_model_3p = Model(inputs=model.input, 
                                outputs=model.layers[5].output)
        intermediate_layer_model_2p = Model(inputs=model.input, 
                                outputs=model.layers[6].output)
        intermediate_layer_model_1p = Model(inputs=model.input, 
                                outputs=model.layers[7].output)
        
    if (num_gpus>1):
        print ('MutiGPU implementation')
        execution_model = multi_gpu_model(model,gpus=num_gpus)
    else:
        execution_model = model
    print ('Device: {0}'.format(BASE_DEVICE))
    if not model_compiled:    
        execution_model.compile(optimizer='adam',loss='binary_crossentropy')


    execution_model.summary()
        
    
    final_data.head()
    n_batches= int(math.ceil(num_train_samples/batch_length))

    #####
    # Para pruebas n_batches=3
    # n_batches = 3

    # Vectores empleados para almacenar los resultados: accuracy y loss.    

    start_sample=start_epoch

    discretizer = ParallelDiscretizer(numbins,10)
    pe = ParallelExecutor(cores=5)
    print("Discretizando todas las entradas\n")
    start_time = time.time()
    x_train_discrete=discretizer.parallelize_discretization(pd.DataFrame(x_train))
    elapsed_time = time.time() - start_time
    print("Tiempo de discretizacion: {0}\n".format(elapsed_time))
    if (not(all(x_train_discrete.dtypes=='category'))):
        warning("Discretizing data from X DataFrame before entropy calculation!")

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)

    # BUCLE ENCARGADO DE GESTIONAR LOS EPOCHS.
    for ind_epoch in range(start_sample, n_epochs): 
        # Aleatorizamos la lista de datos de Train. 
        x_train_idx=np.arange(num_train_samples)
        random.shuffle(x_train_idx)

        # En estas dos variables se almacenarán los scores de los batches ejecutados    
        # train_loss_batch = []
        # BUCLE ENCARGADO DE GESTIONAR LOS BATCHES.
        for ind_batch in range(0, int(n_batches)):
        # for ind_batch in range(0, 3):
            # print ("Batch: {0}\n".format(ind_batch))
            batch_start_time=time.time()
            base_batch_idx=ind_batch*batch_length
            end_batch_idx=np.minimum(base_batch_idx+batch_length,num_train_samples)
            batch_idx=x_train_idx[base_batch_idx:end_batch_idx]
            X_train=x_train[batch_idx,:]
            X_discrete= (x_train_discrete.iloc[batch_idx]).reset_index(drop=True)
            # print(X_train.shape)
            
            start_time=time.time()
            # Aquí se realiza el entrenamiento del modelo batch a batch.
            loss= execution_model.train_on_batch(X_train, X_train)
                                   
            # intermediate_layer_model_1 = Model(inputs=model.input, 
            #                         outputs=model.layers[1].output)
            # intermediate_layer_model_2 = Model(inputs=model.input, 
            #                         outputs=model.layers[2].output)
            # intermediate_layer_model_3 = Model(inputs=model.input, 
            #                         outputs=model.layers[3].output)
            # intermediate_layer_model_Z = Model(inputs=model.input, 
            #                         outputs=model.layers[4].output)
            # intermediate_layer_model_3p = Model(inputs=model.input, 
            #                         outputs=model.layers[5].output)
            # intermediate_layer_model_2p = Model(inputs=model.input, 
            #                         outputs=model.layers[6].output)
            # intermediate_layer_model_1p = Model(inputs=model.input, 
            #                         outputs=model.layers[7].output)
  
            start_time = time.time()
            # Aquí se obtiene la salida de dicha capa intermedia.
            T_out_1 = intermediate_layer_model_1.predict_on_batch(X_train)
            T_out_2 = intermediate_layer_model_2.predict_on_batch(X_train)
            T_out_3 = intermediate_layer_model_3.predict_on_batch(X_train)
            T_out_Z = intermediate_layer_model_Z.predict_on_batch(X_train)
            T_out_3p = intermediate_layer_model_3p.predict_on_batch(X_train)
            T_out_2p = intermediate_layer_model_2p.predict_on_batch(X_train)
            T_out_1p = intermediate_layer_model_1p.predict_on_batch(X_train)
            T_out_0p = execution_model.predict_on_batch(X_train)
            elapsed_time = time.time() - start_time
            #print ("Tiempo de prediccion: {0}".format(elapsed_time))

            start_time = time.time()
            DF_T_out_1=discretizer.parallelize_discretization(pd.DataFrame(T_out_1))
            DF_T_out_2=discretizer.parallelize_discretization(pd.DataFrame(T_out_2))
            DF_T_out_3=discretizer.parallelize_discretization(pd.DataFrame(T_out_3))
            DF_T_out_Z=discretizer.parallelize_discretization(pd.DataFrame(T_out_Z))
            DF_T_out_3p=discretizer.parallelize_discretization(pd.DataFrame(T_out_3p))
            DF_T_out_2p=discretizer.parallelize_discretization(pd.DataFrame(T_out_2p))
            DF_T_out_1p=discretizer.parallelize_discretization(pd.DataFrame(T_out_1p))
            DF_T_out_0p=discretizer.parallelize_discretization(pd.DataFrame(T_out_0p))
            T_discretizacion= time.time() - start_time
            # print ("Tiempo de discretizacion: {0}".format(elapsed_time))   
            
            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)

            #Se crea un pool de datos para la paralelizacion, devoliviendolos en el mismo orden
            # X->X'
            # X_1->X_1'
            # X_2->X_2'
            # X_3->X_3'
            # X->Z
            spool_data=np.array([(X_discrete,DF_T_out_0p),
                                (X_discrete,DF_T_out_Z),
                                (DF_T_out_1,DF_T_out_1p),
                                (DF_T_out_2,DF_T_out_2p),
                                (DF_T_out_3,DF_T_out_3p),
                                ],dtype=object)

            start_time = time.time()
            jedf=pe.parallelize(spool_data,jentropies_aux)
            T_entropias= time.time() - start_time
            # print ("Tiempo de entropias: {0}".format(elapsed_time))

            # snapshot = tracemalloc.take_snapshot()
            # display_top(snapshot)

            spool_data      = None
            DF_T_out_Z      = None
            DF_T_out_1      = None
            DF_T_out_1p     = None
            DF_T_out_2      = None
            DF_T_out_2p     = None
            DF_T_out_3      = None
            DF_T_out_3p     = None

            start_concat=time.time()
            buffer=jedf.copy()

            idx_store_base = (ind_epoch*n_batches+ind_batch)*15
            idx_store_end  = idx_store_base+15
            
            T_batch_creation = time.time()-batch_start_time
            buffer["idx"]  = pd.Series(np.arange(idx_store_base,idx_store_end),index=buffer.index)            
            buffer["epoch"]         = ind_epoch
            buffer["batch"]         = ind_batch
            buffer["Batch Time"]    = T_batch_creation
            buffer["loss"]          = loss
            buffer["input"]         = pd.Series(["X","X","X","X","X","X","T1","T1","T1","T2","T2","T2","T3","T3","T3"],
                                                dtype="category",
                                                index=buffer.index)
            buffer["output"]        = pd.Series(["Xp","Xp","Xp","Z","Z","Z","T1p","T1p","T1p","T2p","T2p","T2p","T3p","T3p","T3p"],
                                                dtype="category",
                                                index=buffer.index)                     
            buffer                  = buffer.reset_index()
            buffer                  = buffer.set_index('idx')
            final_data              = pd.concat([final_data,buffer],ignore_index=True)
            T_buffer_creation       = time.time()-start_concat
            #*********************************************************************
            #*********************************************************************
            #print np.sum(X_intermedio)
            
            # Reseteamos el contenido de los batches (por si acaso)
            X_train = 0
            y_train = 0
            if ((ind_batch+1)%10 == 0 ):
                final_data.to_parquet("{basedir}/{database}_entropy_K_{K}_iteracion_{epoch}_{batch}.npy".format(
                    basedir=basedir,
                    K=K,
                    epoch=ind_epoch,
                    batch=ind_batch,
                    database=database
                    ))
                model.save("{basedir}/model_{database}_entropy_K_{K}_iteracion_{epoch}_{batch}.h5".format(
                    basedir=basedir,
                    K=K,
                    epoch=ind_epoch,
                    batch=ind_batch,
                    database=database
                    ))
            gc.collect()
            print ('Batch number: {0}:{1}\tloss: {2:8.4}\t(Tiempo: \tdiscretizacion: {3:8.4}\tentropias: {4:8.4}\tParcial: {5:8.4}\tTotal:{6:8.4} seg.)'.format(ind_epoch,ind_batch,loss,T_discretizacion,T_entropias,T_buffer_creation,time.time()-batch_start_time))
        

        BASE_DEVICE_CLASS='/cpu:0'
        with tf.device(BASE_DEVICE_CLASS):
            output_dim = nb_classes = 10 
            model_class = Sequential() 
            model_class.add(Dense(output_dim, input_dim=dim_layers[4], activation='softmax')) 
            class_batch_size = 128 
            class_nb_epoch = 20            

        model_class.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        Z_train  = intermediate_layer_model_Z.predict_on_batch(x_train)
        Z_test   = intermediate_layer_model_Z.predict_on_batch(x_test)
        history = model_class.fit(Z_train, y_train, batch_size=class_batch_size, nb_epoch=class_nb_epoch,verbose=1, validation_data=(Z_test, y_test)) 
        score = model_class.evaluate(Z_test, y_test, verbose=0) 
        print("Classifier score: {0}".format(score))

        final_data.to_parquet("{basedir}/{database}_entropy_K_{K}_iteracion_{epoch}.npy".format(
                    basedir=basedir,
                    K=K,
                    epoch=ind_epoch,
                    database=database
                    ))
        model.save("{basedir}/model_{database}_entropy_K_{K}_iteracion_{epoch}.h5".format(
                    basedir=basedir,
                    K=K,
                    epoch=ind_epoch,
                    database=database
                    ))


def check_and_create_dir(directory):
    if not os.path.isdir(directory):
        try:
            os.makedirs(directory)
        except:
            print ("No es posible crear el directorio {0}\n".format(directory))
            exit(-1)

def copy_results(src_dir,target_dir):
    import shutil
    check_and_create_dir(target_dir)
    src_files = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name = os.path.join(src_dir, file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, target_dir)



def main(argv):
    K=4
    basedir="/export/localdata/hmolina/"
    targetdir=None
    n_epochs=10
    batch_length=250
    numbins=32
    database='MNIST'
    start_epoch=0
    num_gpus = 1
    try:
        opts, _ = getopt.getopt(argv[1:], 'D:K:n:s:b:B:T:G:', ['basedir=', 'encoder_final=', 'num_epochs=','batchsize=','numbins=','database=','targetdir=','start_epoch=','num_gpus'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-D', '--basedir'):
            basedir = arg
        if opt in ('-n', '--num_epochs'):
            n_epochs = int(arg)
        if opt in ('-K', '--encoder_final'):
            K = int(arg)
        if opt in ('-s','--batchsize'):
            batch_length=int(arg)
        if opt in ('-b','--numbins'):
            numbins=int(arg)
        if opt in ('-B','--database'):
            database=arg
        if opt in ('-T','--targetdir'):
            targetdir=arg
        if opt in ('-E','--startepoch'):
            start_epoch=int(arg)
        if opt in ('-G','--num_gpus'):
            num_gpus=int(arg)

    check_and_create_dir(basedir)
    training(K,basedir,n_epochs=n_epochs,batch_length=batch_length,numbins=numbins,database=database,start_epoch=start_epoch,num_gpus=num_gpus)
    if not targetdir==None:
        print("Copiando resultados desde {0} a {1}\n".format(basedir,targetdir))    
        copy_results(basedir,targetdir)


if __name__ == "__main__":
    main(sys.argv)