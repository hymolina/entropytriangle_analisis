
# coding: utf-8

# <CENTER><H1>Entorno de Ejecucion de Procesos en Mesos</h1></center>
# Ver: 2.0b1
# 
# 2017/02/07

# Definción del entorno de ejecución y librerias Python 2.7 necesarias



from __future__ import print_function
import os
import sys
import time
import itertools

from Scheduler import SchedulerMaster


# <H1>Variables necesarias para la ejecución de los procesos:</H1>
# Definición de los recursos necesarios. 
# El sistema contiene el proceso en los nodos esclavos a los recursos solicitados y no permite salirse de ese rango.
# 
# <b><i>CPUs</i> - Número de CPUs por proceso</b>: Definir 1 si el proceso no ejecuta o usa recursos en paralelo, o N en caso contrario.<BR>
# Ejemplo de programas con proceso paralelo: Matlab, Python con uso de Numpy.<BR>
# Programas monohilo: Python, R.<BR>
# 
# <b><i>MEMORY</i> - Memoria necesaria por proceso</b>: Se recomienda 2048 * CPUs, con un máximo de 4096

CPUs   = 8          #numero de CPUs usadas para cada tarea en el proceso 1
MEMORY = 65536      # Memoria máxima por proceso, en Megabytes. Recomendado 2048*CPUs
GPUs   = 1

# Definición de las variables de entorno que necesitarán los procesos esclavos.
# Estas variables se exportarán a cada uno de los nodos esclavos.
# 
# En caso de usar MATLAB, hay que definir la variable MATLABPATH, indicando el directorio a partir del cual se encuentra el código de ejecución.
# 
# La Variable <b>ENVIRONMENT_VARS</b> es de tipo Dict

ENVIROMENT_VARS=dict()
ENVIROMENT_VARS={'MATLABPATH':"/export/usuarios01/hmolina/remote/mesos"} 
                                                     # Esta linea fallará, susituir, si es necesario  
                                                     # PATH_LIBRERIAS_MATLAB por el directorio donde se 
                                                     # encuentran las librerias de Matlab a usar

# Definición de los ficheros que se transmitirán al nodo esclavo.
# La variable <B>URIS</B> es un Array de URIs, con la ubicación en el sistema de ficheros de los recursos a enviar. Estos se copiarán en local a los nodos esclavos y en caso de ser un fichero .zip o .tar se expandirán.
# Si el programa o script principal es propietario, se recomienda distribuirlo entre los nodos.
# 
# La definición de la URI pueden ser:<br>
# file:// para ficheros en el sistema de ficheros en red<br>
# http:// para ficheros accesibles desde un servidor web<br>
# https:// Servidores seguros (esto no está probado para casos con la CA no incluida en el sistema)<br>

URIS=[]
URIS.append("/export/clusterdata/hmolina/fva/autoencoder/DeepAutoencoder.py")

# La variable <B>TASK_BASENAME</B> se utiliza para definir el prefijo de las subtareas a ejecutar. El nombre de las subtarea será:
# TASK_BASENAME - TID

TASK_BASENAME="DeepAutoencoder"

# Definición del framework de ejecución.<br>
# Se debe definir:<br>
# <ul>
# <li><B>FRAMEWORK_NAME</B>: Nombre de la tarea en general
# <li><B>FRAMEWORK_PRINCIPAL</B>: Nombre del método principal
# </ul>

FRAMEWORK_NAME="Prueba TensorFlow"       # Sustituir por el nombre del trabajo a lanzar
FRAMEWORK_PRINCIPAL="Proceso Python - No Spark" # Sustituir por el nombre del proceso principal

# La variable <B>MAX_NUM_PROCESOS_SIMULTANEOS</B> define el máximo número de tareas simultaneas.
# Si se desea limitar, definirlo aqui.
# Asignar -1 para no limitarlo

MAX_NUM_PROCESOS_SIMULTANEOS=-1

# <B>MESOS_MASTER</B>: Aqui se define la URI del gestor de recursos de la granja o servidor principal de Mesos:<br>
# (BUG):<BR> 
# Si se ejecuta desde Jupyter, no reconoce bien la URI de zookeeper. Toca configurar directamente el servidor.<br>
# La variable master debe valer una de las siguientes direcciones:<br>
# 
# 192.168.151.234:5050<br>
# 192.168.151.235:5050<br>
# 192.168.151.160:5050<br>
# 
# Si se ejecuta desde línea de comando, usar mejor la configuración ZooKeeper:
# "zk://10.0.12.58:2181,10.0.12.59:2181,10.0.12.60:2181,10.0.12.51:2181,10.0.12.61:2181,10.0.12.52:2181,10.0.12.18:2181/mesos_bastet"

MESOS_MASTER="zk://10.0.12.18:2181,10.0.12.77:2181,10.0.12.60:2181,10.0.12.51:2181,10.0.12.75:2181,10.0.12.76:2181,10.0.12.78:2181/mesos_bastet"
# MESOS_MASTER="macallan.tsc.uc3m.es:5050"

STDOUT_FILE=None
STDERR_FILE=None

# Definición de la linea de comando en cada uno de los nodos esclavos
# La sitaxis es:
# exec COMANDO Parametros fijos %%parametros_a_sustituir
# La sustitución de los parámetros se definen usando la sintaxis de python %X (que está en obsolesencia)
# baseprogram="./TensorFlow_Example-1.1.0.py"
# baseprogram="./test_tf_gpus_4.py"

baseprogram="python3 DeepAutoencoder.py"
DATABASE="FMNIST"
base_cmd='{CMD} -B {database} -K {freeparam1} -n {numepochs} -s {batchsize} -b {numbins} -D /export/localdata/hmolina/{database}/{freeparam1}/ -T /export/clusterdata/hmolina/fva/autoencoder/{database}'.format(
        CMD=baseprogram,
        database=DATABASE,
        numepochs=10,
        batchsize=250,
        targetdir="/export/localdata/hmolina/FMNIST_{param1}",
        freeparam1="{param1}",
        numbins=32
    )

print(base_cmd)

# Definición de los rangos de valores que se van a iterar.
# Deben ser de tipo lista y se pueden mezclar numeros con cadenas de caracteres

parametro1=[4,8,12,16,32]

# Construcción de los comandos a ejecutar:<br>
# Dados los parámetros de ejecución, se crea el producto cartesiano de los parámetros (itertools.product), 
# generando un array de tuplas. Dichas tuplas serán utilizadas para sustituir en la linea de comando los comodines %X por el valor exacto.

cmds = []
lista=parametro1
for i in list(lista):
    final_cmd=base_cmd.format(param1=i)
    cmds.append(final_cmd)
    
T_totales = len(cmds)

print("Numero total de tareas: {0}\n".format(T_totales))
print("Ejemplo de comando a ejecutar: {0}\n".format(cmds[0]))

# <H2><FONT COLOR="#FF00">No Tocar a partir de este punto</FONT></H2>
# <H3>Construccion y ejecución de la tarea</H3>
# Se configura el la salida del Driver a los ficheros definidos previamente

#guarda los datos en un fichero para poder 
#ver los errores y estados de finalización de las tareas
if STDOUT_FILE != None:
    sys.stdout=open(STDOUT_FILE,'w')
if STDERR_FILE != None:
    sys.stderr=open(STDERR_FILE,'w')


# Se instancia el Objeto Scheduler de esta tarea y se configura el número máximo de tareas simultaneas a ejecutar

master = SchedulerMaster(cmds,CPUs,MEMORY,ENVIROMENT_VARS,URIS,basename=TASK_BASENAME,GPUs=GPUs)
master.setMaxNumTaskRunning(MAX_NUM_PROCESOS_SIMULTANEOS)
master.setMaxFailedTasks(5)

# Se inicializan los objetos Mesos Driver y Framework

try:
    framework = master.InitializeFramework(FRAMEWORK_NAME,FRAMEWORK_PRINCIPAL)
    credentials = master.InitializeCredentials()
    driver = master.InitializeDriver(MESOS_MASTER)
except Exception as error:
    e = sys.exc_info()[0]
    print( "Error: {0}".format(e))
    print( "Exception message: {0}".format(repr(error)))

#Instanciacion del Framework con la definición de tareas
tiempo_inicial1 = time.time() #tiempo en el que se inicia el porceso1

# Se lanza el hilo de ejecución paralela que ejecuta y controla los procesos remotos.
#Ejecución y espera de finalización de la tareaa
try:
    exit_val=master.start_framework()
    print("Iniciando ejecucion\nMaster status: {0}".format(master.getSchedulerStatus()))
except Exception as error:
    e = sys.exc_info()[0]
    print( "Error: {0}".format(e))
    print( "Exception message: {0}".format(repr(error)))


# <H1>A partir de este punto los procesos se están distribuyendo.</H1>
# Salvo que se desee ejecutar algo más, o supervisar de alguna manera avanzada la ejecución, dejar los siguientes campos como están.

print("Master status: {0}".format(master.getSchedulerStatus()))

# El siguiente while verifica que el driver esté en ejecución, y si es así, pasa a dormir 1 segundo.

tasks = master.getTasksInfo()




while (master.getSchedulerStatus() == 1):
    time.sleep(1)




#comprueba el cierre de la conexion con nuestro sistema distribuido
status = 0 if driver.run() == master.DRIVER_STOPPED else 1 
#imprime el tiempo de ejecucion proceso 1
print('El tiempo de ejecucion fue: {0}'.format(master.getStatistics())) 
driver.stop()






