
# coding: utf-8

# In[1]:


import os
import cv2
import subprocess
import numpy as np
import random
import shutil
import math
from time import time
from matplotlib import pyplot as plt


def CrearCarpeta():
    crearCarpeta = input("¿Desea crear un nuevo proyecto?[y/N] ")
    if crearCarpeta=='y' or crearCarpeta=='Y':
        nombreCarpetaPro = input("Nombre de la carpeta de proyecto ")
        creaCarpetaPro = "mkdir C:\\ResumenVideo\\"+nombreCarpetaPro
        creaCarpetaEnt = "mkdir C:\\ResumenVideo\\"+nombreCarpetaPro+"\\input"
        creaCarpetaSal = "mkdir C:\\ResumenVideo\\"+nombreCarpetaPro+"\\output"
        os.system(creaCarpetaPro)
        os.system(creaCarpetaEnt)
        os.system(creaCarpetaSal)
        print("El proyecto se ha creado en: C:\\ResumenVideo\\"+nombreCarpetaPro)
    elif crearCarpeta == 'N' or crearCarpeta == 'n' or crearCarpeta == '':
        nombreCarpetaPro = input("Escriba el nombre de la carpeta de proyecto que desea usar ")
        usarCarpetaPro = "C:\\ResumenVideo\\"+nombreCarpetaPro
        print("Usando la carpeta de proyecto creada en: C:\\ResumenVideo\\"+nombreCarpetaPro)
    else:
        print("nombre incorrecto")
        nombreCarpetaPro = CrearCarpeta()
    return nombreCarpetaPro


def recorre_imagenes(H, path):
    #directorio de imagenes (path)
    #diccionario de histogramas rutas
    res=[]
    #itera nombre ficheros de carpeta
    for image in os.listdir(path):
        #toma path fichero
        input_path = os.path.join(path, image)
        
        #lee imagen
        img=cv2.imread(input_path)

        color=('r','g','b')
        #por cada canal
        for i,col in enumerate(color):            
            
            histr=cv2.calcHist([img],[i],None,[H],[0,256])
            
            res.append(histr)
        
    return res

def getImagenes(inpath):  
    res=[]
    for image in os.listdir(inpath):
        #toma path fichero
        input_path = os.path.join(inpath, image)        
        #lee imagen
        img=cv2.imread(input_path)
        res.append(img)
        
    #print(len(res))
    return res
     
def return_intersection(centroide, hist):
    minima = np.minimum(centroide, hist)#el mínimo coincide con la intersección
    intersection = np.true_divide(np.sum(minima), np.sum(hist))#división real para normalizar
    return intersection

def DeterminaK(listaHist, ajuste):
    listaDistanciaAlSiguiente=[]

    for j in range(int(len(listaHist)/3-1)):#tratamos la lista general en bloques de tres
    #histogramas fotograma actual
        histr=listaHist[j*3]
        histg=listaHist[j*3+1]
        histb=listaHist[j*3+2]     
    #histogramas siguiente fotograma
        histNextr=listaHist[(j+1)*3]
        histNextg=listaHist[(j+1)*3+1]
        histNextb=listaHist[(j+1)*3+2]  

        n=0
        #r

        n=n+return_intersection(histr,histNextr)
        #g

        n=n+return_intersection(histr,histNextg)
        #b

        n=n+return_intersection(histr,histNextb)

        listaDistanciaAlSiguiente.append(n)
  
    #calculamos media de diferencias al siguiente
    media=sum(listaDistanciaAlSiguiente)/len(listaDistanciaAlSiguiente)
    #buscamos saltos considerables
    indice = 0
    centrosInicialesPorSalto=[]
    for z in listaDistanciaAlSiguiente:
        indice+=1
        if(indice > len(listaDistanciaAlSiguiente)*0.15 or z< ajuste*media):#% a determinar
            indice=0
            centrosInicialesPorSalto.append(listaHist[listaDistanciaAlSiguiente.index(z)*3])
            centrosInicialesPorSalto.append(listaHist[listaDistanciaAlSiguiente.index(z)*3+1])
            centrosInicialesPorSalto.append(listaHist[listaDistanciaAlSiguiente.index(z)*3+2])
            #incorpora como centro a los tres histogramas de la imagen detectada
        
        
    centrosInicialesPorSalto.append(listaHist[len(listaDistanciaAlSiguiente)*3])
    centrosInicialesPorSalto.append(listaHist[len(listaDistanciaAlSiguiente)*3+1])
    centrosInicialesPorSalto.append(listaHist[len(listaDistanciaAlSiguiente)*3+2])
    #incorpora como centro a los tres últimos histrogramas de la última imagen
    
    print("centros iniciales con heurística para k ->",int(len(centrosInicialesPorSalto)/3))
    return centrosInicialesPorSalto
    
def CalculaCentroidesIniciales(listaHist, K):
    listCentroides=[]
    listaCogidos=[]
    salto = math.floor(len(listaHist)/(3*int(K)))
    indice = 0
    for i in range(int(K)):
        listCentroides.append(listaHist[indice*3])
        listCentroides.append(listaHist[indice*3+1])
        listCentroides.append(listaHist[indice*3+2])
        indice += salto
    return listCentroides

def MasParecidoA(listaHist,listaHistRef):
#distancias a centroides
    distancias=[]
    for j in range(int(len(listaHist)/3)):#tratamos la lista general en bloques de tres
        
        histr=listaHist[j*3]
        histg=listaHist[j*3+1]
        histb=listaHist[j*3+2]        
        
        n=0
        #r
        histCentror=listaHistRef[0]
        n=n+return_intersection(histCentror,histr)
        #g
        histCentrog=listaHistRef[1]
        n=n+return_intersection(histCentrog,histg)
        #b
        histCentrob=listaHistRef[2]
        n=n+return_intersection(histCentrob,histb)
            
        distancias.append(n)
        #print(n)
        #print("----------------------------------")
            
    asignado=distancias.index(max(distancias))#indice del fotograma más parecido
    
    #print(asignado)
    return asignado 
        

def CalculaNuevosCentros(ListaCentros,ListaframesConClasificacion,K,listaHist,H,inpath):
    centrosNuevos=[]    
    imgs=getImagenes(inpath);
    color=('r','g','b')
    for i in range(int(K)):
        #print(i)
        #print(ListaCentros[i])
        grupo=[]
        pos=0
        for j in ListaframesConClasificacion:#index es pos fotograma, valor es centroide + cercano  
           # print('j in Listaframes')
            #print(j)
            if(i==j):
               # print('true')
                grupo.append(imgs[pos])  
                #print(pos)
            pos=pos+1 
        #print(len(grupo))
        if (len(grupo)==0):
            rd=random.randrange(len(imgs))
            #print("ALEATORIO==============================")
            #print(rd)
            #print("==============================")
            img=imgs[rd]
            grupo.append(img)
            accuWeightr= cv2.calcHist([img],[0],None,[H],[0,256])#primer histograma del grupo  
            accuWeightg= cv2.calcHist([img],[1],None,[H],[0,256])#primer histograma
            accuWeightb= cv2.calcHist([img],[2],None,[H],[0,256])#primer histograma
        if(len(grupo)==1):
            
            img=grupo[0]
            grupo.append(img)
            accuWeightr= cv2.calcHist([img],[0],None,[H],[0,256])#primer histograma del grupo  
            accuWeightg= cv2.calcHist([img],[1],None,[H],[0,256])#primer histograma
            accuWeightb= cv2.calcHist([img],[2],None,[H],[0,256])#primer histograma
        if(len(grupo)>1):

            img=grupo[0]
            alpha=1/2#suavizado, estudiar causa

            avgr= cv2.calcHist([img],[0],None,[H],[0,256])#primer histograma del grupo

            #plt.plot(avgr)
           # plt.xlim([0,256])

            avgg= cv2.calcHist([img],[1],None,[H],[0,256])#primer histograma
            avgb= cv2.calcHist([img],[2],None,[H],[0,256])#primer histograma
            indice = 2
            for x in range(1,len(grupo)):
                g=grupo[x]

                h=cv2.calcHist([g],[0],None,[H],[0,256])
                accuWeightr = cv2.accumulateWeighted(h, avgr, alpha)


                h=cv2.calcHist([g],[1],None,[H],[0,256])
                accuWeightg = cv2.accumulateWeighted(h, avgg, alpha)


                h=cv2.calcHist([g],[2],None,[H],[0,256])
                accuWeightb = cv2.accumulateWeighted(h, avgb, alpha)
                indice = indice + 1
                alpha = 1/indice
        ListaHistCentroideFicticio=[accuWeightr,accuWeightg,accuWeightb]  
        #plt.plot(accuWeightr)
        #plt.xlim([0,256])
        #comprobar que imagen del grupo se acerca más a los histogramas medios calculados
        indexNuevoFotogramaCentroide=MasParecidoA(listaHist,ListaHistCentroideFicticio)
        #print("***************")  
        #print(indexNuevoFotogramaCentroide)
        
                                        
        #ya tenemos calculados los histogramas medios por color de cada centro, pero no son reales. ¿buscamos real o al final solo?
        centrosNuevos.append(indexNuevoFotogramaCentroide)     
        #pare evitar repetidos lista m de index repes
        m=[i for i,x in enumerate(centrosNuevos) if centrosNuevos.count(x)>1]
        
        while (len(m)>0):
            centrosNuevos[m[0]]=random.randrange(len(imgs))
            m=[i for i,x in enumerate(centrosNuevos) if centrosNuevos.count(x)>1]
        
    return centrosNuevos

def Clasifica(ListaHist,K,ListaCentro):
    ListaframesConClasificacion=[]
    for j in range(int(len(ListaHist)/3)):#de la lista general, damos tratamiento de tres en tres
        distancias=[]
        histr=ListaHist[j*3]
        histg=ListaHist[j*3+1]
        histb=ListaHist[j*3+2]        
        for i in range(int(K)):
            n=0
            #r
            histCentror=ListaHist[ListaCentro[i]*3]
            n=n+return_intersection(histCentror,histr)
            #g
            histCentrog=ListaHist[ListaCentro[i]*3+1]
            n=n+return_intersection(histCentrog,histg)
            #b
            histCentrob=ListaHist[ListaCentro[i]*3+2]
            n=n+return_intersection(histCentrob,histb)
            
            distancias.append(n)
            
        asignado=distancias.index(max(distancias))
        #print(asignado)
        ListaframesConClasificacion.append(asignado)#index es pos fotograma, valor es centroide + cercano  
    return ListaframesConClasificacion

def CalcularFotogramasClave(INPUTPATH, H, inpath, ajuste):
    #1. ListaFrames ← Lista vacía(se inicializará en la función auxiliar )
    #2. Para cada fotograma (ya selecionados cada T previamente).
    #Añadir a ListaFrames el conjunto de histogramas RGB (de tamaño H) del fotograma f 
    listaHist=recorre_imagenes(H, inpath)
    #print(len(listaHist))
    #3. ListaFramesConClasificacion ← AplicaKmedias(ListaFrames, K) 
    
    ListaframesConClasificacion=[]
    #4. ListaKeyFrames ← CalculaCentroidesClases(ListaFramesConClasificacion, K) 
    #centroides iniciales
    ListaKeyFrames= DeterminaK(listaHist, ajuste)
    #ListaKeyFrames= CalculaCentroidesIniciales(listaHist, K)
    K = int(len(ListaKeyFrames)/3)
    #distancias a centroides
    for j in range(int(len(listaHist)/3)):#de la lista general, damos tratamiento de tres en tres
        distancias=[]
        histr=listaHist[j*3]
        histg=listaHist[j*3+1]
        histb=listaHist[j*3+2]        
        for i in range(int(K)):
            n=0
            #r
            histCentror=ListaKeyFrames[i*3]
            n=n+return_intersection(histCentror,histr)
            #g
            histCentrog=ListaKeyFrames[i*3+1]
            n=n+return_intersection(histCentrog,histg)
            #b
            histCentrob=ListaKeyFrames[i*3+2] 
            n=n+return_intersection(histCentrob,histb)
            
            distancias.append(n)
            
        asignado=distancias.index(max(distancias))
        #print(asignado)
        ListaframesConClasificacion.append(asignado)#index es pos fotograma, valor es centroide + cercano  
    print("primera lista clasificacion completada")
    
    ListaCentros=[]
    for i in range(int(K)):
        ListaCentros.append(i)#los k primeros que hemos tomado como iniciales
    nuevosCentros=CalculaNuevosCentros(ListaCentros,ListaframesConClasificacion,K,listaHist,H,inpath)
    tope=40 #*******************************************meter en petición
    vueltas=1
    print("vueltas:")
    while(sorted(nuevosCentros)!=sorted(ListaCentros) ):
        ListaCentros=nuevosCentros
        ListaframesConClasificacion=Clasifica(listaHist,K,ListaCentros)
        nuevosCentros=CalculaNuevosCentros(ListaCentros,ListaframesConClasificacion,K,listaHist,H,inpath)
        vueltas=vueltas+1
        print(vueltas)
        if vueltas == tope:
            break
            
    #*********************************************************************************************************************

    #5. Devolver ListaKeyFrames y ListaFramesConClasificacion
    return nuevosCentros

def Escribir(ListaKeyFrames,OUTPUTPATH,inpath,outpath):
    cont=1
    for c in sorted(ListaKeyFrames):
        
        l=str(c+1)

        while (len(l)<3):
            l='0'+l
        shutil.copy2(inpath+'\\img_'+l+'.jpg', outpath+'\\img_'+str(cont)+'.jpg') # complete target filename given
        cont=cont+1

    #cmd='C:\\ffmpeg\\bin\\ffmpeg.exe -f image2 -i C:\\Temp\\IA\\outpath\\img_%03d.jpg -r 1 -s 100x100 '+OUTPUTPATH
    cmd = 'C:\\ffmpeg\\bin\\ffmpeg.exe -r 10 -i '+outpath+'\\img_%d.jpg -r 30 -c:v mpeg4 -pix_fmt yuv420p '+OUTPUTPATH
    #cmd = 'ffmpeg.exe -f image2 -framerate 25 -pattern_type sequence -framerate 3 -i "C:\\Temp\\IA\\outpath\\img_%03d.jpg" -s 720x480 '+OUTPUTPATH

    print(cmd)
    os.system(cmd)
   

        

#************************************************************************************

nombreCapetaProyecto = str(CrearCarpeta())
rutaCarpetaProyecto = "C:\\ResumenVideo\\"+nombreCapetaProyecto
INPUTPATH = input("Escriba la ruta completa (incluyendo nombre del archivo) del video que desea tratar ")
print("ruta entrada, " + INPUTPATH)
OUTPUTPATH = input("Escriba la ruta completa (incluyendo nombre del archivo) del video que desea obtener ")
print("ruta salida, " + OUTPUTPATH)
ajuste = input("porcentaje de ajuste (en %): ")
print("ajuste, " + ajuste + "%")
AJUSTE = float(ajuste)/100
H=int(256)
start_time = time()
#************limpiamos carpetas
outpath = rutaCarpetaProyecto+"\\output"
for image in os.listdir(outpath):
        #toma path fichero
        out_path = os.path.join(outpath, image)
        os.remove(out_path)

inpath = rutaCarpetaProyecto+"\\input"
for image in os.listdir(inpath):
        #toma path fichero
        in_path = os.path.join(inpath, image)
        os.remove(in_path)
        
#************leemos video y guardamos según salto de fotograma seleccionado        
cmd='C:\\ffmpeg\\bin\\ffmpeg.exe -i '+INPUTPATH+' -vf "select=not(mod(n\,1))" -vsync vfr -q:v 2 '+inpath+'/img_%03d.jpg'
print(cmd)
os.system(cmd)

#****************

#************ppal
ListaKeyFrames=CalcularFotogramasClave(INPUTPATH, H, inpath, AJUSTE)
Escribir( ListaKeyFrames,OUTPUTPATH,inpath,outpath)
allTime = time() - start_time
print("Tiempo: %0.10f seconds." % allTime)


input("Completado, pulsa intro para continuar")
#***********


#C:/Users/carmenJara/Downloads/Rana.avi
#C:\\Temp\\IA\\hola\\v70.mpg
#C:\\ResumenVideo\\v70.mpg
#C:\\Users\\carmenJara\\Downloads\\database\\database\\v70.mpg


# In[81]:


listaHist=recorre_imagenes(256, "C:\\ResumenVideo\\distribuido\\input")
def DeterminaK(listaHist, T):
    listaDistanciaAlSiguiente=[]

    for j in range(int(len(listaHist)/3-1)):#tratamos la lista general en bloques de tres
    #histogramas fotograma actual
        histr=listaHist[j*3]
        histg=listaHist[j*3+1]
        histb=listaHist[j*3+2]     
    #histogramas siguiente fotograma
        histNextr=listaHist[(j+1)*3]
        histNextg=listaHist[(j+1)*3+1]
        histNextb=listaHist[(j+1)*3+2]  

        n=0
        #r

        n=n+return_intersection(histr,histNextr)
        #g

        n=n+return_intersection(histr,histNextg)
        #b

        n=n+return_intersection(histr,histNextb)
        listaDistanciaAlSiguiente.append(n)
  
    #calculamos media de diferencias al siguiente
    media=sum(listaDistanciaAlSiguiente)/len(listaDistanciaAlSiguiente)
    print(media*0.2)
    #buscamos saltos considerables
    indice = 0
    centrosInicialesPorSalto=[]
    for z in listaDistanciaAlSiguiente:
        indice+=1
        print(z)
        if(z<0.2*media):#% a determinar
            indice=0
            centrosInicialesPorSalto.append(listaDistanciaAlSiguiente.index(z))
            #incorpora como centro a los tres histogramas del grupo detectado
        
        
    centrosInicialesPorSalto.append(len(listaDistanciaAlSiguiente))#incorpora como centro al último del último grupo

    print("centros iniciales con heurística para k ->",int(len(centrosInicialesPorSalto)))
    return centrosInicialesPorSalto
DeterminaK(listaHist, 2)

