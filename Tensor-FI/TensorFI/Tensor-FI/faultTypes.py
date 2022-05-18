# These are the list of fault injection functions for different types of faults
# NOTE: There are separate versions of the scalar and tensor values for portability
# If you add a new fault type, please create both the scalar and tensor functions 

from operator import index, le
from posixpath import split
import re
from ast import Index
from math import floor
from this import d

import numpy as np

# Currently, we support 6 types of faults {None, Rand, Zero, Rand-element, bitFlip-element, bitFlip-tensor} - See fiConfig.py

def randomScalar( dtype, max = 1.0 ):
    "Return a random value of type dtype from [0, max]"
    return dtype.type( np.random.random() * max )

def randomTensor( dtype, tensor):
    "Random replacement of a tensor value with another one"
    # The tensor.shape is a tuple, while rand needs linear arguments
    # So we need to unpack the tensor.shape tuples as arguments using *  
    res = np.random.rand( *tensor.shape ) 
    return dtype.type( res )

def zeroScalar(dtype, val):
    "Return a scalar 0 of type dtype"
    # val is a dummy parameter for compatibility with randomScalar
    return dtype.type( 0.0 )

def zeroTensor(dtype, tensor):
    "Take a tensor and zero it"
    res = np.zeros( tensor.shape ) 
    return dtype.type( res )

def noScalar(dtype, val):
    "Dummy injection function that does nothing"
    return val

def noTensor(dtype, tensor):
    "Dummy injection function that does nothing"
    return tensor

def randomElementScalar( dtype, max = 1.0):
    "Return a random value of type dtype from [0, max]"
    return dtype.type( np.random.random() * max )

def randomElementTensor ( dtype, val):
    "Random replacement of an element in a tensor with another one"
    "Only one element in a tensor will be changed while the other remains unchanged" 
    dim = val.ndim 
    
    if(dim==1):
        index = np.random.randint(low=0 , high=(val.shape[0]))
        val[index] = np.random.random() 
    elif(dim==2):
        index = [np.random.randint(low=0 , high=(val.shape[0])) , np.random.randint(low=0 , high=(val.shape[1]))]
        val[ index[0] ][ index[1] ] = np.random.random()

    return dtype.type( val )



def float2bin(number, decLength = 10): 
    "convert float data into binary expression"
    # we consider fixed-point data type, 32 bit: 1 sign bit, 21 integer and 10 mantissa

    # split integer and decimal part into seperate variables  
    integer, decimal = str(number).split(".") 
    # convert integer and decimal part into integer  
    integer = int(integer)  
    # Convert the integer part into binary form. 
    res = bin(integer)[2:] + "."		# strip the first binary label "0b"

    # 21 integer digit, 22 because of the decimal point "."
    res = res.zfill(22)
    
    def decimalConverter(decimal): 
        "E.g., it will return `x' as `0.x', for binary conversion"
        decimal = '0' + '.' + decimal 
        return float(decimal)

    # iterate times = length of binary decimal part
    for x in range(decLength): 
        # Multiply the decimal value by 2 and seperate the integer and decimal parts 
        # formating the digits so that it would not be expressed by scientific notation
        integer, decimal = format( (decimalConverter(decimal)) * 2, '.10f' ).split(".")    
        res += integer 

    return res 


def randomBitFlip(val):
    "Flip a random bit in the data to be injected" 

    # Split the integer part and decimal part in binary expression
    def getBinary(number):
        # integer data type
        if(floor(number) == number):
            integer = bin(int(number)).lstrip("0b") 
            # 21 digits for integer
            integer = integer.zfill(21)
            # integer has no mantissa
            dec = ''	
        # float point datatype 						
        else:
            binVal = float2bin(number)				
            # split data into integer and decimal part	
            integer, dec = binVal.split(".")	
        return integer, dec

    # we use a tag for the sign of negative val, and then consider all values as positive values
    # the sign bit will be tagged back when finishing bit flip
    negTag = 1
    if(str(val)[0]=="-"):
        negTag=-1

    if(isinstance(val, np.bool_)):	
        # boolean value
        return bool( (val+1)%2 )
    else:	
        # turn the val into positive val
        val = abs(val)
        integer, dec = getBinary(val)

    intLength = len(integer)
    decLength = len(dec)

    # random index of the bit to flip  
    index = np.random.randint(low=0 , high = intLength + decLength)
 
     # flip the sign bit (optional)
    #if(index==-1):
    #	return val*negTag*(-1)

    # bit to flip at the integer part
    if(index < intLength):		
        # bit flipped from 1 to 0, thus minusing the corresponding value
        if(integer[index] == '1'):	val -= pow(2 , (intLength - index - 1))  
        # bit flipped from 0 to 1, thus adding the corresponding value
        else:						val += pow(2 , (intLength - index - 1))
    # bit to flip at the decimal part  
    else:						
        index = index - intLength 	  
        # bit flipped from 1 to 0, thus minusing the corresponding value
        if(dec[index] == '1'):	val -= 2 ** (-index-1)
        # bit flipped from 0 to 1, thus adding the corresponding value
        else:					val += 2 ** (-index-1) 

    return val*negTag

def bitElementScalar( dtype, val ):
    "Flip one bit of the scalar value"   
    return dtype.type( randomBitFlip(val) )

def bitElementTensor( dtype, val):
    "Flip ont bit of a random element in a tensor"
    # flatten the tensor into a vector and then restore the original shape in the end
    valShape = val.shape
    val = val.flatten()
    # select a random data item in the data space for injection
    index = np.random.randint(low=0, high=len(val))
    val[index] = randomBitFlip(val[index])	
    val = val.reshape(valShape)

    return dtype.type( val )

def bitScalar( dtype, val):
    "Flip one bit of the scalar value"
    return dtype.type( randomBitFlip(val) )

def bitTensor ( dtype, val):
    "Flip one bit in all elements within the tensor"
    
    # flatten the tensor into a vector and then restore the original shape in the end
    valShape = val.shape
    val = val.flatten()
    for i in range(len(val)-10):
        val[i] = randomBitFlip(val[i])
    val = val.reshape(valShape)
    return dtype.type( val )

def bulletWake(dtype,val):
    "Flip a bit in a decided position through different feautere map"
    print("entro")
    #print("entro")
    num=[]
    valShape=np.shape(val)
    print(valShape)
    indici=[]
    num=[]
    valShape=np.shape(val)
    print(len(valShape))
    if(len(valShape)==4):
        print("entro")
        g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
        value=g.readline()
        cardinalita=g.readline()
        max=g.readline()
        cardinalita=int(cardinalita)
        indici.append(np.random.randint(low=0,high=valShape[0])-1)
        indici.append(np.random.randint(low=0,high=valShape[2])-1)
        indici.append(np.random.randint(low=0,high=valShape[3])-1)
        valLittle=val[int(indici[0])]
        valLittleShape=np.shape(valLittle)
        valLittle=np.squeeze(valLittle,axis=0) 
        #valLittle=np.split(valLittle,valShape[1])
        index= 0
        currentIndex=0
        if(max=="RANDOM\n"):
            max=g.readline()        #E il valore masssimo per cui possmo iniettazre, e necesasrio
            g.close()
            max=int(max)
            for i in range(cardinalita):
                index= index + np.random.randint(low=0, high=max-1)
                print(index)
                if currentIndex+1==valShape[1]-1:
                    valLittle[valShape[1]-1][index[1]][index[2]]=value 
                    print(valLittle[valShape[1]-1])
                    break
                if index>valShape[1]:
                    index=np.random.randint(low=currentIndex+1, high=valShape[1]-1)
                    print("new index   " )
                    print(index)
                if value=="":    
                    valLittle[index][indici[1]][indici[2]]  = None
                else:
                    valLittle[index][indici[1]][indici[2]] = value
                currentIndex=index
            valLittle=np.reshape(valLittle,valLittleShape)
            val=np.reshape(val,valShape)
            val[int(indici[0])]=valLittle
            return dtype.type(val)
        else:
            num=re.findall(r'\b\d+\b',max)
    
            for i in num:
                print(i)
                if(valShape[1]<=int(i)):
                    print("Lunghezza troppo lunga")
                    break
                if value=="":    
                    valLittle[int(i)][indici[1]][indici[2]]  = None
                else:
                    valLittle[int(i)][indici[1]][indici[2]] = value
            valLittle=np.reshape(valLittle,valLittleShape)
            val=np.reshape(val,valShape)
            val[int(indici[0])]=valLittle
            return dtype.type(val)

    

#DA finire non e ancora chiaro
def sameFeatureRandom(dtype,val):
    "Flip a bit in a random position in a random feature map"
    print("entro")
    #print("entro")
    num=[]
    pastIndex=[]
    valShape=np.shape(val)
    print(valShape)
    indici=[]
    index=0
    valShape=np.shape(val)
    print(len(valShape))
    if(len(valShape)==4):
        print("entro")
        g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
        value=g.readline()
        cardinalita=g.readline()
        max=g.readline()
        cardinalita=int(cardinalita)
        indici.append(np.random.randint(low=0,high=valShape[0])-1)
        indici.append(np.random.randint(low=0,high=valShape[1])-1)
        val=np.split(val,valShape[0])
        valLittle=val[int(indici[0])]
        valLittle=np.squeeze(valLittle,axis=0) 
        valLittle=np.split(valLittle,valShape[1])
        valLittle=valLittle[int(indici[1])]
        valLittleShape=np.shape(valLittle)
        valLittle=np.squeeze(valLittle,axis=0)
        valFlatten=valLittle.flatten()
        if(max=="RANDOM\n"):
            max=g.readline()
            max=int(max)
            print("dentro RANDOm")         
            for i in range(cardinalita):
                minimo=min(max,valShape[2]*valShape[3]-1)
                index=np.random.randint(low=0, high=minimo)
                while(index in pastIndex):
                    index=np.random.randint(low=0, high=minimo)
                if value=="":    
                    valFlatten[index]  = None
                else:
                    valFlatten[index] = value
        else:
            num=re.findall(r'\b\d+\b',max)
            for i in num:
                print(i)
                if(int(i)>valShape[2]*valShape[3]):
                    break
                
                if value=="":    
                    valFlatten[int(i)]  = None
                else:
                    valFlatten[int(i)] = value
        valFlatten=np.reshape(valFlatten,valLittleShape)
        valLittle=valFlatten
        valLittle=np.reshape(valLittle,valLittleShape)
        val=np.reshape(val, valShape)
        val[int(indici[0])][int(indici[1])]=valLittle   
        return dtype.type(val)


def bitTensorColumn(dtype,val):
    "Flip one bit in a random column"
    index=0
    index2=0
    currentIndex=0
    #print("entro")
    num=[]
    valShape=np.shape(val)
    print(valShape)
    indici=[]
    num=[]
    valShape=np.shape(val)
    print(len(valShape))
    print("entro")
    g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
    value=g.readline()
    cardinalita=g.readline()
    max=g.readline()
    cardinalita=int(cardinalita)
    indici.append(np.random.randint(low=0,high=valShape[0])-1)
    indici.append(np.random.randint(low=0,high=valShape[1])-1)
    val=np.split(val,valShape[0])
    valLittle=val[int(indici[0])]
    valLittle=np.squeeze(valLittle,axis=0) 
    valLittle=np.split(valLittle,valShape[1])
    valLittle=valLittle[int(indici[1])]
    valLittleShape=np.shape(valLittle)
    valLittle=np.squeeze(valLittle,axis=0)
    column=valShape[3]
    lengthColumn=valShape[2]
    print(np.shape(valLittle))    
    index2= np.random.randint(low=0, high=column)
    index= np.random.randint(low=0, high=lengthColumn)
    currentIndex=0
    if(max=="RANDOM\n"):
        print("dentro RANDOm")
        max=g.readline()
        max=int(max)
        for i in range(cardinalita):
            index= index + np.random.randint(low=0, high=max)
            print(index)
            if currentIndex+1==lengthColumn-1:
                valLittle[lengthColumn-1][index2]  = value
                print(valLittle[lengthColumn-1])
                break
            if index>lengthColumn-1:
                index=np.random.randint(low=currentIndex+1, high=lengthColumn)
                print("new index   " )
                print(index)
            if value=="":    
                valLittle[index][index2]  = None
            else:
                print(str(index)+"  "+str(index2))
                valLittle[index][index2] = value
                currentIndex=index
        print("finito")
        if(len(valShape)==2):
            return dtype.type(valLittle)
        valLittle=np.reshape(valLittle,valLittleShape)
        if(len(valShape)==2):
            return dtype.type(valLittle)
        val=np.reshape(val,valShape)
        val[int(indici[0])][int(indici[1])]=valLittle
        return dtype.type(val)
    else: 
        #for s in string:
        num=re.findall(r'\b\d+\b',max)
        if(int(num[0])>valShape[0]):
            print("Dimensione 4 troppo grande")
            return dtype.type(val)
        if(int(num[1])>valShape[1]):
            print("!!! feature map non esistente !!! \n")
            print(num)
        for i in num:
            print(i)
            if(lengthColumn<=int(i)):
                print("Lunghezza troppo lunga")
                break
            if value=="":    
                valLittle[int(i)][index2]  = None
            else:
                valLittle[int(i)][index2] = value
        valLittle=np.reshape(valLittle,valLittleShape)
        val=np.reshape(val,valShape)
        val[int(indici[0])][int(indici[1])]=valLittle
        return dtype.type(val)
               
        
    #valShape=np.shape(val)
       
   
       
def singleFlip(dtype, val):
    num=[]
    valShape=np.shape(val)
    
    print("entro")

    g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
    value=g.readline()
    value=float(value)
    if(len(valShape)==4):
        num.append(np.random.randint(low=0,high=valShape[0]))
        num.append(np.random.randint(low=0,high=valShape[1]))
        num.append(np.random.randint(low=0,high=valShape[2]))
        num.append(np.random.randint(low=0,high=valShape[3]))
        if value=="":
            val[num[0]][num[1]][num[2]][num[3]]=None
        else:
            val[num[0]][num[1]][num[2]][num[3]]=value
        return dtype.type(val)
    if(len(valShape)==3):
        num.append(np.random.randint(low=0,high=valShape[0]))
        num.append(np.random.randint(low=0,high=valShape[1]))
        num.append(np.random.randint(low=0,high=valShape[2]))
        if value=="":
            val[num[0]][num[1]][num[2]]=None
        else:
            val[num[0]][num[1]][num[2]]=value
        return dtype.type(val)
    if(len(valShape)==2):
        num.append(np.random.randint(low=0,high=valShape[0]))
        num.append(np.random.randint(low=0,high=valShape[1]))
        if value=="":
            val[num[0]][num[1]]=None
        else:
            val[num[0]][num[1]]=value
        return dtype.type(val)
    if(len(valShape)==1):
        num.append(np.random.randint(low=0,high=valShape[0]))
        if value=="":
            val[num[0]]=None
        else:
            val[num[0]]=value
        return dtype.type(val)


             
#Scrive il valore passato dal file nella feature map selezionata se non e random, scrive questo valore
# per le cardinalita lette nel file, se il valore estratto da MAX e troppo alto seleziona la lunghezza della riga
# si ferma se arriva alla fine della riga, potrebbe non inserire tuute le cardinalita se raggiunge la fine della riga 
def bitTensorRow(dtype,val):
    "Flip one bit in a random row"
    #print("entro")
    indici=[]
    num=[]
    valShape=np.shape(val)
    print(len(valShape))
    if(len(valShape)==4):
        print("entro")
        g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
        value=g.readline()
        cardinalita=g.readline()
        max=g.readline()
        cardinalita=int(cardinalita)
        indici.append(np.random.randint(low=0,high=valShape[0])-1)
        indici.append(np.random.randint(low=0,high=valShape[1])-1)
        val=np.split(val,valShape[0])
        valLittle=val[indici[0]]   #seleziona sempre 0 per la 4 dimensione
        valLittle=np.squeeze(valLittle,axis=0)    
        lengthRow=valShape[3];
        valLittle=np.row_stack(valLittle[int(indici[1])-1])
        valShape2=np.shape(valLittle)
        print(valShape2)
        valFlatten=np.ndarray.flatten(valLittle)
        index=0
        indexRow= np.random.randint(low=0,high=valShape2[0]-1)
        currentIndex=0
        if(max=="RANDOM\n"):
            print("dentro RANDOm")
            max=g.readline()        #E il valore masssimo per cui possmo iniettazre, e necesasrio
            g.close()
            max=int(max)
            for i in range(cardinalita):
                index= index + np.random.randint(low=0, high=max-1)
                print(index)
                if currentIndex+1==lengthRow-1:
                    valFlatten[lengthRow-1] = value
                    valLittle[indexRow][lengthRow-1]=valFlatten[lengthRow-1]
                    print(valLittle[indexRow])
                    break
                if index>lengthRow:
                    index=np.random.randint(low=currentIndex+1, high=lengthRow-1)
                    print("new index   " )
                    print(index)
                if value=="":    
                    valFlatten[index] = None
                else:
                    valFlatten[index] = value
                valLittle[indexRow][index]=valFlatten[index]
                print(valLittle[indexRow])
                currentIndex=index
                print("\n")
            valLittle=np.reshape(valLittle,valShape2)
            val=np.reshape(val,valShape)
            val[int(indici[0])][int(indici[1])]=valLittle
            return dtype.type(val)
        else:
            numero=re.findall(r'\b\d+\b',max)
            print(numero)
            for i in numero:
                print(i)
                if(len(valFlatten))<=int(i):
                    print("Lunghezza troppo lunga")
                    break
                if value=="":    
                    valFlatten[int(i)] = None
                else:
                    valFlatten[int(i)] = value
                valLittle[indexRow][int(i)]=valFlatten[int(i)]
                print(valLittle[indexRow])
            valLittle=np.reshape(valLittle,valShape2)
            val=np.reshape(val,valShape)
            val[int(indici[0])][int(indici[1])]=valLittle
            return dtype.type(val)
    print("esco")    
    return 


def shutterGlass(dtype,val):
    bol=True
    "Flip one bit in a random row"
    #print("entro")
    min=0
    indici=[]
    num=[]
    valore=[]
    valShape=np.shape(val)
    print(len(valShape))
    if(len(valShape)==4):
        print("entro")
        g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
        value=g.readline()
        cardinalita=g.readline()
        max=g.readline()
        cardinalita=int(cardinalita)
        indici.append(np.random.randint(low=0,high=valShape[0]))
        indici.append(np.random.randint(low=0,high=valShape[1]))
        indici.append(np.random.randint(low=0,high=valShape[2]*valShape[3]))
        val=np.split(val,valShape[0])
        valLittle=val[indici[0]]
        valShape2=np.shape(valLittle)
        valLittle=np.squeeze(valLittle,axis=0)
         
        string=max
        string2=string.split("))")            
        for i in string2:
            numero=re.findall(r'-?\d+',i)
            print(i)
            if(i!=')'):
                num.append(int(numero[0]))
        if(int(num[len(numero)-1])>valShape[1]-1):
            indici[1]=0
        else:
            while(int(num[len(numero)-1])+indici[1]<valShape[1]-1):
                indici[1]=np.random.randint(low=0,high=valShape[1])
        numero=[]
        num=[]
        for i in string2:
                
            numero=re.findall(r'-?\d+',i)
            indici[2]=(np.random.randint(low=0,high=valShape[2]*valShape[3]))
            for j in range(len(numero)):
                if(int(numero[j])+indici[2]<0):
                    while(int(numero[j])+indici[2]>0):
                        indici[2]=(np.random.randint(low=0,high=valShape[2]*valShape[3]))
                        for k in range(len(numero)):
                            if(int(numero[k])+indici[2]>valShape[2]*valShape[3]):
                                indici[2]=(np.random.randint(low=0,high=valShape[2]*valShape[3]))
            for j in range(len(numero)):
                if((int(numero[0])+int(indici[1]))>valShape[1]-1):
                    print("in break")
                    break
                if(j==0):
                    print("zero")
                else:
                    if(-int(numero[j])>(valShape[2]*valShape[3])-1):
                        break
                    else:
                        print(indici[2])
                        valShape3= np.shape(valLittle[int(numero[0])+int(indici[1])])
                        valFlatten=np.ndarray.flatten(valLittle[int(numero[0])+int(indici[1])])
                        valFlatten[int(numero[j])+indici[2]]=value
                         
                        valFlatten=np.reshape(valFlatten,valShape3)
                        valLittle[int(numero[0])+int(indici[1])]=valFlatten
                           #valFlatten[int(numero[j])+indici[2]]=value
                           #valFlatten=np.reshape(valFlatten,valShape3)
                           #valLittle[int(numero[0])+int(indici[1])]=valFlatten
    valLittle=np.reshape(valLittle,valShape2)
    val[indici[0]]=valLittle
    val=np.reshape(val,valShape)
    return dtype.type(val)


def  uncategorize(dtype,val):
    "Flip a bit in a random position in a random feature map"
    print("entro")
    i=0
    #print("entro")
    num=[]
    pastIndex=[]
    valShape=np.shape(val)
    print(valShape)
    indici=[]
    index=0
    valShape=np.shape(val)
    print(len(valShape))
    if(len(valShape)<4):
       return singleFlip
    if(len(valShape)==4):
        print("entro")
        g=open("/usr/local/lib/python2.7/dist-packages/TensorFI/variables.txt","r")
        value=g.readline()
        cardinalita=g.readline()
        max=g.readline()
        cardinalita=int(cardinalita)
        indici.append(np.random.randint(low=0,high=valShape[0])-1)
        indici.append(np.random.randint(low=0,high=valShape[1])-1)
        val=np.split(val,valShape[0])
        valLittle=val[int(indici[0])]
        valLittle=np.squeeze(valLittle,axis=0) 
        valLittle=np.split(valLittle,valShape[1])
        valLittle=valLittle[int(indici[1])]
        valLittleShape=np.shape(valLittle)
        valLittle=np.squeeze(valLittle,axis=0)
        valFlatten=valLittle.flatten()
        if(max=="RANDOM\n"):
            max=g.readline()
            max=int(max)
            print("dentro RANDOm")         
            for i in range(cardinalita):
                minimo=min(max,valShape[2]*valShape[3]-1)
                index=np.random.randint(low=0, high=minimo)
                while(index in pastIndex):
                    index=np.random.randint(low=0, high=minimo)
                if value=="":    
                    valFlatten[index]  = None
                else:
                    valFlatten[index] = value
        else:
            num=re.findall(r'\b\d+\b',max)
            for i in num:
                print(i)
                if(int(i)>valShape[2]*valShape[3]):
                    break
                
                if value=="":    
                    valFlatten[int(i)]  = None
                else:
                    valFlatten[int(i)] = value
        valFlatten=np.reshape(valFlatten,valLittleShape)
        valLittle=valFlatten
        valLittle=np.reshape(valLittle,valLittleShape)
        val=np.reshape(val, valShape)
        val[int(indici[0])][int(indici[1])]=valLittle   
        return dtype.type(val)