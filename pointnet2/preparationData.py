import os
import sys
import numpy as np
import h5py
import torch
import json
import pandas as pd
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

#ROOT_DIR = os.path.join(BASE_DIR, 'data')
#ROOT_DIR is the folder of finale files of data set in form .h5 which will be passed during the training

type_data='espression'
if type_data=='espression':

    ROOT_DIR="/data/falhamdoosh/data_espression"
elif type_data=='unicod':

    ROOT_DIR="/data/falhamdoosh/data_unicod"
elif type_data=='id':
    ROOT_DIR= "/data/falhamdoosh/face_with_normal"


if not os.path.exists(ROOT_DIR):
    os.mkdir(ROOT_DIR)
    #'bosphorusReg3DMM'modelnet40_ply_hdf5_2048
#DATA_DIR is the direction to the original data file in format .pt for example 
DATA_DIR = "/data/falhamdoosh/bosphorusReg3DMM"
    
def pc_normalize(pc):
#normalize a single point cloud to zero mean and unit variance
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def get_filenames(DATA_DIR , json_filename):# get all filnames in a directory
    #DATA_DIR:Path ,directory to fetch
    #json_filename: string, name of file json to save
    #creat list of file names
    listofFilename=os.listdir(path=DATA_DIR)
    #write list to json file
    with open(os.path.join(ROOT_DIR, json_filename), "w") as outfile:
        outfile.write(json.dumps(listofFilename))
    print ("Json file is created with length",len(listofFilename))
    
def read_mat_file(matFilename):  
    #matFilename:string, name of the file .mat which should be saved in ROOT_DIR
    #read matlab file (mesh file)
    #return numpy array 
    matFile=os.path.join(ROOT_DIR, matFilename)

    f = h5py.File(matFile,'r')
    dset = f['tri'][:].astype(int) - 1
    print ("matFile")
    return dset
matFilename='triangulation.mat'
faces= read_mat_file(matFilename)


def read_json(json_filename):
    with open(os.path.join(ROOT_DIR, json_filename), 'r') as f:
        listofFilename = json.loads(f.read()) 
    print(listofFilename)



def creat_dataframes(json_filename,dataframName):
    #for identitiy expirment.
    #dataframName:string, name of dataframe to be save at the end.
    #json_filename:string, name of json file which contains all filenames
    #return: data frame with two columns [filname, label]
    #from list of file names in jzson creat a pandas data frame ("filename",lable)
    df= pd.read_json(os.path.join(ROOT_DIR, json_filename))

    df.columns=['filename']
    #delete space in strings
    df['filename'].str.strip()
    #add a new column which copy th id from the file name
    df.insert(1, 'label', df['filename'].str[2:5].astype(int), True)
    df.to_csv(os.path.join(ROOT_DIR, dataframName))
    return df
def creat_dataframes_espressione(json_filename,dataframName):
    #same as creat_dataframes but return diffrents labels
    df= pd.read_json(os.path.join(ROOT_DIR, json_filename))

    df.columns=['filename']
    #delete space in strings
    df['filename'].str.strip()
    #add a new column which copy th id from the file name
    x=df['filename'].str.split("_",n = 3, expand = True) # return es.[bs011,E,HAPPY,0]
    df["label"]=x[2] # select the second column for expressions
    #keep rows which has label value one of the list c .
    c=['ANGER','DISGUST','FEAR','HAPPY','SADNESS','SURPRISE','N']
    df=df[df['label'].isin(c)]     
    print ("len df after dropping a class minority",len(df))
    df.to_csv(os.path.join(ROOT_DIR, dataframName))
    dict_label=get_class_count(df)
    df['label']=df['label'].replace(dict_label)
    df.to_csv(os.path.join(ROOT_DIR, dataframName))
    df.head(10)
    print("mapped done")
    return df
def creat_dataframes_unicod(json_filename,dataframName):
    df= pd.read_json(os.path.join(ROOT_DIR, json_filename))

    df.columns=['filename']
    #delete space in strings
    df['filename'].str.strip()
    #add a new column which copy th id from the file name
    x=df['filename'].str.split("_",n = 3, expand = True)
    df["label"]=x[1]
    
    
    
    print('len',len(df))
    df.to_csv(os.path.join(ROOT_DIR, dataframName))
    dict_label=get_class_count(df)
    df['label']=df['label'].replace(dict_label)
    df.to_csv(os.path.join(ROOT_DIR, dataframName))
    df.head(10)
    print("mapped done")
    return df

def read_csv(dataframName):
    df= pd.read_csv(os.path.join(ROOT_DIR,dataframName))
    return df
    

def get_class_count(df):
    #df: pandas dataframe
    #find the distribution of labels in data frame df
    #map each class to an integer.
    grp=df.groupby(['label']).nunique()
    
    u=df['label'].unique().tolist()
    u=sorted(u)
    dict_label={}
    for i in range(len(u)):
        dict_label[u[i]]=i
    print(dict_label,len(u))
    return dict_label

def split(df,size, firstName,secondeName):
# split one dataframe in two,
#df:pandas dataframe
#size:size of the second data frame
#firstName,secondaName:strings of dataframe names to be saved
    target=df['label']
    train, test= train_test_split(df, test_size=size, random_state=1, stratify=target)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.to_csv(os.path.join(ROOT_DIR, firstName ))
    test.to_csv(os.path.join(ROOT_DIR,secondeName))
    #print(train.columns)
    get_class_count(train)
    get_class_count(test)
    return train,test

def split_all(train,n):
#split recursivly  n times a data frame to obtain stratified splitted dataframes
#train: pandas dataframe to split to two dataframes each with size 50% 

#return a dictionary of dataframe and list of theirnames which identify the dataframe
    train_sets={}
    datafram_names=[]
    train_sets[0]=train
    j=1
    for i in range(n): #n=3
        #0 = 1+2 newj= 1 , newk=2
       
        #1=3,4,,,2,3 ,j=2 ...> newj=3 ,newk=4
        #2=5,6 ,,,3,4, ....new j= 5
        #######
        #3=7,8,,,4,5 ....5+3
        #4=9,10 ,,,5,6
     #   print(i , j , j+1)
        train_sets[j],train_sets[j+1]=split(train_sets[i],0.5 ,("trainDatafram%s.csv" % str(j)),                                                                                  ("trainDatafram%s.csv" % str(j+1)))
        
        datafram_names.append("trainDatafram%s.csv" % str(j))
        datafram_names.append("trainDatafram%s.csv" % str(j+1))
        
        j=j+2
    return train_sets ,datafram_names

def read_ptfile(filename):
    #filname:string, filename.pt
    #read a single point cloud into numpy array and normalize it    
    data_module = torch.load(os.path.join(DATA_DIR,filename)).numpy()
    
    n=pc_normalize(data_module)
    
    return n
def normalize_v3(arr):
    #normalize a normal array
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    #print("lens is",lens )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens                
    return arr

def calc_normal(filename):
#return a a 
    data_module = read_ptfile(filename)
    #print("modul",data_module.shape,type(data_module))
    
    norm = np.zeros( data_module.shape, dtype=data_module.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = data_module[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    #n= normalize_v3(n)
    
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    #print("norm",norm.shape,type(norm))
    #print("befor normalization",norm)
    norm=normalize_v3(norm)
    #print("after normalization",norm)
    data=np.zeros( (data_module.shape[0],data_module.shape[1]+3))
    data[:,:3]= data_module 
    data[:,3:]=norm
    #print("data shape and typr",data.shape,type(data),data)
    
   # npfilename=filename[0:-3]+".npy"
   # np.save(os.path.join(DATA_DIR_NORMAL,npfilename),data)
    return data

faces= read_mat_file("triangulation.mat")    
def read_all(df, normal=False):
    #df: pandas data frame,
    # n :len (df) ,number of pointclouds 
    #all_dataNumpy: (n,numberofpoint,number of channel 3 or 6) n :numer o
    #all_lable: (n,)
    if normal:
        print("normal will be calculate")
       # faces= read_mat_file(matFilename)
        allfiles=[calc_normal(x) for x in df["filename"]]
    else:
        allfiles=[read_ptfile(x) for x in df["filename"]]
    print("length of allfiles",len(allfiles))
    #from list to numpy oobject
    all_dataNumpy=np.stack(allfiles, axis=0)
    print("shape of allfiles after stak numpy array",all_dataNumpy.shape)
    labels=[np.array([y]) for y in df["label"]]
    all_label=np.stack(labels, axis=0)
    print(all_label.shape,type(all_label),type(all_label[0]))
    print("data and numpy array is created")
    return all_dataNumpy,all_label
        
def creat_h5file(data, label ,namefileh5):
    #data:numpy array of our data.
    #label:numpy array of our labels
    #creat one file.h5 from two numpy array (data,label)
    hf = h5py.File(os.path.join(ROOT_DIR,namefileh5), 'w')
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)
    hf.close()
    print(namefileh5,"was created")
    
    
def load_h5(h5_filename):
    #load .h5 for verfication 
    #h5_filename:string, name of file to load
    f = h5py.File(os.path.join(ROOT_DIR,h5_filename),'r')
    print("load succesefully h5",f,f.keys())
    data = f["data"][:]
    label = f['label'][:]
    f.close()
    print(data.shape,type(data),type(data[0]),data[0])
 #   print(label.shape,type(label),label,type(label[0]),label[0])
    return (data)

def creath5fromCSV(file_namecsv,h5filename,normal=False):
    #file_namecsv:string, name of dataframe saved on ROOT_DIR
    #h5filename:string, name of file.h5 to be created from the dataframe
    #normal: if True, calculate Normal 
    # return .h5 file from file.csv
    df= read_csv(file_namecsv)
    data,label=read_all(df,normal)
    creat_h5file(data, label ,h5filename)
    load_h5(h5filename)
    


def main_fun():
    
    dataframName="dataframe.csv"
  
    json_filename="allfilesnames.json"
  
    get_filenames(DATA_DIR , json_filename)

    if type_data=='espression':
        df=creat_dataframes_espressione(json_filename,dataframName)
    elif type_data=='id':
        df=creat_dataframes(json_filename,dataframName)
    elif type_data =='unicod':
        df=creat_dataframes_unicod(json_filename,dataframName)
    train,test=split(df,0.3 ,"train.csv","test.csv")
    creath5fromCSV("test.csv","test.h5",normal=True)
    
    train_sets,dataframNames=split_all(train,3)
    
    print ("traindataframe numbers",train_sets.keys(),dataframNames)
    filenamesh5=[]
    for j in range (3,7):
        fileh5_name="trainset_%s.h5" % str(j)
        creath5fromCSV(dataframNames[j-1],fileh5_name,normal=True)
        filenamesh5.append(fileh5_name)
   
    with open(os.path.join(ROOT_DIR,'trainfiles.txt'), 'w') as f:
        for item in filenamesh5:
            f.write("%s\n" % item)
    with open(os.path.join(ROOT_DIR,'testfiles.txt'), 'w') as f:
        f.write("test.h5")
    print("Done")

main_fun()
