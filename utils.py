from os import path
from enums import*
from scipy.sparse.construct import kron
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.lines as mlines
import collections
import pandas as pd
import numpy as np
import sys

class Utils:
    def __init__(self, fileName):
        dir_path = path.dirname(path.realpath(__file__))
        filePath = dir_path+"/input/"+fileName
        if path.isfile(filePath):
            self.dir = dir_path
            self.fileName,_ = path.splitext(fileName)
            self.df = pd.read_csv(filePath)
            self.X_train=None 
            self.X_test=None  
            self.Y_train=None  
            self.Y_test=None
            self.classGroups={}#represetntation: index
            self.classNames={}#represetntation: unique name
            self.colors={}
            self.markers={}
            self.columns = list(self.df.columns)
            self.columns.append('PREDICTED')
            self.legends=[]
            self.init=True
            
        else:
            print("file did not exist")
            self.init=False

    def SetClassName(self, classifications):
        # newclassGrps={}
        for classification in classifications:
            names=[]
            for itemData in classifications[classification]:
                names.append(itemData[-1])
            newNames=[]
            counter = collections.Counter(names)
            for key in counter:
                newNames.append([key,counter[key]])
                # print(counter[key])
            newNames.sort(key=lambda x:x[1],reverse=True)
            #classGroup name is based on the most dominant number in the group
            self.classNames[classification]=newNames[0][0]


    def SetMedaData(self, k):
        classInd=0
        for c in Colors:
            self.colors[classInd]=c.value
            classInd+=1
            if len(self.colors)==k:
                break
        classInd=0
        for m in Markers:
            self.markers[classInd]=m.value
            classInd+=1
            if len(self.markers)==k:
                break

    def SetLegends(self):
        for classIdx in self.classNames:
            legend = mlines.Line2D([], [], color=self.colors[classIdx], marker=self.markers[classIdx], linestyle='None',
                                    markersize=5, label=self.classNames[classIdx])
            self.legends.append(legend)
      
    def SetGroups(self, k):
        for groupIdx in range(k):
            self.classGroups[groupIdx]=groupIdx
        #TBC(checking if len(self.classGroups)<self.k):

    def Normalize(self,data):
        size = len(data[0])
        tmp = np.arange(0,size-1,1).tolist()
        a = data[:,tmp]
        b = data[:,[-1]]
        normed_matrix = preprocessing.normalize(a, axis=0, norm='l1')
        data = np.append(normed_matrix,b,axis=1)
        return data

    def SetData(self, testRatio):
        x = self.df.iloc[:,].values
        y = self.df.iloc[:, [-1]].values
        #normalize data
        x = self.Normalize(x)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(x, y, test_size=testRatio, random_state=42)    

    def GetData(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def SaveCSV(self, Postfix, saveResults):
        savePath = self.dir + "/output/"+ self.fileName + Postfix
        df1=pd.DataFrame(saveResults, columns=self.columns)
        df1.to_csv(savePath,index=False)

    @staticmethod
    def CheckLabelNums(num):
        if (len(Colors)==len(Markers) and len(Markers) >=num):
            return True
        else:
            return False
    
