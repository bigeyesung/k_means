import matplotlib.pyplot as plt
from operator import truediv
from sklearn import utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, classification_report
from utils import*

class KMeans:
    def __init__(self, csvPath, k=3, tol=0.0001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.utils = Utils(csvPath)
        if self.utils.init:
            self.utils.SetData(0.25)
            X_train, X_test, Y_train, Y_test=self.utils.GetData()
            self.utils.SetGroups(self.k)
            self.utils.SetMedaData(self.k)

    def SetInitCenters(self, norm):
        data = self.utils.X_train
        dimension = len(data[0])-1
        origin = np.array([0]*dimension)
        distances = [np.linalg.norm(featureSet[0:-1]-origin, ord=norm) for featureSet in data]
        for ind in range(len(distances)):
            newData = [[distances[ind]],data[ind].tolist(),ind]
            distances[ind] = newData

        #sort array based on distances with origin
        distances.sort(key=lambda x: x[0])

        #assign to k groups
        grpSize = len(data)//self.k
        for ind in range(self.k):
            group = distances[grpSize*(ind):grpSize*(ind)+grpSize]
            middle = len(group)//2
            initInd = group[middle][-1]
            self.centroids[ind] = data[initInd][0:-1]

    def Fit(self, norm):
        trainCorr=0
        trainErr=0
        grp_true=[]
        grp_pred=[]
        saveResults = []
        data, _, _, _=self.utils.GetData()
        self.centroids = {}

        #Set initial cluster centers based on sorting data and finding middle
        self.SetInitCenters(norm)

        # for i in range(self.k):
        #     self.centroids[i] = data[i][0:-1]
            # self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = collections.defaultdict(list)
            
            for featureSet in data:
                distances = [np.linalg.norm(featureSet[0:-1]-self.centroids[centroid], ord=norm) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureSet)
            prev_centroids = dict(self.centroids)

            #Find the new center by taking the average of the points in each cluster group
            for classification in self.classifications:
                tmpData=[]
                for itemData in self.classifications[classification]:
                    tmpData.append(itemData[0:-1])
                    # tmpData.append(itemData)
                self.centroids[classification] = np.average(tmpData,axis=0)

            optimized = True
            thresHold=0
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # tmp1 = current_centroid*100-original_centroid*100
                # tmp2 = original_centroid*100
                # thresHold = abs(np.sum(tmp1/tmp2))
                # print(np.sum((current_centroid-original_centroid)/original_centroid))
                thresHold = abs(np.sum((current_centroid-original_centroid)/original_centroid,dtype=float))
                if thresHold > self.tol:
                    # print("cur: ", np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                # print("cur revised: ", tmp)
                break
                
        #Set classGroup Names
        self.utils.SetClassName(self.classifications)

        classGroupNames=list(self.utils.classNames.values())
        for classification in self.classifications:
            # print(classification)
            predicted = self.utils.classGroups[classification]
            for featureSet in self.classifications[classification]:
                grp_true.append(featureSet[-1])
                grp_pred.append(self.utils.classNames[predicted])
                newList=featureSet.tolist()
                newList.append(self.utils.classNames[predicted])
                saveResults.append(newList)
                if self.utils.classNames[predicted]==featureSet[-1]:
                    trainCorr+=1
                else:
                    trainErr+=1
                plt.scatter(featureSet[0], featureSet[1], marker=self.utils.markers[predicted], color=self.utils.colors[predicted], s=150, linewidths=5)

        #Get training accuracy
        print("Train accu: ",float(trainCorr/(trainCorr+trainErr))*100)
        print("Train err: ",float(trainErr/(trainCorr+trainErr))*100)
        report = classification_report(grp_true, grp_pred, labels=classGroupNames)
        print(report)
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1],
                        marker="*", color="k", s=150, linewidths=5)
        self.utils.SetLegends()
        plt.legend(handles=self.utils.legends)                        
        plt.savefig(self.utils.dir + '/images/'+ self.utils.fileName+ '_train_predicted.jpg',bbox_inches='tight')
        plt.show()

        #save train result
        self.utils.SaveCSV('_train_predicted.csv', saveResults)

    def Predict(self, data, norm):
        tmpData = data[0:-1]
        distances = [np.linalg.norm(tmpData-self.centroids[centroid], ord=norm) for centroid in self.centroids]
        classification = distances.index(min(distances))
        # predictGrp=self.utils.classGroups[classification]
        return classification

    def Evaluate(self, norm):
        grp_true=[]
        grp_pred=[]
        testCorr=0
        testErr=0
        saveResults=[]
        _, X_test, _, _=self.utils.GetData()
        for ind in range(len(X_test)):
            featureSet=X_test[ind]
            grp_true.append(featureSet[-1])
            pred=self.utils.classNames[self.Predict(featureSet, norm)]
            grp_pred.append(pred)
            newList=featureSet.tolist()
            newList.append(pred)
            saveResults.append(newList)
            if pred==featureSet[-1]:
                testCorr+=1
            else:
                testErr+=1

        #Get test accuracy
        print("Test accu: ",float(testCorr/(testCorr+testErr))*100)
        print("Test err: ",float(testErr/(testCorr+testErr))*100)        
        classGroupNames=list(self.utils.classNames.values())
        report = classification_report(grp_true, grp_pred, labels=classGroupNames)
        print(report)

        #save test result
        self.utils.SaveCSV('_test_predicted.csv', saveResults)


        #######
        # mat = confusion_matrix(grp_true, grp_pred, labels=tags)
        # matSize = len(tags)
        # # print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
        # # print(homogeneity_score(grp_true, grp_pred))
        # # print(completeness_score(grp_true, grp_pred))
        # # print(v_measure_score(grp_true, grp_pred))
        # # print(adjusted_rand_score(grp_true, grp_pred))
        # # print(adjusted_mutual_info_score(grp_true, grp_pred))
        # for ind in range(len(tags)):
        #     print("this is: ", tags[ind])
        #     tp,tn,fp,fn=0,0,0,0
        #     tp = mat[ind][ind]
        #     for row in range(matSize):
        #         for col in range(matSize):
        #             if row!=ind and col !=ind:
        #                 tn+=mat[row][col]
        #     for col in range(matSize):
        #         if col!=ind:
        #             fp+=mat[ind][col]
        #     for row in range(matSize):
        #         if row!=ind:
        #             fn+=mat[row][ind]
        #     # print("tp: ",tp )
        #     # print("tn: ",tn )
        #     # print("fp: ",fp )
        #     # print("fn: ",fn )
        #     print("precision: ",tp/(tp+fp) )
        #     print("recall: ", tp/(tp+fn))
        #     Precision = tp/(tp+fp)
        #     Recall=tp/(tp+fn)
        #     F1 = 2 * Precision * Recall / (Precision + Recall)
        #     print("F1: ", F1)