import numpy as np

class KNN(object):
    def __init__(self):
        pass

    def knn_distances(self,xTest,k):
        distances = -2 * xTrain@xTest.T + np.sum(xTest**2,axis=1) + np.sum(xTrain**2,axis=1)[:, np.newaxis]
        #because of float precision, some small numbers can become negatives. Need to be replace with 0.
        distances[distances < 0] = 0
        distances = np.sqrt(distances)
        indices = np.argsort(distances, 0) #get indices of sorted items
        distances = np.sort(distances,0) #distances sorted in axis 0
        #returning the top-k closest distances.
        return indices[0:k,:], distances[0:k,:]
    
    def knn_predictions(self, xTrain,yTrain,xTest,k=3):
        import numpy as np
        indices, distances = self.knn_distances(xTrain,xTest,k)
        yTrain = yTrain.flatten()
        rows, columns = indices.shape
        predictions = list()
        for j in range(columns):
            temp = list()
            for i in range(rows):
                cell = indices[i][j]
                temp.append(yTrain[cell])
            predictions.append(max(temp,key=temp.count)) #this is the key function, brings the mode value
        predictions=np.array(predictions)
        return predictions