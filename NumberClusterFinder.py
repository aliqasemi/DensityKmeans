import numpy as np
from sklearn.neighbors import NearestNeighbors


class NumberClusterFinder:
    def __init__(self, data):
        self.data = data
        self.distances = None
        self.eps = None

    def getData(self):
        return self.data

    def findMinPts(self):
        return int(len(self.data) / 200)

    def generateDistance(self):
        neigh = NearestNeighbors(n_neighbors=self.findMinPts())
        nbrs = neigh.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)
        distances = np.sort(distances, axis=0)
        self.distances = distances[:, 1]
        return distances

    def getDistances(self):
        return self.distances

    def findDiffStd(self):
        diff = np.array([])
        for key, value in enumerate(self.distances):
            if key != len(self.distances) - 1:
                diff = np.append(diff, abs(value - self.distances[key + 1]))

        diffStd = diff.std()

        for key, value in enumerate(diff):
            if value > 3 * diffStd:
                self.eps = self.distances[key]
                break

    def getEps(self):
        return self.eps

    def find(self):
        self.findMinPts()
        self.generateDistance()
        self.findDiffStd()
        return self.getEps()
