from NumberClusterFinder import NumberClusterFinder
import numpy as np
import pandas as pd
from math import dist
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


class DensityKmeans:
    def __init__(self, data):
        self.data = data
        self.min_pts = NumberClusterFinder(data).findMinPts()
        self.cluster_value = None
        self.dbscan = None
        self.dense_point = None
        self.low_dense_point = None
        self.delete_point = []
        self.remain_point = []
        self.center_point = None
        self.labels_ = None
        self.generateEps()
        self.kmeans = None

    def generateEps(self):
        n = 2
        labelsNumber = -1
        while labelsNumber <= 0:
            self.eps = NumberClusterFinder(self.getData()).find(n)
            self.dbscan = DBSCAN(eps=self.getEps(), min_samples=self.min_pts)
            self.dbscan.fit(self.getData())
            labelsNumber = self.dbscan.labels_.max()
            # print("labelsNumber")
            # print(labelsNumber)
            n += 1

    def getData(self):
        return self.data

    def getEps(self):
        return self.eps

    def getClusterValue(self):
        return self.cluster_value

    def getDensePoint(self):
        return self.dense_point

    def getLowDensePoint(self):
        return self.low_dense_point

    def getDeletePoint(self):
        return self.delete_point

    def getRemainPoint(self):
        return self.remain_point

    def getCenterPoint(self):
        return self.center_point

    def GenerateClusterValue(self):
        self.dbscan = DBSCAN(eps=self.getEps(), min_samples=self.min_pts)
        self.dbscan.fit(self.getData())
        self.cluster_value = pd.concat(
            [pd.DataFrame(self.getData()), pd.DataFrame(self.dbscan.labels_, columns=['cluster'])],
            axis=1)

    def calculateMemberCorePoint(self, cluster_values, cluster_number):
        core_point_arrays = cluster_values.groupby('cluster').get_group(cluster_number).drop('cluster',
                                                                                             axis=1).to_numpy()
        core_points = dict()
        for pi in core_point_arrays:
            point = 0
            for pj in core_point_arrays:
                if dist(pi, pj) < self.getEps():
                    point += 1
            core_points[point] = pi
        return core_points

    def findCenterPoint(self, point1, point2):
        n = len(point1)
        result = [0 for i in range(n)]
        for i in range(n):
            result[i] = (point1[i] + point2[i]) / 2
        return [result[i] for i, v in enumerate(result)]

    def compressData(self, dens_values):
        final_point = []
        not_check = []
        iterate = 0
        middlePoint = []
        for kpi, pi in dens_values.items():
            middlePoint = pi
            dens = 0
            continues = 0
            for key, check in enumerate(not_check):
                if (pi == check).all():
                    continues = 1
                    break
            if continues == 1:
                continue
            for kpj, pj in dens_values.items():
                if (dist(middlePoint, pj) < self.getEps() and dist(middlePoint, pj) != 0):
                    middlePoint = self.findCenterPoint(middlePoint, pj)
                    not_check.append(pj)
                    dens += 1
            middlePoint = np.append(middlePoint, dens)
            final_point.append(np.ndarray.tolist(np.array(middlePoint)))
        return final_point

    def generateDensePoint(self):
        all_data = []
        for i in range(self.dbscan.labels_.max() + 1):
            dens_value = dict(sorted(self.calculateMemberCorePoint(self.getClusterValue(), i).items(), reverse=True))
            all_data.append(self.compressData(dens_value))

        result = []
        for i, vi in enumerate(all_data):
            for j, vj in enumerate(vi):
                result.append(vj)

        arrays_dens = np.array(result)

        self.dense_point = arrays_dens[np.where(arrays_dens[:, len(arrays_dens[0, :]) - 1] >= 3)]
        self.low_dense_point = arrays_dens[np.where(arrays_dens[:, len(arrays_dens[0, :]) - 1] < 3)]
        # print("self.dense_point")
        # print(self.dense_point)
        # print("self.low_dense_point")
        # print(self.low_dense_point)

    def cleaningLowDensePoint(self):
        low_dense_point = self.getLowDensePoint()
        zero_dens_point = low_dense_point[:, 0:len(low_dense_point[0, :]) - 1]
        for key, value in enumerate(self.getData()):
            for zkey, zval in enumerate(zero_dens_point):
                if dist(zval, value) < self.getEps():
                    self.delete_point.append(value)

        # print("len(self.getData())")
        # print(len(self.getData()))
        # print("len(self.delete_point)")
        # print(len(self.delete_point))
        for key, value in enumerate(self.getData()):
            length = 0
            for dkey, dval in enumerate(self.delete_point):
                if (dval == value).all():
                    continue
                else:
                    length += 1
            if length == len(self.delete_point):
                self.remain_point.append(value)
        # print("self.remain_point")
        # print(self.remain_point)
        self.remain_point = np.array(self.remain_point)

    def findCenterPointCluster(self, cluster_values, cluster_number):
        center_points = []
        for i in range(cluster_number):
            mid_point = []
            denses = cluster_values.groupby('cluster').get_group(i).drop('cluster', axis=1).to_numpy()
            for key, value in enumerate(denses):
                if key == 0:
                    mid_point = value
                if (key != len(denses)):
                    mid_point = self.findCenterPoint(mid_point, value)
            center_points.append(mid_point)
        return center_points

    def generateInitialPoint(self):
        m = DBSCAN(eps=self.getEps(), min_samples=self.min_pts)
        m.fit(self.getRemainPoint())
        dens_cluster_value = pd.concat(
            [pd.DataFrame(self.getRemainPoint()), pd.DataFrame(m.labels_, columns=['cluster'])],
            axis=1)
        self.center_point = self.findCenterPointCluster(dens_cluster_value, max(m.labels_) + 1)
        self.center_point = np.array(self.center_point)

    def kmeansClustering(self):
        kmeans = KMeans(n_clusters=len(self.getCenterPoint()), init=self.getCenterPoint(),
                        n_init=1)
        # kmeans = KMeans(n_clusters=len(self.getCenterPoint()), init='k-means++')
        kmeans.fit(self.getData())
        self.kmeans = kmeans
        return kmeans.labels_

    def getKmeans(self):
        return self.kmeans

    def fit(self):
        self.GenerateClusterValue()
        self.generateDensePoint()
        self.cleaningLowDensePoint()
        self.generateInitialPoint()
        self.labels_ = self.kmeansClustering()
