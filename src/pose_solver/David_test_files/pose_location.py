import numpy as np

class PoseLocation:
    def __init__(self):
        self.__RVEC = np.zeros(3)
        self.__TVEC = np.zeros(3)

        self.__RVEC_list = []
        self.__TVEC_list = []

    def add_RVEC(self, new_RVEC):
        self.__RVEC_list.append(new_RVEC)

        RVEC_mean = np.zeros(3)
        for i in range(3):
            RVEC_mean[i] = np.mean([rvec[i] for rvec in self.__RVEC_list])

        self.__RVEC = RVEC_mean

    def add_TVEC(self, new_TVEC):
        self.__TVEC_list.append(new_TVEC)

        TVEC_mean = np.zeros(3)
        for i in range(3):
            TVEC_mean[i] = np.mean([tvec[i] for tvec in self.__TVEC_list])

        self.__TVEC = TVEC_mean

    def get_RVEC(self):
        return np.array(self.__RVEC)

    def get_TVEC(self):
        return np.array(self.__TVEC)


"""
pose = PoseLocation()
pose.add_RVEC([1,2,3])
pose.add_RVEC([4,5,6])

pose.add_TVEC([4,4,4])
pose.add_TVEC([4,5,6])

RVEC = pose.get_RVEC()
TVEC = pose.get_TVEC()
print(RVEC)
print(TVEC)

"""

