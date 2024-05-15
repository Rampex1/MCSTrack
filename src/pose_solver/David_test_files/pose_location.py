class PoseLocation:
    def __init__(self):

        self.__RVEC = 0
        self.__TVEC = 0

        self.__RVEC_list = []
        self.__TVEC_list = []

    def add_RVEC(self, new_RVEC):
        self.__RVEC_list.append(new_RVEC)

        RVEC_mean = []
        RVEC_sum = 0
        for i in range(3):
            for j in range(len(self.__RVEC_list)):
                RVEC_sum += self.__RVEC_list[j][i]
            RVEC_mean.append(RVEC_sum / len(self.__RVEC_list))
            RVEC_sum = 0

        self.__RVEC = RVEC_mean

    def add_TVEC(self, new_TVEC):
        self.__TVEC_list.append(new_TVEC)

        TVEC_mean = []
        TVEC_sum = 0
        for i in range(3):
            for j in range(len(self.__TVEC_list)):
                TVEC_sum += self.__TVEC_list[j][i]
            TVEC_mean.append(TVEC_sum / len(self.__TVEC_list))
            TVEC_sum = 0

        self.__TVEC = TVEC_mean

    def get_RVEC(self):
        return self.__RVEC

    def get_TVEC(self):
        return self.__TVEC

"""
pose = PoseLocation(1)
pose.add_RVEC([1,2,3])
pose.add_RVEC([4,5,6])

pose.add_TVEC([4,4,4])
pose.add_TVEC([4,5,6])

RVEC = pose.get_RVEC()
TVEC = pose.get_TVEC()
print(RVEC)
print(TVEC)
"""

