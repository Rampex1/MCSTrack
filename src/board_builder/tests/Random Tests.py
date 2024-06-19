def threeSum( nums: list[int]) -> list[list[int]]:
    # O(n^2)

    # for num in nums: target = num
    # 2sum hash dic to find if the rest equal to target
    output = []
    for index, target in enumerate(nums):
        hashDict = {}

        for otherIndex, otherNum in enumerate(nums):
            difference = target - otherNum
            if otherIndex == index:
                continue
            elif difference in hashDict:
                output.append([target, otherNum, hashDict[difference]])
            else:
                hashDict[difference] = otherNum
        continue

    return output

threeSum([-1,0,1,2,-1,-4])