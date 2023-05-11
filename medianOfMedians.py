import math
def partition5(list, left, right):
    i = left + 1
    while i <= right :
        j = i
        while (j > left) and (list[j-1] > list[j]) :
            temp = list[j-1]    #swap list[j−1] and list[j]
            list[j-1] = list[j]
            list[j] = temp
            j = j - 1
        i =  i + 1
    return math.floor((left + right) / 2)

def select(list, left, right, n):
    while 1>0:
        if left == right:
            return left
        pivotIndex = pivot(list, left, right)
        pivotIndex = partition(list, left, right, pivotIndex, n)
        if n == pivotIndex:
            return n
        elif n < pivotIndex:
            right = pivotIndex - 1
        else:
            left = pivotIndex + 1

def pivot(list, left, right):
    #for 5 or less elements just get median
    if (right - left) < 5:
        return partition5(list, left, right)
    #otherwise move the medians of five-element subgroups to the first n/5 positions
    for i in range(left, right, 5):
        # get the median position of the i'th five-element subgroup
        subRight = i + 4
        if subRight > right:
            subRight = right
        median5 = partition5(list, i, subRight)
        temp = list[median5]    #swap list[median5] and list[left + floor((i − left)/5)]
        list[median5] = list[left + math.floor((i-left)/5)]
        list[left + math.floor((i-left)/5)] = temp

    #compute the median of the n/5 medians-of-five
    mid = (right - left) / 10 + left + 1
    return select(list, left, left + math.floor((right-left)/5), mid)


def partition(list, left, right, pivotIndex, n):
    pivotValue = list[pivotIndex]
    temp = list[pivotIndex]         #swap list[pivotIndex] and list[right]  // Move pivot to end
    list[pivotIndex] = list[right]
    list[right] = temp
    storeIndex = left
    #Move all elements smaller than the pivot to the left of the pivot
    for i in range(left, right):
        if list[i] < pivotValue:
            temp = list[storeIndex]         #swap list[storeIndex] and list[i]
            list[storeIndex] = list[i]
            list[i] = temp
            storeIndex +=1
    # Move all elements equal to the pivot right after
    # the smaller elements
    storeIndexEq = storeIndex
    for i in range (storeIndex, right):
        if list[i] == pivotValue:
            temp = list[storeIndexEq]       #swap list[storeIndexEq] and list[i]
            list[storeIndexEq] = list[i]
            list[i] = temp
            storeIndexEq+=1
    temp = list[right]
    list[right] = list[storeIndexEq]
    list[storeIndexEq] = temp       #    swap list[right] and list[storeIndexEq]  // Move pivot to its final place
    # Return location of pivot considering the desired location n
    if n < storeIndex :
        return storeIndex  # n is in the group of smaller elements
    if n <= storeIndexEq :
        return n  # n is in the group equal to pivot
    return storeIndexEq # n is in the group of larger elements


