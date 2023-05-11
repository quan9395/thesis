from kthsmallest import kthSmallest
# Python3 implementation of worst case 
# linear time algorithm to find
# k'th smallest element
# Returns k'th smallest element in arr[l..r]
# in worst case linear time.
# ASSUMPTION: ALL ELEMENTS IN ARR[] ARE DISTINCT
from operator import itemgetter
def wtmedian(arr, l, r, balance, k):
    # If k is smaller than number of
    # elements in array
    if (k > 0 and k <= r - l + 1):
         
        # Number of elements in arr[l..r]
        n = r - l + 1
 
        # Divide arr[] in groups of size 5,
        # calculate median of every group
        # and store it in median[] array.
        median = []
 
        i = 0
        while (i < n // 5):
            median.append(findMedian(arr, l + i * 5, 5))
            i += 1
 
        # For last group with less than 5 elements
        if (i * 5 < n):
            median.append(findMedian(arr, l + i * 5, n % 5))
            i += 1
 
        # Find median of all medians using recursive call.
        # If median[] has only one element, then no need
        # of recursive call
        if i == 1:
            medOfMed = median[i - 1]
        else:
            medOfMed = kthSmallest(median, 0, i - 1, i // 2)
        # Partition the array around a medOfMed
        # element and get position of pivot
        # element in sorted array
        pos = partition(arr, l, r, medOfMed)

        #calculate wtsum1, wtsum2, wtsum3
        wtsum1 = 0
        wtsum2 = 0
        wtsum3 = 0
        for i in range(l,r+1):
            if(i<pos):
                wtsum1 += arr[i]['f']
            elif(i == pos):
                wtsum2 += arr[i]['f']
            else:
                wtsum3 += arr[i]['f']

        if(wtsum1 + balance >= wtsum2 + wtsum3):
            return(wtmedian(arr, l, pos-1, balance - wtsum2 - wtsum3, k))
        elif(wtsum1 + wtsum2 + balance >= wtsum3):
            return arr[pos]
        else:
            return(wtmedian(arr, pos+1, r, balance + wtsum1 + wtsum2, k))
 
    # If k is more than the number of
    # elements in the array
    return 999999999999
 
def swap(arr, a, b):
    temp = arr[a]
    arr[a] = arr[b]
    arr[b] = temp
 
# It searches for x in arr[l..r], 
# and partitions the array around x.
def partition(arr, l, r, x):
    for i in range(l, r):
        if arr[i]['x'] == x:
            swap(arr, r, i)
            break
 
    x = arr[r]['x']
    i = l
    for j in range(l, r):
        if (arr[j]['x'] <= x):
            swap(arr, i, j)
            i += 1
    swap(arr, i, r)
    return i
 
# A simple function to find
# median of arr[] from index l to l+n
def findMedian(arr, l, n):
    lis = []
    for i in range(l, l + n):
        lis.append(arr[i])   
    # Sort the array
    d = sorted(lis, key=itemgetter('x'))
    # Return the middle element
    return d[n // 2]
 
