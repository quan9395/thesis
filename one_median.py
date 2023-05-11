import numpy as np
#import matplotlib.pyplot as plt
#from wtmedian import wtmedian
from kthsmallest import kthSmallest
import medianOfMedians as mm

import random
from operator import attrgetter
import math
import timeit
import matplotlib.pyplot as plt


#generate array of f, m is size of array
def generate_f(m):
    f = np.random.randint(1, 9999999, m)
    s = sum(f)
    r = [ i/s for i in f ]
    return r

def generate_f2(m):
    d = 10**(len(str(m)) + 2)
    rand = [1]*m
    rand2 = generate_f(m)   #generate array of randoms, sum is 1
    for i in range(0,m):
        rand[i] = rand2[i]
    rnd_array = np.random.multinomial(d, rand, size=m)[np.random.randint(0,m)]
    rnd = [i/(d) for i in rnd_array]
    return rnd

def calculate_slope(I, pivot, flag):
    global temp
    global all_p
    if flag == 0:
        for k in all_p:
            for l in k:
                if l['left'].x < pivot.x and l['right'].x >= pivot.x:
                    pivot.sumI = pivot.sumI + l['f']
        for i in I:
            if i.side == 1 and pivot.x > i.x:
                pivot.sumLeft = pivot.sumLeft + i.f
        for j in reversed(I):
            if j.side == 0 and pivot.x <= j.x:
                pivot.sumRight = pivot.sumRight + j.f
        print("%.2f" % pivot.sumLeft, "%.2f" % pivot.sumRight, "%.2f" % pivot.sumI)
    if flag == 1:
        temp2 = temp
        I.append(temp2)
        for i in I:
            if i.side == 1 and pivot.x < i.x:
                temp.sumLeft = temp.sumLeft - i.f
        pivot.sumLeft = temp.sumLeft
        for i in reversed(I):
            if i.side == 0 and pivot.x <= i.x:
                pivot.sumRight = pivot.sumRight + i.f
            pivot.sumRight = pivot.sumRight + temp.sumRight
    if flag == 2:
        temp2 = temp
        I.append(temp2)
        for i in I:
            if i.side == 1 and pivot.x > i.x:
                pivot.sumLeft = pivot.sumLeft + i.f
        pivot.sumLeft = pivot.sumLeft + temp.sumLeft
        for i in I:
            if i.side == 0 and pivot.x > i.x:
                temp.sumRight = temp.sumRight - i.f
        pivot.sumRight = temp.sumRight
    temp = pivot
    slope = pivot.sumLeft - pivot.sumRight
    return slope

def base_case(endpoint):
    global all_p
    for k in all_p:
        for l in k:
            if l['left'].x < endpoint.x and l['right'].x >= endpoint.x:
                endpoint.sumI = endpoint.sumI + l['f']
            if l['right'].x < endpoint.x:
                endpoint.sumLeft = endpoint.sumLeft + l['f']
            if l['left'].x >= endpoint.x:
                endpoint.sumRight = endpoint.sumRight + l['f']        
        # print("%.2f" % endpoint.sumLeft, "%.2f" % endpoint.sumRight, "%.2f" % endpoint.sumI)
    slope = endpoint.sumLeft - endpoint.sumRight
    return slope

def findMedian(I):
    print(len(I))
    if(len(I) < 5):
        I.sort(key=lambda x: x.x, reverse=False)
        for b in range(len(I) - 1):
            slope = base_case(I[b])
            slope2 = base_case(I[b+1])
            if(slope * slope2 < 0):
                print('median found base')
                return I[b]
            if((b == len(I) - 2) and slope * slope2 >= 0):
                return 99999

    I2 = []
    for i in I:
        I2.append(i.x)

    p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
    pivot = I[0]
    for i in I:
        if(p == i.x):
            pivot = i


    I_l=[]
    I_r=[]
    I_c=[]

    for i in I:     #partition into 3 array I
        if(i.x < pivot.x):
            I_l.append(i)
        if(i.x > pivot.x):
            I_r.append(i)
    I_c.append(pivot)

    slope = calculate_slope(I, pivot, 0)
    

    if(len(I_r) == 0):
        print("No neighbor")
        print("%.2f" % slope)
        return pivot
    else:
        neighbor = min(I_r, key=attrgetter('x'))
    if(pivot.side == 1 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft + pivot.f
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 1 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight
    print("%.2f" % slope, "%.2f" % slope2)
    if(slope * slope2 < 0):
        print("Median found")
        return pivot

    if(slope < 0):
        return(findMedianR(I_r))

    if(slope > 0):
       return(findMedianL(I_l))

def findMedianL(I):
    print(len(I))
    if(len(I) < 5):
        I.sort(key=lambda x: x.x, reverse=False)
        for b in range(len(I) - 1):
            slope = base_case(I[b])
            print(slope)
            slope2 = base_case(I[b+1])
            print(slope2)
            if(slope * slope2 < 0):
                print('median found base 1')
                return I[b]
            if((b == len(I) - 2) and slope * slope2 >= 0):
                return 99999

    if(len(I) == 1):
        return I[0]
    I2 = []
    for i in I:
        I2.append(i.x)

    p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
    pivot = I[0]
    for i in I:
        if(p == i.x):
            pivot = i


    I_l=[]
    I_r=[]
    I_c=[]

    for i in I:     #partition into 3 array I
        if(i.x < pivot.x):
            I_l.append(i)
        if(i.x > pivot.x):
            I_r.append(i)
    I_c.append(pivot)


    slope = calculate_slope(I, pivot, 1)


    if(len(I_r) == 0):
        print("No neighbor1")
        print("%.2f" % slope)
        return pivot
    else:
        neighbor = min(I_r, key=attrgetter('x'))
    if(pivot.side == 1 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft + pivot.f
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 1 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight
    print("%.2f" % slope, "%.2f" % slope2)

    if(slope * slope2 < 0):
        print("Median found1")
        return pivot

    if(slope < 0):
       return(findMedianR(I_r))

    if(slope > 0):
       return(findMedianL(I_l))

def findMedianR(I):
    print(len(I))
    if(len(I) < 5):
        I.sort(key=lambda x: x.x, reverse=False)
        for b in range(len(I) - 1):
            slope = base_case(I[b])
            print(slope)
            slope2 = base_case(I[b+1])
            print(slope2)
            if(slope * slope2 < 0):
                print("median found base 2")
                return I[b]
            if((b == len(I) - 2) and slope * slope2 >= 0):
                return 99999
    if(len(I) == 1):
        return I[0]
    I2 = []
    for i in I:
        I2.append(i.x)

    p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
    pivot = I[0]
    for i in I:
        if(p == i.x):
            pivot = i



    I_l=[]
    I_r=[]
    I_c=[]

    for i in I:     #partition into 3 array I
        if(i.x < pivot.x):
            I_l.append(i)
        if(i.x > pivot.x):
            I_r.append(i)
    I_c.append(pivot)

    slope = calculate_slope(I, pivot, 2)

    if(len(I_r) == 0):
        print("No neighbor2")
        print("%.2f" % slope)
        return pivot
    else:
        neighbor = min(I_r, key=attrgetter('x'))
    if(pivot.side == 1 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft + pivot.f
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 1 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 0):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight
    if(pivot.side == 0 and neighbor.side == 1):
        neighbor.sumLeft = pivot.sumLeft
        neighbor.sumRight = pivot.sumRight - pivot.f
        slope2 = neighbor.sumLeft - neighbor.sumRight    

    print("%.2f" % slope, "%.2f" % slope2)
    if(slope * slope2 < 0):
        print("Median found2")
        return pivot

    if(slope < 0):
       return(findMedianR(I_r))

    if(slope > 0):
       return(findMedianL(I_l))

def sample_floats(low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result

class endPoint:
    def __init__(self, x, sumLeft, sumRight, sumI, f, side):
        self.x = x
        self.sumLeft = sumLeft
        self.sumRight = sumRight
        self.sumI = sumI
        self.f = f
        self.side = side

#def findMedian(I, sumLeft, sumRight):

print("How many uncertain points? (n) ")
n = int(input())
print("How many locations for each point? (m) ")
m = int(input())

rows, cols = (n, m)
all_f = [[0 for i in range(cols)] for j in range(rows)]

for j in range(0, n):       #generate f array
    f_array = generate_f2(m)
    for i in range(0, m):
        all_f[j][i] = f_array[i]


arr_I = []
arr_Ii = []
all_p = []
randX = []
randY = []
coordinates_x1 = np.random.uniform(-m*n, m*n, m*n)
coordinates_x = sorted(coordinates_x1)
coordinates_y = np.random.uniform(-m*n, m*n, m*n)
a=0
for j in range(0,n):        #generate all_p
    p1=[]
    for i in range(0,m):
        left = endPoint(0,0,0,0,0,0)
        right = endPoint(0,0,0,0,0,1)
        sample_point = {'x':0, 'y':0, 'f':f_array[0], 'left':left, 'right':right}
        new = dict(sample_point)
        new['x'] = coordinates_x[a]
        new['y'] = coordinates_y[a]
        a+=1
        new['f'] = all_f[j][i]
        new['left'].x = new['x'] - abs(new['y'])  #todo: include negative value for y (abs)
        new['left'].f = new['f']
        new['right'].x = new['x'] + abs(new['y'])
        new['right'].f = new['f']
        p1.append(new)
    #new_p = sorted(p1, key=itemgetter('x'))
    all_p.append(p1)

for p in all_p:    #put endpoints in arrays
    for i in p:
        arr_I.append(i['left'])
        arr_I.append(i['right'])

for j in range (0,n):
    for i in range(0,m):
        print(all_p[j][i]['x'],all_p[j][i]['y'],all_p[j][i]['f'], all_p[j][i]['left'].x, all_p[j][i]['right'].x)
    print("======")

all_locations = []         # all locations listed
for i in all_p:
    for j in i:
        all_locations.append(j)
        #print(j['left'].x)

arr_I = []     
for i in all_locations:
    arr_I.append(i['left'])
    arr_I.append(i['right'])

temp = arr_I[0]

start = timeit.default_timer()

result = findMedian(arr_I)

stop = timeit.default_timer()
print(result)
print('Time: ', stop - start)  

# print("%.2f" % result.x)

x_axis = [4,
8,
16,
32,
64,
128,
256,
512,
1024,
2048,
4096,
8192,
16384,
32768,
65536]
y_axis = [0.0014139,
0.0036763,
0.0016631,
0.0019943,
0.0083076,
0.0034168,
0.0140393,
0.0192641,
0.0196574,
0.0318777,
0.0549914,
0.0922377,
0.1992713,
0.4066119,
0.9010901]

plt.plot(x_axis, y_axis)
plt.show()