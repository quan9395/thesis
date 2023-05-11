from base64 import b16decode
from binascii import b2a_base64
from unittest import result
import numpy as np
# import matplotlib.pyplot as plt
# from wtmedian import wtmedian
from kthsmallest import kthSmallest
import medianOfMedians as mm

import random
from operator import attrgetter
import math
import timeit
import matplotlib.pyplot as plt
import copy

# generate array of f, m is size of array
def generate_f(m):
    f = np.random.randint(1, 9999999, m)
    s = sum(f)
    r = [i / s for i in f]
    return r


def generate_f2(m):
    d = 10 ** (len(str(m)) + 2)
    rand = [1] * m
    rand2 = generate_f(m)  # generate array of randoms, sum is 1
    for i in range(0, m):
        rand[i] = rand2[i]
    rnd_array = np.random.multinomial(d, rand, size=m)[np.random.randint(0, m)]
    rnd = [i / (d) for i in rnd_array]
    return rnd




def sample_floats(low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(0, k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result


class endPoint:
    def __init__(self, x, o_x, y, link, flag, sumLeft, sumRight, sumInside, f, side, a, b, b_1, b_2, b_3):
        self.x = x
        self.o_x = o_x
        self.y = y
        self.link = link
        self.flag = flag
        self.sumLeft = sumLeft
        self.sumRight = sumRight
        self.sumInside = sumInside
        self.f = f
        self.side = side
        self.a = a
        self.b = b
        self.b_1 = b_1
        self.b_2 = b_2
        self.b_3 = b_3

def remove_endpoint(arr_Ii, size, n, i):  #n is number of uncertain points, i is number of locations
    arr_Ii[n][i], arr_Ii[n][size[n]-1] = arr_Ii[n][size[n]-1], arr_Ii[n][i]
    # new = swap_pos(arr_Ii[n], i, int(size[n]-1))
    # arr_Ii[n] = new
    size[n] = size[n] - 1
    arr_Ii[n][size[n]-1].a = 1
    return arr_Ii, size

def remove_uncertain_point(all_p, size_P, i):
    global global_arr_Ii
    del global_arr_Ii[i]
    del all_p[i]
    size_P = size_P - 1
    return all_p, size_P

def calculate_ed(I_i, endpoint, size, I_index):
    global sumL, sumR, sumI, b1, b2, b3
    ed_i = [0, 0]
    for l in range(0, size):
        if(I_i[l].side == 0 and I_i[l].x <= endpoint.x and I_i[l].link.x >= endpoint.x and I_i[l].flag == 0):
            endpoint.sumInside = endpoint.sumInside + I_i[l].f
            endpoint.b_3 = endpoint.b_3 + I_i[l].f * I_i[l].y
            I_i[l].link.flag = 1
            I_i[l].flag = 1
        if(I_i[l].side == 1 and I_i[l].x >= endpoint.x and I_i[l].link.x <= endpoint.x and I_i[l].flag == 0):
            endpoint.sumInside = endpoint.sumInside + I_i[l].f
            endpoint.b_3 = endpoint.b_3 + I_i[l].f * I_i[l].y
            I_i[l].link.flag = 1
            I_i[l].flag = 1
        if(I_i[l].side == 1 and I_i[l].x < endpoint.x and I_i[l].flag == 0):
            endpoint.sumLeft = endpoint.sumLeft + I_i[l].f
            endpoint.b_2 = endpoint.b_2 + I_i[l].f * I_i[l].o_x
            I_i[l].link.flag = 1
            I_i[l].flag = 1
        if(I_i[l].side == 0 and I_i[l].x > endpoint.x and I_i[l].flag == 0):
            endpoint.sumRight = endpoint.sumRight + I_i[l].f
            endpoint.b_1 = endpoint.b_1 + I_i[l].f * I_i[l].o_x
            I_i[l].link.flag = 1
            I_i[l].flag = 1
    endpoint.sumLeft = endpoint.sumLeft + sumL[I_index]
    endpoint.sumRight = endpoint.sumRight + sumR[I_index]
    endpoint.sumInside = endpoint.sumInside + sumI[I_index]
    endpoint.b_1 = endpoint.b_1 + b1[I_index]
    endpoint.b_2 = endpoint.b_2 + b2[I_index]
    endpoint.b_3 = endpoint.b_3 + b3[I_index]
    ed_i[0] = endpoint.sumLeft - endpoint.sumRight
    ed_i[1] = endpoint.b_1 - endpoint.b_2 + endpoint.b_3
    # print(format(endpoint.sumLeft, ".2f"), format(endpoint.sumRight, ".2f"), format(endpoint.sumInside, ".2f"))
    return ed_i


def findCenter(all_p, I, n, m):
    global sumL
    global sumR
    global sumI
    global b1
    global b2
    global b3
    global pruned_P

    count = 0
    for a in pruned_P:
        if(a == 0):
            count += 1
    if(count < 9):        #base case
        arr_Ii = I
        size = [2*m]*n
        x_L = -999999.0
        x_R = 999999.0
        for inner_loop in range(0, math.ceil(math.log2(m))+2): #logm
            total_size = 0
            for i in range(0,n):
                if(pruned_P[i] == 1):
                    continue
                total_size = total_size + size[i]

            I2 = []
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    continue
                for j in range(size[i]):
                    I2.append(arr_Ii[i][j].x)
            
            p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
            pivot = arr_Ii[0]
            for q in arr_Ii:
                for i in q:
                    if(p == i.x):
                        pivot = i
            # print(I2)


            all_ed = [] #find all ed_function that pivot goes through
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_ed.append(None)
                    continue
                temp = copy.deepcopy(pivot)
                temp_Ii = copy.deepcopy(arr_Ii)
                result = calculate_ed(temp_Ii[i], temp, size[i], i)
                all_ed.append(result)
            all_y = []  #calculate all y values and pick the highest
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_y.append(-math.inf)
                    continue
                temp = 0
                temp = all_ed[i][0]*pivot.x + all_ed[i][1]
                all_y.append(temp)
            highest_y_index = np.argmax(all_y)  

            if(all_ed[highest_y_index][0] < 0): #if slope negative
                #prune all endpoint to the left of pivot
                x_L = pivot.x
                a=0
                for i in range(0, n):
                    if(pruned_P[i] == 1):
                        continue
                    j = 0
                    while j < size[i]:
                        if(arr_Ii[i][j].x < x_L):
                            # print(arr_Ii[i][j].x, x_L)
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].flag == 0):
                                sumL[i] = sumL[i] + arr_Ii[i][j].f
                                b2[i] = b2[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].link.x <= x_L and arr_Ii[i][j].flag == 0):
                                sumL[i] = sumL[i] + arr_Ii[i][j].f
                                b2[i] = b2[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].link.x >= x_R and arr_Ii[i][j].flag == 0):
                                sumI[i] = sumI[i] + arr_Ii[i][j].f
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                                b3[i] = b3[i] + arr_Ii[i][j].f*arr_Ii[i][j].y  
                            
                            arr_Ii, size = remove_endpoint(arr_Ii, size, i, j)
                            a+=1
                        else:
                            j = j + 1

            elif(all_ed[highest_y_index][0] > 0):
                #prune all endpoint to the right of pivot
                x_R = pivot.x
                a = 0
                for i in range(0, n):
                    if(pruned_P[i] == 1):
                        continue
                    j = 0
                    while j < size[i]:
                        if(arr_Ii[i][j].x > x_R):
                            # print(arr_Ii[i][j].x, x_R)
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].flag == 0):
                                sumR[i] = sumR[i] + arr_Ii[i][j].f
                                b1[i] = b1[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].link.x >= x_R and arr_Ii[i][j].flag == 0):
                                sumR[i] = sumR[i] + arr_Ii[i][j].f
                                b1[i] = b1[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].link.x < x_L and arr_Ii[i][j].flag == 0):
                                sumI[i] = sumI[i] + arr_Ii[i][j].f
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                                b3[i] = b3[i] + arr_Ii[i][j].f*arr_Ii[i][j].y
                            arr_Ii, size = remove_endpoint(arr_Ii, size, i, j)
                            a+=1
                        else:
                            j = j + 1
                
            else:
                return pivot

        P_star_index = []
        for i in range(0, n):
            if(pruned_P[i] == 1):
                continue
            if(size[i] == 0):
                P_star_index.append(i)
        all_intersection = []
        for i in range(0, len(P_star_index)-1):
            for j in range(i + 1, len(P_star_index)):
                temp_intersection = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                a_one = sumL[P_star_index[i]] - sumR[P_star_index[i]]
                a_two = sumL[P_star_index[j]] - sumR[P_star_index[j]]
                b_one = b1[P_star_index[i]] - b2[P_star_index[i]] + b3[P_star_index[i]]
                b_two = b1[P_star_index[j]] - b2[P_star_index[j]] + b3[P_star_index[j]]
                temp_intersection.x = (a_one - a_two) / (b_two - b_one)
                temp_intersection.y = a_one*temp_intersection.x + b_one
                all_intersection.append(temp_intersection)
        all_results = []
        max_index=999
        for point in all_intersection:
            all_ed = [] #find all ed_function that goes through pivot
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_ed.append(None)
                    continue
                temp = copy.deepcopy(point)
                temp_Ii = copy.deepcopy(arr_Ii)
                result = calculate_ed(temp_Ii[i], temp, size[i], i)
                all_ed.append(result)

            all_y = []  #calculate all y values and pick the highest
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_y.append(-math.inf)
                    continue
                temp = 0
                temp = all_ed[i][0]*point.x + all_ed[i][1]
                all_y.append(temp)
            highest_y = max(all_y)
            all_results.append(highest_y)
        # print("len all res: ", len(all_results))
        q_star = min(all_results)
        # print(all_results)
        # print(q_star)
        min_index = all_results.index(q_star)
        # print("!!!!!", min_index)
        return(all_intersection[min_index].x)

    else:   #if not base case
        arr_Ii = I
        size = [2*m]*n
        x_L = -999999.0
        x_R = 999999.0
        for inner_loop in range(0, math.ceil(math.log2(m))+2): #logm

            total_size = 0
            for i in range(0,n):
                if(pruned_P[i] == 1):
                    continue
                total_size = total_size + size[i]
            # print(total_size)
            I2 = []
            for i in range(0, n): 
                if(pruned_P[i] == 1):
                    continue
                for j in range(size[i]):
                    I2.append(arr_Ii[i][j].x)
            
            p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
            pivot = arr_Ii[0]
            for q in arr_Ii:
                for i in q:
                    if(p == i.x):
                        pivot = i
            # print(I2)

            all_ed = [] #find all ed_function that pivot goes through
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_ed.append(None)
                    continue
                temp = copy.deepcopy(pivot)
                temp_Ii = copy.deepcopy(arr_Ii)
                result = calculate_ed(temp_Ii[i], temp, size[i], i)
                all_ed.append(result)

            all_y = []  #calculate all y values and pick the highest
            for i in range(0, n):
                if(pruned_P[i] == 1):
                    all_y.append(-math.inf)
                    continue
                temp = 0
                temp = all_ed[i][0]*pivot.x + all_ed[i][1]
                all_y.append(temp)
            highest_y_index = np.argmax(all_y)  

            if(all_ed[highest_y_index][0] < 0): #if slope negative
                #prune all endpoint to the left of pivot
                x_L = pivot.x
                a=0
                for i in range(0, n):
                    if(pruned_P[i] == 1):
                        continue
                    j = 0
                    while j < size[i]:
                        if(arr_Ii[i][j].x < x_L):
                            # print(arr_Ii[i][j].x, x_L)
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].flag == 0):
                                sumL[i] = sumL[i] + arr_Ii[i][j].f
                                b2[i] = b2[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].link.x <= x_L and arr_Ii[i][j].flag == 0):
                                sumL[i] = sumL[i] + arr_Ii[i][j].f
                                b2[i] = b2[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].link.x >= x_R and arr_Ii[i][j].flag == 0):
                                sumI[i] = sumI[i] + arr_Ii[i][j].f
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                                b3[i] = b3[i] + arr_Ii[i][j].f*arr_Ii[i][j].y  
                            
                            arr_Ii, size = remove_endpoint(arr_Ii, size, i, j)
                            a+=1
                        else:
                            j = j + 1

            elif(all_ed[highest_y_index][0] > 0):
                #prune all endpoint to the right of pivot
                x_R = pivot.x
                a = 0
                for i in range(0, n):
                    if(pruned_P[i] == 1):
                        continue
                    j = 0
                    while j < size[i]:
                        if(arr_Ii[i][j].x > x_R):
                            if(arr_Ii[i][j].side == 0 and arr_Ii[i][j].flag == 0):
                                sumR[i] = sumR[i] + arr_Ii[i][j].f
                                b1[i] = b1[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].link.x >= x_R and arr_Ii[i][j].flag == 0):
                                sumR[i] = sumR[i] + arr_Ii[i][j].f
                                b1[i] = b1[i] + arr_Ii[i][j].f*arr_Ii[i][j].o_x
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                            if(arr_Ii[i][j].side == 1 and arr_Ii[i][j].link.x < x_L and arr_Ii[i][j].flag == 0):
                                sumI[i] = sumI[i] + arr_Ii[i][j].f
                                arr_Ii[i][j].flag = 1
                                arr_Ii[i][j].link.flag = 1
                                b3[i] = b3[i] + arr_Ii[i][j].f*arr_Ii[i][j].y
                            
                            arr_Ii, size = remove_endpoint(arr_Ii, size, i, j)
                            a+=1
                        else:
                            j = j + 1
                
            else:
                return pivot

        P_star_index = []
        for i in range(0, n):
            if(pruned_P[i] == 1):
                continue
            if(size[i] == 0):
                P_star_index.append(i)

        if(len(P_star_index) == 5):
            all_intersection = []
            for i in range(0, len(P_star_index), 2):
                temp_intersection = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                if(i == len(P_star_index) - 1):
                    a_one = sumL[P_star_index[i]] - sumR[P_star_index[i]]
                    a_two = sumL[P_star_index[i-1]] - sumR[P_star_index[i-1]]
                    b_one = b1[P_star_index[i]] - b2[P_star_index[i]] + b3[P_star_index[i]]
                    b_two = b1[P_star_index[i-1]] - b2[P_star_index[i-1]] + b3[P_star_index[i-1]]
                    temp_intersection.x = (a_one - a_two) / (b_two - b_one)
                    temp_intersection.y = a_one*temp_intersection.x + b_one
                    all_intersection.append(temp_intersection)
                    break
                a_one = sumL[P_star_index[i]] - sumR[P_star_index[i]]
                a_two = sumL[P_star_index[i+1]] - sumR[P_star_index[i+1]]
                b_one = b1[P_star_index[i]] - b2[P_star_index[i]] + b3[P_star_index[i]]
                b_two = b1[P_star_index[i+1]] - b2[P_star_index[i+1]] + b3[P_star_index[i+1]]
                temp_intersection.x = (a_one - a_two) / (b_two - b_one)
                temp_intersection.y = a_one*temp_intersection.x + b_one
                all_intersection.append(temp_intersection)
        else:
            all_intersection = []
            for i in range(0, len(P_star_index), 2):
                temp_intersection = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                if(i == len(P_star_index) - 1):
                    break
                a_one = sumL[P_star_index[i]] - sumR[P_star_index[i]]
                a_two = sumL[P_star_index[i+1]] - sumR[P_star_index[i+1]]
                b_one = b1[P_star_index[i]] - b2[P_star_index[i]] + b3[P_star_index[i]]
                b_two = b1[P_star_index[i+1]] - b2[P_star_index[i+1]] + b3[P_star_index[i+1]]
                temp_intersection.x = (a_one - a_two) / (b_two - b_one)
                temp_intersection.y = a_one*temp_intersection.x + b_one
                all_intersection.append(temp_intersection)
        
        

        I2 = []
        for i in range(len(all_intersection)):
            I2.append(all_intersection[i].x)
        p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
        pivot = all_intersection[0]
        for i in all_intersection:
            if(p == i.x):
                pivot = i

        all_ed = [] #find all ed_function that pivot goes through
        for i in range(0, n):
            if(pruned_P[i] == 1):
                all_ed.append(None)
                continue
            temp = copy.deepcopy(pivot)
            temp_Ii = copy.deepcopy(arr_Ii)
            result = calculate_ed(temp_Ii[i], temp, size[i], i)
            all_ed.append(result)

        all_y = []  #calculate all y values and pick the highest
        for i in range(0, n):
            if(pruned_P[i] == 1):
                all_y.append(-math.inf)
                continue
            temp = 0
            temp = all_ed[i][0]*pivot.x + all_ed[i][1]
            all_y.append(temp)
        highest_y_index = np.argmax(all_y)  

        if(all_ed[highest_y_index][0] < 0): #if slope negative
            temp1 = 0
            for i in range(len(all_intersection)):
                if(all_intersection[i].x <= pivot.x):
                    if(len(P_star_index) == 5 and i == 2):
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] < sumL[P_star_index[2*i-1]] - sumR[P_star_index[2*i-1]]):
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            pruned_P[P_star_index[2*i-1]] = 1
                        temp1+=1 
                    else:  
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] < sumL[P_star_index[2*i+1]] - sumR[P_star_index[2*i+1]]):
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            pruned_P[P_star_index[2*i+1]] = 1
                        temp1+=1
        elif(all_ed[highest_y_index][0] > 0): #if slope positive
            temp2 = 0
            for i in range(len(all_intersection)):
                if(all_intersection[i].x >= pivot.x):
                    if(len(P_star_index) == 5 and i == 2):
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] > sumL[P_star_index[2*i-1]] - sumR[P_star_index[2*i-1]]):
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            pruned_P[P_star_index[2*i-1]] = 1
                        temp2+=1
                    else:
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] > sumL[P_star_index[2*i+1]] - sumR[P_star_index[2*i+1]]):
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            pruned_P[P_star_index[2*i+1]] = 1
                        temp2+=1
        else:
            return pivot.x
        
        findCenter(all_p, I, n, m)
                    
        
        
        














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
global_arr_Ii = []
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
        left = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        right = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        sample_point = {'x':0, 'o_x':0, 'y':0, 'f':f_array[0], 'left':left, 'right':right}
        new = dict(sample_point)
        new['x'] = coordinates_x[a]
        new['y'] = coordinates_y[a]
        a+=1
        new['f'] = all_f[j][i]
        new['left'].x = new['x'] - abs(new['y'])
        new['left'].o_x = new['x']
        new['left'].y = new['y']
        new['left'].f = new['f']
        new['right'].x = new['x'] + abs(new['y'])
        new['right'].o_x = new['x']
        new['right'].y = new['y']
        new['right'].f = new['f']
        new['right'].link = new['left']
        new['left'].link = new['right']
        p1.append(new)
    all_p.append(p1)
sumL = [0]*n
sumR = [0]*n
sumI = [0]*n
b1 = [0]*n
b2 = [0]*n
b3 = [0]*n
pruned_P = [0]*n
for p in all_p:    #put endpoints in arrays
    temp = []
    for i in p:
        arr_I.append(i['left'])
        arr_I.append(i['right'])
        temp.append(i['left'])
        temp.append(i['right'])
    global_arr_Ii.append(temp)


start = timeit.default_timer()

x = findCenter(all_p, global_arr_Ii, n, m)

stop = timeit.default_timer()


print('Time: ', stop - start)  


x_axis = [16,
25,
36,
49,
64,
81,
100,
121,
144,
169,
196,
225,
256,
289,
324,
1024]
y_axis = [0.0276803,
0.0483245,
0.1042969,
0.115845,
0.2920408,
0.4728452,
0.6370588,
0.9841723,
1.4445859,
1.698481,
2.9762274,
4.0777136,
5.6589318,
6.7934823,
8.3717052,
1024]

# plt.plot(x_axis, y_axis)
# plt.show()