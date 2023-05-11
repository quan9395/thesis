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
    print("|P|: ", count)
    if(count < 9):        #base case
        arr_Ii = I
        size = [2*m]*n
        x_L = -999999.0
        x_R = 999999.0
        for inner_loop in range(0, math.ceil(math.log2(m))+2): #logm
            print(size)
            total_size = 0
            for i in range(0,n):
                if(pruned_P[i] == 1):
                    continue
                total_size = total_size + size[i]
            print(total_size)
            print("X_range: ", format(x_L, ".2f"), format(x_R, ".2f"))
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
            print("pivot: ", pivot.x, "at index: ", I2.index(p))

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
                print(all_ed)
                return pivot
            print("remove: ", a)

        print(size)
        print("X_range: ", format(x_L, ".2f"), format(x_R, ".2f"))
        P_star_index = []
        for i in range(0, n):
            if(pruned_P[i] == 1):
                continue
            if(size[i] == 0):
                P_star_index.append(i)
        print("Length P*: ", len(P_star_index), ", ", P_star_index)
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
        print("len all inter: ", len(all_intersection))
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
        print("q* x coordinate is: ", all_intersection[min_index].x)
        return(all_intersection[min_index].x)

    else:   #if not base case
        arr_Ii = I
        size = [2*m]*n
        x_L = -999999.0
        x_R = 999999.0
        for inner_loop in range(0, math.ceil(math.log2(m))+2): #logm
            print(size)
            total_size = 0
            for i in range(0,n):
                if(pruned_P[i] == 1):
                    continue
                total_size = total_size + size[i]
            print(total_size)
            print("X_range: ", format(x_L, ".2f"), format(x_R, ".2f"))
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
            print("pivot: ", pivot.x, "at index: ", I2.index(p))

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
                print(all_ed)
                return pivot
            print("remove: ", a)

        print(size)
        print("X_range: ", format(x_L, ".2f"), format(x_R, ".2f"))
        P_star_index = []
        for i in range(0, n):
            if(pruned_P[i] == 1):
                continue
            if(size[i] == 0):
                P_star_index.append(i)
        print("Length P*: ", len(P_star_index), ", ", P_star_index)

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
        
        print("len all inter: ", len(all_intersection))
        

        I2 = []
        for i in range(len(all_intersection)):
            I2.append(all_intersection[i].x)
        p = kthSmallest(I2, 0, len(I2) - 1, math.ceil(len(I2)/2))
        pivot = all_intersection[0]
        for i in all_intersection:
            if(p == i.x):
                pivot = i
        print("pivot: ", pivot.x, "at index: ", I2.index(p))

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
                            if(pruned_P[P_star_index[2*i]] == 1):
                                print("1a!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            if(pruned_P[P_star_index[2*i-1]] == 1):
                                print("1b!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i-1]] = 1
                        temp1+=1 
                    else:  
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] < sumL[P_star_index[2*i+1]] - sumR[P_star_index[2*i+1]]):
                            if(pruned_P[P_star_index[2*i]] == 1):
                                print("1a!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            if(pruned_P[P_star_index[2*i+1]] == 1):
                                print("1b!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i+1]] = 1
                        temp1+=1
            print("remove1: ", temp1)
        elif(all_ed[highest_y_index][0] > 0): #if slope positive
            temp2 = 0
            for i in range(len(all_intersection)):
                if(all_intersection[i].x >= pivot.x):
                    if(len(P_star_index) == 5 and i == 2):
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] > sumL[P_star_index[2*i-1]] - sumR[P_star_index[2*i-1]]):
                            if(pruned_P[P_star_index[2*i]] == 1):
                                print("2a!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            if(pruned_P[P_star_index[2*i-1]] == 1):
                                print("2b!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i-1]] = 1
                        temp2+=1
                    else:
                        if(sumL[P_star_index[2*i]] - sumR[P_star_index[2*i]] > sumL[P_star_index[2*i+1]] - sumR[P_star_index[2*i+1]]):
                            if(pruned_P[P_star_index[2*i]] == 1):
                                print("2a!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i]] = 1
                        else:
                            if(pruned_P[P_star_index[2*i+1]] == 1):
                                print("2b!!!!!!!!!!!!!!!!!!!!!!", i)
                            pruned_P[P_star_index[2*i+1]] = 1
                        temp2+=1
            print("remove2: ", temp2)
        else:
            return pivot.x
        
        findCenter(all_p, I, n, m)
                    
        
        
        













n = 16
m = 32


rows, cols = (n, m)
all_f = [[0 for i in range(cols)] for j in range(rows)]
all_f = [[0.0237, 0.0382, 0.0019, 0.0422, 0.044, 0.0277, 0.0013, 0.0228, 0.0069, 0.0226, 0.0522, 0.0261, 0.0443, 0.059, 0.0586, 0.0013, 0.0289, 0.0508, 0.0593, 0.0397, 0.0447, 0.0107, 0.0273, 0.0148, 0.0603, 0.0083, 0.041, 0.0023, 0.0627, 0.0289, 0.0311, 0.0164], [0.0314, 0.0308, 0.0352, 0.0349, 0.036, 0.0066, 0.0356, 0.033, 0.0482, 0.0014, 0.0247, 0.006, 0.0476, 0.0216, 0.0123, 0.056, 0.0376, 0.0045, 0.0179, 0.0352, 0.0089, 0.0559, 0.0369, 0.0425, 0.0265, 0.0383, 0.0526, 0.0679, 0.0136, 0.0211, 0.0536, 0.0257], [0.0248, 0.0505, 0.0044, 0.0029, 0.0578, 0.0689, 0.002, 0.0121, 0.0268, 0.0031, 0.0188, 0.0371, 0.0405, 0.0271, 0.0285, 0.0461, 0.0364, 0.0217, 0.0508, 0.042, 0.0341, 0.0217, 0.0488, 0.0432, 0.0233, 0.0286, 0.0319, 0.0072, 0.0499, 0.025, 0.043, 0.041], [0.0045, 
0.0176, 0.0489, 0.02, 0.0223, 0.0596, 0.0343, 0.0575, 0.0205, 0.0311, 0.0067, 0.0319, 0.0593, 0.0133, 0.0152, 0.0192, 0.0273, 0.0618, 0.0278, 0.0141, 0.0437, 0.063, 0.0027, 0.03, 0.0102, 0.0344, 0.0296, 0.0485, 0.0434, 0.0165, 0.0598, 0.0253], [0.0232, 0.0373, 0.0186, 0.0347, 0.0221, 0.0335, 0.0141, 0.0478, 0.0337, 0.0532, 0.0095, 0.0097, 0.0263, 0.065, 0.0397, 0.0025, 0.0059, 0.0563, 0.0399, 0.0596, 0.0466, 0.0256, 0.0109, 0.0182, 0.0571, 0.0445, 0.0172, 0.0246, 0.0547, 0.0089, 0.0448, 0.0143], [0.006, 0.0031, 0.055, 0.052, 0.0401, 0.019, 0.0136, 0.0053, 0.0597, 0.0438, 0.0286, 0.0277, 0.0467, 0.03, 0.0423, 0.0115, 0.0214, 0.0433, 0.0284, 0.0422, 0.0513, 0.0222, 0.0181, 0.0405, 0.02, 0.0188, 0.0368, 0.0307, 0.0434, 0.0164, 0.0329, 0.0492], [0.0364, 0.0388, 0.04, 0.0458, 0.0525, 0.0194, 0.0162, 0.0516, 0.0519, 0.0366, 0.0457, 0.0363, 0.0212, 0.036, 0.0519, 0.0238, 0.0162, 0.0294, 0.0308, 0.0165, 0.0352, 0.0061, 0.0482, 0.009, 0.0319, 0.0095, 0.0259, 0.0363, 0.0067, 0.0336, 0.051, 0.0096], [0.0283, 0.0087, 0.0509, 0.0018, 0.0311, 0.0547, 0.0563, 0.0282, 0.0387, 0.0587, 0.0031, 0.0512, 0.0034, 0.0511, 0.0449, 0.027, 0.0089, 0.0263, 0.0475, 0.0276, 0.0085, 0.0403, 0.0026, 0.0639, 0.0523, 0.0077, 0.0654, 0.0054, 0.023, 0.0083, 0.0607, 0.0135], [0.069, 0.0346, 0.0468, 0.008, 0.0416, 0.0051, 0.0138, 0.0692, 0.0152, 0.0209, 0.0237, 0.0438, 0.0337, 0.0572, 0.0382, 0.0232, 0.0059, 0.0378, 0.0404, 0.0108, 0.0589, 0.0438, 0.006, 0.0112, 0.0225, 0.0282, 0.0264, 0.0061, 0.0573, 0.047, 0.0488, 0.0049], [0.0239, 0.0315, 0.0044, 0.0292, 0.0158, 0.0129, 0.017, 0.0435, 0.0441, 0.051, 0.0097, 0.0468, 0.0166, 0.046, 0.0294, 0.0392, 0.0364, 0.0205, 0.0467, 0.0071, 0.0375, 0.0325, 0.0543, 0.0514, 0.0531, 0.0381, 0.0522, 0.0254, 0.014, 0.0246, 0.0338, 0.0114], [0.053, 0.0495, 0.004, 0.0252, 0.0465, 0.037, 0.0204, 0.0541, 0.0205, 0.0533, 0.0273, 0.0192, 0.0107, 0.0124, 0.0256, 0.0067, 0.0515, 0.0416, 0.0045, 0.0371, 0.0103, 0.0583, 0.036, 0.028, 0.0424, 0.0113, 0.0466, 0.0267, 0.0594, 0.0542, 0.0159, 0.0108], [0.0006, 0.0049, 0.0681, 0.0211, 0.0194, 0.0355, 0.0157, 0.003, 0.0499, 0.0126, 0.0023, 0.0479, 0.0407, 0.0083, 0.0163, 0.0181, 0.0561, 0.0017, 0.071, 0.0656, 0.0006, 0.0222, 0.0523, 0.0388, 0.0134, 0.0627, 0.0441, 0.0447, 0.0275, 0.0638, 0.0594, 0.0117], [0.0323, 0.0003, 0.041, 0.0054, 0.0372, 0.0406, 0.0626, 0.0353, 0.0133, 0.0159, 0.0101, 0.0367, 0.0163, 0.0506, 0.055, 0.0179, 0.0289, 0.0125, 0.0509, 0.0406, 0.046, 0.0382, 0.0152, 0.0102, 0.0609, 0.0181, 0.0561, 0.0411, 0.031, 0.0157, 0.051, 0.0131], [0.0415, 0.0064, 0.0526, 0.0256, 0.0667, 0.0166, 0.0205, 0.0587, 0.0342, 0.0151, 0.0016, 0.0463, 0.0006, 0.0065, 0.0076, 0.029, 0.0667, 0.0361, 0.072, 0.0209, 0.0215, 0.0386, 0.009, 0.0552, 0.0271, 0.0644, 0.0046, 0.0447, 0.0574, 0.0439, 0.0012, 0.0072], [0.0554, 0.0385, 0.0327, 0.0194, 0.033, 0.0603, 0.0275, 0.033, 0.026, 0.002, 0.0118, 0.003, 0.0456, 0.0592, 0.0047, 0.0319, 0.046, 0.039, 0.0007, 0.0452, 0.0525, 0.0622, 0.0175, 0.0499, 0.0096, 0.0559, 0.0023, 0.0173, 0.0403, 0.0259, 0.0511, 0.0006], [0.0627, 0.0182, 0.0038, 0.0513, 0.0276, 0.0359, 0.0448, 0.0045, 0.0405, 0.0586, 0.0309, 0.0394, 0.0443, 0.046, 0.0308, 0.0255, 0.0479, 0.0249, 0.0037, 0.0197, 0.0423, 0.0217, 0.0411, 0.0519, 0.003, 0.0187, 0.0141, 0.0267, 0.0123, 0.0377, 0.031, 0.0385]]

arr_I = []
global_arr_Ii = []
all_p = []
randX = []
randY = []
coordinates_x = [-510.77013986273676, -509.84848616724923, -505.7335944689828, -505.18615568352004, -503.5969326483049, -503.1256951854816, -502.6868903328767, -501.3050845696416, -497.26019696139247, -493.76098538586655, -493.22313238243737, -490.47151622509443, -489.3740307722678, -487.29245472809816, -486.4924338832684, -479.38193574395075, -478.92991142714595, -478.81882497039453, -477.31911035385554, -474.8300637329015, -472.7071198523157, -466.8024066811678, -465.8848755227441, -465.20021482211337, -464.6957398137573, -463.6523622821529, -463.2514524551191, -460.53926236941425, -455.77371852963756, -453.53021523331427, -453.0121638922765, -452.3011925451764, -446.82766935627365, -445.06369622988234, -437.7878970025513, -434.696224970457, -433.68861898500245, -430.6861683554233, -430.4394554159845, -429.12524980951036, -427.31984205346146, -424.67079506552295, -424.1059585185087, -421.44343311537034, -421.3581700148511, -421.150976059002, -418.6121352813674, -417.6045478912813, -413.95307664913537, -411.4877841297874, -405.81780962165055, -404.1501999696279, -400.7611000836695, -397.58332195158687, -397.3858814613899, -395.64533597528145, -394.17201169712234, -389.93574821226264, -385.87313298910317, -385.0543807545506, -382.39825193263914, -382.07640075966106, -381.87535672123965, -379.4158982442252, -378.62054035450535, -376.73102640482773, -374.0733984707457, -371.2895682702348, -371.2053998518303, -361.10992743343036, -359.7362523179527, -358.3389138636709, -356.00476335408257, -354.1509585828386, -352.0680097743549, -346.0598652201111, -345.22589168249783, -344.53146044351433, -342.9498513810464, -341.2227062176157, -338.47770939769623, -332.8627516665763, -332.14326951045166, -332.0143281223129, -331.95472791536656, -331.7472784185678, -330.6581067513913, -328.7464383067786, -328.1387344317852, -326.4890995544014, -325.3520113973699, -323.2389515214086, -321.94599961896415, -321.41943883202214, -311.27386896963037, -310.96410131518235, -305.28837247892216, -302.8323514116721, -302.4892297473267, -298.87066024985404, -297.0581790073767, -295.29537202617746, -294.46440188882343, -293.4527937509565, -293.36600968079085, -293.0972680454199, -288.978690301455, -285.86062375355937, -285.04826182476575, -285.02243018755644, -283.39673393782937, -283.3599710519517, -282.2170599586135, -281.4476516936437, -280.6519045396203, -280.61323754710077, -278.6774774436751, -273.02764212554996, -272.93128968737517, -270.5674818837001, -270.34394030255567, -270.03275619920043, -269.5499005040524, -269.1658155062913, -265.6462637961538, -260.14485669961186, -260.13990389752064, -258.22908772229755, -256.51022061107153, -251.80421740758743, -250.1357894355075, -249.76245059791927, -246.72944191918077, -245.91987251329817, -244.6277429211833, -244.0131096049406, -236.7716008571059, -236.26488787128085, -235.5416083049613, -234.9554154810869, -234.53463092855793, -226.89075113461672, -226.1598676235261, -225.33818584373807, -220.4386205424422, -220.02023059829673, -219.5845377898563, -218.87625454537454, -218.10683171901007, -216.37812769892867, -212.64649001469445, -209.27779794067055, -206.07828941833247, -199.43238837876095, -197.58543270795735, -187.94513099424125, -187.78844925379076, -187.01555910791035, -184.41063849472914, -183.5883746194803, -171.2573495385534, -165.609247369753, -165.58738221570707, -156.17175975648627, -156.16825748028282, -154.87784597223344, -154.11328050963198, -152.2934649322898, -152.17583778039102, -151.1636580600383, -148.7978436100442, -147.99449542883133, -145.19481731557175, -143.86324993825997, -141.53595305030717, -139.4290855141063, -135.0544382036776, -133.93384976059713, -131.9850564338093, -125.97437405332596, -125.96134809512023, -125.70707334705025, -125.66379542041, -125.05847278722933, -124.8242938800031, -124.40243361520288, -123.83839907995821, -120.79423837279069, -119.33472535664396, -115.0660605757547, -111.77113369904635, -108.97589726432875, -108.18376921010258, -102.77534137669261, -100.80765997483729, -100.76440075906874, -98.80864790492615, -98.44774356055927, -97.58506807578487, -97.35831194908826, -93.46966948071122, -92.02072050464028, -84.13351122179347, -83.7104474927429, -83.32614266238897, -82.66510621862551, -78.45235682785176, -76.6920304035674, -75.90253289303462, -65.29513390994896, -63.55700697785824, -53.144477906952375, -52.75834653019956, -47.4180348002518, -46.98578202273529, -46.00460675347779, -45.800433774779776, -43.60219553125057, -43.351311411907545, -43.27562113649094, -41.8568563291451, -41.50477386526666, -37.88548760252513, -36.88656796800058, -34.391316960163294, -33.50253908263926, -33.208651882613594, -30.752723197583464, -26.290391177845095, -22.239108145537557, -19.357997294997517, -17.194204642576437, -14.910621322021825, -12.394996311038994, -10.317018836634361, -10.189094237217773, -7.5125490910860435, -5.886772759360042, -1.3642722239515024, -0.5956897444895048, 3.7015255093650694, 3.8769871523526263, 11.854116084542284, 12.519467420693331, 17.336163558336125, 
17.40581160952081, 17.957094129090706, 20.835593839416106, 25.808658744724994, 27.710172478233858, 29.550494829434683, 30.768583660700642, 35.77053549270977, 36.0431673180542, 36.907973520490714, 37.468745995488575, 38.24488230894451, 39.73685543289844, 41.39486996442622, 45.66528800824722, 47.83989781891307, 50.57408874086423, 50.661357165817776, 50.70745966720074, 51.50610083442473, 56.84822314960377, 57.00322917485505, 59.57160834913748, 64.1092038943201, 64.99731741345283, 66.56569033519861, 67.16873712573886, 68.04816054639355, 69.55601848502226, 70.56131097158686, 73.02015861926145, 76.4811258954129, 79.59716917567596, 81.81975632109379, 83.49062958956256, 85.2742356448947, 86.4274794942952, 86.52738501699423, 87.49371060568888, 88.33933475308754, 91.50167871080782, 91.60810605439701, 91.80172090962355, 94.55684430417489, 99.62708509461083, 100.17090909039621, 100.23721560105207, 102.45562804166218, 104.56250526444478, 106.7239063811885, 107.51780509918046, 109.82525347838839, 112.81007303608476, 115.20592233518698, 119.03104557426695, 127.36394784722972, 127.4005836214186, 129.74086961922637, 130.15641710358045, 137.09958464426927, 137.85375635497985, 138.9858057108073, 140.9194973134887, 140.9273653231944, 141.86125428415573, 142.92167921309056, 142.9772259799479, 143.45566521996523, 144.27079322435918, 145.07516507077014, 148.9522661976664, 149.0456358246289, 149.1543835066907, 150.42719541079725, 150.73826090501154, 151.3630622469576, 153.47542434801392, 153.69281632222908, 155.46664902842224, 155.56168190459982, 158.7246822155222, 160.65931025168072, 165.45455746062862, 165.985276392846, 170.1927516803737, 175.54099595806554, 177.40635015120677, 179.84819748330392, 180.79393164220699, 182.02986519119577, 184.42919950107614, 185.38740118005114, 186.68180398117488, 188.1592660083711, 191.16291915799047, 195.90817998313594, 204.17821600770708, 206.09595216211676, 206.1852174680879, 206.8646518931996, 210.50690385386622, 210.55196734476783, 210.95948541717178, 212.86760247073323, 220.9924049544917, 221.70453905323643, 224.43459847115116, 225.1915059944513, 226.91778904643706, 230.59479986562826, 232.5758707155279, 233.99372099218056, 238.94857847769492, 238.98584119599252, 239.39682155413004, 240.59978188507137, 241.52560074866688, 244.6521940380968, 244.94681493487997, 245.07196724747223, 247.32435284383962, 248.54718471894978, 248.63449486168338, 260.37756003196614, 261.26501067517415, 261.6969493109342, 262.80471488348667, 262.92321388715004, 263.9360837816307, 268.07687331698776, 272.902603119938, 273.326470214507, 275.85286631191127, 276.3523673132828, 277.5766055614863, 280.62935771049024, 283.5204917065812, 288.0796245165468, 293.9379091474955, 296.9517350304927, 299.6575539352916, 300.47149442299667, 302.5877021025551, 302.7398963522686, 304.8612786914964, 310.77483927207993, 312.774866057145, 314.3938844503425, 315.69011261047547, 316.02611964381094, 316.8386133817462, 321.6320133626623, 321.7875748300954, 322.9620259324289, 326.51208402694726, 327.9235482748593, 328.6744404040328, 329.0614032997247, 331.27802879916646, 331.4527958616329, 332.2345516403378, 332.53417386697254, 333.98617709574444, 335.98667738653296, 337.2756870274691, 337.8102813336726, 338.49440328305354, 342.7810988868981, 346.28916134212625, 346.6426730310982, 349.03583472316177, 350.4291653289049, 354.33198936987503, 355.05170725492553, 358.71586658306296, 360.1356641345143, 364.34568167229554, 366.1494144481934, 369.7471814288024, 370.36458001459675, 371.38457907640566, 373.5204578314688, 377.0362810669436, 379.643626887054, 381.3464362295956, 384.969790531479, 385.1326567314925, 385.5189210094918, 387.82314954489607, 388.25671899126974, 388.4379054689126, 389.9953697806119, 390.03687861750143, 390.08515458283455, 391.1159157916944, 394.27002310410546, 394.38409478403196, 395.07108708656676, 401.6799156678271, 404.6263751052337, 404.6732770012768, 405.227386180571, 409.1206861809768, 411.28713232241125, 411.4527203769429, 411.5096441815424, 415.4082870500464, 415.58699378536437, 417.6164211357544, 418.7208279960455, 421.3402759582518, 423.65691094180295, 424.2862758139189, 426.1418850137069, 429.6202577244633, 431.6694643664853, 434.8902791735695, 435.4959577813896, 435.60922789521453, 437.7593850398813, 438.1898800601367, 440.1108807301998, 440.3663046403932, 442.9823079891901, 444.79425919424443, 450.26682291910765, 452.76591831510837, 454.1465502809565, 454.6745356585126, 454.7503424634235, 456.51752015657314, 457.7669288850484, 457.81654040204455, 462.7361053537371, 462.80258814482545, 464.61418488923175, 465.53252950621015, 467.2792115071594, 467.99184782119016, 469.2894806883203, 472.27320310153095, 475.9675256771735, 477.0925400590936, 478.7045158593129, 479.6479419657527, 482.3268024109225, 484.58237166340996, 484.79775241806635, 485.240267382609, 486.60614342083227, 487.9952930027623, 488.4518188533908, 489.9412457852335, 492.5857820636239, 494.7970067675898, 501.6086643462521, 503.51172145153953, 503.6059015480248, 503.9894895757593, 504.60842082325416, 506.84595783901466, 507.9287246107725, 508.04380790426364, 509.5785191886986, 510.1770017478526, 510.2079595091974, 511.87646497059666]

coordinates_y = [-370.25685888,  149.86445241,  193.38499896,  432.60684813,  127.40518202,
  -21.75514137,  -98.66535165,  359.53868602, -178.27688226, -348.72855891,
 -411.15771462,  319.60702148,  338.84564728,  255.18593454,   -3.13699939,
   84.8144034,  -263.76092932, -143.16458599,  339.60571642,  171.64163559,
  378.11406438, -436.51977562, -331.92119815, -312.84643948,  226.53577277,
  251.55621829,  214.86342367, -202.46240555, -415.71282348,  341.75542127,
  310.77116232,  400.86818897,  425.79105713,  292.57359639,  149.95063214,
  329.14672894,  -85.96861308, -445.63008785,  -61.39187065,  443.7305599,
 -336.72417267,  138.87179886,  356.39137468,   44.06822068, -234.12251725,
  122.4922344,  -254.79916641,  -96.56551946,  302.33746069, -227.21554604,
  208.69898389,  143.95839631,  191.13058138, -239.09190854, -283.22359106,
 -247.80833964,  -47.09587529,  245.43879864,   81.79576967, -267.65759264,
   77.67062202,   -6.59399597,  475.82622252,  455.68517169,  457.5574802,
  464.81645313,   77.36827055,  432.33012064,  393.55938907, -397.83587201,
  250.94229502,   45.80244816, -454.80557683,  394.28926312,  171.34894112,
  483.6986709,   -15.88611719,  441.03440159,  122.57165948,   13.32887988,
 -320.67252675, -335.12388731, -353.08547151,  -61.93929108,  288.89198399,
  295.79063549,  332.65821298,  380.30266428,   17.75719523,   77.10198618,
 -355.81250947,   66.26642066,   47.77916083,  427.59035084,  -43.47570011,
  132.49380023, -340.18881582,  480.71432831, -143.05690102, -307.58258488,
 -278.56609202, -133.56989258,  477.43725304,  384.67748875, -257.48920346,
 -371.23117771, -244.27621711,  -90.95894098,  152.4433042,  -378.59784747,
  317.00849399, -418.74547396,   37.03819511, -454.89450202,    8.27146523,
 -288.45555972,  224.77771642, -225.82310419, -237.45397667,  -47.45448589,
  -17.98507565, -496.37765359, -388.11055286, -400.06680219,   51.30507069,
  383.98873704,  310.72864782,   12.59840957,  179.64569217, -351.47356932,
  465.32915561,  476.37813569, -367.30850707,  406.54494226,  431.09320854,
   31.97596391,  159.0871058,  -141.75281245,  457.62676009,   81.42931361,
 -223.08457473,  408.09232052, -193.37055799,   -7.68548436,   96.92713927,
 -191.34027889, -451.09130116,  455.08681773, -493.04146146, -266.38520331,
   93.96786547, -198.20639792, -332.11751974, -173.56962492,  389.59498312,
 -324.66537166, -470.4864433,   253.2099438,   491.12019471,  -25.78289435,
 -267.82685008, -131.90896179,    8.85181037, -259.38491981,  286.01902473,
  274.69985656,  122.39984045,  437.99057852, -190.2076301,  -217.74446581,
  199.44982098,  -48.87143965,   22.54803364, -299.23390149,  270.20847188,
  336.36098219, -432.2245767,   367.56695369,  495.40774519, -439.44310684,
  411.88517045,  427.24182156, -506.35580135,  233.03115959,  -49.22374256,
   65.84700707, -385.95269431, -392.41684443,  290.96647256, -156.7881508,
  487.72332942,   41.63934024, -201.88156894,   36.63311985,  -15.58976878,
 -316.77358938,  283.84674605,  224.72031257,  332.4904691,  -185.33886216,
 -212.38415231, -243.67703124, -336.02125686,   34.08032105, -214.41977989,
  438.17181126,  -56.82195349,  274.88326408,  417.72575305, -148.52105821,
 -127.57305913, -407.15318251, -162.32612018, -103.15637298, -474.28177924,
 -320.5321021,    63.42753657,  -88.92205516, -475.10298618, -139.99742842,
   42.21426311,  -78.50419287,  169.91093681, -237.78172564, -504.56955276,
 -404.471232,    296.43347411,  177.30659746,  -52.2558569,   429.8968294,
 -229.75158954,  229.33141519,  228.59239174,   51.90868302, -156.04851162,
 -251.79646399,  252.14195436, -250.60721235, -137.7925428,   415.09968389,
  235.98226946, -378.23909942, -293.80106984,   88.00858739,  511.31878293,
 -168.34367775,  176.88102944, -136.3602329,   132.15747542,  478.03332171,
  -29.82937295, -116.10992764, -392.86276386,  419.04432285,   64.77728377,
 -104.52806277,  252.88747546,   61.77397136,  391.10581577, -361.17318538,
 -233.72369675, -318.13280531,  409.37442503, -109.35273808,  171.65802551,
  143.24427949,  493.28852024, -139.44691496, -399.13938201,   67.64388748,
 -158.29488329,   28.20931541, -503.16337241,  143.57452854, -395.03768951,
  273.12180005,  414.4171334,   334.74278487, -362.3119606,   193.42266738,
 -267.99297878, -485.92507654, -468.75110626, -296.68986237,  439.49684111,
   70.56941783, -184.08680522,  225.20028006, -440.09393711, -300.41705432,
  163.03657112,  122.89758679,  278.61996822, -290.51959312, -404.28519408,
 -279.55934404,  221.90617844, -467.91845215,  126.72191164,  210.16001761,
  296.90454047,  184.55265368, -245.69421043,  331.16799153,   48.39034057,
 -257.34140059,  427.2815375,  -413.74395586, -248.12212416, -305.17115511,
 -246.35151307,  258.62821584,  411.48208048, -140.30839156,  151.23733998,
 -509.18962134,  412.20359927,  253.62578905,  424.1663072,  -316.65682574,
  491.08579454,  374.81351827,  373.60855365,  243.06497398,  -25.98204279,
 -258.62542954,  144.93921416,  416.38293581,  219.78592165, -478.99319425,
 -211.48263636, -204.63879144, -443.36740953,  303.36053981,  187.29258555,
  410.87473155,   95.07941125, -364.02772035, -507.35836698,  235.22906313,
 -235.06238191, -393.74138104, -355.98392474,  287.87399218,   14.27830378,
  444.65014712,  384.25158282,  117.95276459, -476.92014378,  220.61558783,
  495.19548319, -119.10332418,  153.724857,    144.67751009, -259.16078705,
  397.09548213,  146.77572066,  -28.77338084,  392.39011738,    9.89289472,
  422.41298098,  -33.43758852,  232.96483245,  365.24483645, -249.60204989,
 -468.03414657, -191.20699721,   89.55525774,  361.28721157, -188.72192622,
 -386.31765453,   36.38809096,  432.53313173, -209.19558974,   56.72383734,
 -479.25743751,   97.47319411,  216.85590412,  397.50504384,  256.65639835,
  507.32693736,  491.3489588,   142.96314746, -109.24863582,   38.63843534,
  296.3294366,   131.10992559, -377.47120237, -256.517764,    328.06154567,
 -222.65432257, -205.50910901,  369.55746834,  348.43927469,   20.99516821,
  445.66815443, -130.91354542, -469.05043676, -229.52826102,  176.19001929,
 -248.94075837,  397.5869148,   208.21560605, -307.37253167,  296.46644038,
   32.27718405, -209.20150697,  234.22677315, -345.76583249,  -99.4243554,
 -201.25902633, -419.74501188, -114.35135688,  201.512449,   -194.06491846,
 -354.06619222, -200.710189,    451.69069707,  -34.47289127,  132.58507386,
 -386.73323459, -295.98691861,   34.11925309,  329.38596115, -425.54837606,
   73.92587357,  -38.33393472,  445.13475715,  -95.04849801, -215.18841918,
   89.73759117,  393.79653561, -104.59791553, -356.19924629, -260.72671215,
  344.07054274,  418.70857668,  -96.71723326,  365.03746189,  475.72114749,
  260.1974462,   296.31483719, -233.53412726,  399.08435632,  270.96196028,
  339.40614234,  494.46644057,  291.2587202,  -270.6488971,  -327.92296981,
   -8.46849292,   26.92605915, -249.37222109,  222.12176954,  179.95477877,
 -349.71881258,  171.39491651,  251.18869053, -174.17129853, -152.35079383,
   45.93640286,  -79.91159291,  463.13405164, -207.95365855,  132.02541562,
  469.50250057,  362.68103706,  133.87430326, -126.4176441,   238.23095481,
 -289.4107282,  -384.06002572, -335.87928606, -126.22213405,  158.42388095,
  486.57828778,   84.86939397,  264.64496543,  376.27386929,  262.62314721,
  -58.69100205, -167.30554187,  225.67656209,   50.34554596, -496.95712651,
  254.49313752, -182.21895419, -119.92029621, -247.01529255,  118.63359892,
  -29.56535255, -461.05364492,   61.56164641, -459.5326234,  -273.60304555,
  410.01437215, -444.67361384,  -40.65719147,  -39.76490309,  465.29348798,
   83.86419307,  156.22892086,  168.69862446, -338.98254994, -340.15065135,
  240.06015178, -459.33340609,  314.85533981, -185.19370807, -308.3425807,
 -310.98523722, -136.99641055]

a=0
for j in range(0,n):        #generate all_p
    p1=[]
    for i in range(0,m):
        left = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        right = endPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        sample_point = {'x':0, 'o_x':0, 'y':0, 'f':all_f[0][0], 'left':left, 'right':right}
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


for j in range(0, n):
    for i in range(0, m):
        print(format(all_p[j][i]['x'],".2f"), format(all_p[j][i]['y'],".2f"), format(all_p[j][i]['f'],".2f"), format(all_p[j][i]['left'].x,".2f"), format(all_p[j][i]['right'].x,".2f"))
    print("======")

x = findCenter(all_p, global_arr_Ii, n, m)
print(x)

