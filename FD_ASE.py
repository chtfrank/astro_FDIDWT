import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os


log_cnt = 1
log_file = os.path.join('RESULTS', 'FDASE_tmp.log.1')
while os.path.exists(log_file):
    log_cnt += 1
    log_file = log_file.split('FDASE_tmp.log.')[0] + str(log_cnt)
os.makedirs(os.path.dirname(log_file), exist_ok=True)



def myprint(print_str, logfile):
    with open(logfile, 'a') as fp:
        fp.write(print_str + '\n')


# box counting
def computeSr(data, r):
    N, E = np.shape(data)
    cells = {}
    Sr = 0
    for i in range(N): # each point in data
        ck_key = ''
        for k in range(E):
            ck = int(np.ceil(data[i, k]/r))
            ck_key += (str(ck) + '_')
        if ck_key in cells.keys():
            cells[ck_key] += 1
        else:
            cells[ck_key] = 1
    for p in cells.values():
        Sr += (p * p)
    return Sr


# compute the partial intrinsic dimension of dataset or individual contribution of one attribute
def computepD(data, R, return_fit=False, fit_length=5):
    if len(data) == 0:
        return 0
    
    rr = []
    Sqr = []
    for j in range(R+1):
        r = 1/np.power(2.0, j)
        Sr = computeSr(data, r)
        Sqr.append(np.log(Sr))
        rr.append(np.log(r))
    logR = np.reshape(rr[:], (-1,1))
    logSqr = np.reshape(Sqr[:], (-1,1))

    fit_length = fit_length
    nfit = len(logR)-fit_length+1
    regr = LinearRegression()
    slopes = []
    predicts = []
    fit_range = []
    for i in range(nfit):
        fit_start = i
        fit_stop = i + fit_length
        regr.fit(logR[fit_start:fit_stop], logSqr[fit_start:fit_stop])
        slope = np.squeeze(regr.coef_)
        predict = regr.predict(logR[fit_start:fit_stop])
        slopes.append(slope)
        predicts.append(predict)
        fit_range.append([fit_start, fit_stop])
    inx_slope = np.argmax(slopes)
    slope = np.max(slopes)
    best_predict = predicts[inx_slope]
    best_range = fit_range[inx_slope]

    if return_fit == True:
        return slope, logR, logSqr, best_predict, best_range
    else: return slope


def cal_iC(data, R, small_ic=0.1, show_fit=False, fit_length=5):
    N, E = np.shape(data)
    print('calculating individual contribution ...')
    myprint('calculating individual contribution ...', log_file)
    A = {} # indecies of attributes and iC values
    if show_fit == False:
        for i in range(E):
            attr_data = data[:, [i]] # data for one attribute
            iC = computepD(attr_data, R) # individual contribution
            if iC >= small_ic:
                A[i] = iC
                print('iC %d / %d: %.5f' % (i, E-1, iC))
                myprint('iC {} / {}: {:.5f}'.format(i, E-1, iC), log_file)
            else:
                print('remove attributes {} with small ic {}'.format(i, iC))
                myprint('remove attributes {} with small ic {}'.format(i, iC), log_file)
    else:
        rr = []
        ss = []
        pp = []
        bb = []
        for i in range(E):
            attr_data = data[:, [i]]
            iC, logR, logSqr, predict, best_range = computepD(attr_data, R, return_fit=True, fit_length=fit_length)
            if iC >= small_ic:
                A[i] = iC
                print('iC %d / %d: %.5f' % (i, E-1, iC))
                myprint('iC {} / {}: {:.5f}'.format(i, E-1, iC), log_file)
            else:
                print('remove attributes {} with small ic {}'.format(i, iC))
                myprint('remove attributes {} with small ic {}'.format(i, iC), log_file)
            rr.append(logR)
            ss.append(logSqr)
            pp.append(predict)
            bb.append(best_range)

        fig = plt.figure()
        cols = 3
        rows = int(np.ceil(len(rr)/cols))
        for i in range(len(rr)):
            axs = fig.add_subplot(rows,cols,i+1)
            axs.scatter(rr[i], ss[i], color='k')
            axs.plot(rr[i][bb[i][0]:bb[i][1]], pp[i], '*:', color='r', linewidth=2)
        plt.savefig('R_{}.png'.format(R))

    # A': the sorted indecies according to decending iC values
    A_ = np.array([], dtype=np.int32)
    iCs = np.array([], dtype=np.float64)
    sorted_A = sorted(A.items(),key=lambda x:x[1], reverse=True)
    E_ = len(sorted_A)
    for i in range(E_):
        A_ = np.append(A_, sorted_A[i][0])
        iCs = np.append(iCs, sorted_A[i][1])

    print('\nsorted iC A\':')
    myprint('\nsorted iC A\':', log_file)
    for i in range(E_):
        print('attribute %d: %.5f' % (A_[i], iCs[i]))
        myprint('attribute {}: {:.5f}'.format(A_[i], iCs[i]), log_file)
    iD = computepD(data, R)
    print('partial intrinsic dimension D = %.5f\n' % (iD))
    myprint('partial intrinsic dimension D = {:.5f}\n'.format(iD), log_file)

    return A_, iCs, iD


# find the correlation group and correlation base
def findGroup(data, SG, R, epsilon):
    print('<findGroup> SG:', SG)
    myprint('<findGroup> SG: {}'.format(SG), log_file)
    # step 1
    G = np.copy(SG) # {a1, a2, ..., ak}
    B = np.array([], dtype=np.int32)

    pd_k = computepD(data[:, [SG[-1]]], R) # iC(ak)
    for attr_indx in range(len(SG)-1):
        G_remove_aj = np.copy(SG) # {a1, a2, ..., ak}
        G_remove_aj = np.delete(G_remove_aj, attr_indx) # {a1, a2, ..., a(j-1), a(j+1), ..., ak}
        aj = SG[attr_indx] # the attribute to check
        G_remove_aj_append_aj = np.append(G_remove_aj, aj) # {a1, a2, ..., a(j-1), a(j+1), ..., ak, aj}

        pd_j = computepD(data[:, [aj]], R) # iC(aj)
        pd_test1 = computepD(data[:, G_remove_aj_append_aj], R) # pD({a1, a2, ..., a(j-1), a(j+1), ..., ak, aj}
        pd_test2 = computepD(data[:, G_remove_aj], R) # pD({a1, a2, ..., a(j-1), a(j+1), ..., ak})
        if np.absolute(pd_test1 - pd_test2) >= (epsilon*pd_j):
            aj_indx = list(G).index(aj)
            print('<findGroup> remove {} from {}'.format(G[aj_indx], G))
            myprint('<findGroup> remove {} from {}'.format(G[aj_indx], G), log_file)
            G = np.delete(G, aj_indx)
            
        else:
            if True:
                G_remove_aj_k = np.copy(G_remove_aj) # {a1, a2, ..., a(j-1), a(j+1), ..., ak}
                G_remove_aj_k = np.delete(G_remove_aj_k, -1) # pD({a1, a2, ..., a(j-1), a(j+1), ..., a(k-1)})
                pd_test3 = computepD(data[:, G_remove_aj_k], R) # pD({a1, a2, ..., a(j-1), a(j+1), ..., a(k-1)})
                if np.absolute(pd_test3-pd_test2) < epsilon*pd_k:
                    print('<findGroup> recurrence with SG: {}'.format(G_remove_aj))
                    myprint('<findGroup> recurrence with SG: {}'.format(G_remove_aj), log_file)
                    G0, B0 = findGroup(data, G_remove_aj, R, epsilon)
                    print('---> <findGroup> back from recurrence with G: {}, B: {}'.format(G0, B0))
                    print('---> <findGroup> back from recurrence with G: {}, B: {}'.format(G0, B0), log_file)

                    # # step 5
                    if len(G0) != 0:
                        return G0, B0
    print('<findGroup> now G: {}'.format(G))
    myprint('<findGroup> now G: {}'.format(G), log_file)


    # step 6-7: construct correlation base
    B = findBase(data, G, R, epsilon)
    print('<findGroup> B: {}'.format(B))
    myprint('<findGroup> B: {}'.format(B), log_file)

    # step 8
    remain = [x for x in G if x not in B]
    if len(remain) == 0:
        G = np.array([], dtype=np.int32)
        print('<findGroup> G == B !')
        myprint('<findGroup> G == B !', log_file)

    # step 9-10
    else:
        print('<findGroup> check G - B: {} '.format(remain))
        myprint('<findGroup> check G - B: {} '.format(remain), log_file)
        for g in range(len(remain)):
            B_ag_list = np.copy(B) # {B}
            B_ag_list = np.append(B_ag_list, remain[g]) # {B}U{ag}
            pd_B_ag = computepD(data[:, B_ag_list], R) # pD({B}U{ag})
            remove_flag = False
            for b in range(len(B)):
                B_ag_no_ab = np.copy(B) # {B}
                B_ag_no_ab = np.delete(B_ag_no_ab, b) # {B-ab}
                B_ag_no_ab = np.append(B_ag_no_ab, remain[g]) # {B-ab}U{ag}
                pd_B_ag_no_ab = computepD(data[:, B_ag_no_ab], R) # pD({B-ab}U{ag})
                pd_ab = computepD(data[:, [B[b]]], R) # pD(ab)
                if np.absolute(pd_B_ag_no_ab - pd_B_ag) >= epsilon*pd_ab:
                    remove_flag = True
                    break
            if remove_flag == True:
                remove_indx = list(G).index(remain[g])
                
                print('<findGroup> remove {} from G {}'.format(remain[g], G))
                myprint('<findGroup> remove {} from G {}'.format(remain[g], G), log_file)
                G = np.delete(G, remove_indx)
    return G, B



def findBase(data, G, R, epsilon):
    B = np.array([], dtype=np.int32)
    
    for g in range(len(G)):
        B_ag_list = np.copy(B)
        B_ag_list = np.append(B_ag_list, G[g]) # {B}U{ag}
        pd_B_ag = computepD(data[:, B_ag_list], R) # pD({B}U{ag})
        pd_B = computepD(data[:, B], R) # pD(B)
        pd_ag = computepD(data[:, [G[g]]], R) # pD(ag)
        if np.absolute(pd_B_ag - pd_B) >= epsilon*pd_ag:
            print('<findBase> add {} to Base {}'.format(G[g], B))
            myprint('<findBase> add {} to Base {}'.format(G[g], B), log_file)
            B = np.append(B, G[g])
    return B





# FD_ASE algorithm to return correlation groups and bases
def FD_ASE(data, R, epsilon, A_):

    valid_result = True

    N, E = np.shape(data)
    print('start to run the FD_ASE algorithm to search correlation groups !\n\n')
    myprint('start to run the FD_ASE algorithm to search correlation groups !\n\n', log_file)

    Gs = [] # the correlation groups
    Bs = [] # the correlation bases

    while(1):

        array_list_k1 = [A_[0]]
        array_list_k2 = [A_[0]]

        # step 6
        # always remenber the attributes in A_ have the descending order
        k = 1
        while(k < len(A_)):
            array_list_k2.append(A_[k])
            attr_data_k1 = data[:, array_list_k1] # data{a1, a2, ..., ak-1}
            attr_data_k2 = data[:, array_list_k2] # data{a1, a2, ..., ak}
            pd_k1 = computepD(attr_data_k1, R) # pD({a1, a2, ..., ak-1})
            pd_k2 = computepD(attr_data_k2, R) # pD({a1, a2, ..., ak})
            pd_k = computepD(data[:, [A_[k]]], R) # iC(ak)
            diff_pd = np.absolute(pd_k2 - pd_k1)
            if diff_pd < epsilon * pd_k:
                break
            else:
                array_list_k1.append(A_[k])
                k += 1
        
        # step 7
        if k == len(A_) and diff_pd >= epsilon * pd_k:
            print('<FD_ASE> finished !')
            myprint('<FD_ASE> finished !', log_file)
            valid_result = True
            break
        
        print('<FD_ASE> find k: {}, attri list: {}'.format(k, array_list_k2))
        myprint('<FD_ASE> find k: {}, attri list: {}'.format(k, array_list_k2), log_file)
        # step 8
        SGc = np.copy(array_list_k2)
        # step 9
        Gc, Bc = findGroup(data, SGc, R, epsilon) # find the correlation group and correlation base
        print('<FD_ASE> return from <findGroup> G:', Gc, 'B:', Bc)
        myprint('<FD_ASE> return from <findGroup> G: {}, B: {}'.format(Gc, Bc), log_file)
        if len(Bc) == 0:
            print('<FD_ASE> stop with error ?!')
            myprint('<FD_ASE> stop with error ?!', log_file)
            valid_result = False
            break

        # step 10
        # assert k+1 < len(A_)
        remained = [A_[i] for i in range(k+1, len(A_))]
        if len(remained) == 0:
            print('<FD_ASE> no attributes remained, G: {}, B: {}'.format(Gc, Bc))
            myprint('<FD_ASE> no attributes remained, G: {}, B: {}'.format(Gc, Bc), log_file)
        else:
            print('<FD_ASE> checking the remained attributes:\n{}'.format(remained))
            myprint('<FD_ASE> checking the remained attributes:\n{}'.format(remained), log_file)
            pd_base = computepD(data[:, Bc], R) # pD(Bc)
            for j in range(k+1, len(A_)):
                pd_j = computepD(data[:, [A_[j]]], R) # iC(aj)
                test_list = np.copy(Bc)
                test_list = np.append(test_list, A_[j]) # {Bc}U{aj}
                pd_Bc_aj = computepD(data[:, test_list], R) # pD({Bc}U{aj})
                # step 11
                if np.absolute(pd_Bc_aj-pd_base) < epsilon*pd_j:
                    # step 12
                    add_flag = True
                    for b in range(len(Bc)):
                        pd_b = computepD(data[:, [Bc[b]]], R) # iC(ab)
                        Bc_no_ab = np.copy(Bc)
                        Bc_aj_no_ab = np.copy(Bc)
                        Bc_no_ab = np.delete(Bc_no_ab, b) # Bc - {ab}
                        Bc_aj_no_ab[b] = A_[j] # {Bc - {ab}}U{aj}
                        pd_Bc_no_ab = computepD(data[:, Bc_no_ab], R) # pD(Bc - {ab})
                        pd_Bc_aj_no_ab = computepD(data[:, Bc_aj_no_ab], R) # pD({Bc - {ab}}U{aj})
                        if (np.absolute(pd_Bc_aj-pd_Bc_aj_no_ab) >= epsilon*pd_b) or (np.absolute(pd_Bc_no_ab-pd_Bc_aj_no_ab) < epsilon*pd_j):
                            add_flag = False
                            break
                    if add_flag == True:
                        print('<FD_ASE> add attribute {} to G: {}'.format(A_[j], Gc))
                        myprint('<FD_ASE> add attribute {} to G: {}'.format(A_[j], Gc), log_file)
                        Gc = np.append(Gc, A_[j])
                else:
                    print('<FD_ASE> attribute {} is not correlated to B: {}'.format(A_[j],Bc))
                    myprint('<FD_ASE> attribute {} is not correlated to B: {}'.format(A_[j],Bc), log_file)
            print('<FD_ASE> after checking the remained attributes, G: {}, B: {}'.format(Gc,Bc))
            myprint('<FD_ASE> after checking the remained attributes, G: {}, B: {}'.format(Gc,Bc), log_file)

        # step 13
        Gc_no_Bc = [x for x in Gc if x not in Bc]
        if len(Gc_no_Bc) == 0:
            print('<FD_ASE> stop with error ?!')
            myprint('<FD_ASE> stop with error ?!', log_file)
            valid_result = False
            break

        print('<FD_ASE> remove {} from A\': {}'.format(Gc_no_Bc, A_))
        myprint('<FD_ASE> remove {} from A\': {}'.format(Gc_no_Bc, A_), log_file)
        A_ = np.array([x for x in A_ if x not in Gc_no_Bc], dtype=np.int32)
        print('<FD_ASE> new A\': {}'.format(A_))
        myprint('<FD_ASE> new A\': {}'.format(A_), log_file)
            
        # step 14
        Gs.append(Gc)
        Bs.append(Bc)


    if not valid_result and len(Gs) == 0 and len(Bs) == 0:
        return None, None, None
    else:
        return Gs, Bs, A_



def cal_importance(data, R, A_, core):
    importance_dict = {}
    not_core = list(set(A_) - set(core))

    # calculate core importance
    for i in range(len(core)):
        pd_ = computepD(data[:, [core[i]]], R)
        importance_dict[core[i]] = pd_
    core_pd = computepD(data[:, core], R)

    # importance of the remaining attributes
    for atrr in not_core:
        cal_attrs = list(np.copy(core))
        cal_attrs.append(atrr)
        new_pd = computepD(data[:, cal_attrs], R)
        # print('attrs: {}, pd: {}'.format(cal_attrs, new_pd))
        importance = np.absolute(new_pd - core_pd)
        importance_dict[atrr] = importance
    
    importance_dict_sorted = {}
    importance_list_sorted = sorted(importance_dict.items(),  key=lambda d: d[1], reverse=True)
    for atrr, importance in importance_list_sorted:
        # print(atrr, importance)
        importance_dict_sorted[atrr] = importance
    # pds = pds[1:]
    return importance_dict_sorted




