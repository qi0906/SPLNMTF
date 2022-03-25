# coding=gbk

import sys
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import precision_recall_curve

R_23_mat1 = sys.argv[1]
cv_R_23_mat1 = sys.argv[2]
pv_index1 = sys.argv[3]

R_23_mat2 = sys.argv[4]
cv_R_23_mat2 = sys.argv[5]
pv_index2 = sys.argv[6]

R_23_mat3 = sys.argv[7]
cv_R_23_mat3 = sys.argv[8]
pv_index3 = sys.argv[9]

R_23_mat4 = sys.argv[10]
cv_R_23_mat4 = sys.argv[11]
pv_index4 = sys.argv[12]

R_23_mat5 = sys.argv[13]
cv_R_23_mat5 = sys.argv[14]
pv_index5 = sys.argv[15]

raw_R_23_mat = sys.argv[16]

## R23 Raw data
f = open(raw_R_23_mat,'r',encoding='gbk')
new_R_23_mat = []
for line in f:
    line = line.replace(',','\n')
    new_R_23_mat.append(line.split('\n'))
f.close()
for i in range(len(new_R_23_mat)):
    for j in range(len(new_R_23_mat[i])):
            if '' in  new_R_23_mat[i]:
                new_R_23_mat[i].remove('')
raw_R_23_array = np.array(new_R_23_mat).astype(int)



########   1-fold CV
f1 = open(R_23_mat1,'r',encoding='gbk')
new_R_23_mat1 = []
for line in f1:
    line = line.replace(',','\n')
    new_R_23_mat1.append(line.split('\n'))
f1.close()
for i in range(len(new_R_23_mat1)):
    for j in range(len(new_R_23_mat1[i])):
            if '' in  new_R_23_mat1[i]:
                new_R_23_mat1[i].remove('')
R_23_array1 = np.array(new_R_23_mat1).astype(float)


##  Index of positive samples in R_23_mat1
fv1 = open(pv_index1,'r',encoding='gbk')
new_pv_index1 = []
for line in fv1:
    line = line.replace(',','\n')
    new_pv_index1.append(line.split('\n'))
fv1.close()

for i in range(len(new_pv_index1)):
    for j in range(len(new_pv_index1[i])):
        if '' in  new_pv_index1[i]:
            new_pv_index1[i].remove('')
pv_index_array1 = np.array(new_pv_index1).astype(int)
pv_index_array1 = pv_index_array1 -1 
new_pv_index_array1 = tuple(pv_index_array1[:,:])


## Index of negative samples in raw_R_23_mat
po_num1 = np.array(new_pv_index_array1).shape[1]
ngv_index1 = np.array(np.where(raw_R_23_array == 0))
np.random.seed(20)
np.random.shuffle(ngv_index1.T)
ne_index1 = tuple(ngv_index1[:, :po_num1])


##  Positive and negative samples in raw_R_23_mat
real_score1 = np.column_stack((np.mat(raw_R_23_array[new_pv_index_array1].flatten()), np.mat(raw_R_23_array[ne_index1].flatten())))


## In 1-fold CV, positive and negative samples in R_23_mat1
pre_score1 = np.column_stack((np.mat(R_23_array1[new_pv_index_array1].flatten()), np.mat(R_23_array1[ne_index1].flatten())))


## Computing the AUC for the first cross-validation
fpr1, tpr1, thresholds1 = metrics.roc_curve(real_score1.T, pre_score1.T, pos_label=1)
ACU1 = metrics.auc(fpr1, tpr1)
print('AUC1:',metrics.auc(fpr1, tpr1))


## Computing the AUPR for the first cross-validation
p1, r1, _ = precision_recall_curve(real_score1.T, pre_score1.T)
AUPR1 = metrics.auc(r1, p1)
print('AUPR1:',AUPR1)


## Computing the MSE for  1-fold CV
cv1 = open(cv_R_23_mat1,'r',encoding='gbk')
new_cv_R_23_mat1 = []
for line in cv1:
    line = line.replace(',','\n')
    new_cv_R_23_mat1.append(line.split('\n'))
cv1.close()
for i in range(len(new_cv_R_23_mat1)):
    for j in range(len(new_cv_R_23_mat1[i])):
            if '' in  new_cv_R_23_mat1[i]:
                new_cv_R_23_mat1[i].remove('')
cv_R_23_array1 = np.array(new_cv_R_23_mat1).astype(int)

MSE1 = []
for m in range(cv_R_23_array1.shape[0]):
    for n in range(cv_R_23_array1.shape[1]):
        if cv_R_23_array1[m,n] == 1:
            MSE1.append((cv_R_23_array1[m,n] - R_23_array1[m,n])**2)

avg_MSE1 = sum(MSE1)/len(MSE1)
print('MSE1:',avg_MSE1)


## Computing the RMSE for  1-fold CV
avg_RMSE1 = avg_MSE1 **0.5
print('RMSE1:',avg_RMSE1)



########  2-fold CV
f2 = open(R_23_mat2,'r',encoding='gbk')
new_R_23_mat2 = []
for line in f2:
    line = line.replace(',','\n')
    new_R_23_mat2.append(line.split('\n'))
f2.close()
for i in range(len(new_R_23_mat2)):
    for j in range(len(new_R_23_mat2[i])):
            if '' in  new_R_23_mat2[i]:
                new_R_23_mat2[i].remove('')
R_23_array2 = np.array(new_R_23_mat2).astype(float)


## Index of positive samples in R_23_mat2
fv2 = open(pv_index2,'r',encoding='gbk')
new_pv_index2 = []
for line in fv2:
    line = line.replace(',','\n')
    new_pv_index2.append(line.split('\n'))
fv2.close()

for i in range(len(new_pv_index2)):
    for j in range(len(new_pv_index2[i])):
        if '' in  new_pv_index2[i]:
            new_pv_index2[i].remove('')
pv_index_array2 = np.array(new_pv_index2).astype(int)
pv_index_array2 = pv_index_array2 -1 
new_pv_index_array2 = tuple(pv_index_array2[:,:])


## Index of negative samples in raw_R_23_mat
po_num2 = np.array(new_pv_index_array2).shape[1]
ngv_index2 = np.array(np.where(raw_R_23_array == 0))
np.random.seed(20)
np.random.shuffle(ngv_index2.T)
ne_index2 = tuple(ngv_index2[:, :po_num2])


## Positive and negative samples in raw_R_23_mat
real_score2 = np.column_stack((np.mat(raw_R_23_array[new_pv_index_array2].flatten()), np.mat(raw_R_23_array[ne_index2].flatten())))


## In 2-fold CV, positive and negative samples in R_23_mat2
pre_score2 = np.column_stack((np.mat(R_23_array2[new_pv_index_array2].flatten()), np.mat(R_23_array2[ne_index2].flatten())))


## Computing the AUC for 2-fold CV
fpr2, tpr2, thresholds2 = metrics.roc_curve(real_score2.T, pre_score2.T, pos_label=1)
ACU2 = metrics.auc(fpr2, tpr2)
print('AUC2:',metrics.auc(fpr2, tpr2))


## Computing the AUPR for 2-fold CV
p2, r2, _ = precision_recall_curve(real_score2.T, pre_score2.T)
AUPR2 = metrics.auc(r2, p2)
print('AUPR2:',AUPR2)


## Computing the MSE for 2-fold CV
cv2 = open(cv_R_23_mat2,'r',encoding='gbk')
new_cv_R_23_mat2 = []
for line in cv2:
    line = line.replace(',','\n')
    new_cv_R_23_mat2.append(line.split('\n'))
cv2.close()
for i in range(len(new_cv_R_23_mat2)):
    for j in range(len(new_cv_R_23_mat2[i])):
            if '' in  new_cv_R_23_mat2[i]:
                new_cv_R_23_mat2[i].remove('')
cv_R_23_array2 = np.array(new_cv_R_23_mat2).astype(int)

MSE2 = []
for m in range(cv_R_23_array2.shape[0]):
    for n in range(cv_R_23_array2.shape[1]):
        if cv_R_23_array2[m,n] == 1:
            MSE2.append((cv_R_23_array2[m,n] - R_23_array2[m,n])**2)

avg_MSE2 = sum(MSE2)/len(MSE2)
print('MSE2:',avg_MSE2)


## Computing the RMSE for 2-fold CV
avg_RMSE2 = avg_MSE2 **0.5
print('RMSE2:',avg_RMSE2)



######## 3-fold CV
f3 = open(R_23_mat3,'r',encoding='gbk')
new_R_23_mat3 = []
for line in f3:
    line = line.replace(',','\n')
    new_R_23_mat3.append(line.split('\n'))
f3.close()
for i in range(len(new_R_23_mat3)):
    for j in range(len(new_R_23_mat3[i])):
            if '' in  new_R_23_mat3[i]:
                new_R_23_mat3[i].remove('')
R_23_array3 = np.array(new_R_23_mat3).astype(float)


## Index of positive samples in R_23_mat3
fv3 = open(pv_index3,'r',encoding='gbk')
new_pv_index3 = []
for line in fv3:
    line = line.replace(',','\n')
    new_pv_index3.append(line.split('\n'))
fv3.close()

for i in range(len(new_pv_index3)):
    for j in range(len(new_pv_index3[i])):
        if '' in  new_pv_index3[i]:
            new_pv_index3[i].remove('')
pv_index_array3 = np.array(new_pv_index3).astype(int)
pv_index_array3 = pv_index_array3 -1 
new_pv_index_array3 = tuple(pv_index_array3[:,:])


## Index of negative samples in raw_R_23_mat
po_num3 = np.array(new_pv_index_array3).shape[1]
ngv_index3 = np.array(np.where(raw_R_23_array == 0))
np.random.seed(20)
np.random.shuffle(ngv_index3.T)
ne_index3 = tuple(ngv_index3[:, :po_num3])


## Positive and negative samples in raw_R_23_mat
real_score3 = np.column_stack((np.mat(raw_R_23_array[new_pv_index_array3].flatten()), np.mat(raw_R_23_array[ne_index3].flatten())))


## In 3-fold CV, positive and negative samples in R_23_mat3
pre_score3 = np.column_stack((np.mat(R_23_array3[new_pv_index_array3].flatten()), np.mat(R_23_array3[ne_index3].flatten())))


## Computing the AUC for 3-fold CV
fpr3, tpr3, thresholds3 = metrics.roc_curve(real_score3.T, pre_score3.T, pos_label=1)
ACU3 = metrics.auc(fpr3, tpr3)
print('AUC3:',metrics.auc(fpr3, tpr3))


## Computing the AUPR for the third cross-validation
p3, r3, _ = precision_recall_curve(real_score3.T, pre_score3.T)
AUPR3 = metrics.auc(r3, p3)
print('AUPR3:',AUPR3)


## Computing the MSE for 3-fold CV
cv3 = open(cv_R_23_mat3,'r',encoding='gbk')
new_cv_R_23_mat3 = []
for line in cv3:
    line = line.replace(',','\n')
    new_cv_R_23_mat3.append(line.split('\n'))
cv3.close()
for i in range(len(new_cv_R_23_mat3)):
    for j in range(len(new_cv_R_23_mat3[i])):
            if '' in  new_cv_R_23_mat3[i]:
                new_cv_R_23_mat3[i].remove('')
cv_R_23_array3 = np.array(new_cv_R_23_mat3).astype(int)

MSE3 = []
for m in range(cv_R_23_array3.shape[0]):
    for n in range(cv_R_23_array3.shape[1]):
        if cv_R_23_array3[m,n] == 1:
            MSE3.append((cv_R_23_array3[m,n] - R_23_array3[m,n])**2)

avg_MSE3 = sum(MSE3)/len(MSE3)
print('MSE3:',avg_MSE3)


## Computing the RMSE for 3-fold CV
avg_RMSE3 = avg_MSE3 **0.5
print('RMSE3:',avg_RMSE3)



######## 4-fold CV
f4 = open(R_23_mat4,'r',encoding='gbk')
new_R_23_mat4 = []
for line in f4:
    line = line.replace(',','\n')
    new_R_23_mat4.append(line.split('\n'))
f4.close()
for i in range(len(new_R_23_mat4)):
    for j in range(len(new_R_23_mat4[i])):
            if '' in  new_R_23_mat4[i]:
                new_R_23_mat4[i].remove('')
R_23_array4 = np.array(new_R_23_mat4).astype(float)


## Index of positive samples in R_23_mat
fv4 = open(pv_index4,'r',encoding='gbk')
new_pv_index4 = []
for line in fv4:
    line = line.replace(',','\n')
    new_pv_index4.append(line.split('\n'))
fv4.close()

for i in range(len(new_pv_index4)):
    for j in range(len(new_pv_index4[i])):
        if '' in  new_pv_index4[i]:
            new_pv_index4[i].remove('')
pv_index_array4 = np.array(new_pv_index4).astype(int)
pv_index_array4 = pv_index_array4 -1 
new_pv_index_array4 = tuple(pv_index_array4[:,:])


## Index of negative samples in raw_R_23_mat
po_num4 = np.array(new_pv_index_array4).shape[1]
ngv_index4 = np.array(np.where(raw_R_23_array == 0))
np.random.seed(20)
np.random.shuffle(ngv_index4.T)
ne_index4 = tuple(ngv_index4[:, :po_num4])


## Positive and negative samples in raw_R_23_mat
real_score4 = np.column_stack((np.mat(raw_R_23_array[new_pv_index_array4].flatten()), np.mat(raw_R_23_array[ne_index4].flatten())))


## In 4-fold CV, positive and negative samples in R_23_mat4
pre_score4 = np.column_stack((np.mat(R_23_array4[new_pv_index_array4].flatten()), np.mat(R_23_array4[ne_index4].flatten())))


## Computing the AUC for 4-fold CV
fpr4, tpr4, thresholds4 = metrics.roc_curve(real_score4.T, pre_score4.T, pos_label=1)
ACU4 = metrics.auc(fpr4, tpr4)
print('AUC4:',metrics.auc(fpr4, tpr4))


## Computing the AUPR for 4-fold CV
p4, r4, _ = precision_recall_curve(real_score4.T, pre_score4.T)
AUPR4 = metrics.auc(r4, p4)
print('AUPR4:',AUPR4)


## Computing the MSE for 4-fold CV
cv4 = open(cv_R_23_mat4,'r',encoding='gbk')
new_cv_R_23_mat4 = []
for line in cv4:
    line = line.replace(',','\n')
    new_cv_R_23_mat4.append(line.split('\n'))
cv4.close()
for i in range(len(new_cv_R_23_mat4)):
    for j in range(len(new_cv_R_23_mat4[i])):
            if '' in  new_cv_R_23_mat4[i]:
                new_cv_R_23_mat4[i].remove('')
cv_R_23_array4 = np.array(new_cv_R_23_mat4).astype(int)

MSE4 = []
for m in range(cv_R_23_array4.shape[0]):
    for n in range(cv_R_23_array4.shape[1]):
        if cv_R_23_array4[m,n] == 1:
            MSE4.append((cv_R_23_array4[m,n] - R_23_array4[m,n])**2)

avg_MSE4 = sum(MSE4)/len(MSE4)
print('MSE4:',avg_MSE4)


## Computing the RMSE for 4-fold CV
avg_RMSE4 = avg_MSE4 **0.5
print('RMSE4:',avg_RMSE4)



######## 5-fold CV
f5 = open(R_23_mat5,'r',encoding='gbk')
new_R_23_mat5 = []
for line in f5:
    line = line.replace(',','\n')
    new_R_23_mat5.append(line.split('\n'))
f5.close()
for i in range(len(new_R_23_mat5)):
    for j in range(len(new_R_23_mat5[i])):
            if '' in  new_R_23_mat5[i]:
                new_R_23_mat5[i].remove('')
R_23_array5 = np.array(new_R_23_mat5).astype(float)


## Index of positive samples in R_23_mat5
fv5 = open(pv_index5,'r',encoding='gbk')
new_pv_index5 = []
for line in fv5:
    line = line.replace(',','\n')
    new_pv_index5.append(line.split('\n'))
fv5.close()

for i in range(len(new_pv_index5)):
    for j in range(len(new_pv_index5[i])):
        if '' in  new_pv_index5[i]:
            new_pv_index5[i].remove('')
pv_index_array5 = np.array(new_pv_index5).astype(int)
pv_index_array5 = pv_index_array5 -1 
new_pv_index_array5 = tuple(pv_index_array5[:,:])


## Index of negative samples in raw_R_23_mat
po_num5 = np.array(new_pv_index_array5).shape[1]
ngv_index5 = np.array(np.where(raw_R_23_array == 0))
np.random.seed(20)
np.random.shuffle(ngv_index5.T)
ne_index5 = tuple(ngv_index5[:, :po_num5])


## Positive and negative samples in raw_R_23_mat
real_score5 = np.column_stack((np.mat(raw_R_23_array[new_pv_index_array5].flatten()), np.mat(raw_R_23_array[ne_index5].flatten())))


## In 5-fold CV, positive and negative samples in R_23_mat5
pre_score5 = np.column_stack((np.mat(R_23_array5[new_pv_index_array5].flatten()), np.mat(R_23_array5[ne_index5].flatten())))


## Computing the AUC for 5-fold CV
fpr5, tpr5, thresholds5 = metrics.roc_curve(real_score5.T, pre_score5.T, pos_label=1)
ACU5 = metrics.auc(fpr5, tpr5)
print('AUC5:',metrics.auc(fpr5, tpr5))


## Computing the AUPR for 5-fold CV
p5, r5, _ = precision_recall_curve(real_score5.T, pre_score5.T)
AUPR5 = metrics.auc(r5, p5)
print('AUPR5:',AUPR5)


## Computing the MSE for 5-fold CV
cv5 = open(cv_R_23_mat5,'r',encoding='gbk')
new_cv_R_23_mat5 = []
for line in cv5:
    line = line.replace(',','\n')
    new_cv_R_23_mat5.append(line.split('\n'))
cv5.close()
for i in range(len(new_cv_R_23_mat5)):
    for j in range(len(new_cv_R_23_mat5[i])):
            if '' in  new_cv_R_23_mat5[i]:
                new_cv_R_23_mat5[i].remove('')
cv_R_23_array5 = np.array(new_cv_R_23_mat5).astype(int)

MSE5 = []
for m in range(cv_R_23_array5.shape[0]):
    for n in range(cv_R_23_array5.shape[1]):
        if cv_R_23_array5[m,n] == 1:
            MSE5.append((cv_R_23_array5[m,n] - R_23_array5[m,n])**2)

avg_MSE5 = sum(MSE5)/len(MSE5)
print('MSE5:',avg_MSE5)


## Computing the RMSE for 5-fold CV
avg_RMSE5 = avg_MSE5 **0.5
print('RMSE5:',avg_RMSE5)



### AVERAGED RESULTS
AUC = [ACU1,ACU2,ACU3,ACU4,ACU5]
AUC_std = np.std(AUC)
AUC_avg = sum(AUC) / 5
print('AUC(std):',str(AUC_avg) + '\u00B1' +  str(AUC_std))

AUPR = [AUPR1,AUPR2,AUPR3,AUPR4,AUPR5]
AUPR_std = np.std(AUPR)
AUPR_avg = sum(AUPR) / 5
print('AUPR(std):',str(AUPR_avg) + '\u00B1' +  str(AUPR_std))

tol_MSE = [avg_MSE1,avg_MSE2,avg_MSE3,avg_MSE4,avg_MSE5]
MSE_std = np.std(tol_MSE)
MSE_avg = sum(tol_MSE) / 5
print('MSE(std):',str(MSE_avg) + '\u00B1' +  str(MSE_std))

tol_RMSE = [avg_RMSE1,avg_RMSE2,avg_RMSE3,avg_RMSE4,avg_RMSE5]
RMSE_std = np.std(tol_RMSE)
RMSE_avg = sum(tol_RMSE) / 5
print('RMSE(std):',str(RMSE_avg) + '\u00B1' +  str(RMSE_std))



#### ROC curve
auc1 = metrics.auc(fpr1, tpr1)
auc2 = metrics.auc(fpr2, tpr2)
auc3 = metrics.auc(fpr3, tpr3)
auc4 = metrics.auc(fpr4, tpr4)
auc5 = metrics.auc(fpr5, tpr5)

plt.plot(fpr1, tpr1, 'blue',label='auc1 = %0.3f'% auc1,linewidth = 1.0)
plt.plot(fpr2, tpr2, 'green',label='auc2 = %0.3f'% auc2,linewidth = 1.0)
plt.plot(fpr3, tpr3, 'darkred',label='auc3 = %0.3f'% auc3,linewidth = 1.0)
plt.plot(fpr4, tpr4, 'cyan',label='auc4 = %0.3f'% auc4,linewidth = 1.0)
plt.plot(fpr5, tpr5, 'magenta',label='auc5 = %0.3f'% auc5,linewidth = 1.0)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')

plt.ylim((0,1))
x = np.linspace(0,1,100)

plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')  
plt.title('ROC curve')

plt.savefig('ROC.svg')
plt.show()



###### PR curve
aupr1 = metrics.auc(r1, p1)
aupr2 = metrics.auc(r2, p2)
aupr3 = metrics.auc(r3, p3)
aupr4 = metrics.auc(r4, p4)
aupr5 = metrics.auc(r5, p5)

plt.plot(r1, p1, 'blue',label = 'aupr1 = %0.3f' % aupr1,linewidth = 1.0)
plt.plot(r2, p2, 'green',label = 'aupr2 = %0.3f' % aupr2,linewidth = 1.0)
plt.plot(r3, p3, 'darkred',label = 'aupr3 = %0.3f' % aupr3,linewidth = 1.5)
plt.plot(r4, p4, 'cyan',label = 'aupr4 = %0.3f' % aupr4,linewidth = 1.5)
plt.plot(r5, p5, 'magenta',label = 'aupr5 = %0.3f' % aupr5,linewidth = 1.5)

plt.legend(loc='upper left')
plt.plot([0, 1], [0.5, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0.5, 1.01])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')

plt.savefig('PR.svg')
plt.show()
