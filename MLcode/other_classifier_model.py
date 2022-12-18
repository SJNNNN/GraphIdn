import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score,precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  precision_recall_curve
# area = metrics.auc(recall, precision)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgbm
from xgboost import plot_importance
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
path= "./IPVP_Datasets/SeqVec/"
dataset="iPVP data_SeqVecElastic175.txt"
idx_features_labels = np.genfromtxt("{}{}".format(path, dataset),
                               dtype=np.dtype(str))
print(idx_features_labels.shape)
X_train=idx_features_labels[0:400, 0:-1].tolist()
Y_train=idx_features_labels[0:400, -1].tolist()
X_test=idx_features_labels[400:548,0:-1].tolist()
Y_test=idx_features_labels[400:548,-1].tolist()
X_train_smo=[]
X_test_smo=[]
for i in range(len(X_train)):
       x_train1 = []
       m=X_train[i]
       for j in range(len(m)):
              x_train1.append(float(m[j]))
       X_train_smo.append(x_train1)
for i in range(len(X_test)):
       x_test1 = []
       m=X_test[i]
       for j in range(len(m)):
              x_test1.append(float(m[j]))
       X_test_smo.append(x_test1)
y_train_smo=[]
y_test_smo=[]
for i in Y_train:
    y_train_smo.append(float(i))
for i in Y_test:
    y_test_smo.append(float(i))
#朴素贝叶斯
lr  = GaussianNB()
lr.fit(X_train, Y_train)  # 调用LogisticRegression中的fit函数训练模型参数
lr_pres = lr.predict(X_test)
lr_score = lr.predict_proba(X_test)# 使用训练好的模型lr对X_test进行预测
test_acc = accuracy_score(Y_test, lr_pres)
test_prec = precision_score(Y_test, lr_pres,pos_label='1')
test_recall = recall_score(Y_test, lr_pres,pos_label='1')
test_f1 = f1_score(Y_test, lr_pres,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, lr_pres)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----GaussianNB----")
# f = open("./GAN_data/GaussianNB_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p1 = lr_score[:, 1].tolist()
l1 = y_test_smo
# for j in range(len(p1)):
#   f.writelines(str(p1[j]) + " " + str(int(l1[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
# LR模型预测
lr = LogisticRegression()  #初始化LogisticRegression
lr.fit(X_train, Y_train)  # 调用LogisticRegression中的fit函数训练模型参数
lr_pres = lr.predict(X_test)
lr_score = lr.predict_proba(X_test)# 使用训练好的模型lr对X_test进行预测
test_acc = accuracy_score(Y_test, lr_pres)
test_prec = precision_score(Y_test, lr_pres,pos_label='1')
test_recall = recall_score(Y_test, lr_pres,pos_label='1')
test_f1 = f1_score(Y_test, lr_pres,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, lr_pres)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----LR----")
# f = open("./GAN_data/LR_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p2 = lr_score[:, 1].tolist()
l2= y_test_smo
# for j in range(len(p2)):
#   f.writelines(str(p2[j]) + " " + str(int(l2[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
#SVM模型预测
model=svm.SVC(C=2,kernel='linear',gamma=10,decision_function_shape='ovr',probability=True)
model.fit(X_train,Y_train)
lr_pres = model.predict(X_test)
lr_score = model.predict_proba(X_test)# 使用训练好的模型lr对X_test进行预测
test_acc = accuracy_score(Y_test, lr_pres)
test_prec = precision_score(Y_test, lr_pres,pos_label='1')
test_recall = recall_score(Y_test, lr_pres,pos_label='1')
test_f1 = f1_score(Y_test, lr_pres,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, lr_pres)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----SVM----")
# f = open("./GAN_data/SVM_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p3 = lr_score[:, 1].tolist()
l3 = y_test_smo
# for j in range(len(p3)):
#   f.writelines(str(p3[j]) + " " + str(int(l3[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
#RF模型预测
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
pre_test = rf.predict(X_test)
lr_score = rf.predict_proba(X_test)
test_acc = accuracy_score(Y_test, pre_test)
test_prec = precision_score(Y_test, pre_test,pos_label='1')
test_recall = recall_score(Y_test, pre_test,pos_label='1')
test_f1 = f1_score(Y_test, pre_test,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, pre_test)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----RF----")
# f = open("./GAN_data/RF_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p4 = lr_score[:, 1].tolist()
l4 = y_test_smo
# for j in range(len(p4)):
#   f.writelines(str(p4[j]) + " " + str(int(l4[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
#LGBM
lgbm=lgbm.LGBMClassifier(num_leaves=60,learning_rate=0.5,n_estimators=40)
lgbm.fit(X_train,Y_train)
y_pre=lgbm.predict(X_test)
lr_score = lgbm.predict_proba(X_test)
test_acc = accuracy_score(Y_test, y_pre)
test_prec = precision_score(Y_test,y_pre,pos_label='1')
test_recall = recall_score(Y_test, y_pre,pos_label='1')
test_f1 = f1_score(Y_test, y_pre,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, y_pre)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----LightGBM----")
# f = open("./GAN_data/LGBM_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p5 = lr_score[:,1].tolist()
l5 = y_test_smo
# for j in range(len(p5)):
#   f.writelines(str(p5[j]) + " " + str(int(l5[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
##GBDT
params = {'n_estimators': 500,   # 弱分类器的个数
          'max_depth': 3,       # 弱分类器（CART回归树）的最大深度
          'min_samples_split': 5,  # 分裂内部节点所需的最小样本数
          'learning_rate': 0.5,  # 学习率
          'loss': 'exponential'}
GBDTreg = GradientBoostingClassifier(**params)
GBDTreg.fit(X_train, Y_train)
y_pre=GBDTreg.predict(X_test)
lr_score = GBDTreg.predict_proba(X_test)
test_acc = accuracy_score(Y_test, y_pre)
test_prec = precision_score(Y_test,y_pre,pos_label='1')
test_recall = recall_score(Y_test, y_pre,pos_label='1')
test_f1 = f1_score(Y_test, y_pre,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, y_pre)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----GBDT----")
# f6 = open("./GAN_data/GBDT_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p6 = lr_score[:,1].tolist()
l6 = y_test_smo
# for j in range(len(p6)):
#   f.writelines(str(p6[j]) + " " + str(int(l6[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
##MLP
clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(30,20,10),random_state=1)
clf.fit(X_train,Y_train)
Y_pre=clf.predict(X_test)
lr_score = clf.predict_proba(X_test)
# acc=accuracy_score(Y_test,Y_pre)
# print("the acc: %.5f"%acc)
# print('MLP准确率：',accuracy_score(Y_test,Y_pre))
# print('MLP精确率：',precision_score(Y_test,Y_pre,pos_label='1'))
# print('MLP召回率：',recall_score(Y_test,Y_pre,pos_label='1'))
test_acc = accuracy_score(Y_test, Y_pre)
test_prec = precision_score(Y_test,Y_pre,pos_label='1')
test_recall = recall_score(Y_test, Y_pre,pos_label='1')
test_f1 = f1_score(Y_test, Y_pre,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, Y_pre)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----MLP----")
# f = open("./GAN_data/MLP_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p7 = lr_score[:, 1].tolist()
l7 = y_test_smo
# for j in range(len(p7)):
#   f.writelines(str(p7[j]) + " " + str(int(l7[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
#KNN
knn = KNeighborsClassifier()    #实例化KNN模型
knn.fit(X_train, Y_train)
Y_pre=knn.predict(X_test)
lr_score = knn.predict_proba(X_test)
# print('kNN准确率：',accuracy_score(Y_test,Y_pre))
# print('KNN精确率：',precision_score(Y_test,Y_pre,pos_label='1'))
# print('KNN召回率：',recall_score(Y_test,Y_pre,pos_label='1'))
test_acc = accuracy_score(Y_test, Y_pre)
test_prec = precision_score(Y_test,Y_pre,pos_label='1')
test_recall = recall_score(Y_test, Y_pre,pos_label='1')
test_f1 = f1_score(Y_test, Y_pre,pos_label='1')
test_auc = roc_auc_score(Y_test, lr_score[:,1])
con_matrix = confusion_matrix(Y_test, Y_pre)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
print("----KNN----")
# f = open("./GAN_data/KNN_Fusion_NOSMOTE.txt", 'w', encoding="utf-8")
p8 = lr_score[:,1].tolist()
l8 = y_test_smo
# for j in range(len(p8)):
#   f.writelines(str(p8[j]) + " " + str(int(l8[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
#XGboost
print("----XGboost独立测试----")
model = XGBClassifier(learning_rate=0.5,
                        n_estimators=300,         # 树的个数--1000棵树建立xgboost
                        max_depth=6,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                        subsample=0.8,             # 随机选择80%样本建立决策树
                        colsample_btree=0.8,       # 随机选择80%特征建立决策树
                        objective='binary:logitraw', # 指定损失函数
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=27,   # 随机数
                       )
model.fit(np.array(X_train_smo),np.array(y_train_smo),eval_set = [(np.array(X_test_smo), np.array(y_test_smo))],eval_metric = "logloss",early_stopping_rounds = 10,verbose = True)


# fig,ax = plt.subplots(figsize=(15,15))
# plot_importance(model,height=0.5,ax=ax,max_num_features=64)
# plt.show()

### make prediction for test data
Y_pred = model.predict(np.array(X_test_smo))
Y_score = model.predict_proba(np.array(X_test_smo))
test_acc = accuracy_score(np.array(y_test_smo), Y_pred)
test_prec = precision_score(np.array(y_test_smo),Y_pred,pos_label=1)
test_recall = recall_score(np.array(y_test_smo), Y_pred,pos_label=1)
test_f1 = f1_score(np.array(y_test_smo), Y_pred,pos_label=1)
test_auc = roc_auc_score(np.array(y_test_smo), Y_score[:,1])
con_matrix = confusion_matrix(np.array(y_test_smo), Y_pred)
test_spec = con_matrix[0][0]/(con_matrix[0][0]+con_matrix[0][1])
test_mcc = (con_matrix[0][0]*con_matrix[1][1]-con_matrix[0][1]*con_matrix[1][0])/(((con_matrix[1][1]+con_matrix[0][1])*(con_matrix[1][1]+con_matrix[1][0])*(con_matrix[0][0]+con_matrix[0][1])*(con_matrix[0][0]+con_matrix[1][0]))**0.5)
# f = open("./PP/1.txt", 'w', encoding="utf-8")
p9 = Y_score[:,1].tolist()
l9 = y_test_smo
# for j in range(len(p9)):
#   f.writelines(str(p9[j]) + " " + str(int(l9[j])) + "\n")
# f.close()
print("acc: ",test_acc," ; prec: ",test_prec," ; recall: ",test_recall," ; f1: ",test_f1," ; auc: ",test_auc," ; spec:",test_spec," ; mcc: ",test_mcc)
fpr1, tpr1, threshold1 = roc_curve(l1, p1)
fpr2, tpr2, threshold2 = roc_curve(l2, p2)  ###计算真正率和假正率
fpr3, tpr3, threshold3 = roc_curve(l3, p3)
fpr4, tpr4, threshold4 = roc_curve(l4, p4)  ###计算真正率和假正率
fpr5, tpr5, threshold5 = roc_curve(l5, p5)
fpr6, tpr6, threshold6 = roc_curve(l6, p6)  ###计算真正率和假正率
fpr7, tpr7, threshold7 = roc_curve(l7, p7)
fpr8, tpr8, threshold8 = roc_curve(l8, p8)  ###计算真正率和假正率
fpr9, tpr9, threshold9 = roc_curve(l9, p9)
roc_auc1 = auc(fpr1, tpr1)###计算auc的值
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)###计算auc的值
roc_auc4 = auc(fpr4, tpr4)
roc_auc5 = auc(fpr5, tpr5)###计算auc的值
roc_auc6 = auc(fpr6, tpr6)
roc_auc7 = auc(fpr7, tpr7)###计算auc的值
roc_auc8 = auc(fpr8, tpr8)
roc_auc9 = auc(fpr9, tpr9)###计算auc的值
lw = 2
plt.figure(figsize=(8, 5))
plt.plot(fpr1, tpr1, color='cornflowerblue',
         lw=lw,alpha=0.9, label=' GaussianNB ROC curve (area = %0.6f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='lightsteelblue',
         lw=lw, label='LR ROC curve (area = %0.6f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='plum',
         lw=lw, label='SVM ROC curve (area = %0.6f)' % roc_auc3)
plt.plot(fpr4, tpr4, color='purple',
         lw=lw, label='RF ROC curve (area = %0.6f)' % roc_auc4)
plt.plot(fpr5, tpr5, color='cyan',
         lw=lw, label='LightGBM ROC curve (area = %0.6f)' % roc_auc5)
plt.plot(fpr6, tpr6, color='m',
         lw=lw, label='GBDT ROC curve (area = %0.6f)' % roc_auc6)
plt.plot(fpr7, tpr7, color='red',
         lw=lw, label='MLP ROC curve (area = %0.6f)' % roc_auc7)
plt.plot(fpr8, tpr8, color='yellow',
         lw=lw, label='KNN ROC curve (area = %0.6f)' % roc_auc8)
plt.plot(fpr9, tpr9, color='gold',
         lw=lw, label='XGBoost ROC curve (area = %0.6f)' % roc_auc9)
###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], lw=lw, color='navy',linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right",fontsize=7.2)
plt.show()
def draw_pr(y_true, y_pred, label=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    area=auc(recall,precision)
    print(area)
    plt.plot(recall, precision, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")

# 这里也是要用的是概率值，y_train_pro_lr[:,1]取了概率值中的正例的概率
draw_pr(l1, p1, 'GaussianNB')
draw_pr(l2, p2, 'LR')
draw_pr(l3, p3, 'SVM')
draw_pr(l4, p4, 'RF')
draw_pr(l5, p5, 'LightGBM')
draw_pr(l6, p6, 'GBDT')
draw_pr(l7, p7, 'MLP')
draw_pr(l8, p8, 'KNN')
draw_pr(l9, p9, 'XGBoost')
plt.legend()
plt.show()
