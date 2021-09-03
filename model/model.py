import numpy as np
import torch
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from torch import nn

data = pd.read_csv('../data/Mean_Cope1.csv', header=0)
print(data)

# X_train = data['i1_r'].values.reshape(-1,1)
# Y_train = data['o12_r'].values.reshape(-1,1)
# X_train = data[['i1_r','i2_r','i3_r','i4_r','i5_r','i6_r','i7_r','i8_r','t1_r','t2_r','t3_r','t4_r']].values.reshape(-1,1)
# Y_train = data[['o1_r','o2_r','o3_r','o4_r','o5_r','o6_r','o7_r','o8_r','o9_r','o10_r','o11_r','o12_r']].values.reshape(-1,1)
Y_train = data['o1_r'].values
data = data.drop(columns=['o1_r', 'o2_r', 'o3_r', 'o4_r', 'o5_r', 'o6_r', 'o7_r', 'o8_r', 'o9_r', 'o10_r', 'o11_r', 'o12_r'])
# 特征值
X_train = data.values

model_LinearRegression = linear_model.LinearRegression()

model_LinearRegression.fit(X_train,Y_train)
y_pred = model_LinearRegression.predict(X_train)
# print(y_pred)
# plt.figure(figsize=(14,4),dpi=150)
# plt.scatter(X_train,Y_train,color='r')
# plt.plot(X_train,y_pred,color='y')
# plt.show()
print(y_pred)
print("MAPE:",metrics.mean_absolute_percentage_error(Y_train,y_pred))

# print(data.isnull().sum())
# print(data.shape)
# print(data.head())
print(data)
print(data.describe())

# data.corr()[['o1_r','o2_r','o3_r','o4_r','o5_r','o6_r','o7_r','o8_r','o9_r','o10_r','o11_r','o12_r']]
# plt.figure(facecolor='grey')
# corr = data.corr()
# corr = corr['o1_r']
# corr[abs(corr)>0].sort_values().plot.bar()
# plt.show()
#
# y = np.array(data['o1_r'])
# x = np.array(data['i1_r'])
#
# # print(x)
# # print(y)
#
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=0)
#
# # print(X_train,X_test,Y_train,Y_test)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train = min_max_scaler.fit_transform(X_train.reshape(-1,1))
# Y_train = min_max_scaler.fit_transform(Y_train.reshape(-1,1))
# X_test = min_max_scaler.fit_transform(X_test.reshape(-1,1))
# Y_test = min_max_scaler.fit_transform(Y_test.reshape(-1,1))
#
# # print(X_train,X_test,Y_train,Y_test)
# from sklearn.linear_model import  LinearRegression
# lr = LinearRegression()
# lr.fit(X_train,Y_train)
# Y_test_pred = lr.predict(X_test)
#
# from sklearn.metrics import mean_absolute_percentage_error,r2_score
# Y_train_pred = lr.predict(X_train)
#
# MAPE = mean_absolute_percentage_error((Y_train, [np.mean(Y_train)] * len(Y_train)), mean_absolute_percentage_error(Y_train, Y_train_pred))
#
# print("MAPE: ",MAPE)

# x = data['i8_r'].values.reshape(-1,1)
# y = data['o1_r'].values.reshape(-1,1)
#
# #数据规范化
# from sklearn.preprocessing import MinMaxScaler
# mm_scale = MinMaxScaler()
# x = mm_scale.fit_transform(x)
#
# #切分数据集
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
# x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
# y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
# y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
# print(y_test)
#
# #构造网络
# model =nn.Sequential(
#     nn.Linear(1,1),
#     nn.ReLU(),
#     nn.Linear(1,1)
# )
#
# #定义优化器和损失函数
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#
# #训练
# max_epoch = 500
# for i in range(max_epoch):
#     #前向传播
#     y_pred = model(x_train)
#     #计算loss
#     loss = criterion(y_pred,y_train)
#     print(loss)
#     #梯度清0
#     optimizer.zero_grad()
#     #反向传播
#     loss.backward()
#     #权重调整
#     optimizer.step()
#
# #测试
# output = model(x_test)
# predict_list = output.detach().numpy()
# print(predict_list)


