import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

#读入数据
df_train_origin = pd.read_csv('.\\train.csv')
df_test_origin = pd.read_csv('.\\test.csv')
#整合数据
df_dataset = pd.concat([df_train_origin,df_test_origin],sort=False)

# print(df_dataset.tail())
# df = df.drop(['PassengerId','Name','Ticket','Embarked'],axis=1)
df = df_dataset.drop(['PassengerId'],axis=1)                 #去掉序号列
df.loc[(df['Fare'].isnull()),'Fare'] = df[(df['Pclass']==3)&(df['Embarked']=='S')]['Fare'].mean() #补充’Fare'的空值
df['Embarked'] = df['Embarked'].fillna('C')                  #补充目的地的空值
df['Parch'] = df['SibSp'] + df['Parch']                      #将有关系的个数整合在一起
df = df.drop('SibSp',axis=1)                                 #去掉整合后的其中一列

df['Age'] = df['Age'].fillna(np.mean(df['Age'])) # 年龄可以做区间区分，空值做预测
# df['Age'] = df['Age'].fillna('U')
df['Name'] = df['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0].str.strip() #将姓名中的称呼提取出来
df['Cabin_label'] = df['Cabin'].str[0]
df['Cabin_label'] = df['Cabin_label'].fillna('U')            #将序号作为属性
df['Cabin'] = df['Cabin'].str.split(' ').str.len()           #将作为的个数作为预测的属性
df['Cabin'] = df['Cabin'].fillna(0)
df = df.drop(['Cabin','Cabin_label'],axis=1)                 #去掉了’cabin‘列
df['Sex'] = df['Sex'].replace({'male':1,'female':0})         #将性别特征化
ticket_dict = dict(df['Ticket'].value_counts())
df['Ticket'] = pd.DataFrame(df['Ticket'].map(ticket_dict))
'''
def group(x):
    if x >4:
        return 2
    elif x > 2:
        return 1
    else:
        return 0
df['Ticket'] = df['Ticket'].apply(group)

def family(x):
    if x > 4:
        return 2
    elif x > 2:
        return 1
    else:
        return 0
df['Parch'] = df['Parch'].apply(family)

Title_dic = {}
Title_dic.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_dic.update(dict.fromkeys(['Don', 'Sir', 'the Countess','Dona','Lady'], 'Royalty'))
Title_dic.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))
Title_dic.update(dict.fromkeys(['Mlle','Miss'],'Miss'))
Title_dic.update(dict.fromkeys(['Mr'],'Mr'))
Title_dic.update(dict.fromkeys(['Master','Jonkheer'],'Master'))
df['Name'] = df['Name'].map(Title_dic)

df_for_age = df[['Age','Pclass','Sex','Name']]
df_for_age = pd.get_dummies(df_for_age)
features_age = df_for_age[df_for_age['Age'].notnull()].drop('Age', axis = 1)#这种写法不错
target_age = df_for_age[df_for_age['Age'].notnull()]['Age']
features_no_age = df_for_age[df_for_age['Age'].isnull()].drop('Age', axis = 1)
GDBT_age = GradientBoostingRegressor()
GDBT_age.fit(features_age, target_age)
predict_age = GDBT_age.predict(features_no_age)
df.loc[(df['Age'].isnull()),'Age'] = predict_age
'''
# print(df)
# print(np.where(df.isna()))
# print(df.head())
# print(df.iloc[:,-2])
'''
writer_1 = pd.ExcelWriter('data_for_visualization.xlsx')
sheet_2 = pd.DataFrame(df)
sheet_2.to_excel(writer_1)
writer_1.close()
'''
vec=DictVectorizer(sparse=False) #sparse=False意思是不用稀疏矩阵表示，将一些特征向量化
np_vec=vec.fit_transform(df.to_dict(orient='record'))
# print(pd.DataFrame(test))
# print(type(np_vec))
df_vec = pd.DataFrame(np_vec)                                #向量化后再转成pandas
# print(vec.feature_names_)
# print(type(vec.feature_names_))
df_vec.columns = vec.feature_names_                          #加上列名
# print(len(vec.feature_names_))
# print(df_vec)
df_train = pd.DataFrame(df_vec[0:891])
df_test = pd.DataFrame(df_vec[891:])
# print(df_train.keys())
# print(df_test.shape)
# df_test = pd.concat([df_test.iloc[:,:-2],df_test.iloc[:,-1]],axis=1)
df_test = df_test.drop('Survived',axis=1)
# print(np.where(df_test.isna()))
# print(df_test.shape)
# print(df_test)
# train_features = pd.concat([df_train.iloc[:,:-2],df_train.iloc[:,-1]],axis=1)
train_features = df_train.drop('Survived',axis=1)
train_labels = df_train['Survived']
# print(len(train_features.keys()))
# print(train_labels)
clf = RandomForestClassifier(n_estimators=26, max_depth=6,min_samples_split=2, random_state=10,warm_start =True,
                             criterion='gini')               #随机森林参数设置,'MAX_DEPTH'调了这个参发现好了很多！

##验证
train, test, train_suv, test_suv = train_test_split(train_features,train_labels,
                                                          test_size=0.33,
                                                          random_state=42)
clf.fit(train,train_suv)
pred = clf.predict(test)
print(accuracy_score(test_suv,pred))

cro_clf = cross_val_score(clf,train_features,train_labels,cv=10)
print(cro_clf)
print(np.mean(cro_clf))
'''

clf.fit(train_features,train_labels)
pratice = clf.predict(df_test)
# sheet_1=pd.DataFrame({'PassengerId':df_test_origin['PassengerId'],'Survived':pratice})
writer = pd.ExcelWriter('pratice_16.xlsx')
sheet_1 = pd.DataFrame(pratice)
# sheet_1.to_csv('.\\pratice_11_sub.csv',index=False)
sheet_1.to_excel(writer)
writer.close()
'''

'''
print('\nDesicion Tree')
clf_1 = DecisionTreeClassifier()
clf_1.fit(train,train_suv)
pred_1 = clf_1.predict(test)
print(accuracy_score(test_suv,pred_1))
cro_clf_1 = cross_val_score(clf_1,train_features,train_labels,cv=10)
print(cro_clf_1)

print('\nSVM')
clf_2 = SVC()
clf_2.fit(train,train_suv)
pred_2 = clf_2.predict(test)
print(accuracy_score(test_suv,pred_2))
cro_clf_2 = cross_val_score(clf_2,train_features,train_labels,cv=10)
print(cro_clf_2)
'''
'''尝试使用xgboost
dtrain=xgb.DMatrix(train,label=train_suv)
param = {'max_depth':4, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
watchlist = [(dtrain,'train')]
# print(dtrain.get_label())
bst=xgb.train(param,dtrain,num_boost_round=100,evals=watchlist)
# dtest = xgb.DMatrix(df_test)
dtest = xgb.DMatrix(test)
train_preds = bst.predict(dtest)
train_predictions = [round(value) for value in train_preds]
# print(train_predictions)
# y_train = dtrain.get_label() #值为输入数据的第一行
train_accuracy = accuracy_score(test_suv, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))

'''
