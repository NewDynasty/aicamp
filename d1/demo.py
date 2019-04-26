from keras import Sequential, optimizers
from keras.layers import Dense

# 引入读取csv文件的库类
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
# %matplotlib inline

headers = [
    "销售日期", "销售价格", "卧室数",
    "浴室数", "房屋面积", '停车面积',
    "楼层数", "房屋评分", "建筑面积",
    "地下室面积", "建筑年份", "修复年份",
    '纬度', '经度'
]
# data = pd.read_csv("~/datasets/housing/kc_train.csv", header=None, names=headers)
data = pd.read_csv("~/Documents/workspace/code_python/aicamp/d1/school_of_ai_workshops/workshop0/data/美国King County房价预测训练赛/kc_train.csv", header=None, names=headers)
display(data.head(5))


print("数据总量: {}".format(len(data)))

# 我们将数据集以9比1的比例分割
test_ratio = 0.1
test_size = int(len(data) * test_ratio)
train_size = len(data) - test_size

train_data = data[:train_size]
test_data = data[train_size:]
print("分割后数据大小: 训练 {} | 测试 {}".format(len(train_data), len(test_data)))
display(train_data.head(5))
display(test_data.head(5))

def input_output_split(dataset):
    y = dataset[['销售价格']]
    x = dataset.iloc[:, dataset.columns != '销售价格']
    return x, y

train_x, train_y = input_output_split(train_data)
test_x, test_y = input_output_split(test_data)

print(train_x.shape, train_y.shape)

print("显示输入数据:")
display(train_x.head(5))
print("显示输出数据:")
display(train_y.head(5))

def norm(x):
    return (x - x.mean()) / x.std()

train_x = norm(train_x)
test_x = norm(test_x)

train_y = train_y / 10000
test_y = test_y / 10000

display(train_x.head(5))
display(train_y.head(5))

train_x, train_y = train_x.values, train_y.values
test_x, test_y = test_x.values, test_y.values


model = Sequential()
model.add(Dense(20, input_shape=(13,), activation='relu'))
model.add(Dense(1, activation='relu'))

opt = optimizers.adam(lr=.001)
model.compile(optimizer=opt, loss='mse')

H = model.fit(train_x, train_y, epochs=100, batch_size=16, verbose=0)
print("训练完成!!")
losses = H.history['loss']
plt.plot(losses)

