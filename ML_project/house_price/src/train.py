import numpy 
import pandas 
from sklearn.decomposition import PCA  

# # ========== 动态自动路径定位（永久解决路径问题） ==========
# # 获取当前train.py代码文件的绝对路径
# current_file = os.path.abspath(__file__)
# # 定位到项目根目录 house_price
# project_root = os.path.dirname(os.path.dirname(current_file))
# # 自动拼接数据、输出文件完整路径
# train_path = os.path.join(project_root, "data", "train.csv")
# test_path = os.path.join(project_root, "data", "test.csv")
# save_path = os.path.join(project_root, "output", "submission.csv")

# 读取数据：训练集和测试集（pandas提供的read_csv方法返回类似excel表格类型的数据,DataFrame类型）
train_set = pandas.read_csv("house_price/data/train.csv")
test_set = pandas.read_csv("house_price/data/test.csv")

# print(type(train_set))  -->> pandas.core.frame.DataFrame

# 验证是否成功导入文件
# print(train_set.shape)
# print(test_set.shape)

# 获取训练集房价数值并放到y数组（numpy数组）中
y = train_set["SalePrice"].values
# print(type(y))  -->> numpy.ndarray

# 对数化处理（避免房价右偏）
y = numpy.log1p(y)  # 等价于y = numpy.log(y+1)  （这样处理的原因？）

# 训练集删除Id、SalePrice列，剩下的作为输入
X = train_set.drop(["Id","SalePrice"],axis=1)

# 测试集删除Id列，剩下的作为输入
X_test = test_set.drop(["Id"],axis=1)

# 测试集id，提交的时候一并提交
test_id = test_set["Id"].values



# 数据预处理

# 拼接训练集和测试集(按行拼接)
all_data = pandas.concat([X,X_test],axis=0)

# 缺失值处理（用0填充缺失值）  
# 缺失值填充0未尝不好：数据本身没问题，缺失值肯能就代表没有，如果填充其他值反而改变原有的意思
# all_data = all_data.fillna(0)

# # 缺失值：数值列用中位数填充，类别列用众数填充（效果反而下降）类别列用none填充效果会好一些
# 获取数值列索引号集合
numerical_cols = all_data.select_dtypes(include=['int64','float64']).columns
# # 中位数填充
all_data[numerical_cols] = all_data[numerical_cols].fillna(all_data[numerical_cols].median())
# 获取类别列索引号集合
categorical_cols = all_data.select_dtypes(include=['object']).columns
# # 众数填充
# all_data[categorical_cols] = all_data[categorical_cols].fillna(all_data[categorical_cols].mode().iloc[0])
# 类别列：缺失统一填None
all_data[categorical_cols] = all_data[categorical_cols].fillna("None")


# 独热编码处理（drop_first=True表示删除第一个类别，依旧可以判断类别）
all_data = pandas.get_dummies(all_data,drop_first=True)

# 彻底确保数据类型是浮点型（解决int/float混合导致的np.std报错）
all_data = all_data.astype(numpy.float64)

# 再拆分训练集和测试集（切片）  注意后面加上.values是获取数值，不能只获取表格
X = all_data[:len(train_set)].values
X_test = all_data[len(train_set):].values


# 特征缩放：Z-score标准化（减去均值，除以标准差）

# 求均值(numpy求均值方法，axis=0表示按列求均值，得到一个行向量数组)
mean = numpy.mean(X,axis=0)

#求标准差
std = numpy.std(X,axis=0)

# 将标准差为0的改为1，避免除零错误
std[std == 0]  = 1

# 训练集和测试集都用训练集的均值和标准差来标准化，避免数据泄露
X = (X-mean)/std
X_test = (X_test-mean)/std


# 添加：pca降维（直接用工具PCA，不复现数学原理） 注意必须先标准化

# 保留95%的方差，即保留95%的特征信息量（返回的pca可以看作是帮我们做主成分分析的工具）
pca=PCA(n_components=0.95)

# 对训练集进行降维，只在训练集上拟合（相当于打磨工具）
X=pca.fit_transform(X)

# 测试集只能用训练集拟合好的pca
X_test=pca.transform(X_test)

print("经过独热编码后，降维前特征数：", pca.n_features_in_)
print("降维后特征数：", pca.n_components_)


# 添加偏置项θ0，并与构造一个特征1与之相乘（符合向量内积格式）

# 先获取训练集行数和列数
m,n = X.shape

# 先再每一列行前面加上1，即在第一列之前添加一个全1列
# 先创建一个m行1列的全1矩阵，用于拼接
all_1 = numpy.ones((m,1))
# 横向拼接
X = numpy.hstack([all_1,X])
# 测试集也做相同处理
m,n = X_test.shape
all_1 = numpy.ones((m,1))
X_test = numpy.hstack([all_1,X_test])


# 初始化权重：全零初始化
theta = numpy.zeros(n+1)


# 假设函数：h_θ(x) = θ^T x = θ0 + θ1x1 + ... + θnxn
def Linear_prediction(X,theta):
    return numpy.dot(X,theta)


# 代价函数：J(θ) = 1/(2m) * sum( (h-y)^2 ) + λ/(2m) sum(θ[1:]^2 )  （λ为正则化参数,初始为0）
def cost_function(theta,X,y,lambda_reg=0):
    # 样本数量
    m = len(y)
    h = Linear_prediction(X,theta)
    error = h - y

    cost = numpy.sum(error**2) / (2*m)

    # 正则项（注意：不正则化偏置项θ0）
    reg_term = (lambda_reg / (2*m)) *numpy.sum(theta[1:]**2)

    return cost + reg_term


# 梯度计算
def gradient(theta,X,y,lambda_reg=0): 
    m = len(y)
    h = Linear_prediction(X,theta)
    error = h-y

    # 梯度计算
    grad = numpy.dot(X.T,error) / m

    # 正则项（注意：不正则化偏置项θ0）
    reg_term = (lambda_reg / m) * theta[1:]  # 只对特征项进行正则化

    grad[1:] += reg_term

    return grad


# 批量梯度下降
def BGD(X, y, theta, learing_rate, iterations, lambda_reg=0):

    # 记录每次迭代的代价函数值
    cost_history = []

    # 迭代
    for i in range(iterations):

        #计算梯度
        grad = gradient(theta,X,y,lambda_reg)
        # 更新权重
        theta = theta -learing_rate * grad

        # 计算代价函数值
        cost = cost_function(theta,X,y,lambda_reg)
        cost_history.append(cost)

        # 每迭代1000次查看记录
        if(i % 1000 ==0):
            print(f"第{i:5d}次迭代,代价函数值：{cost:.6f}")

    return theta , cost_history
    

# 训练
learing_rate = 0.05
iterations = 5000
lambda_reg = 5

theta_opt , cost_history = BGD(X, y, theta, learing_rate, iterations, lambda_reg)

# 带入测试集（训练好的权重）  含对数化处理
y_pred_log = Linear_prediction(X_test,theta_opt)

# 取反对数化处理(numpy.expm1相当于 numpy.exp(y_pred_log) - 1)
y_pred = numpy.expm1(y_pred_log)


# 生成提交文件
# 生成提交文件的DataFrame
sub = pandas.DataFrame({
    "Id" : test_id,
    "SalePrice" : y_pred
})

# 生成提交文件CVS文件（注意不要加行号，否则提交格式有误）
sub.to_csv("house_price/output/submission.csv",index=False)

print("\n提交文件已生成：output/submission.csv")










