# ===================== 1. 基础环境配置 & 库导入 =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier

# 深度学习
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# 模型可解释性
import shap
from lime import lime_tabular

from IPython.display import display, HTML

# 绘图设置
from matplotlib import font_manager as fm
import os
# 更稳健地加载中文字体：优先使用系统微软雅黑，如不存在则使用多个候选回退
font_path = r'C:\Windows\Fonts\msyh.ttf'
if os.path.exists(font_path):
    fp = fm.FontProperties(fname=font_path)
    font_name = fp.get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.sans-serif'] = [font_name]
else:
    # 尝试常见中文字体名作为回退（按顺序尝试）
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']

    
plt.rcParams['axes.unicode_minus'] = False
print("库导入完成 | TensorFlow版本:", tf.__version__)

# ===================== 2. 加载数据 & 基础信息查看 =====================
# Kaggle自带糖尿病数据集（T2DM临床数据）
df = pd.read_csv('./diabetes.csv')
print("数据集形状:", df.shape)
print("\n数据集前5行:")
print(df.head())
print("\n数据类型&缺失值:")
print(df.info())

# ===================== 3. 临床导向数据预处理 =====================
# T2DM临床常识：血糖、血压、皮厚、胰岛素、BMI=0为无效值，替换为中位数
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

# 划分特征与标签
X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns.tolist()

# 数据标准化（深度学习必备）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# 训练集/测试集划分（8:2）
'''
test_size=0.2:将20%的数据作为测试集，80%作为训练集。
random_state=42:可复现随机打乱
stratify=y:保持训练集和测试集中目标变量的分布一致，适用于不平衡数据集
'''
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n预处理完成 | 训练集形状:", X_train.shape, "测试集形状:", X_test.shape)

# ===================== 4. 描述性统计 + 可视化 + 相关性分析 =====================
# 4.1 描述性统计
print("\n===== 描述性统计 =====")
print(df.describe().T)

'''
箱线图(Boxplot):
中间线 = 中位数
超出部分标为孤立点(异常值)

四分位数(quartile)是指在统计学中把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值。
四分位数也被称为四分位点，它是将全部数据分成相等的四部分，
其中每部分包括 25%的数据，处在各分位点的数值就是四分位数。
四分位数有三个，第一个四分位数是下四分位数，第二个四分位数就是中位数，
第三个四分位数称为上四分位数，分别用 Q1、Q2、Q3表示

四分位距IQR:是Q3-Q1,也就是说上下四分位数的差值。
上下限:上下限并不是整个数据样本的最大值和最小值
上限 = 去除异常值的最大值(Q3+1.5IQR)和 下限 = 去除异常值的最小值(Q1-1.5IQR)
在上下限这里分别划出两条线段作为异常值的分界点。

那么在箱线图中，上下限之间就是数据样本的正常分布区间，超出上下限就定义为异常值。
'''
# 4.2 特征分布可视化（箱线图+直方图）
# 创建保存图像的目录
plot_dir = os.path.join(os.getcwd(), 'debug', 'tensorflow_deep_learn', 'plots')
os.makedirs(plot_dir, exist_ok=True)

plt.figure(figsize=(16, 12))
for i, col in enumerate(feature_names, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f'{col} vs 糖尿病标签')
plt.tight_layout()
boxplot_path = os.path.join(plot_dir, 'boxplots_by_outcome.png')
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"箱线图已保存至: {boxplot_path}")
# plt.show()

# 4.3 相关性热力图（临床特征关联性）
plt.figure(figsize=(10, 8))
corr = df.corr()
# 限定色阶为 [-1, 1]，并以 0 为中心，适合相关系数矩阵
sns.heatmap(corr.clip(-1, 1), annot=True, cmap='coolwarm', fmt='.2f',
            vmin=-1, vmax=1, center=0,
            cbar_kws={'ticks': [-1, -0.5, 0, 0.5, 1]})
plt.title('特征相关性热力图')
heatmap_path = os.path.join(plot_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"相关性热力图已保存至: {heatmap_path}")
# plt.show()

# ===================== 5. 双递归特征消除(RFE)筛选最优特征子集 =====================
# 目标：剔除冗余/噪声特征，保留T2DM高相关关键指标
print("\n===== 两种RFE特征筛选 =====")

'''
RFE 概念:
使用 RFE（递归特征消除）按模型重要性反复剔除最不重要的特征，直到剩下指定数量（代码中 n_features_to_select=6）。
每次基于所用估计器（如 LogisticRegression 或 XGBClassifier）
计算特征权重/重要性，RFE.fit() 后可通过 support_ 得到被保留的特征索引。
'''


# 方法1：RFE + 逻辑回归（线性基准）
# 用某估计器递归筛到 6 个特征。
'''
RFE筛选流程
1. 用当前特征集训练基估计器（如线性模型或树）。
2. 依据估计器给出的重要性（线性模型的系数绝对值或树的 feature_importances_）为每个特征打分。
    打分标准: 不同的模型不同 - 线性模型：特征系数的绝对值（|coef|）越大，重要性越高；树模型：特征重要性（feature_importances_）越高，重要性越高。
3. 删除得分最低的特征(默认每次删 1 个，可通过 step 参数加速删除多个)
4. 用剩余特征重复步骤 1-3,直到达到 n_features_to_select
'''
rfe_lr = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=6)
# 在训练集上拟合并返回降维后的训练数据。
X_rfe_lr = rfe_lr.fit_transform(X_train, y_train)
lr_selected = [feature_names[i] for i in np.where(rfe_lr.support_)[0]] # 获取被选中特征的名称列表
# 打印逻辑回归 RFE 在最终保留特征上的重要性（绝对系数）
try:
    lr_coef = np.abs(rfe_lr.estimator_.coef_).ravel()
    lr_idx = np.where(rfe_lr.support_)[0]
    print("Logistic RFE - selected feature importances (abs coef):")
    for idx, val in zip(lr_idx, lr_coef):
        print(f"  {feature_names[idx]}: {val:.4f}")
except Exception as e:
    print("无法打印 Logistic RFE 系数:", e)

# 方法2：RFE + XGBoost（树模型基准，更适配医疗数据）
rfe_xgb = RFE(estimator=XGBClassifier(random_state=42), n_features_to_select=6)
X_rfe_xgb = rfe_xgb.fit_transform(X_train, y_train)
xgb_selected = [feature_names[i] for i in np.where(rfe_xgb.support_)[0]]
# 打印 XGBoost RFE 在最终保留特征上的重要性
try:
    xgb_imp = rfe_xgb.estimator_.feature_importances_
    xgb_idx = np.where(rfe_xgb.support_)[0]
    print("XGBoost RFE - selected feature importances:")
    for idx, val in zip(xgb_idx, xgb_imp):
        print(f"  {feature_names[idx]}: {val:.4f}")
except Exception as e:
    print("无法打印 XGBoost RFE 特征重要性:", e)

# 取交集 → 最优临床特征子集（高稳定性）
best_features = list(set(lr_selected) & set(xgb_selected))
print(f"逻辑回归RFE选中特征: {lr_selected}")
print(f"XGBoost RFE选中特征: {xgb_selected}")
print(f"✅ 最优特征子集: {best_features}")

# 筛选最终特征
X_train_final = X_train[best_features]
X_test_final = X_test[best_features]
print(f"筛选后特征数: {len(best_features)}")

# ===================== 6. 结合T2DM临床机制构建特征交互项(升维) =====================
# 临床公认交互：胰岛素抵抗(Glucose×BMI)、年龄代谢(Age×BMI)、血糖血压联合作用
def build_clinical_interaction(X, features):
    X_inter = X.copy()
    # 1. 血糖 × BMI（T2DM核心致病交互）
    if 'Glucose' in features and 'BMI' in features:
        X_inter['Glucose_BMI'] = X['Glucose'] * X['BMI']
    # 2. 年龄 × BMI（中老年肥胖风险）
    if 'Age' in features and 'BMI' in features:
        X_inter['Age_BMI'] = X['Age'] * X['BMI']
    # 3. 血糖 × 血压（代谢综合征核心）
    if 'Glucose' in features and 'BloodPressure' in features:
        X_inter['Glucose_BP'] = X['Glucose'] * X['BloodPressure']
    return X_inter

# 构建交互特征
X_train_clinical = build_clinical_interaction(X_train_final, best_features)
X_test_clinical = build_clinical_interaction(X_test_final, best_features)
print(f"\n构建临床交互后训练集形状: {X_train_clinical.shape}")
print(f"最终输入特征: {X_train_clinical.columns.tolist()}")

# 适配深度学习输入格式
'''
X_train_clinical(pandas.DataFrame).shape[1]:特征维度（输入层神经元数量）
X_train_np是纯数值矩阵、无列名但更接近深度学习框架的输入格式
'''
input_dim = X_train_clinical.shape[1]
print(f"输入层维度（特征数）: {input_dim}")
X_train_np = X_train_clinical.values
print(f"训练集输入格式: {type(X_train_np)}, 形状: {X_train_np.shape}")
X_test_np = X_test_clinical.values
print(f"测试集输入格式: {type(X_test_np)}, 形状: {X_test_np.shape}")


# ===================== 7. 模型构建：基准模型 + 2个指定深度学习模型 =====================
# 通用评估函数
def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    res = {
        '模型': model_name, '准确率': round(acc,4), 'AUC': round(auc,4),
        'F1': round(f1,4), '精确率': round(precision,4), '召回率': round(recall,4)
    }
    return res

results = []

# -------------------- 基准模型：XGBoost（强基准） -------------------- 
# 只用了传统机器学习（XGBoost）
xgb_base = XGBClassifier(random_state=42)
xgb_base.fit(X_train_clinical, y_train)
y_pred_xgb = xgb_base.predict(X_test_clinical)
y_pred_proba_xgb = xgb_base.predict_proba(X_test_clinical)[:,1]
results.append(evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost基准'))

print("\n基准模型评估结果:")
print(results[0])


# -------------------- 深度学习1：Inception网络 + LSTM --------------------

'''
build_inception_lstm(input_dim) 构建并返回一个已编译的 Keras 二分类模型
先用 Inception 风格的多卷积分支抽取局部模式，再用 LSTM 聚合序列信息，最后接全连接输出概率。
'''

'''
Inception（多尺度并行卷积）用于并行提取不同感受野的局部模式
    Inception 通过在同一层并行使用不同大小的卷积核（以及池化/1×1卷积）来同时观察不同尺度的局部模式
    然后把这些多尺度特征拼接（concatenate）起来，从而“并行提取不同感受野”的信息
    把输入同时用小放大镜、中放大镜、大放大镜看一遍，三个视角都能发现不同尺度的“图案”再把三者合起来给后续模块判断。
再用 LSTM 对提取到的“序列/步长”信息做时序/依赖建模，最后汇总为分类/回归输出
'''

def build_inception_lstm(input_dim):
    # 重塑为序列格式 (样本, 时间步, 特征)
    # 张量是一个二维表(矩阵) 大小batch_size(这组样本的数量(人数)) × input_dim，每行是一个样本的特征向量
    '''
    batch size 指的是 一次参数更新时送进模型的样本数，也就是“一小批”样本的大小，不是整个数据集的总样本数。
    举个例子：

    总样本数 = 1000
    batch size = 8 	每批放 8 条数据
    那么每个 epoch 会分成大约 
    1000/8≈125，也就是 125 个 batch（最后一个可能不足 8）。 共分成 125 批次
    模型每处理完一个 batch，就会做一次反向传播和参数更新。
    '''
    inputs = layers.Input(shape=(input_dim,))
    '''
    x 是重塑后的中间张量(Tensor),作为后续卷积/LSTM 层的输入.
    把每个样本的特征向量看成长度为 input_dim 的一维序列
    Reshape 要保持每个样本的元素总数不变 为input_dim个
    '''
    # x = layers.Reshape((1, input_dim))(inputs)
    x = layers.Reshape((input_dim,1))(inputs)
    
    # Inception多分支卷积
    '''
    ilters=32:该卷积分支输出的通道数 (即每个时间步输出向量长度)每个分支产生 (batch_size, timesteps, 32)
        每个滤波器会沿时间/位置维滑动并在每个位置产生一个数值
        （形成一个长度为 timesteps 的特征图）。所以每个滤波器输出 timesteps 个值
        因此总的输出元素数（每个样本）是 timesteps × filters;但“通道数”仍然等于 filters
    kernel_size=1/3/5 卷积核宽度,放大镜大小,越大感受野越大,能捕捉更大范围的特征模式
    padding='same': 输出时间长度与输入时间长度保持一致（边界使用零填充）,这里是input_dim=8
    activation='relu': 非线性激活函数，增加表达能力并引入稀疏激活。
        ReLU 会把负值置为0，导致很多神经元在某些输入下输出0，形成稀疏激活，有助于简化表示并具备一定正则化效果。
        f(x) = max(0, x) 负数输出0，正数保持不变，简单高效且能缓解梯度消失问题。    

    设 input_dim=8
    Conv1D(32,1) 参数 = 1*8*32 + 32 = 288
    Conv1D(32,3) 参数 = 3*8*32 + 32 = 800
    Conv1D(32,5) 参数 = 5*8*32 + 32 = 1296
    每个分支输出形状：(batch, timesteps, 32)

    这段卷积在“学什么”
    k=1 学单特征位置的线性变换（更像逐位置通道投影）
    k=3 学相邻 3 个特征之间的局部组合
    k=5 学更宽范围（5 个特征）的组合关系

    '''
    # 卷积层+过了激活函数
    branch1 = layers.Conv1D(32, 1, padding='same', activation='relu')(x)
    branch2 = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    branch3 = layers.Conv1D(32, 5, padding='same', activation='relu')(x)
    # 沿通道维拼接，结果形状 (batch, timesteps, 32*3)，即 (batch,8,96) 96个通道
    concat = layers.concatenate([branch1, branch2, branch3], axis=-1)
    
    # RNN层
    # LSTM提取时序特征
    '''
    建一个有 64 个隐藏单元的 LSTM 层,return_sequences=False 表示只输出最后一个时刻的隐藏状态
    输出形状为 (batch, 64)，然后接全连接层做分类。
    前面提取的“局部模式”进一步整合成一个全局表示
    LSTM 确实顺序读了 1 到 8 步。
    return_sequences=False 时，输出的是最后时刻的隐藏状态 h8。
    这个 h8 不是“第8步原始输入 x8”，而是融合了前面 1 到 8 步信息后的摘要。
    '''
    lstm = layers.LSTM(64, return_sequences=False)(concat)

    # 全连接层
    # 对输入做线性变换 z = xW + b 并经过激活函数，学习特征组合映射
    # kernel_regularizer=regularizers.l2(0.01) 在权重上加 L2 惩罚，抑制过拟合。输出形状为 (batch, 32)。
    dense = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(lstm)
    '''
    Dropout(0.3) 在训练时随机丢弃 30% 神经元，减少共适应，作为正则化手段,一般为30%-50%
    '''
    drop = layers.Dropout(0.3)(dense)
    '''
    输出层:Dense(1, activation='sigmoid') 将最后的标量映射到 (0,1)，表示二分类预测的概率
    只有 1 个神经元，对应二分类只需输出一个概率值
    将任意实数压缩到 (0, 1) 区间，代表"患糖尿病的概率"
    接收上一层 Dropout 的输出作为输入
    '''
    outputs = layers.Dense(1, activation='sigmoid')(drop)

    # 把前面定义的所有层"串联"成一个完整的计算图：
    model = Model(inputs, outputs)
    '''
    optimizer='adam'	优化器，自适应学习率，负责每轮更新模型权重
    loss ='binary_crossentropy'	损失函数，衡量预测概率与真实标签的差距，专用于二分类
    metrics=['accuracy']	训练过程中额外监控的指标，这里是准确率（不影响优化，只用于展示）
    '''
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 训练基础深度学习模型
# 耐心值 5,连续 5 轮 验证集 val_loss 没有改善，就提前停止训练。
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Inception-LSTM
model_incep = build_inception_lstm(input_dim)
# 最大轮数：30 轮
'''
validation_split=0.1	0.1	从训练集末尾划出 10% 作为验证集，用于监控 val_loss（不参与训练）
callbacks=[early_stop]	EarlyStopping每轮结束后触发回调，val_loss 连续 5 轮无改善则提前停止
训练过程静默，不打印进度条（1=进度条，2=每轮一行）
'''
model_incep.fit(X_train_np, y_train, epochs=30, batch_size=8,
                validation_split=0.1, callbacks=[early_stop], verbose=0)
y_pred_incep = (model_incep.predict(X_test_np, verbose=0) > 0.5).astype(int).flatten()
y_pred_proba_incep = model_incep.predict(X_test_np, verbose=0).flatten()
results.append(evaluate_model(y_test, y_pred_incep, y_pred_proba_incep, 'Inception-LSTM'))

print("\nInception-LSTM评估结果:")
print(results[1])