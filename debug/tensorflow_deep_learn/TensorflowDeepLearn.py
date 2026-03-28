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
plt.show()

# 4.3 相关性热力图（临床特征关联性）
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('特征相关性热力图')
heatmap_path = os.path.join(plot_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
print(f"相关性热力图已保存至: {heatmap_path}")
plt.show()

# ===================== 5. 双递归特征消除(RFE)筛选最优特征子集 =====================
# 目标：剔除冗余/噪声特征，保留T2DM高相关关键指标
print("\n===== 两种RFE特征筛选 =====")

