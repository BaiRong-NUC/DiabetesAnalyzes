import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 重新加载数据
df = pd.read_csv('./diabetes.csv')

# 检查目标变量分布
print("目标变量分布:")
print(df['Outcome'].value_counts())
print(f"\n糖尿病患病率: {df['Outcome'].mean():.2%}")

# 检查异常值(0值在医学上不合理的特征)
print("\n异常值检查(0值统计):")
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} 个0值 ({zero_count/len(df):.2%})")

# 处理异常值：用中位数替换0值
df_clean = df.copy()
for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

print("\n处理后的异常值检查:")
for col in zero_columns:
    zero_count = (df_clean[col] == 0).sum()
    print(f"{col}: {zero_count} 个0值")

# 按年龄分组分析
print("\n按年龄分组分析:")

'''
作用概述：这行代码用 pd.cut 把连续的 df_clean['Age'] 列离散化，生成一个新的分组列 df_clean['Age_Group']（类别型），表示年龄段。

参数含义：

df_clean['Age']：要分箱的连续数值列。
bins=[0,30,40,50,60,100]：定义箱的边界，产生 5 个区间：(0,30], (30,40], (40,50], (50,60], (60,100](默认 right=True,区间右端闭)
labels=['<30','30-40','40-50','50-60','>60']：对应每个区间的标签，赋给新列的类别值。
注意点（常见疑惑）：
因为默认是右闭区间，年龄等于 30 会落到 '<30'(实际上是 ≤30);年龄 60 会落到  '50-60'（因为包含右端）。
若希望左闭右开或把 30 分到下一组，可用 right=False 或调整 bins / labels,或用 include_lowest=True 控制边界包含行为。
示例：

Age=25 → '<30';Age=30 → '<30'(因右闭);Age=61 → '>60'
'''

df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '>60'])
age_group_stats = df_clean.groupby('Age_Group').agg({
    'Outcome': ['count', 'mean'],
    'Glucose': 'mean',
    'BMI': 'mean',
    'BloodPressure': 'mean'
}).round(2)
print(age_group_stats)

# 特征相关性分析
print("\n特征相关性分析:")
# axis=1 表示按列删除；axis=0 表示按行删除
# corr 是 pandas 的方法，用来计算列之间的成对相关系数(默认是 Pearson 相关系数)
correlation_matrix = df_clean.drop('Age_Group', axis=1).corr()

# 保存完整的相关性矩阵到文件，方便用 Excel 或其它工具查看
corr_csv_path = './debug/feature_correlation_analysis/correlation_matrix_full.csv'
correlation_matrix.to_csv(corr_csv_path, float_format='%.6f')
print(f"\n已保存完整相关矩阵到: {corr_csv_path}")

print("与Outcome的相关性:")
print(correlation_matrix['Outcome'].sort_values(ascending=False))

print("\n与Age的相关性:")
print(correlation_matrix['Age'].sort_values(ascending=False))