import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json


# 重新加载和预处理数据
df = pd.read_csv('./diabetes.csv')
df_clean = df.copy()
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

# 准备特征和目标变量
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']
# 用 X(除 Outcome 外的所有列)和 y(Outcome 列)准备特征和目标变量
feature_names = X.columns.tolist() # 获取特征名称列表，方便后续分析和输出

print("=== 特征选择分析 ===")

# 方法1：基于统计检验的特征选择 (SelectKBest with f_classif)
print("\n1. 基于统计检验的特征选择 (ANOVA F-value):")
'''
线性关系
将" 类间变异 "与"类内变异"做比值，衡量不同类别之间特征均值差异相对于组内波动的大
selector_f = SelectKBest(f_classif, k='all'): 创建一个 SelectKBest 实例，
用 f_classif(ANOVA F-value)作为评分函数,k='all' 表示不删除特征值,计算并返回所有特征的得分
F 值大，说明该特征对区分类别有帮助
'''
selector_f = SelectKBest(f_classif, k='all')
selector_f.fit(X, y) #在数据 X(特征向量)和 y(标签)上计算每个特征的 F-value 与对应统计量
scores_f = selector_f.scores_ # 从已拟合的选择器中取出每个特征对应的 F 分数(scores_ 是一个与特征数量相同的一维数组)
print("特征得分:")
for name, score in zip(feature_names, scores_f):
    print(f"{name}: {score:.4f}")

# 方法2：基于互信息的特征选择
print("\n2. 基于互信息的特征选择:")
'''
mutual_info_classif 是计算特征与分类目标之间互信息(Mutual Information, MI)的评分函数
用于 SelectKBest 之类的单变量过滤器中,度量“特征能减少多少关于标签的不确定性
MI 表示知道特征 X 后对标签 Y 不确定性减少的程度（信息增益）值越大,说明该特征包含越多关于类别的信息
当 X 与 Y 独立时 I(X;Y)=0
'''
selector_mi = SelectKBest(mutual_info_classif, k='all')
selector_mi.fit(X, y)
scores_mi = selector_mi.scores_
print("特征得分:")
for name, score in zip(feature_names, scores_mi):
    print(f"{name}: {score:.4f}")

# 方法3：递归特征消除 (RFE)
print("\n3. 递归特征消除 (RFE):")
'''
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
创建一个随机森林分类器作为基学习器。n_estimators=100 指森林中树的数量,random_state=42 固定随机性以便复现。
随机森林能捕捉非线性与特征间交互，且基于树的特征重要性度量（如基尼或信息增益的平均下降）。
随机森林不是越多越好:增加 n_estimators 能降低模型方差、提高稳定性与性能（尤其在小/中等树数时提升明显），但到某个点后性能提升变得很小（收益递减）
'''
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
'''
使用 estimator 做递归特征消除。
n_features_to_select=1 表示最终只保留 1 个特征（因此 RFE 会继续消除直到剩 1 个），同时会为每个特征生成完整的排名（1..p）。
step=1 表示每次迭代去掉 1 个最不重要的特征（step 也可以是百分比）。
过程是贪心的、基于模型的重要性度量，顺序删除会影响最终排名（不是全局最优但常用且有效）。
败者树
'''
selector_rfe = RFE(estimator, n_features_to_select=1, step=1)
selector_rfe.fit(X, y)
print("特征排名 (1=最重要):")
for name, rank in zip(feature_names, selector_rfe.ranking_):
    print(f"{name}: 排名 {rank}")

# 综合三种方法的特征重要性
print("\n=== 综合特征重要性分析 ===")
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'ANOVA_F': scores_f,
    'Mutual_Info': scores_mi,
    'RFE_Rank': selector_rfe.ranking_
})

# 归一化ANOVA和互信息得分
'''
ANOVA 和互信息得分量纲/范围不同，先除以各自的最大值把它们缩放到 [0,1]，便于按权重合并；这是一种简单的最值归一化（max-normalization）。
RFE 给出的是“排名”（1 为最好），通过取倒数把排名转换为“分数”：排名越小，RFE_Score 越大（例如 rank=1 → score=1；rank=5 → score=0.2）。
综合得分 Overall_Score 是三种方法得分的加权平均，权重可以根据实际情况调整（这里示例是 0.3, 0.3, 0.4）。

归一化: 统一量程
'''
feature_importance['ANOVA_F_Norm'] = feature_importance['ANOVA_F'] / feature_importance['ANOVA_F'].max()
feature_importance['Mutual_Info_Norm'] = feature_importance['Mutual_Info'] / feature_importance['Mutual_Info'].max()
feature_importance['RFE_Score'] = 1 / feature_importance['RFE_Rank']  # 排名越低分数越高

# 计算综合得分
'''
权重为经验值，最好用交叉验证或基于下游模型性能来调整权重或直接用学习器做特征选择。
'''
feature_importance['Overall_Score'] = (
    feature_importance['ANOVA_F_Norm'] * 0.3 +
    feature_importance['Mutual_Info_Norm'] * 0.3 +
    feature_importance['RFE_Score'] * 0.4
)

# 按综合得分排序
feature_importance = feature_importance.sort_values('Overall_Score', ascending=False)
print(feature_importance[['Feature', 'Overall_Score']].round(4))

# 准备训练数据
print("\n=== 准备模型训练数据 ===")
# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
# test_size=0.2: 常用比例，保留约20%作为独立评估集合
# 训练集与测试集在样本量上有平衡（不是硬性规则，可根据数据量调整）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 保存前10条样例数据用于可视化
sample_data = []
for i in range(min(10, len(df_clean))):
    sample_data.append({
        'Pregnancies': int(df_clean.iloc[i]['Pregnancies']),
        'Glucose': int(df_clean.iloc[i]['Glucose']),
        'BloodPressure': int(df_clean.iloc[i]['BloodPressure']),
        'SkinThickness': int(df_clean.iloc[i]['SkinThickness']),
        'Insulin': int(df_clean.iloc[i]['Insulin']),
        'BMI': float(df_clean.iloc[i]['BMI']),
        'DiabetesPedigreeFunction': float(df_clean.iloc[i]['DiabetesPedigreeFunction']),
        'Age': int(df_clean.iloc[i]['Age']),
        'Outcome': int(df_clean.iloc[i]['Outcome'])
    })

print(f"\n样例数据（前10条）已准备完成")

# 保存特征重要性数据
feature_importance_data = []
for _, row in feature_importance.iterrows():
    feature_importance_data.append({
        'feature': row['Feature'],
        'score': float(row['Overall_Score']),
        'anova_f': float(row['ANOVA_F']),
        'mutual_info': float(row['Mutual_Info']),
        'rfe_rank': int(row['RFE_Rank'])
    })

print(f"特征重要性数据已准备完成")

pd.DataFrame(feature_importance_data).to_csv('./debug/model_building/feature_importance.csv',index=False)