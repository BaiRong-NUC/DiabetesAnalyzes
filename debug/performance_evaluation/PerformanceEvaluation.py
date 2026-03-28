import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

# 性能评估

# 重新加载和预处理数据
df = pd.read_csv('./diabetes.csv')
df_clean = df.copy()
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, df_clean[col].median())

# 准备特征和目标变量
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# 标准化特征
# 使用 StandardScaler 对所有特征做 z-score 标准化（均值 0、方差 1）平稳
# 梯度下降类方法收敛更快、更稳定；特征缩放可减少数值溢出/条件数问题
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("=== 模型训练与评估 ===")

# 1. XGBoost基准模型
print("\n1. XGBoost模型:")
'''
GBoost(eXtreme Gradient Boosting)是基于梯度提升树(GBDT)的高效实现
支持列采样、正则化和并行计算，常用于二分类、多分类与回归任务

加法模型: 每一步拟合前一步残差，最终模型为多个弱学习器的线性组合，形式上可写为...
目标函数: 包含训练损失 + 正则化项（对叶节点分数和树结构的惩罚），通过二阶泰勒展开近似一阶/二阶梯度来高效优化

常用二分类指标：
logloss:对数损失(binary logistic logloss)。
error:分类错误率（预测类别与真实不同时计数）。
auc:ROC 曲线下面积(Area Under ROC)。
aucpr:PR 曲线下面积(Area Under Precision-Recall)，部分版本可用。

多分类指标/组合多个指标等省略
'''
# 创建XGBoost分类器，设置随机种子和评估指标
# use_label_encoder=False：禁止 XGBoost 的旧标签编码器以避免警告（sklearn 兼容性设置）。
# eval_metric='logloss': 训练期间用对数损失作为评估指标（仅用于训练/早停输出，不影响 sklearn 的 fit 返回）

xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train) # 在训练集上拟合模型（默认不使用早停，除非传入验证集和 early_stopping_rounds）

# 预测
y_pred_xgb = xgb_model.predict(X_test) # 输出类别预测（0/1）

# 阈值
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1] # 输出正类的预测概率，用于计算 roc_auc_score

# 评估
'''
真阳性 (TP): 预测为正类且真实为正类的样本数。
真阴性 (TN): 预测为负类且真实为负类的样本数。
假阳性 (FP): 预测为正类但真实为负类的样本数 (Type I 错误)。
假阴性 (FN): 预测为负类但真实为正类的样本数 (Type II 错误)。
'''
xgb_results = {
    'model': 'XGBoost',
    'accuracy': float(accuracy_score(y_test, y_pred_xgb)), # 反映总体正确率=(TP+TN)/样本总数
    'precision': float(precision_score(y_test, y_pred_xgb)), # 反映正类预测的准确性=TP/(TP+FP)
    'recall': float(recall_score(y_test, y_pred_xgb)), # 反映正类的覆盖率=TP/(TP+FN)
    'f1_score': float(f1_score(y_test, y_pred_xgb)), # F1分数=2*(precision*recall)/(precision+recall) 精确率和召回率的调和平均，综合两者性能
    'auc': float(roc_auc_score(y_test, y_prob_xgb)) # AUC=ROC曲线下的面积
}

'''
ROC 曲线下面积,衡量模型按概率排序正负样本的能力.
AUC 取值范围 [0, 1],越接近 1 表示模型性能越好,0.5 表示随机猜测。
ROC曲线 横轴: 假阳性率 (FPR) = FP/(FP+TN)，纵轴: 真阳性率 (TPR) = TP/(TP+FN)
点对应一个概率阈值 t —— 当模型预测概率 ≥ t 时判定为正类。
把所有阈值的 (FPR, TPR) 连起来就是 ROC 曲线;AUC 是该曲线下的面积，表示总体排序能力
'''

print(f"准确率: {xgb_results['accuracy']:.4f}")
print(f"精确率: {xgb_results['precision']:.4f}")
print(f"召回率: {xgb_results['recall']:.4f}")
print(f"F1分数: {xgb_results['f1_score']:.4f}")
print(f"AUC: {xgb_results['auc']:.4f}")

# 2. 随机森林模型
print("\n2. 随机森林模型:")
'''
随机森林通过对训练集做有放回的自助采样(bootstrap)产生每棵树的训练子集
每次节点分裂时,只在随机选取的一部分特征上寻找最佳分裂（特征子采样）,增加树之间差异性，降低方差并提高泛化能力
每棵树通常不剪枝(或只受 max_depth 限制),是高方差低偏差的基学习器,通过集成平均减少过拟合


n_estimators=100: 森林中树的数量（较大数量通常更稳定，但更慢）
稳定指的是:模型预测结果和性能指标在随机性（不同 bootstrap 样本、不同树的随机性）下的波动变小
也就是方差下降，结果更可重复、更平滑
随机森林是把多棵随机化的树的预测取平均/投票

random_state=42: 随机种子，保证实验可重复。
max_depth, max_features, min_samples_leaf 等）使用默认值，可能不是最优

对特征尺度不敏感，能处理连续与类别特征。
抗过拟合能力较强（比单棵决策树稳健）。
可以直接获取特征重要性 (rf_model.feature_importances_)。
少量参数需要调优即可得到不错结果。

多类分类: 支持 >2 类问题，直接用 RandomForestClassifier。
回归: 用 RandomForestRegressor 预测连续目标（房价、温度等）。
多输出任务: 处理多个目标变量（多输出回归/分类），可用 MultiOutputRegressor/MultiOutputClassifier 包装。
特征选择/重要性: 用 feature_importances_ 快速筛选重要特征；也可结合 SHAP/PD。
'''
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test) # 对每棵树进行分类投票，返回多数投票类别（0 或 1）
y_prob_rf = rf_model.predict_proba(X_test)[:, 1] # 返回被判为正类（1）的估计概率（各树概率平均或投票占比），用于计算 ROC/AUC。

rf_results = {
    'model': 'Random Forest',
    'accuracy': float(accuracy_score(y_test, y_pred_rf)),
    'precision': float(precision_score(y_test, y_pred_rf)),
    'recall': float(recall_score(y_test, y_pred_rf)),
    'f1_score': float(f1_score(y_test, y_pred_rf)),
    'auc': float(roc_auc_score(y_test, y_prob_rf))
}

print(f"准确率: {rf_results['accuracy']:.4f}")
print(f"精确率: {rf_results['precision']:.4f}")
print(f"召回率: {rf_results['recall']:.4f}")
print(f"F1分数: {rf_results['f1_score']:.4f}")
print(f"AUC: {rf_results['auc']:.4f}")

# 3. 逻辑回归模型
print("\n3. 逻辑回归模型:")
'''
max_iter=1000: 最大迭代次数，防止在数据较大或特征相关时出现收敛警告；默认较小时可能不收敛所以显式增大
数据需要使用StandardScaler进行标准化,这是必要且合适的步骤,因为逻辑回归对尺度敏感,尤其使用正则化时

二分类问题的基线模型（医学诊断、信用评分、用户流失预测等）。
需要可解释性（系数可解释为方向和强度）的场景。
特征与目标在对数几率上近似线性关系时效果好。
'''
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

lr_results = {
    'model': 'Logistic Regression',
    'accuracy': float(accuracy_score(y_test, y_pred_lr)),
    'precision': float(precision_score(y_test, y_pred_lr)),
    'recall': float(recall_score(y_test, y_pred_lr)),
    'f1_score': float(f1_score(y_test, y_pred_lr)),
    'auc': float(roc_auc_score(y_test, y_prob_lr))
}

print(f"准确率: {lr_results['accuracy']:.4f}")
print(f"精确率: {lr_results['precision']:.4f}")
print(f"召回率: {lr_results['recall']:.4f}")
print(f"F1分数: {lr_results['f1_score']:.4f}")
print(f"AUC: {lr_results['auc']:.4f}")

# 4. SVM模型
print("\n4. SVM模型:")
'''
SVC 是 scikit-learn 的支持向量机分类器(默认核为 RBF)
probability=True 启用概率估计（内部用 Platt scaling 校准），会在 fit 时额外做交叉验证/拟合
random_state=42 固定随机性（某些内部随机过程），便于可重复性


probability=True 会降低训练速度并增加内存开销；若只关心决策边界可设为 False，但无法直接得概率。
若数据量大且只需线性决策边界，考虑 LinearSVC（不支持 predict_proba）或 SGDClassifier（可用 loss='hinge' 或 log）。
由于 SVM 对特征尺度非常敏感，必须先做标准化 StandardScaler。

适合小到中等规模的数据集，二分类问题效果好；在特征维度较高（如文本、基因组）且样本数量相对较少时表现优异
因为它能在高维空间找到分隔超平面,当类别非常多时，通常更推荐树模型或线性方法
'''
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

svm_results = {
    'model': 'SVM',
    'accuracy': float(accuracy_score(y_test, y_pred_svm)),
    'precision': float(precision_score(y_test, y_pred_svm)),
    'recall': float(recall_score(y_test, y_pred_svm)),
    'f1_score': float(f1_score(y_test, y_pred_svm)),
    'auc': float(roc_auc_score(y_test, y_prob_svm))
}

print(f"准确率: {svm_results['accuracy']:.4f}")
print(f"精确率: {svm_results['precision']:.4f}")
print(f"召回率: {svm_results['recall']:.4f}")
print(f"F1分数: {svm_results['f1_score']:.4f}")
print(f"AUC: {svm_results['auc']:.4f}")

# 模型比较
print("\n=== 模型性能比较 ===")
models_results = [xgb_results, rf_results, lr_results, svm_results]
comparison_df = pd.DataFrame(models_results)
print(comparison_df.round(4))

# 找出最佳模型
best_auc_model = max(models_results, key=lambda x: x['auc'])
best_f1_model = max(models_results, key=lambda x: x['f1_score'])
best_acc_model = max(models_results, key=lambda x: x['accuracy'])

print(f"\n最佳AUC模型: {best_auc_model['model']} (AUC: {best_auc_model['auc']:.4f})")
print(f"最佳F1分数模型: {best_f1_model['model']} (F1: {best_f1_model['f1_score']:.4f})")
print(f"最佳准确率模型: {best_acc_model['model']} (准确率: {best_acc_model['accuracy']:.4f})")

# 准备模型调参 - XGBoost超参调优
'''
功能定位：
sklearn (sklearn.ensemble)：是一个涵盖多种传统机器学习算法（回归、分类、聚类、降维等）的通用框架。
XGBoost (xgboost.sklearn)：专专注于梯度提升（Gradient Boosting）决策树的高效工具库。
接口一致性：xgboost.sklearn.XGBClassifier 或 XGBRegressor 完全遵循 sklearn 的 Estimator API 规范
    可以直接在 Pipeline 或 GridSearchCV 中使用。
xgboost.sklearn 下的XGBClassifier

超参数（hyperparameter）是在训练前设置的模型配置项，不由训练数据直接学习得到。
超参数调优就是通过系统方法寻找一组最优的超参数，使模型在指定评估指标上表现最好
    - 网格搜索（Grid Search）：穷举指定参数组合，配合交叉验证评估。
    - 随机搜索（Randomized Search）：随机采样参数空间，效率更高（高维时）。
    - 贝叶斯优化（Bayesian Optimization）：基于先验/后验模型智能选择候选参数。
    - 进化算法、超带宽搜索等更高级方法。

n_estimators: 弱分类器的数量。
booster：用于指定弱学习器的类型，默认值为 ‘gbtree’，表示使用基于树的模型进行计算。还可以选择为 ‘gblinear’ 表示使用线性模型作为弱学习器。
learning_rate：指定学习率。默认值为0.3。推荐的候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]。实现shrinkage的参数，论文中描述：shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model。（降低一颗树在模型中的影响）
gamma：指定叶节点进行分支所需的损失减少的最小值，默认值为0。设置的值越大，模型就越保守。推荐的候选值为：[0, 0.05 ~ 0.1, 0.3, 0.5, 0.7, 0.9, 1]。每一次分裂时，都会计算损失减少值，如果最大的减少值依然少于gamma，则不分裂，可以避免过拟合。
reg_alpha：L1正则化权重项，增加此值将使模型更加保守。推荐的候选值为：[0, 0.01~0.1, 1]
reg_lambda：L2正则化权重项，增加此值将使模型更加保守。推荐的候选值为：[0, 0.1, 0.5, 1]
max_depth：指定树的最大深度，默认值为6，合理的设置可以防止过拟合。[3, 5, 6, 7, 9, 12]。
min_child_weight：就是叶子上的最小样本数。推荐的候选值为：。[1, 3, 5, 7]
colsample_bytree: 列采样比例。在构建一棵树时，会采样一个特征集合，采样比例通过colsample_bytree控制，默认为1，即使用全部特征。
'''

'''
线性模型可以超参:
penalty: 正则化类型（'l1','l2','elasticnet','none'，不同 solver 支持不同选项）
C: 正则化强度的倒数（越小正则化越强，常用对数刻度搜索）。初始可用 [1e-3, 1e3] 粗调，再细化
solver: 求解器（常用 'lbfgs','liblinear','saga','newton-cg'，与 penalty 有兼容性约束）。
max_iter: 最大迭代次数。默认 100–1000；若出现收敛警告，增大到 1000–5000。
tol: 收敛阈值。常用 1e-4（默认）到 1e-6（更严格）
class_weight: 处理类别不平衡（None 或 'balanced' 或字典）。
fit_intercept: 是否学习截距（True/False）。通常 True（除非数据已中心化）
warm_start: 是否热启动（对增量调参有用）。
l1_ratio: 仅当 penalty='elasticnet' 且 solver='saga' 时有效（L1/L2 比例）。
dual: 在特殊 solver/问题下使用（通常不常用）。
random_state: 某些 solver（如 saga）的随机种子。
'''
print("\n=== XGBoost模型调参 ===")
param_grid = {
    'n_estimators': [50, 100, 200], # 弱学习器（树）的数量；值越大训练更充分但更慢、易过拟合
    'max_depth': [3, 5, 7], # 树的最大深度；值越大模型越复杂，可能过拟合
    'learning_rate': [0.01, 0.1, 0.2], # 学习率；值越小训练越慢但更稳健
    'subsample': [0.8, 0.9, 1.0] # 每棵树训练样本的比例；值越小增加随机性，防止过拟合
} # 组合构成穷举网格，GridSearchCV 会尝试所有组合。

'''
estimator: 使用的是 xgb.XGBClassifier
cv=5: 5 折交叉验证（默认是普通 KFold）；
    将训练集分成 5 份（每份约为训练集的 20%）；每次用 4 份做训练、1 份做验证，循环 5 次，覆盖所有子集作为验证集。
    对每组超参数组合计算 5 次验证得分，取平均作为该组合的交叉验证得分，用于比较和选择最优参数。
    在分类任务中，sklearn 对 cv=int 会自动使用分层折（StratifiedKFold），以保持每折中各类比例接近总体比例。
    优点：更稳定、鲁棒的性能估计；缺点：计算成本约为训练次数的 5 倍。
    可替换为其它策略（例如 cv=StratifiedKFold(n_splits=5)、更大/更小的 n_splits 或 RandomizedSearchCV）
    根据数据量和计算资源调整。
scoring='f1': scoring='f1' 告诉 GridSearchCV 用 F1 分数作为超参数组合优劣的评价指标（交叉验证时计算并取平均）
    最终选择使该指标最大的参数组合
    F1分数=2*(precision*recall)/(precision+recall) 精确率和召回率的调和平均，综合两者性能

n_jobs=-1: 并行化所有可用 CPU 核心加速搜索。可以根据机器资源调整

GridSearchCV 默认会在内层用训练折做验证并在所有候选上计算交叉验证得分
最后 refit=True（默认）会用整个训练集训练出 best_estimator_
'''
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证F1分数: {grid_search.best_score_:.4f}")

# 使用最佳参数重新训练
xgb_tuned = grid_search.best_estimator_
y_pred_xgb_tuned = xgb_tuned.predict(X_test)
y_prob_xgb_tuned = xgb_tuned.predict_proba(X_test)[:, 1]

xgb_tuned_results = {
    'model': 'XGBoost (Tuned)',
    'accuracy': float(accuracy_score(y_test, y_pred_xgb_tuned)),
    'precision': float(precision_score(y_test, y_pred_xgb_tuned)),
    'recall': float(recall_score(y_test, y_pred_xgb_tuned)),
    'f1_score': float(f1_score(y_test, y_pred_xgb_tuned)),
    'auc': float(roc_auc_score(y_test, y_prob_xgb_tuned))
}

print(f"\n调参后的XGBoost性能:")
print(f"准确率: {xgb_tuned_results['accuracy']:.4f}")
print(f"精确率: {xgb_tuned_results['precision']:.4f}")
print(f"召回率: {xgb_tuned_results['recall']:.4f}")
print(f"F1分数: {xgb_tuned_results['f1_score']:.4f}")
print(f"AUC: {xgb_tuned_results['auc']:.4f}")

# 比较调参前后
print("\n=== XGBoost调参前后比较 ===")
comparison_tuned = pd.DataFrame([xgb_results, xgb_tuned_results])
print(comparison_tuned.round(4))

# 保存结果数据
model_comparison_data = []
for result in [xgb_results, rf_results, lr_results, svm_results, xgb_tuned_results]:
    model_comparison_data.append({
        'model': result['model'],
        'accuracy': result['accuracy'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1_score': result['f1_score'],
        'auc': result['auc']
    })

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

# 绘制 ROC 曲线（比较 XGBoost, Random Forest, Logistic Regression）
plt.figure(figsize=(8, 6))

# 列表中包含 (显示名称, 预测概率, AUC 值)
roc_models = [
    ('XGBoost', y_prob_xgb, xgb_results['auc']),
    ('Random Forest', y_prob_rf, rf_results['auc']),
    ('Logistic Regression', y_prob_lr, lr_results['auc'])
]

for name, y_prob, auc_score in roc_models:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机猜测 (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC 曲线比较')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存到项目的 performance_evaluation 目录
plt.savefig('./debug/performance_evaluation/roc_comparison.png', dpi=150)
plt.show()

print(f"\n模型比较数据已准备完成")