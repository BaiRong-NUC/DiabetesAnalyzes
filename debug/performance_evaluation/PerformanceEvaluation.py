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
XGBoost: 一个高效的梯度提升树(Gradient Boosted Trees)实现
擅长处理表格型分类/回归问题，具有列抽样、正则化、并行化等优化。
'''
# 创建XGBoost分类器，设置随机种子和评估指标
# use_label_encoder=False：禁止 XGBoost 的旧标签编码器以避免警告（sklearn 兼容性设置）。
# eval_metric='logloss': 训练期间用对数损失作为评估指标（仅用于训练/早停输出，不影响 sklearn 的 fit 返回）

xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train) # 在训练集上拟合模型（默认不使用早停，除非传入验证集和 early_stopping_rounds）

# 预测
y_pred_xgb = xgb_model.predict(X_test) # 输出类别预测（0/1）
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
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

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

# 准备模型调参 - XGBoost调参
print("\n=== XGBoost模型调参 ===")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

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