# 糖尿病预测与分析项目 (Diabetes Prediction & Analysis)

## 简介

- 本项目基于公开的糖尿病数据集，整合数据预处理、特征工程、多种机器学习与深度学习模型训练、超参数调优、模型可解释性（SHAP / LIME）以及模型持久化与导出。
- 目标：构建可解释且能用于临床参考的二分类预测模型（是否患糖尿病）。

## 主要功能

- 数据清洗与临床导向异常值处理（将医学上不合理的0值替换为中位数）。
- 多种特征选择方法（ANOVA / 互信息 / RFE）与临床交互特征构建（如 `Glucose_BMI`）。
- 基准模型：XGBoost、随机森林、逻辑回归、SVM。
- 深度学习模型：Inception-LSTM、Attention-CNN，并使用 Keras Tuner 做搜索。
- 模型可解释性：SHAP 全局解释、LIME 局部解释。
- 导出 artifacts（模型文件、预处理器 `preprocessor.joblib`、`manifest.json`）。

## 依赖（建议在虚拟环境中安装）

- Python 3.8+
- 常用库：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `keras-tuner`, `shap`, `lime`, `matplotlib`, `seaborn`, `joblib`

示例安装命令：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost tensorflow keras keras-tuner shap lime matplotlib seaborn joblib
```

## 仓库结构（摘要）

- `predict.ipynb`：主流程 Notebook，包含数据加载、预处理、模型训练、评估、可解释性与模型导出。
- `diabetes.csv`：原始数据集（Kaggle/T2DM 常用示例数据）。
- `artifacts/`：训练并导出后的模型与预处理文件（例如 `diabetes_model.keras` 或 `diabetes_model.joblib`、`preprocessor.joblib`、`manifest.json`）。
- `debug/`, `model_building/`, `performance_evaluation/`, `tuner/`：项目中用于分析、调参与评估的辅助脚本与中间结果。

## 快速开始（使用 Notebook）

1. 克隆或将工作目录切换到本项目根目录。保证已安装依赖并激活虚拟环境。
2. 启动 Jupyter Notebook / Lab：

```bash
jupyter lab
# 或
jupyter notebook
```

3. 打开并运行 `predict.ipynb`。
    - 推荐按单元顺序执行；若已存在 artifacts，可跳过训练直接加载导出模型（在第 12 节模型导出单元附近有说明）。

## 重要单元说明（在 `predict.ipynb` 中）

- 数据预处理：替换不合理的 0 值（`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`），并做 `StandardScaler` 标准化。
- 特征选择与交互：RFE（逻辑回归 / XGBoost）取交集，构建临床交互特征（如 `Glucose_BMI`、`Age_BMI`）。
- 模型训练：包含基准 XGBoost 与深度学习模型（Inception-LSTM、Attention-CNN）。
- 超参数调优：使用 `RandomizedSearchCV`（XGBoost）与 `keras-tuner`（深度学习）。
- 可解释性：使用 SHAP （全局）与 LIME （局部）来解释最优模型。
- 模型导出：将预处理器与模型保存到 `artifacts/`，并生成 `manifest.json`。

## 如何复现实验 / 复训练模型

- 若要从头复现训练，请在 `predict.ipynb` 中从头运行相关训练单元（注意 GPU 环境下可加速深度学习训练）。
- 若只希望加载已导出的模型进行预测：检查 `artifacts/manifest.json`，加载 `preprocessor.joblib` 与模型文件（`.joblib` 或 `.keras`），按预处理步骤（0值替换、标准化、构建交互特征）再进行预测。

示例（伪代码）：

```python
import joblib
import pandas as pd
from pathlib import Path
ART = Path('artifacts')
pre = joblib.load(ART / 'preprocessor.joblib')
model = joblib.load(ART / 'diabetes_model.joblib')  # or load Keras model
# 按 manifest 中的规则进行预处理，再 model.predict
```

## 部署建议

- 若使用 sklearn / xgboost 模型：可将 `preprocessor.joblib` 与模型文件一起打包，使用 Flask / FastAPI 提供 REST API。
- 若使用 Keras 模型：保存为 `.keras`（或 TensorFlow SavedModel），并在生产端加载，注意 TensorFlow 版本兼容性。
- 在生产环境中：使用相同的 Python 及库版本，严格按照 `manifest.json` 中的 `final_input_columns` 进行输入排列与预处理。

## 可视化与结果复现

- Notebook 中提供 ROC、SHAP summary、dependence plot、LIME 个体解释等可视化单元，运行对应单元会生成图表。

## 注意事项

- 本项目示例数据较小，深度学习模型可能出现过拟合，请根据需要调整架构、正则化与训练策略。
- 若需在无 GUI 的服务器环境运行 Notebook，可导出或在脚本中禁用 `plt.show()` 并保存图片到文件。

## 版权与联系

- 本项目为分析与教育示例。若用于临床决策，请在临床专家指导下严格验证模型性能与可解释性。
- 如需帮助或有改进建议，请在仓库中打开 issue 或联系维护人。

---

已生成：`predict.ipynb`、`artifacts/` 里包含模型与预处理器的导出示例。阅读并运行 `predict.ipynb` 以获得详细步骤与交互式结果。
