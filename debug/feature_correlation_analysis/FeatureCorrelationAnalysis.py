import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# 更稳健地加载中文字体：优先使用系统微软雅黑，如不存在则使用候选回退
# 尝试按优先级找到一个可用中文字体，并创建 FontProperties 供绘图时使用
ch_font = None

# 常见系统中文字体文件路径优先尝试（Windows）
common_paths = [
    r'C:\Windows\Fonts\msyh.ttf',
    r'C:\Windows\Fonts\msyh.ttf',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
    r'C:\Windows\Fonts\simsun.ttf',
    r'C:\Windows\Fonts\STFANGSO.ttf'
]

def try_set_font_from_path(path):
    if path and os.path.exists(path):
        try:
            fp = fm.FontProperties(fname=path)
            name = fp.get_name()
            plt.rcParams['font.family'] = name
            plt.rcParams['font.sans-serif'] = [name]
            return fp
        except Exception:
            return None
    return None

for p in common_paths:
    ch_font = try_set_font_from_path(p)
    if ch_font is not None:
        break

if ch_font is None:
    # 根据已安装字体列表匹配常见中文字体名或文件名
    tokens = ['microsoft yahei', 'msyh', 'simhei', 'simsun', 'fangsong', 'kai', 'wqy', 'noto']
    for f in fm.fontManager.ttflist:
        fname_l = (f.fname or '').lower()
        name_l = (f.name or '').lower()
        for t in tokens:
            if t in fname_l or t in name_l:
                try:
                    ch_font = fm.FontProperties(fname=f.fname)
                    name = ch_font.get_name()
                    plt.rcParams['font.family'] = name
                    plt.rcParams['font.sans-serif'] = [name]
                    break
                except Exception:
                    ch_font = None
        if ch_font is not None:
            break

if ch_font is None:
    # 最后尝试使用 findfont 查找已知字体族
    family_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    for fam in family_candidates:
        try:
            font_path = fm.findfont(fam, fallback_to_default=False)
            if font_path and os.path.exists(font_path):
                ch_font = fm.FontProperties(fname=font_path)
                name = ch_font.get_name()
                plt.rcParams['font.family'] = name
                plt.rcParams['font.sans-serif'] = [name]
                break
        except Exception:
            continue

if ch_font is None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    print('警告：未找到可用中文字体，中文可能无法正确显示。')

# 关闭负号被显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

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

# 从保存的 CSV 读取相关矩阵并画热力图
corr_csv_path = './debug/feature_correlation_analysis/correlation_matrix_full.csv'
heatmap_path = './debug/feature_correlation_analysis/correlation_matrix_heatmap.png'
corr = pd.read_csv(corr_csv_path, index_col=0)
plt.figure(figsize=(10, 8))
sns.set(font_scale=0.9)
ax = sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    annot_kws={'fontsize':10, 'fontproperties': ch_font} if ch_font is not None else {'fontsize':10},
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={'shrink': 0.8}
)
# 标题和刻度显式使用已找到的中文字体，避免方块或缺字
if ch_font is not None:
    ax.set_title('特征相关性热力图', fontproperties=ch_font, fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=ch_font)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=ch_font)
else:
    ax.set_title('特征相关性热力图', fontsize=14)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(heatmap_path, dpi=300)
print(f"已保存热力图到: {heatmap_path}")
plt.show()

# 为每列绘制数值分布的柱状图并保存
output_dir = './debug/feature_correlation_analysis/column_distributions'
os.makedirs(output_dir, exist_ok=True)

for col in df_clean.columns:
    # 跳过辅助分组列（如果存在）
    if col == 'Age_Group':
        continue
    plt.figure(figsize=(8, 6))
    if pd.api.types.is_numeric_dtype(df_clean[col]):
        unique_vals = df_clean[col].nunique()
        # 对离散或取值较少的数值列直接计数绘图
        if unique_vals <= 20:
            counts = df_clean[col].value_counts().sort_index()
            counts.plot(kind='bar', color='C0')
            xlabel = col
            title = f"{col} 值分布"
        else:
            # 连续型变量按 10 个箱分箱后绘制柱状图
            binned = pd.cut(df_clean[col], bins=10)
            counts = binned.value_counts().sort_index()
            counts.plot(kind='bar', color='C0')
            xlabel = f"{col} (分箱)"
            title = f"{col} 值分布（分箱）"
    else:
        counts = df_clean[col].value_counts()
        counts.plot(kind='bar', color='C0')
        xlabel = col
        title = f"{col} 值分布"

    if ch_font is not None:
        plt.xlabel(xlabel, fontproperties=ch_font)
        plt.ylabel('计数', fontproperties=ch_font)
        plt.title(title, fontproperties=ch_font)
        plt.xticks(rotation=45, ha='right', fontproperties=ch_font)
    else:
        plt.xlabel(xlabel)
        plt.ylabel('count')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    savef = os.path.join(output_dir, f"{col}_distribution.png")
    plt.savefig(savef, dpi=200)
    plt.close()

print(f"已保存每列分布柱状图到: {output_dir}")