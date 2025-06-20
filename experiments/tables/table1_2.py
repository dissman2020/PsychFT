import pandas as pd
import re
import numpy as np


def excel_to_latex_table(excel_path):
    """
    从Excel文件读取数据并转换为LaTeX表格格式，每行最小值加粗

    参数:
    excel_path (str): Excel文件路径

    返回:
    str: 格式化后的LaTeX表格代码
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path, sheet_name='Sheet')

    # 清理列名（去除前后空格）
    df.columns = [col.strip() for col in df.columns]

    # 自动识别列名 - 查找包含"task"的列作为任务类别列
    task_col = None
    for col in df.columns:
        if 'task' in col.lower():
            task_col = col
            break

    if task_col is None:
        raise ValueError("未找到任务类别列（包含'task'的列）")

    # 自动识别模型列 - 查找包含模型名称的列
    model_cols = []
    # 修正关键词：cenaur → centaur
    model_keywords = ['llama', 'cognitive', 'centaur', 'adamca', 'model']
    for col in df.columns:
        if col.lower() == task_col.lower():
            continue
        for keyword in model_keywords:
            if keyword in col.lower():
                model_cols.append(col)
                break

    if not model_cols:
        raise ValueError("未找到模型性能列（包含'llama', 'cognitive', 'centaur'或'adamca'的列）")

    # 打印检测到的列名用于验证
    print(f"检测到的任务类别列: {task_col}")
    print(f"检测到的模型列: {', '.join(model_cols)}")

    # 选择需要的列
    selected_cols = [task_col] + model_cols
    df = df[selected_cols]

    # 处理特殊字符（如德语变音符号）
    df[task_col] = df[task_col].apply(
        lambda x: re.sub(r'枚枚', 'öö', x) if isinstance(x, str) else x
    )

    # 将模型列转换为数值类型，处理NaN值
    for col in model_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 找出每行的最小值（忽略NaN）
    min_indices = []
    for i, row in df.iterrows():
        # 找出最小值
        min_val = None
        min_cols = []

        for col in model_cols:
            value = row[col]
            # 跳过NaN值
            if pd.isna(value):
                continue

            # 初始化最小值
            if min_val is None or value < min_val:
                min_val = value
                min_cols = [col]
            elif value == min_val:
                min_cols.append(col)

        min_indices.append(min_cols)

    # 生成LaTeX表格内容
    latex_rows = []
    for i, row in df.iterrows():
        # 处理任务类别名称
        category = format_category(row[task_col])

        # 构建行内容
        values = []
        for col in model_cols:
            value = row[col]
            formatted = format_value(value)

            # 如果当前列是当前行的最小值，则加粗
            if col in min_indices[i]:
                values.append(f"\\textbf{{{formatted}}}")
            else:
                values.append(formatted)

        latex_rows.append(f"{category} & {' & '.join(values)} \\\\")

    # 构建完整LaTeX表格
    latex_table = f"""\\begin{{table}}[th]
\\centering
\\caption{{Caption}}
\\label{{tab:my_label}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{l|{'c' * len(model_cols)}}}
\\toprule
Task category & {' & '.join(model_cols)} \\\\
\\midrule
{"\n".join(latex_rows)}
\\bottomrule 
\\end{{tabular}}
}}
\\end{{table}}"""

    return latex_table


def format_value(value):
    """格式化数值为小数点后4位"""
    try:
        # 处理NaN值
        if pd.isna(value):
            return 'nan'

        # 转换为浮点数
        num = float(value)

        # 非常小的值使用科学计数法表示
        if abs(num) < 0.0001:
            return f"{num:.4e}"

        # 格式化为小数点后4位
        return f"{num:.4f}"
    except:
        return str(value)


def format_category(category):
    """格式化任务类别名称"""
    if not isinstance(category, str):
        return ""

    # 特殊处理已知缩写
    special_cases = {
        'n-back': 'N-back',
        'cpc18': 'CPC18',
        'things': 'THINGS',
        'go/no-go': 'Go/no-go',
        'nback': 'N-back',
        'grammar judgement': 'Grammar judgment',
        'grammar judgment': 'Grammar judgment',
        'decisions from description': 'Decisions from description',
        'decisions from experience': 'Decisions from experience'
    }

    # 检查特殊处理
    lower_cat = category.lower()
    for key, value in special_cases.items():
        if key in lower_cat:
            # 保留原始名称中的大小写差异（如首字母大写）
            return value + category[len(key):]

    # 常规处理：每个单词首字母大写
    words = category.split()
    formatted_words = []
    for word in words:
        # 保留特殊字符的大小写（如"from"保持小写）
        if word.lower() in ['from', 'and', 'to', 'in', 'of', 'the', 'a', 'an']:
            formatted_words.append(word.lower())
        else:
            formatted_words.append(word.capitalize())

    return ' '.join(formatted_words)


# 使用示例
if __name__ == "__main__":
    # 输入Excel文件路径
    excel_file = "最终实验结果latex输入版.xlsx"

    # 生成LaTeX表格
    try:
        latex_output = excel_to_latex_table(excel_file)

        # 直接打印输出而不是保存到文件
        print(latex_output)

    except Exception as e:
        print(f"Error: {str(e)}")