import re
from typing import List

# 预编译正则表达式提升匹配效率
CHOICE_PATTERN = re.compile(r'<<([A-Za-z])>>')


def extract_content(text: str) -> str:
    """从文本中提取<<X>>模式的字符并拼接成字符串

    Args:
        text: 需要处理的原始文本

    Returns:
        由所有<<X>>内容拼接的字符串，如'CNC'
    """
    return ''.join(CHOICE_PATTERN.findall(text))


def decaying_weight_similarity(a: str, b: str, beta: float = 0.8) -> float:
    """衰减权重相似度算法

    Args:
        a: 基准字符串（真实决策序列）
        b: 对比字符串（预测序列）
        beta: 衰减系数（0 < beta < 1），默认0.8

    Returns:
        相似度得分（0~1之间），满足：
        - 当b包含完整a前缀时返回1.0
        - 其他情况按指数衰减权重计算匹配度
    """
    len_a, len_b = len(a), len(b)

    # 特殊处理b包含完整a前缀的情况
    if len_b >= len_a and b.startswith(a):
        return 1.0

    max_len = max(len_a, len_b)
    numerator = 0.0
    denominator = 0.0
    current_weight = 1.0  # 初始权重beta^0=1

    for i in range(max_len):
        denominator += current_weight

        # 获取对应位置字符（超出长度返回None）
        char_a = a[i] if i < len_a else None
        char_b = b[i] if i < len_b else None

        if char_a == char_b and char_a is not None:
            numerator += current_weight

        # 更新权重：beta^i = beta^(i-1) * beta
        current_weight *= beta

    return numerator / denominator if denominator else 0.0


def compute_similarity(label: str, predictions: List[str], beta: float = 0.5) -> List[float]:
    """批量计算相似度

    Args:
        label: 包含真实决策的文本
        predictions: 预测文本列表
        beta: 衰减系数

    Returns:
        各预测文本与基准的相似度列表
    """
    label_choices = extract_content(label)
    extracted_predictions = [extract_content(p) for p in predictions]

    return [decaying_weight_similarity(label_choices, p, beta) for p in extracted_predictions]


if __name__ == "__main__":
    # 测试数据

    labels = """You will be shown several examples of geometric objects.Your task is to learn a rule that allows you totell whether an object <<belongs to the C or N category.For each presented object, you will be asked to makea category judgment by pressing the corresponding key and >>then you will receive feedback.You willencounter four different problems with different rules."
    You encounter a new problem with a new rule determining which objects belong to each category:
     You see a big white square. You press <<C>>. The correct category is N.
     You see a big black triangle. You press <<C>>. The correct category is C.
     You see a small white square. You press <<N>>. The correct category is N.
     You see a big white triangle. You press <<N>>. The correct category is C.
     You see a small black square. You press <<C>>. The correct category is N.
     You see a small white triangle. You press <<C>>. The correct category is C."""

    predictions = ["""You will be shown several examples of geometric objects.Your task is to learn a rule that allows you totell whether an object belongs to the C or N category.For each presented object, you will be asked to makea category judgment by pressing the corresponding key and then you will receive feedback.You willencounter four different problems with different rules."
    You encounter a new problem with a new rule determining which objects belong to each category:
     You see a big white square. You press <<C>>. The correct category is N.
     You see a big black triangle. You press <<C>>. The correct category is C.
     You see a small white square. You press <<N>>. The correct category is N.
     You see a big white triangle. You press <<C>>. The correct category is C.
     You see a small black square. You press <<N>>. The correct category is N.
     You see a small white triangle. You press <<C>>. The correct category is C.""",
                   """You will be shown several examples of geometric objects.Your task is to learn a rule that allows you totell whether an object belongs to the C or N category.For each presented object, you will be asked to makea category judgment by pressing the corresponding key and then you will receive feedback.You willencounter four different problems with different rules."
                       You encounter a new problem with a new rule determining which objects belong to each category:
                        You see a big white square. You press <<C>>. The correct category is N.
                        You see a big black triangle. You press <<C>>. The correct category is C.
                        You see a small white square. You press <<N>>. The correct category is N.
                        You see a big white triangle. You press <<C>>. The correct category is C.
                        You see a small black square. You press <<N>>. The correct category is N.
                        You see a small white triangle. You press <<N>>. The correct category is C."""]

    # 执行计算
    similarity_scores = compute_similarity(labels, predictions)
    print(f"Similarity scores: {similarity_scores}")
