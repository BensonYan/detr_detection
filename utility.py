import torch
from torchvision.ops import box_iou
import torch.nn.functional as F


def minimize_iou_overlap_loss(pred_boxes, max_iou=0.5):
    """
    损失函数：最小化预测框之间的重叠（通过最小化 IoU 重叠）。

    参数：
    - pred_boxes: 预测框的边界框张量，形状为 [batch, num_predictions, 4]，格式为 [x_min, y_min, w, h]
    - max_iou: 最大允许的 IoU 阈值，超过此值的预测框对会受到惩罚

    返回值：
    - loss: IoU 重叠损失
    """
    batch_size, num_preds, _ = pred_boxes.size()
    loss = 0.0

    # 将 [x_min, y_min, w, h] 转换为 [x_min, y_min, x_max, y_max]
    x_min = pred_boxes[..., 0]
    y_min = pred_boxes[..., 1]
    x_max = x_min + pred_boxes[..., 2]  # x_max = x_min + w
    y_max = y_min + pred_boxes[..., 3]  # y_max = y_min + h
    pred_boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=-1)  # [batch, num_predictions, 4]

    for b in range(batch_size):
        # 获取当前批次的预测框
        boxes = pred_boxes_xyxy[b]  # [num_predictions, 4]

        # 计算所有预测框之间的 IoU
        iou_matrix = box_iou(boxes, boxes)  # [num_predictions, num_predictions]

        # 遍历上三角矩阵以避免重复计算
        for i in range(num_preds):
            for j in range(i + 1, num_preds):
                # 仅惩罚 IoU 大于 max_iou 的情况
                if iou_matrix[i, j] > max_iou:
                    loss += (iou_matrix[i, j] - max_iou) ** 2  # 使用平方惩罚

    # 平均损失
    total_pairs = num_preds * (num_preds - 1) / 2
    return loss / (batch_size * total_pairs)  # 对批次和预测框对进行平均

def size_constraint_loss(pred_boxes, min_size=(0.2, 0.2), max_size=(0.5, 0.5)):
    """
    损失函数：对超出指定比例范围的预测框进行惩罚，限制预测框的相对尺寸。

    参数：
    - pred_boxes: 预测框的张量，形状为 [batch, num_predictions, 4]，格式为 [x_min, y_min, w, h]
    - image_size: 图像尺寸 (image_width, image_height)
    - min_size: 允许的最小比例尺寸 (min_width_ratio, min_height_ratio)，值在 [0, 1] 之间
    - max_size: 允许的最大比例尺寸 (max_width_ratio, max_height_ratio)，值在 [0, 1] 之间

    返回值：
    - loss: 尺寸限制损失
    """


    # 获取预测框的宽度和高度
    widths = pred_boxes[..., 2]
    heights = pred_boxes[..., 3]

    # 计算宽度和高度的限制损失
    min_width, min_height = min_size
    max_width, max_height = max_size
    # 惩罚小于最小尺寸的预测框
    width_loss = torch.clamp(min_width - widths, min=0) ** 2
    height_loss = torch.clamp(min_height - heights, min=0) ** 2

    # 惩罚大于最大尺寸的预测框
    width_loss += torch.clamp(widths - max_width, min=0) ** 2
    height_loss += torch.clamp(heights - max_height, min=0) ** 2

    # 计算总的尺寸限制损失
    size_loss = width_loss + height_loss
    return size_loss.mean()


def extract_and_normalize_features(feature_map, mapped_boxes, input_image_size,device):
    """
    从特征图中提取每个批次的目标框特征并归一化。目标框先从原始图像大小缩放到特征图大小。

    参数：
    - feature_map: 特征图张量，形状为 [b, c, h, w]
    - mapped_boxes: 映射到输入图像的目标框，形状为 [b, num_boxes, 4]，格式为 [x_min, y_min, h, w]
    - input_image_size: 输入图像的尺寸 (image_width, image_height)
    - feature_map_size: 特征图的尺寸 (feature_map_width, feature_map_height)

    返回值：
    - normalized_features: 归一化后的特征，形状为 [b, num_boxes, c]
    """
    b, c, h, w = feature_map.size()
    image_width, image_height = input_image_size
    feature_map_width, feature_map_height = w, h

    # 计算缩放比例
    scale_x = feature_map_width / image_width
    scale_y = feature_map_height / image_height

    features = []

    for i in range(b):
        # 当前批次的特征图 [c, h, w]
        current_feature_map = feature_map[i]

        # 存储每个目标框的特征
        batch_features = []

        for j in range(mapped_boxes.size(1)):
            # 获取并缩放当前目标框的坐标
            x_min = int(mapped_boxes[i, j, 0] * scale_x)
            y_min = int(mapped_boxes[i, j, 1] * scale_y)
            box_height = int(mapped_boxes[i, j, 2] * scale_y)
            box_width = int(mapped_boxes[i, j, 3] * scale_x)

            # 计算 x_max 和 y_max
            x_max = x_min + box_width
            y_max = y_min + box_height

            # 检查边界并裁剪到特征图大小范围
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(feature_map_width - 1, x_max)
            y_max = min(feature_map_height - 1, y_max)

            # 提取目标框对应的特征区域
            region = current_feature_map[:, y_min:y_max+1, x_min:x_max+1]  # [c, h', w']

            # 如果区域为空，跳过处理
            if region.numel() == 0:
                pooled_feature = torch.zeros(c)
            else:
                # 使用全局平均池化将区域特征聚合为固定大小的特征向量 [c]
                pooled_feature = region.mean(dim=[1, 2])  # 平均池化为 [c]

            # 归一化特征向量
            normalized_feature = F.normalize(pooled_feature, p=2, dim=0)  # 在通道维度上归一化
            batch_features.append(normalized_feature.to(device))

        # 将当前批次的所有特征堆叠为 [num_boxes, c]
        batch_features = torch.stack(batch_features)
        features.append(batch_features)

    # 将所有批次的特征堆叠为 [b, num_boxes, c]
    return torch.stack(features)

def compute_cosine_similarity(features):
    """
    计算特征之间的余弦相似度。

    参数：
    - features: 特征张量，形状为 [b, num_boxes, c]

    返回值：
    - similarity_matrices: 余弦相似度矩阵，形状为 [b, num_boxes, num_boxes]
    """
    b, num_boxes, c = features.size()

    # 计算每对特征的余弦相似度
    similarity_matrices = torch.bmm(features, features.transpose(1, 2))  # [b, num_boxes, num_boxes]

    # 将相似度归一化到 [0, 1] 范围（如果需要）
    similarity_matrices = (similarity_matrices + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]

    return similarity_matrices



def similarity_loss(similarity_matrices, lower_threshold=0.3, upper_threshold=0.7):
    """
    对特定范围的相似度进行损失惩罚。

    参数：
    - similarity_matrices: 相似度矩阵，形状为 [b, num_boxes, num_boxes]
    - lower_threshold: 下限阈值
    - upper_threshold: 上限阈值

    返回值：
    - loss: 相似度惩罚损失
    """
    b, num_boxes, _ = similarity_matrices.size()

    # 计算大于 upper_threshold 的惩罚
    high_similarity_penalty = torch.clamp(similarity_matrices - upper_threshold, min=0) ** 2

    # 计算小于 lower_threshold 的惩罚
    low_similarity_penalty = torch.clamp(lower_threshold - similarity_matrices, min=0) ** 2

    # 总损失
    loss = (high_similarity_penalty + low_similarity_penalty).sum() / (b * num_boxes * (num_boxes - 1))

    return loss