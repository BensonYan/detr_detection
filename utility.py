import torch
from torchvision.ops import box_iou
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
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



def similarity_loss(similarity_matrices, lower_threshold=0, upper_threshold=0.5):
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
    # 取上三角区域（排除对角线）
    mask = torch.triu(torch.ones(num_boxes, num_boxes), diagonal=1).bool().to(similarity_matrices.device)
    upper_triangle = similarity_matrices[:, mask]  # [b, num_pairs]

    # 计算大于 upper_threshold 的惩罚
    high_similarity_penalty = torch.clamp(upper_triangle - upper_threshold, min=0) ** 2

    # 计算小于 lower_threshold 的惩罚
    low_similarity_penalty = torch.clamp(lower_threshold - upper_triangle, min=0) ** 2

    # 总损失
    loss = (high_similarity_penalty + low_similarity_penalty).sum() / (b * num_boxes * (num_boxes - 1))

    return loss


def convert_boxes_format(boxes):
    """
    将边界框从 [x1, y1, w, h] 格式转换为 [xc, yc, w, h] 格式。

    参数：
    - boxes: 输入边界框张量，形状为 [batch_size, num_boxes, 4]，格式为 [x1, y1, w, h]

    返回值：
    - converted_boxes: 转换后的边界框张量，形状为 [batch_size, num_boxes, 4]，格式为 [xc, yc, w, h]
    """
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]

    # 计算中心坐标 xc 和 yc
    xc = (x1 + w / 2) / 256
    yc = (y1 + h / 2) / 256
    w1 = w / 256
    h1 = h / 256

    # 构建转换后的边界框张量
    converted_boxes = torch.stack([xc, yc, w1, h1], dim=-1)
    return converted_boxes


def extract_and_create_graph_per_sample(extracted_features, all_boxes, device, k=10, output_size=(7, 7)):
    """
    对每个样本的 10 个 cropped region 生成 KNN 图。

    参数：
    - extracted_features_1ch: 输入特征图，形状为 [batch_size, 1, H_feat, W_feat]
    - all_boxes: 预测框的坐标，形状为 [batch_size, num_queries, 4]，格式为 [xc, yc, w, h]
    - k: KNN 的最近邻参数
    - output_size: 每个区域特征重采样的目标大小 (h, w)

    返回值：
    - batched_graph_data: PyTorch Geometric 的批次图数据对象
    """
    # 初始化列表
    cropped_regions = []
    batch_indices = []
    batch_size = extracted_features.shape[0]
    num_queries = all_boxes.shape[1]
    H_feat, W_feat = extracted_features.shape[-2:]
    graph_data_list = []  # 每个样本的图数据

    for i in range(batch_size):
        feat = extracted_features[i]  # [1, H_feat, W_feat]
        boxes = all_boxes[i]  # [num_queries, 4]

        # 获取预测框坐标
        x_c, y_c, w, h = boxes.unbind(-1)
        x_min = (x_c - 0.5 * w) * W_feat
        y_min = (y_c - 0.5 * h) * H_feat
        x_max = (x_c + 0.5 * w) * W_feat
        y_max = (y_c + 0.5 * h) * H_feat

        # 裁剪坐标到特征图范围
        x_min = x_min.clamp(0, W_feat - 1).round().long()
        y_min = y_min.clamp(0, H_feat - 1).round().long()
        x_max = x_max.clamp(0, W_feat - 1).round().long()
        y_max = y_max.clamp(0, H_feat - 1).round().long()

        # 提取区域特征
        for j in range(num_queries):  # 取 N 个查询
            x1, y1, x2, y2 = x_min[j], y_min[j], x_max[j], y_max[j]
            if x2 >= x1 and y2 >= y1:
                # 提取区域特征
                region = feat[:, y1:y2 + 1, x1:x2 + 1]  # [1, h, w]
            else:
                # 无效框使用零填充
                region = torch.zeros((feat.size(0), 1, 1))
            cropped_regions.append(region)
            batch_indices.append(i)

    # 调整区域大小
    resized_regions = []
    for region in cropped_regions:
        resized_region = F.interpolate(region.unsqueeze(0), size=output_size, mode='bilinear',
                                       align_corners=False)
        resized_region = resized_region.squeeze(0)
        resized_regions.append(resized_region)
    # 准备特征向量
    feature_vectors = [region.view(-1) for region in resized_regions]
    feature_vectors = torch.stack(feature_vectors)  # [total_num_regions, k * k]
    # 构建 KNN 图
    batch_index = torch.tensor(batch_indices, dtype=torch.long).to(device)
    edge_index = knn_graph(feature_vectors, k, batch=batch_index, loop=False)

    # 构建图数据对象
    data = Data(x=feature_vectors, edge_index=edge_index)
    data.batch = batch_index


    return data
