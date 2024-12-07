import torch
from torchvision.ops import box_iou
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
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


def extract_and_create_graph_per_sample(extracted_features, all_boxes, device, k=30, output_size=(7, 7)):
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

def extract_and_create_graph_per_sample1(extracted_features_1ch, all_boxes, device, k=4, output_size=(7, 7)):
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
    batch_size = extracted_features_1ch.shape[0]
    num_queries = all_boxes.shape[1]
    feature_channels, H_feat, W_feat = extracted_features_1ch.shape[1:]
    graph_data_list = []  # 每个样本的图数据
    scl_features = []

    for i in range(batch_size):
        feat = extracted_features_1ch[i]  # [1, H_feat, W_feat]
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

        cropped_regions = []
        all_region_nodes = []
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

        # 调整区域大小

        for region in cropped_regions:
            scl_features.append(region.reshape(feature_channels, -1).mean(dim=1))
            resized_region = F.interpolate(region.unsqueeze(0), size=output_size, mode='bilinear',
                                           align_corners=False)
            resized_region = resized_region.squeeze(0)
            node_features = resized_region.view(feature_channels, -1).mean(dim=1)
            all_region_nodes.append(node_features)
        all_nodes = torch.stack(all_region_nodes, dim=0)
        # 构建 KNN 图
        x = all_nodes.view(-1, feature_channels)  # [num_regions, C]
        edge_index = knn_graph(x, k=k, batch=None, loop=False)  # 当前样本内部构建 KNN 图

        # 创建图数据对象
        data = Data(x=x, edge_index=edge_index)
        graph_data_list.append(data)

    batch_feature = torch.stack(scl_features, dim=0).reshape(batch_size, -1)
    # 使用 Batch 合并所有样本的数据
    batch_data = Batch.from_data_list(graph_data_list)

    # x = batch_data.x  # [B * R, C]
    # batch = batch_data.batch  # [B * R]
    #
    # # 将 x 按照 batch 进行拆分，每个样本有 R 个节点
    # x_list = x.split(num_queries, dim=0)  # 长度为 B 的列表，每个元素形状为 [R, C]
    #
    # # 将每个样本的节点特征展平成一维向量，并堆叠
    # batch_x = torch.stack([xi.view(-1) for xi in x_list], dim=0)  # [B, R * C]

    return batch_data.to(device), batch_feature.to(device)


def build_graph(extracted_features, all_boxes, device, output_size=(7, 7)):
    batch_size = extracted_features.shape[0]
    num_queries = all_boxes.shape[1]
    feature_channels, H_feat, W_feat = extracted_features.shape[1:]
    num_nodes = int(num_queries * feature_channels)

    # center_list = []
    # 用于存储每个样本的组合区域和坐标
    all_combined_regions = []
    all_combined_coords = []
    scl_features = []

    for i in range(batch_size):
        feat = extracted_features[i]  # [1, H_feat, W_feat]
        boxes = all_boxes[i]  # [num_queries, 4]

        # 获取预测框坐标
        x_c, y_c, w, h = boxes.unbind(-1)
        # center_list.append(torch.stack([x_c, y_c], dim=1).repeat_interleave(feature_channels, dim=0))
        x_min = (x_c - 0.5 * w) * W_feat
        y_min = (y_c - 0.5 * h) * H_feat
        x_max = (x_c + 0.5 * w) * W_feat
        y_max = (y_c + 0.5 * h) * H_feat
        x_c = x_c * W_feat
        y_c = y_c * H_feat


        # print(x_min, y_min, x_max, y_max)
        # 裁剪坐标到特征图范围
        x_min = x_min.clamp(0, W_feat - 1).round().long()
        y_min = y_min.clamp(0, H_feat - 1).round().long()
        x_max = x_max.clamp(0, W_feat - 1).round().long()
        y_max = y_max.clamp(0, H_feat - 1).round().long()
        # 将中心坐标转换为整数
        x_c = x_c.round().long()
        y_c = y_c.round().long()

        cropped_regions = []
        queries_regions = []
        image_coords = []
        # 提取区域特征
        for j in range(num_queries):  # 取 N 个查询
            x1, y1, x2, y2 = x_min[j], y_min[j], x_max[j], y_max[j]
            # print(x1, y1, x2, y2)
            if x2 >= x1 and y2 >= y1:
                # 提取区域特征
                region = feat[:, y1:y2 + 1, x1:x2 + 1]  # [1, h, w]
            else:
                # 无效框使用零填充
                region = torch.zeros((feat.size(0), 1, 1)).to(device)
                print("Found invalid box")
            cropped_regions.append(region)

            x_center = x_c[j]
            y_center = y_c[j]

            # 计算半尺寸
            half_w = output_size[1] // 2
            half_h = output_size[0] // 2

            # 初始起始和结束索引
            x_start = x_center - half_w
            x_end = x_center + half_w - 1  # 包含在内

            y_start = y_center - half_h
            y_end = y_center + half_h - 1  # 包含在内

            # 处理左边界（x_start < 0）
            if x_start < 0:
                x_start = 0
                x_end = x_start + output_size[1] - 1

            # 处理上边界（y_start < 0）
            if y_start < 0:
                y_start = 0
                y_end = y_start + output_size[0] - 1

            # 处理右边界（x_end >= W_feat）
            if x_end >= W_feat:
                x_end = W_feat - 1
                x_start = x_end - output_size[1] + 1
                if x_start < 0:
                    x_start = 0
                    x_end = x_start + output_size[1] - 1

            # 处理下边界（y_end >= H_feat）
            if y_end >= H_feat:
                y_end = H_feat - 1
                y_start = y_end - output_size[0] + 1
                if y_start < 0:
                    y_start = 0
                    y_end = y_start + output_size[0] - 1

            # 确保区域大小正确
            width = x_end - x_start + 1
            height = y_end - y_start + 1

            if width != output_size[1] or height != output_size[0]:
                # 如果调整后区域尺寸仍然不足，跳过该区域
                print("Not valid region after adjustment")
                continue

            # 生成并保存对应的坐标
            x_coords = torch.arange(x_start, x_end + 1)
            y_coords = torch.arange(y_start, y_end + 1)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]
            coords = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
            coords = coords.to(feat.device)
            image_coords.append(coords)

        # 调整区域大小
        for region in cropped_regions:
            scl_features.append(region.reshape(feature_channels, -1).mean(dim=1))
            resized_region = F.interpolate(region.unsqueeze(0), size=output_size, mode='bilinear',
                                           align_corners=False)
            resized_region = resized_region.squeeze(0)
            queries_regions.append(resized_region)

        if len(queries_regions) > 0:
            # 在 x 轴方向组合所有预测区域和坐标
            combined_region = torch.cat(queries_regions, dim=2)  # [C, H, num_queries * W]
            combined_coords = torch.cat(image_coords, dim=2)  # [2, H, num_queries * W]

            # 保存每个样本的组合结果
            all_combined_regions.append(combined_region)
            all_combined_coords.append(combined_coords)
        else:
            # 没有可用的区域
            print(f"No enough {num_queries} regions.")

    batch_feature = torch.stack(scl_features, dim=0).reshape(batch_size, -1)

    # features = torch.stack(all_regions, dim=0).reshape(batch_size, num_nodes, -1)
    features = torch.stack(all_combined_regions, dim=0)

    all_centers = torch.stack(all_combined_coords, dim=0)

    # data_list = []
    alpha = 0.7
    # sigma_feature = 1.0
    # sigma_spatial = 1.0
    data_list = build_from_adjacency_matrix(features,all_centers,alpha = alpha)
    # for i in range(batch_size):
    #     feature = features[i]  # [num_nodes, feature_dim]
    #     num_nodes = features.size(0)
    #
    #     # 计算特征差异
    #     diff_features = feature.unsqueeze(1) - feature.unsqueeze(0)  # [num_nodes, num_nodes, feature_dim]
    #     dist_features = torch.norm(diff_features, dim=2) ** 2  # [num_nodes, num_nodes]
    #     feature_weights = torch.exp(-dist_features / sigma_feature ** 2)
    #
    #     # 计算空间位置差异
    #     centers = all_centers[i]  # [num_nodes, 2]
    #     diff_positions = centers.unsqueeze(1) - centers.unsqueeze(0)  # [num_nodes, num_nodes, 2]
    #     dist_spatial = torch.norm(diff_positions, dim=2) ** 2  # [num_nodes, num_nodes]
    #     spatial_weights = torch.exp(-dist_spatial / sigma_spatial ** 2)
    #
    #     # 融合权重
    #     edge_weights = alpha * feature_weights + (1 - alpha) * spatial_weights  # [num_nodes, num_nodes]
    #
    #     # 构建边索引（全连接图，去除自环）
    #     row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    #     edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)  # [2, num_edges]
    #     mask = edge_index[0] != edge_index[1]  # 去除自环
    #     edge_index = edge_index[:, mask]
    #     edge_attr = edge_weights[edge_index[0], edge_index[1]]  # [num_edges]
    #
    #     # 构建 Data 对象
    #     graph_data = Data(
    #         x=feature,  # [num_nodes, feature_dim]
    #         edge_index=edge_index,  # [2, num_edges]
    #         edge_attr=edge_attr  # [num_edges]
    #     )
    #
    #     data_list.append(graph_data)

    # 批量化所有图数据
    batch_data = Batch.from_data_list(data_list)

    return batch_data.to(device), batch_feature.to(device)

def visualize_graph(graph_data):
    """
    可视化图结构，包括节点、边和边的权重。
    :param graph_data: torch_geometric.data.Data 对象
    """
    # 将 PyTorch Geometric 数据对象转换为 networkx 图
    G = to_networkx(graph_data, edge_attrs=['edge_attr'], node_attrs=['x'])

    # 获取边权重
    edge_weights = nx.get_edge_attributes(G, 'edge_attr')

    # 使用 spring 布局
    pos = nx.spring_layout(G)

    # 绘制节点
    node_colors = [d for _, d in G.nodes(data='x')]  # 以节点特征作为颜色信息（如高维特征可降维）
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=30)

    # 绘制边
    edges, weights = zip(*edge_weights.items())
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Blues, width=2)

    # 添加节点标签
    nx.draw_networkx_labels(G, pos, font_size=2, font_color='black')

    # 添加边权重标签
    edge_labels = {e: f"{w:.2f}" for e, w in edge_weights.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph Visualization")
    plt.show()

def build_from_adjacency_matrix(feature_map, coords, alpha):
    batch_size, c, h, w4 = feature_map.shape  # feature_map: [batch_size, c, h, w*4]
    num_nodes = h * w4  # 每个样本的节点数量

    # 将特征图展平，获取节点特征
    feature_map_flat = feature_map.view(batch_size, c, -1).transpose(1, 2)  # [batch_size, num_nodes, c]
    # 将坐标展平
    coords_flat = coords.view(batch_size, 2, -1).transpose(1, 2)  # [batch_size, num_nodes, 2]

    data_list = []

    for i in range(batch_size):
        features = feature_map_flat[i]  # [num_nodes, c]
        positions = coords_flat[i].float()  # [num_nodes, 2]

        num_nodes = features.size(0)

        # 获取上三角索引，避免重复计算
        row_indices, col_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)

        # 计算特征距离和空间距离
        # feature_diff = features[row_indices] - features[col_indices]
        # w_feature = torch.norm(feature_diff, p=2, dim=1)
        # 计算点积
        dot_product = torch.sum(features[row_indices] * features[col_indices], dim=1)  # [num_edges]

        # 归一化：除以 sqrt(d)
        normalized_dot_product = dot_product / torch.sqrt(torch.tensor(c, dtype=torch.float32))
        w_feature = torch.sigmoid(normalized_dot_product)

        spatial_diff = positions[row_indices] - positions[col_indices]
        w_spatial = torch.norm(spatial_diff, p=2, dim=1)

        spatial_diff = positions[row_indices] - positions[col_indices]
        w_spatial = torch.norm(spatial_diff, p=2, dim=1)

        # 组合边权重
        edge_weight = alpha * w_feature + (1 - alpha) * w_spatial

        # 构建边索引
        edge_index = torch.stack([row_indices, col_indices], dim=0)

        # 因为图是无向的，添加对称的边
        edge_index_sym = torch.stack([col_indices, row_indices], dim=0)
        edge_index = torch.cat([edge_index, edge_index_sym], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)


        features = feature_map_flat[i]  # [num_nodes, c]

        data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)
        data_list.append(data)
    return data_list
