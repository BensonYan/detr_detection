import os
from skimage.metrics import structural_similarity as ssim
import re
from collections import defaultdict
import cv2
import numpy as np
import json
import shutil
from tqdm import tqdm

# output_path = "/data/Deepfake/ff_c23_object/train"
# output_jsonl_path = "/data/Deepfake/ff_c23_object/metadata.jsonl"
# def natural_sort_key(filename):
#     # 使用正则表达式提取文件名中的数字部分
#     numbers = re.findall(r'\d+', filename)
#     return [int(num) for num in numbers] if numbers else [0]
#
#
# def write_jsonl(name, bbox_list,specific_method=None,category=None):
#     if  category == "real":
#         label = [1]
#         classes = [1, 1, 1, 1]
#     else:
#         label = [0]
#         name = specific_method + "_" + name
#         classes = [0, 0, 0, 0]
#     try:
#         assert len(bbox_list) == 4
#     except AssertionError as e:
#         print(f"Warning: {e}")
#         print(name)
#     bboxes = [[float(value) for value in bbox] for bbox in bbox_list]
#     data = {
#         "file_name": name,
#         "objects": {
#             "bbox": bboxes,
#             "categories": label,  # 假设类别为1
#             "class": classes
#         }
#     }
#     data['objects']['bbox'] = [[float(coord) for coord in box] for box in data['objects']['bbox']]
#
#     with open(output_jsonl_path, 'a') as jsonl_file:
#         jsonl_file.write(json.dumps(data) + "\n")
#
# def SSIM(img_real,img_fake,specific_method):
#     if img_real is None or img_fake is None:
#         print("无法读取图片，请检查文件路径。")
#         exit()
#
#     # 确保两张图片尺寸相同
#     if img_real.shape != img_fake.shape:
#         # print("图片尺寸不一致，正在调整大小...")
#         img_fake = cv2.resize(img_fake, (img_real.shape[1], img_real.shape[0]))
#
#     # 2. 转换为灰度图像
#     gray_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
#     gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
#
#     # 3. 计算 SSIM
#     score, ssim_map = ssim(gray_real, gray_fake, full=True)
#     # print(f"总体 SSIM: {score}")
#
#     # 4. 获取不相似度图像
#     dissimilarity_map = 1 - ssim_map
#     # 将不相似度图像的值范围缩放到 [0, 255]，并转换为 uint8 类型
#     dissimilarity_map_uint8 = (dissimilarity_map * 255).astype(np.uint8)
#
#
#     # 3. 滑动窗口计算不相似度区域
#     window_size = 35  # 窗口大小，可以根据需要调整
#     step_size = 1  # 滑动步长，可以根据需要调整
#
#     height, width = dissimilarity_map.shape
#     regions = []
#
#     for y in range(0, height - window_size + 1, step_size):
#         for x in range(0, width - window_size + 1, step_size):
#             # 提取窗口内的不相似度
#             window = dissimilarity_map[y:y + window_size, x:x + window_size]
#             # 计算窗口内的不相似度平均值
#             mean_dissimilarity = np.mean(window)
#             # 保存窗口信息
#             regions.append({
#                 'bbox': [x, y, window_size, window_size],
#                 'mean_dissimilarity': mean_dissimilarity
#             })
#
#     # 4. 根据不相似度排序窗口
#     regions_sorted = sorted(regions, key=lambda x: x['mean_dissimilarity'], reverse=True)
#     if specific_method != "NeuralTextures":
#         # 计算所有窗口的不相似度值列表
#         mean_dissimilarities = [region['mean_dissimilarity'] for region in regions_sorted]
#
#         # 计算中位数
#         median_dissimilarity = np.median(mean_dissimilarities)
#         # print(f"平均不相似度的中位数：{median_dissimilarity:.4f}")
#
#         # 定义一个范围，例如中位数的±10%
#         lower_bound = median_dissimilarity * 0.9
#         upper_bound = median_dissimilarity * 1.1
#
#         # 筛选出平均不相似度在该范围内的区域
#         regions_sorted = [region for region in regions_sorted if lower_bound <= region['mean_dissimilarity'] <= upper_bound]
#         # print(f"满足条件的区域数量：{len(middle_regions)}")
#     # 5. 选取不重叠的窗口
#     top_regions = []
#     selected_bboxes = []  # 已选取的 bbox 列表
#
#     def compute_iou(box1, box2):
#         x1, y1, w1, h1 = box1
#         x2, y2, w2, h2 = box2
#         xa = max(x1, x2)
#         ya = max(y1, y2)
#         xb = min(x1 + w1, x2 + w2)
#         yb = min(y1 + h1, y2 + h2)
#         inter_area = max(0, xb - xa) * max(0, yb - ya)
#         box1_area = w1 * h1
#         box2_area = w2 * h2
#         iou = inter_area / (box1_area + box2_area - inter_area)
#         return iou
#
#     # 设置 IoU 阈值为 0，表示不允许重叠
#     iou_threshold = 0
#
#     for region in regions_sorted:
#         bbox = region['bbox']
#         overlap = False
#         for selected_bbox in selected_bboxes:
#             iou = compute_iou(bbox, selected_bbox)
#             if iou > iou_threshold:
#                 overlap = True
#                 break
#         if not overlap:
#             top_regions.append(region)
#             selected_bboxes.append(bbox)
#         if len(top_regions) >= 4:
#             break
#
#     return selected_bboxes
#
#
#
#
# # img_real = cv2.imread('/media/bosheng/One Touch/ff_c23_new/face/train/real/001_1.png')
# # img_fake = cv2.imread('/media/bosheng/One Touch/ff_c23_new/face/train/fake/NeuralTextures/001_870_1.png')
# # SSIM(img_real,img_fake,"NeuralTextures")
#
# # 定义真图和假图文件夹路径
# real_images_folder = "/media/bosheng/One Touch/ff_c23_new/face/train/real"
# fake_images_base_folder = "/media/bosheng/One Touch/ff_c23_new/face/train/fake"
# method = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
# start_index = 889
#
# all_real_files = os.listdir(real_images_folder)
# grouped_real_files = defaultdict(list)
# for filename in all_real_files:
#     prefix = filename.split('_')[0]  # 提取前缀
#     grouped_real_files[prefix].append(filename)
#
# # 对每个前缀的真图进行排序
# sorted_real_files = {prefix: sorted(files, key=natural_sort_key) for prefix, files in grouped_real_files.items()}
#
# # # 初始化存储SSIM值的字典
# # ssim_results = {f'method_{i+1}': [] for i in range(4)}
#
#
# grouped_fake_files = {method[i]: defaultdict(list) for i in range(4)}
# for method_idx in range(4):
#     fake_images_folder = os.path.join(fake_images_base_folder, method[method_idx])
#     fake_files = os.listdir(fake_images_folder)
#     for filename in fake_files:
#         prefix = filename.split('_')[0]  # 提取前缀
#         grouped_fake_files[method[method_idx]][prefix].append(filename)
#
# sorted_real_files = {prefix: sorted(files, key=natural_sort_key) for prefix, files in grouped_real_files.items()}
# sorted_fake_files = {
#     method: {prefix: sorted(files, key=natural_sort_key) for prefix, files in method_files.items()}
#     for method, method_files in grouped_fake_files.items()
# }
#
# for prefix, real_files in tqdm(sorted_real_files.items(), desc="Processing groups"):
#     # 检查假图文件夹中是否有对应前缀
#     for method_idx in range(4):
#         method_name = method[method_idx]
#         fake_images_folder = os.path.join(fake_images_base_folder, method[method_idx])
#         if prefix not in sorted_fake_files[method_name]:
#             continue  # 跳过没有对应前缀的伪造方法
#
#         # 获取对应的假图
#         fake_files = sorted_fake_files[method_name][prefix]
#
#         # 计算真图的起始和结束索引
#         start_idx = method_idx * 35
#         end_idx = start_idx + 35
#         real_batch = real_files[start_idx:end_idx]  # 真图的每35张为一批
#
#         # 逐一读取并计算SSIM
#         for real_name, fake_name in zip(real_batch, fake_files[:35]):
#             real_number = int(re.findall(r'\d+', real_name)[0])
#             fake_number = int(re.findall(r'\d+', fake_name)[0])
#             if real_number < start_index or fake_number < start_index:
#                 continue
#             real_image_path = os.path.join(real_images_folder, real_name)
#             fake_image_path = os.path.join(fake_images_folder, fake_name)
#             # print(real_image_path + "-----" + "fake:" + fake_image_path)
#             img_real = cv2.imread(real_image_path)
#             img_fake = cv2.imread(fake_image_path)
#             specific_method = fake_image_path.split('/')[-2]
#             bboxes = SSIM(img_real,img_fake,specific_method)
#             shutil.copy(real_image_path,
#                         output_path + '/' + real_name)
#             shutil.copy(fake_image_path,
#                         output_path + '/' + specific_method+ '_' + fake_name)
#             write_jsonl(real_name,bboxes,category="real")
#             write_jsonl(fake_name,bboxes,specific_method,"fake")

# output_path = "/data/Deepfake/ff_c23_object/train"
# output_jsonl_path = "/data/Deepfake/ff_c23_object/train/metadata.jsonl"
# output_jsonl_path1 = "/data/Deepfake/ff_c23_object/metadata.jsonl"
# def convert_bbox_to_float(jsonl_path, output_path):
#     with open(jsonl_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
#         for line in infile:
#             # 解析每一行的JSON对象
#             data = json.loads(line.strip())
#             # num_bboxes = len(data['objects']['bbox'])
#             # # 检查并转换bbox字段
#             # if 'objects' in data:
#             #     if 'bbox' in data['objects']:
#             #         data['objects']['bbox'] = [[float(coord) for coord in box] for box in data['objects']['bbox']]
#
#             #加class 这个key
#             if 'categories' in data['objects']:
#                 category_value = data['objects']['categories']
#                 if category_value[0] == 0:
#                     data['objects']['class'] = [0] * 4
#                 elif category_value[0] == 1:
#                     data['objects']['class'] = [1] * 4
#
#             # 将转换后的JSON对象写入新的文件
#             outfile.write(json.dumps(data) + '\n')
#
#
# # 使用方法，指定输入JSONL文件路径和输出文件路径
# convert_bbox_to_float(output_jsonl_path, output_jsonl_path1)

def check_empty_bbox(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(infile, start=1):
            data = json.loads(line.strip())
            if 'bbox' in data['objects']:
                if not data['objects']['bbox']:  # 检查 bbox 是否为空
                    print(f"Line {line_number}: bbox is empty")
            else:
                print(f"Line {line_number}: No bbox key found")

# 使用方法，指定输入 JSONL 文件路径
check_empty_bbox("/data/Deepfake/ff_c23_object/train/metadata.jsonl")