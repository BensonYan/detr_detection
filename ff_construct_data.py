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
output_jsonl_path = "/data/Deepfake/ff_c23_object/metadata20class_test.jsonl"#"/data/Deepfake/ff_c23_object/train/metadata20class.jsonl"
output_jsonl_path1 = "/data/Deepfake/ff_c23_object/metadata8class_test.jsonl"
def convert_bbox_to_float(jsonl_path, output_path):
    with open(jsonl_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 解析每一行的JSON对象
            data = json.loads(line.strip())
            # num_bboxes = len(data['objects']['bbox'])
            # # 检查并转换bbox字段
            # if 'objects' in data:
            #     if 'bbox' in data['objects']:
            #         data['objects']['bbox'] = [[float(coord) for coord in box] for box in data['objects']['bbox']]

            #加class 这个key
            if 'categories' in data['objects']:
                category_value = data['objects']['categories']
                if category_value[0] == 0:
                    data['objects']['class'] = [0, 1, 2, 3]
                elif category_value[0] == 1:
                    data['objects']['class'] = [4, 5, 6, 7]

            # 将转换后的JSON对象写入新的文件
            outfile.write(json.dumps(data) + '\n')


# 使用方法，指定输入JSONL文件路径和输出文件路径
# convert_bbox_to_float(output_jsonl_path, output_jsonl_path1)
#
# def check_empty_bbox(jsonl_path):
#     with open(jsonl_path, 'r', encoding='utf-8') as infile:
#         for line_number, line in enumerate(infile, start=1):
#             data = json.loads(line.strip())
#             if 'bbox' in data['objects']:
#                 if not data['objects']['bbox']:  # 检查 bbox 是否为空
#                     print(f"Line {line_number}: bbox is empty")
#             else:
#                 print(f"Line {line_number}: No bbox key found")
#
# # 使用方法，指定输入 JSONL 文件路径
# check_empty_bbox("/data/Deepfake/ff_c23_object/train/metadata.jsonl")


def replace_empty_bbox_with_previous(jsonl_path, output_path):
    previous_bboxes = None  # 用于存储上一行的 bbox 值

    with open(jsonl_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line_number, line in enumerate(infile, start=1):
            data = json.loads(line.strip())


            if 'bbox' in data['objects']:
                for box in data['objects']['bbox']:
                # 检查 bbox 是否为 [0.0, 0.0, 0.0, 0.0]
                    if box == [0.0, 0.0, 0.0, 0.0]:
                        if previous_bboxes:
                            data['objects']['bbox'] = previous_bboxes  # 替换为上一行的 bbox
                            print(f"Line {line_number}: bbox was empty and replaced with the previous bbox")
                        else:
                            print(f"Line {line_number}: bbox is empty, but no previous bbox to replace")
                        break
            else:
                print(f"Line {line_number}: No bbox key found in an object")

            # 更新 previous_bboxes 为当前行的 bbox 值（如果存在且不为空）
            if 'objects' in data:
                current_bboxes = data['objects']['bbox']
                if current_bboxes:  # 存在 bbox 的情况下更新
                    previous_bboxes = current_bboxes

            # 写入修改后的数据到输出文件
            # outfile.write(json.dumps(data) + '\n')


# replace_empty_bbox_with_previous(output_jsonl_path, output_jsonl_path1)

def check_zero_bbox(input_file_path,output_file_path):
    """
    Checks for bbox entries where width (w) or height (h) is zero.
    Saves the lines with invalid bbox to a new file for review.

    Args:
    - input_file_path (str): Path to the input JSONL file.
    - output_file_path (str): Path to save the lines with invalid bbox.
    """
    invalid_lines = []

    with open(input_file_path, "r") as infile:
        for line in infile:
            data = json.loads(line)
            objects = data.get("objects", {})
            bboxes = objects.get("bbox", [])

            # Check for zero width (w) or height (h) in bboxes
            for bbox in bboxes:
                x, y, w, h = bbox
                if w == 0 or h == 0:
                    invalid_lines.append(line)
                    break  # No need to check other bboxes in this line

    # Save the invalid lines to a new file
    # with open(output_file_path, "w") as outfile:
    #     for line in invalid_lines:
    #         outfile.write(line)

    print(f"Found {len(invalid_lines)} lines with invalid bbox.")
    print(f"Invalid lines saved to: {output_file_path}")

# check_zero_bbox("/data/Deepfake/ff_c23_object/train/metadata.jsonl","/data/Deepfake/ff_c23_object/invalid_metadata.jsonl")

# def replace_zero_bbox(input_file_path, output_file_path):
#     """
#     Fixes bbox entries where width (w) or height (h) is zero by replacing it with the non-zero value
#     from the same bbox if available.
#
#     Args:
#     - input_file_path (str): Path to the input JSONL file.
#     - output_file_path (str): Path to save the updated JSONL file.
#     """
#     fixed_lines = []
#
#     with open(input_file_path, "r") as infile:
#         for line in infile:
#             data = json.loads(line)
#             objects = data.get("objects", {})
#             bboxes = objects.get("bbox", [])
#
#             updated_bboxes = []
#             for bbox in bboxes:
#                 x, y, w, h = bbox
#
#                 # Replace zero width (w) or height (h) with the non-zero counterpart
#                 if w == 0 and h > 0:
#                     w = h  # Replace w with h if h is non-zero
#                 elif h == 0 and w > 0:
#                     h = w  # Replace h with w if w is non-zero
#                 updated_bboxes.append([x, y, w, h])
#
#             # Update the data object with fixed bboxes
#             objects["bbox"] = updated_bboxes
#             data["objects"] = objects
#             fixed_lines.append(data)
#
#     # Save the fixed data to a new JSONL file
#     with open(output_file_path, "w") as outfile:
#         for line in fixed_lines:
#             outfile.write(json.dumps(line) + "\n")
#
#     print(f"Updated bounding boxes saved to: {output_file_path}")
#
# replace_zero_bbox("/data/Deepfake/ff_c23_matafile/test/metadata8class_1.jsonl","/data/Deepfake/ff_c23_object/metadata.jsonl")
# check_zero_bbox("/data/Deepfake/ff_c23_object/metadata.jsonl","/data/Deepfake/ff_c23_object/invalid_metadata.jsonl")

#for copy different manipulation method images in FF++
def filter_metadata(input_file_path, output_file_path, source_folder, destination_folder,mani_method):
    """
    Filters the JSONL file and copies the corresponding files based on the following rules:
    1. For blocks of lines with category=1, copy the first 35 lines.
    2. Copy all lines where file_name starts with "Deepfakes".
    3. Copies the files corresponding to the selected lines into the destination folder.

    Args:
    - input_file_path (str): Path to the input JSONL file.
    - output_file_path (str): Path to save the filtered JSONL file.
    - source_folder (str): Path to the folder containing the original files.
    - destination_folder (str): Path to the folder where selected files will be copied.
    """
    selected_lines = []
    category_1_count = 0
    within_category_1_block = False  # Tracks whether we're in a category=1 block
    selected_file_names = set()

    with open(input_file_path, "r") as infile:
        lines = list(infile)  # Read all lines
        for line in lines:
            data = json.loads(line)
            file_name = data.get("file_name", "")
            category = data.get("objects", {}).get("categories", [])

            # Rule 1: Handle category=1 blocks
            if 1 in category:
                if not within_category_1_block:
                    # Start of a new category=1 block
                    within_category_1_block = True
                    category_1_count = 0
                if category_1_count < 35:
                    selected_lines.append(line)
                    selected_file_names.add(file_name)
                    category_1_count += 1
            else:
                # Reset category block flag if not in a category=1 line
                within_category_1_block = False

            # Rule 2: Handle "Deepfakes" lines
            if file_name.startswith(mani_method):
                selected_lines.append(line)
                selected_file_names.add(file_name)

    # if not os.path.exists(destination_folder):
    #     os.makedirs(destination_folder)
    # Write the selected lines to the output JSONL file
    with open(output_file_path, "w") as outfile:
        outfile.writelines(selected_lines)

    # Copy the corresponding files to the destination folder


    # for file_name in selected_file_names:
    #     source_path = os.path.join(source_folder, file_name)
    #     destination_path = os.path.join(destination_folder, file_name)
    #     if os.path.exists(source_path):
    #         shutil.copy(source_path, destination_path)
    #     else:
    #         print(f"File not found: {source_path}")

    print(f"Filtered JSONL file saved to: {output_file_path}")
    print(f"Selected files copied to: {destination_folder}")

# input_file = "/data/Deepfake/ff_c23_object/test/metadata.jsonl"  # Replace with the actual input file path
# output_file = "/data/Deepfake/df_object/test/metadata.jsonl"  # Replace with the desired output file path
# source_folder = "/data/Deepfake/ff_c23_object/train"  # Replace with the folder containing the original files
# destination_folder = "/data/Deepfake/f2f_object/train"  # Replace with the folder to copy selected files
# filter_metadata(input_file, output_file, source_folder, destination_folder,mani_method="Deepfakes")


def update_invalid_bboxes(input_file_path, output_file_path):
    """
    Fix invalid bounding boxes ([0, 0, 0, 0]) in the first image of each category.
    - If the first file in a category has invalid bboxes, find another image in the
      same category where all bboxes are valid.
    - Replace the first file's bboxes with valid bboxes from the found image.

    Args:
    - input_file_path (str): Path to the input JSONL file.
    - output_file_path (str): Path to save the updated JSONL file.
    """
    category_map = {}  # Maps category keys to first file and its bbox validity
    updated_lines = []  # List to store updated JSONL lines

    def get_category_key(file_name, category):
        """Determine the category key based on the filename and category."""
        parts = file_name.split("_")
        if category == 1 and len(parts) > 0:
            return parts[0]  # Use the first part for category 1
        elif category == 0 and len(parts) > 1:
            return f"{parts[0]}_{parts[1]}"  # Use first two parts for category 0
        return None

    # Read and process the JSONL file
    with open(input_file_path, "r") as infile:
        lines = list(infile)

        # Iterate through all lines to map category and track files
        for line in lines:
            data = json.loads(line)
            file_name = data.get("file_name", "")
            objects = data.get("objects", {})
            category = objects.get("categories", [])[0] if "categories" in objects else None
            bboxes = objects.get("bbox", [])

            # Determine the category key
            category_key = get_category_key(file_name, category)
            if not category_key:
                updated_lines.append(data)
                continue

            # Track the first image in each category
            if category_key not in category_map:
                invalid_bbox = any(all(value == 0 for value in bbox) for bbox in bboxes)
                category_map[category_key] = {
                    "first_file_name": file_name,
                    "first_invalid": invalid_bbox,
                    "valid_bboxes": None,  # Placeholder for valid bboxes
                }

            # Check for fully valid bboxes in subsequent files
            if not any(all(value == 0 for value in bbox) for bbox in bboxes):  # All bboxes valid
                if category_map[category_key]["valid_bboxes"] is None:
                    category_map[category_key]["valid_bboxes"] = bboxes

            updated_lines.append(data)

    # Update the first file in each category if needed
    for i, data in enumerate(updated_lines):
        file_name = data.get("file_name", "")
        objects = data.get("objects", {})
        category = objects.get("categories", [])[0] if "categories" in objects else None
        bboxes = objects.get("bbox", [])
        category_key = get_category_key(file_name, category)

        if category_key in category_map and file_name == category_map[category_key]["first_file_name"]:
            # If the first file is invalid and a valid bbox is found, update it
            if category_map[category_key]["first_invalid"] and category_map[category_key]["valid_bboxes"]:
                print(f"Updating {file_name} in category {category_key}")
                data["objects"]["bbox"] = category_map[category_key]["valid_bboxes"]

            updated_lines[i] = data  # Update the line

    # Save the updated JSONL file
    with open(output_file_path, "w") as outfile:
        for data in updated_lines:
            outfile.write(json.dumps(data) + "\n")

    print(f"Updated JSONL file saved to: {output_file_path}")

# input_file = "/data/Deepfake/ff_c23_matafile/test/metadata20class.jsonl"  # Replace with your input JSONL file path
# output_file = "/data/Deepfake/ff_c23_matafile/updated_metadata_test.jsonl"  # Replace with your desired output JSON file path
# update_invalid_bboxes(input_file, output_file)


# def check_invalid_bboxes(input_file_path, output_file_path):
#     """
#     Check for invalid bounding boxes ([0, 0, 0, 0]) in a JSONL file.
#     - For `category = 1`: Group by the first part of the filename before "_".
#     - For `category = 0`: Group by the first two parts of the filename before "_".
#
#     Args:
#     - input_file_path (str): Path to the JSONL file to check.
#     - output_file_path (str): Path to save the invalid categories and their corresponding file names.
#     """
#     invalid_categories = {}  # Dictionary to store invalid categories
#
#     def get_category_key(file_name, category):
#         """Determine the category key based on the filename and category."""
#         parts = file_name.split("_")
#         if category == 1 and len(parts) > 0:
#             return parts[0]  # Use the first part for category 1
#         elif category == 0 and len(parts) > 1:
#             return f"{parts[0]}_{parts[1]}"  # Use first two parts for category 0
#         return None
#
#     # Process the JSONL file
#     processed_categories = set()
#     with open(input_file_path, "r") as infile:
#         for line in infile:
#             data = json.loads(line)
#             file_name = data.get("file_name", "")
#             objects = data.get("objects", {})
#             category = objects.get("categories", [])[0] if "categories" in objects else None
#             bboxes = objects.get("bbox", [])
#
#             # Determine the category key
#             category_key = get_category_key(file_name, category)
#             if not category_key or category_key in processed_categories:
#                 continue  # Skip if no valid key or already processed
#
#             # Check if any bbox is invalid ([0, 0, 0, 0])
#             invalid_bbox = any(all(value == 0 for value in bbox) for bbox in bboxes)
#             if invalid_bbox:
#                 invalid_categories[category_key] = file_name
#
#             # Mark this category key as processed
#             processed_categories.add(category_key)
#
#     # Save invalid categories to output file
#     with open(output_file_path, "w") as outfile:
#         json.dump(invalid_categories, outfile, indent=4)
#
#     print(f"Invalid categories saved to: {output_file_path}")
#
# input_file = "/data/Deepfake/ff_c23_matafile/updated_metadata.jsonl"  # Replace with your input JSONL file path
# output_file = "/data/Deepfake/ff_c23_matafile/invalid_categories.jsonl"  # Replace with your desired output JSON file path
# check_invalid_bboxes(input_file, output_file)

def update_invalid_bboxes_and_copy(input_file_path, output_file_path, source_folder, invalid_folder):
    """
    1. Checks for invalid bounding boxes ([0, 0, 0, 0]) in the first image of each category.
    2. If all bboxes in a category are invalid, moves all files of that category to another folder.

    Args:
    - input_file_path (str): Path to the input JSONL file.
    - output_file_path (str): Path to save the updated JSONL file.
    - source_folder (str): Path to the folder containing the original files.
    - invalid_folder (str): Path to the folder where invalid files will be moved.
    """
    category_map = {}  # Maps category keys to metadata
    updated_lines = []  # Stores the updated lines for output
    invalid_categories = set()  # Tracks categories where all bboxes are invalid

    def get_category_key(file_name, category):
        """Determine the category key based on the filename and category."""
        parts = file_name.split("_")
        if category == 1 and len(parts) > 0:
            return parts[0]  # Use the first part for category 1
        elif category == 0 and len(parts) > 1:
            return f"{parts[0]}_{parts[1]}"  # Use first two parts for category 0
        return None

    # Read the JSONL file and process each line
    with open(input_file_path, "r") as infile:
        lines = list(infile)
        category_files = {}  # Tracks all files belonging to each category
        for line in lines:
            data = json.loads(line)
            file_name = data.get("file_name", "")
            objects = data.get("objects", {})
            category = objects.get("categories", [])[0] if "categories" in objects else None
            bboxes = objects.get("bbox", [])

            # Determine the category key
            category_key = get_category_key(file_name, category)
            if not category_key:
                updated_lines.append(line)  # Keep the line unchanged
                continue

            # Initialize tracking for this category if not already present
            if category_key not in category_map:
                category_map[category_key] = {"invalid": True, "file_name": file_name}
                category_files[category_key] = []

            # Track all files in this category
            category_files[category_key].append(file_name)

            # Check if bboxes in this file are valid
            valid_bbox_found = any(not all(value == 0 for value in bbox) for bbox in bboxes)
            if valid_bbox_found:
                category_map[category_key]["invalid"] = False

            # Update the first invalid file if necessary
            if category_map[category_key]["invalid"] and valid_bbox_found:
                first_file_name = category_map[category_key]["file_name"]
                for i, updated_line in enumerate(updated_lines):
                    updated_data = json.loads(updated_line)
                    if updated_data["file_name"] == first_file_name:
                        updated_data["objects"]["bbox"] = bboxes
                        updated_lines[i] = json.dumps(updated_data) + "\n"
                        category_map[category_key]["invalid"] = False
                        break

            # Add the line to updated output
            updated_lines.append(line)

    # Move invalid files to the invalid folder
    if not os.path.exists(invalid_folder):
        os.makedirs(invalid_folder)

    for category_key, metadata in category_map.items():
        if metadata["invalid"]:  # If all bboxes in the category are invalid
            print(f"Category '{category_key}' is invalid. Moving files to {invalid_folder}")
            for file_name in category_files[category_key]:
                source_path = os.path.join(source_folder, file_name)
                destination_path = os.path.join(invalid_folder, file_name)
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
                else:
                    print(f"File not found: {source_path}")

    # Save the updated JSONL file
    # with open(output_file_path, "w") as outfile:
    #     outfile.writelines(updated_lines)

    print(f"Updated JSONL file saved to: {output_file_path}")
    print(f"Invalid files moved to: {invalid_folder}")

# Example usage

input_file = "/data/Deepfake/nt_object/test/metadata.jsonl"  # Replace with your input JSONL file path
output_file = "/data/Deepfake/ff_c23_matafile/invalid_categories_remove_test.jsonl"   # Replace with your desired output JSON file path
source_folder = "/data/Deepfake/nt_object/test/"  # Folder containing original files
invalid_folder = "/data/Deepfake/ff_c23_remove/nt/test/"  # Folder to move invalid files
update_invalid_bboxes_and_copy(input_file, output_file, source_folder, invalid_folder)


def validate_and_update_bboxes(input_file_path, output_file_path):
    """
    Updates bounding boxes in each category of the JSONL file:
    - Checks all images in a category for valid bounding boxes.
    - If a bbox is invalid, replaces it with the bbox from the first image of that category (e.g., ..._1.png).
    - Skips the category if the bbox in the first image is invalid.

    Args:
    - input_file_path (str): Path to the input JSONL file.
    - output_file_path (str): Path to save the updated JSONL file.
    """
    category_map = {}  # Maps category keys to their first image (..._1.png) and bbox
    updated_lines = []  # List to store updated JSONL lines

    def get_category_key(file_name, category):
        """Determine the category key based on the filename and category."""
        parts = file_name.split("_")
        if category == 1 and len(parts) > 0:
            return parts[0]  # Use the first part for category 1
        elif category == 0 and len(parts) > 1:
            return f"{parts[0]}_{parts[1]}"  # Use first two parts for category 0
        return None


    # Read and process the JSONL file
    with open(input_file_path, "r") as infile:
        lines = list(infile)

        for line in lines:
            data = json.loads(line)
            file_name = data.get("file_name", "")
            objects = data.get("objects", {})
            category = objects.get("categories", [])[0] if "categories" in objects else None
            bboxes = objects.get("bbox", [])

            # Determine the category key
            category_key = get_category_key(file_name,category)

            # Identify the first image (..._1.png) for each category
            if category_key not in category_map and file_name.endswith("_1.png"):
                valid_bbox = next((bbox for bbox in bboxes if not all(value == 0 for value in bbox)), None)
                category_map[category_key] = {"file_name": file_name, "bbox": valid_bbox}

            updated_lines.append(data)

    # Validate and update bboxes in each category
    for data in updated_lines:
        file_name = data.get("file_name", "")
        objects = data.get("objects", {})
        bboxes = objects.get("bbox", [])
        category = objects.get("categories", [])[0] if "categories" in objects else None
        category_key = get_category_key(file_name,category)

        # Get the bbox from the first image in the category
        first_bbox = category_map[category_key]["bbox"] if category_key in category_map else None

        # If the first bbox is valid, update invalid bboxes in this file
        if first_bbox:
            updated_bboxes = []
            for bbox in bboxes:
                if all(value == 0 for value in bbox):  # Invalid bbox
                    updated_bboxes.append(first_bbox)  # Replace with the first bbox
                else:
                    updated_bboxes.append(bbox)
            data["objects"]["bbox"] = updated_bboxes

    # Save the updated JSONL file
    with open(output_file_path, "w") as outfile:
        for data in updated_lines:
            outfile.write(json.dumps(data) + "\n")

    print(f"Updated JSONL file saved to: {output_file_path}")

# Example usage

# input_file = "/data/Deepfake/ff_c23_matafile/invalid_categories_remove_test.jsonl"  # Replace with your input JSONL file path
# output_file = "/data/Deepfake/ff_c23_object/metadata20class_test.jsonl"  # Replace with your desired output JSON file path
# validate_and_update_bboxes(input_file, output_file)