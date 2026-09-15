import os
import shutil
from PIL import Image
import json


def create_groundtruth(query_paths, dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        if img.endswith((".jpg", ".png", ".jpeg")):
            data['imlist'].append(img)

    if all(os.path.isdir(path) for path in query_paths) and (dataset == "ILIAS" or dataset == "ILIAS_Test"):
        for path in query_paths:
            temp_queries = []
            class_pos_images = []
            temp_path_query = os.path.join(path, "query")
            temp_path_pos = os.path.join(path, "pos")

            # A. Gather all positive images for this specific class first
            for file in sorted(os.listdir(temp_path_pos)):
                if file.endswith((".jpg", ".png", ".jpeg")):
                    pos_file = os.path.join("queries", os.path.basename(path), 'pos', file)
                    data['imlist'].append(pos_file)
                    class_pos_images.append(pos_file)

            # B. Find the text description for this class
            class_text = ""
            for file in sorted(os.listdir(temp_path_query)):
                if file.endswith(".txt") and file.startswith("T"):
                    with open(os.path.join(temp_path_query, file), 'r', encoding='utf-8') as f:
                        class_text = f.read().strip()
                    break

            # C. Process Image Queries and inject the text
            for file in sorted(os.listdir(temp_path_query)):
                if file.endswith((".jpg", ".png", ".jpeg")):
                    query_file = os.path.join(os.path.basename(path), 'query', file)
                    query_name = os.path.splitext(file)[0]
                    temp_queries.append(query_file)

                    if query_name not in query_info:
                        bbx_path = os.path.join(temp_path_query, query_name + '_bbox.txt')
                        with open(bbx_path, 'r') as f:
                            content = f.read().strip()
                            raw_bbx = list(map(float, content.split()))
                            x, y, w, h = raw_bbx
                            x2 = x + w
                            y2 = y + h
                            bbx = [x, y, x2, y2]

                        query_info[query_file] = {
                            'query': query_file,
                            'bbx': bbx,
                            'text': class_text,
                            'ok': class_pos_images.copy(),
                            'good': [],
                            'junk': []
                        }
                        data['qimlist'].append(query_file)
    else:
        # Iterate over all image files in the directory
        for filename in sorted(query_paths):
            query_name = os.path.basename(filename)
            category = 'query'

            if query_name not in query_info:
                query_info[query_name] = {'query': None, 'bbx': None, 'text': "",
                                          'ok': [], 'good': [], 'junk': []}

            if category == 'query':
                query_info[query_name][category] = query_name
                w, h = Image.open(filename).size
                query_info[query_name]['bbx'] = [0, 0, w, h]
                data['qimlist'].append(query_name)

            if category in ['ok', 'good', 'junk']:
                query_info[query_name][category].append(None)

    for query_name, info in query_info.items():
        data['gnd'].append(info)

    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


def create_groundtruth_from_txt(dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        if img.endswith((".jpg", ".png", ".jpeg")):
            data['imlist'].append(img)

    groundtruth_dir = os.path.join(dir_path, dataset, "groundtruth")
    for filename in sorted(os.listdir(groundtruth_dir)):
        if filename.endswith('.txt'):
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])
            category = parts[-1][:-4]

            with open(os.path.join(groundtruth_dir, filename), 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.split()
                if query_name not in query_info:
                    query_info[query_name] = {'query': None, 'bbx': None, 'text': "",
                                              'ok': [], 'good': [], 'junk': []}

                if category == 'query':
                    file_name = parts[0][5:] + '.jpg'
                    query_info[query_name][category] = file_name
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(file_name)

                    src_path = os.path.join(dir_path, dataset, file_name)
                    dst_path = os.path.join(dir_path, dataset, "queries", file_name)
                    shutil.copy(src_path, dst_path)

                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

    for query_name, info in query_info.items():
        data['gnd'].append(info)

    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data


def create_groundtruth_imagenet10(dir_path, dataset):
    """
    Creates a ground truth JSON file based on the flat ImageNet-10 Active Learning structure.
    - Database images: Sitting directly in the dataset root (e.g., Category_1_Global/*.jpg)
    - Queries: Sitting in the 'queries' subfolder.
    - Text: Extracted from a paired .txt file in the queries folder, or falls back to a template.
    """
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': os.path.join(dir_path, dataset)}
    query_info = {}

    target_classes = {
        "n02128385": "Leopard", "n02128757": "Snow Leopard",
        "n02130308": "Cheetah", "n02129604": "Tiger", "n02129165": "Lion",
        "n02114367": "Timber Wolf", "n02114548": "Arctic Wolf",
        "n02114855": "Coyote", "n02117135": "Hyena", "n02116738": "African Hunting Dog"
    }

    base_path = os.path.join(dir_path, dataset)
    queries_path = os.path.join(base_path, "queries")

    # 1. Populate the Database (imlist) from the flat category root
    # Note: os.path.isfile ensures we don't accidentally append the 'queries' folder
    for img in sorted(os.listdir(base_path)):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(base_path, img)):
            data['imlist'].append(img)

    # 2. Process Queries and Inject Text Modality
    if os.path.exists(queries_path):
        for q_img in sorted(os.listdir(queries_path)):
            if q_img.lower().endswith(('.jpg', '.jpeg', '.png')):
                q_rel_path = os.path.join("queries", q_img).replace('\\', '/')
                data['qimlist'].append(q_rel_path)

                # Extract the WNID dynamically (e.g., n02128757 from n02128757_mistake1.jpg)
                wnid = next((w for w in target_classes.keys() if w in q_img), q_img.split('_')[0])

                # Find all true positive matches in the database using the clean WNID
                ok_matches = [img for img in data['imlist'] if wnid in img]

                # Retrieve BBox dimensions (Default to full image for ImageNet)
                try:
                    w, h = Image.open(os.path.join(queries_path, q_img)).size
                except Exception:
                    w, h = 224, 224

                    # ---------------------------------------------------------
                # TEXTUAL DESCRIPTION LOGIC (Crucial for SigLIP)
                # ---------------------------------------------------------
                text_query = ""
                txt_file_path = os.path.join(queries_path, os.path.splitext(q_img)[0] + '.txt')

                # If you wrote a custom query description (e.g., n02128757_mistake1.txt), use it.
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                        text_query = f.read().strip()
                else:
                    # Fallback template if a text file hasn't been written yet
                    class_name = target_classes.get(wnid, "animal")
                    text_query = f"A photo of a {class_name}"

                query_info[q_rel_path] = {
                    'query': q_rel_path,
                    'bbx': [0, 0, w, h],
                    'text': text_query,
                    'ok': ok_matches,
                    'good': [],
                    'junk': []
                }
                data['gnd'].append(query_info[q_rel_path])

    # 3. Save the RevisitOP-compatible JSON
    output_file = os.path.join(base_path, f'gnd_{dataset}.json')
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data