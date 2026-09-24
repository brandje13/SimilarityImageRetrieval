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
    Strictly maps true positives by reading the 'true_Class' string and isolates C/R edge cases.
    """
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': os.path.join(dir_path, dataset)}
    query_info = {}

    target_classes = {
        "n02128385": "Leopard",
        "n02128757": "Snow_Leopard",
        "n02130308": "Cheetah",
        "n02129604": "Tiger",
        "n02129165": "Lion",
        "n02114367": "Timber_Wolf",
        "n02114548": "Arctic_Wolf",
        "n02114855": "Coyote",
        "n02117135": "Hyena",
        "n02116738": "African_Hunting_Dog"
    }

    base_path = os.path.join(dir_path, dataset)
    queries_path = os.path.join(base_path, "queries")

    print(f"[INFO] Building Ground Truth for: {dataset}")

    # 1. Populate the Database (imlist) from the flat category root
    for img in sorted(os.listdir(base_path)):
        if img.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(base_path, img)):
            data['imlist'].append(img)

    # 2. Process Queries & Map True Positives
    if os.path.exists(queries_path):
        for q_img in sorted(os.listdir(queries_path)):
            if q_img.lower().endswith(('.jpg', '.jpeg', '.png')):
                data['qimlist'].append(q_img)
                query_name = os.path.splitext(q_img)[0]

                # STRICT FIX: Force match only on the "true_" half of the string
                true_wnid = None
                for wnid, class_name in target_classes.items():
                    if f"true_{class_name}" in q_img:
                        true_wnid = wnid
                        break

                # Fallback if filename lacks a class string
                if not true_wnid:
                    true_wnid = q_img.split('_')[0]

                # Start with all images matching the true WNID
                ok_matches = [img for img in data['imlist'] if img.startswith(true_wnid)]

                # CATEGORY FILTERING FIX: Restrict positives to the actual targeted edge cases
                if "Category_1_Global" in dataset:
                    ok_matches = [img for img in ok_matches if "_C_" in img]
                elif "Category_5_Semantic" in dataset:
                    ok_matches = [img for img in ok_matches if "_R_" in img]
                # Categories 2, 3, and 4 keep the full list because they search for organic/starved standard images

                # ---------------------------------------------------------
                # BOUNDING BOX LOGIC
                # ---------------------------------------------------------
                bbx_path = os.path.join(queries_path, query_name + '_bbox.txt')
                if os.path.exists(bbx_path):
                    with open(bbx_path, 'r') as f:
                        raw_bbx = list(map(float, f.read().strip().split()))
                        x, y, w, h = raw_bbx
                        bbx = [x, y, x + w, y + h]
                else:
                    try:
                        w, h = Image.open(os.path.join(queries_path, q_img)).size
                    except Exception:
                        w, h = 224, 224
                    bbx = [0, 0, w, h]

                # ---------------------------------------------------------
                # TEXTUAL DESCRIPTION LOGIC
                # ---------------------------------------------------------
                text_query = ""
                txt_file_path = os.path.join(queries_path, query_name + '.txt')

                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r', encoding='utf-8') as f:
                        text_query = f.read().strip()
                else:
                    display_name = target_classes.get(true_wnid, "animal").replace("_", " ")
                    text_query = f"A photo of a {display_name}"

                query_info[q_img] = {
                    'query': q_img,
                    'bbx': bbx,
                    'text': text_query,
                    'ok': ok_matches,
                    'good': [],
                    'junk': []
                }
                data['gnd'].append(query_info[q_img])

    # 3. Save the RevisitOP-compatible JSON
    output_file = os.path.join(base_path, f'gnd_{dataset}.json')
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data