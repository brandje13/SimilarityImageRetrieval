import os
import shutil
from PIL import Image
import json


def create_groundtruth(query_paths, dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Determine the category based on the filename
    # parts = filename.split('_')
    if os.name == 'nt':
        split = "\\"
    elif os.name == 'posix':
        split = "/"
    else:
        split = "_"  # TODO: Better fix for this

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg, .jpeg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg"):
            # Add the file to the list
            data['imlist'].append(img)

    if all(os.path.isdir(path) for path in query_paths) and (dataset == "ILIAS" or dataset == "ILIAS_Test"):
        for path in query_paths:
            temp_queries = []
            temp_path = os.path.join(path, "query")

            # Iterate through each file in the query directory
            for file in sorted(os.listdir(temp_path)):
                # Check if the file ends with .jpg, .jpeg or .png
                # Leave extentions to handle multiple data types
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    query_file = os.path.join(path.split('\\')[-1], 'query', file)
                    query_name = file.split('.')[0]
                    temp_queries.append(query_file)

                    if query_name not in query_info:
                        bbx_path = os.path.join(path, 'query', query_name + '_bbox.txt')
                        with open(bbx_path, 'r') as f:
                            content = f.read().strip()
                            raw_bbx = list(map(float, content.split()))
                            x, y, w, h = raw_bbx
                            x2 = x + w
                            y2 = y + h

                            bbx = [x, y, x2, y2]

                        query_info[query_file] = {'query': query_file, 'bbx': bbx,
                                                  'ok': [], 'good': [], 'junk': []}
                        data['qimlist'].append(query_file)

            # Iterate through each file in the pos directory
            temp_path = os.path.join(path, "pos")
            for file in sorted(os.listdir(temp_path)):
                # Check if the file ends with .jpg, .jpeg or .png
                # Leave extentions to handle multiple data types
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    pos_file = os.path.join("queries", path.split('\\')[-1], 'pos', file)
                    data['imlist'].append(pos_file)

                    for img in temp_queries:
                        query_info[img]['ok'].append(pos_file)
    else:
        # Iterate over all image files in the directory
        for filename in sorted(query_paths):
            query_name = filename.split(split)[-1]  # Extract query name
            category = 'query'

            # Process each line in the text file
            if query_name not in query_info:
                query_info[query_name] = {'query': None, 'bbx': None,
                                          'ok': [], 'good': [], 'junk': []}

            # Check if the line indicates a query
            if category == 'query':
                query_info[query_name][category] = query_name
                w, h = Image.open(filename).size
                query_info[query_name]['bbx'] = [0, 0, w, h]
                data['qimlist'].append(query_name)

            # Populate data dictionary based on category
            if category in ['ok', 'good', 'junk']:
                query_info[query_name][category].append(None)

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)


# Create groundtruth based on provided txt files
def create_groundtruth_from_txt(dir_path, dataset):
    data = {'imlist': [], 'qimlist': [], 'gnd': [], 'path': str}
    query_info = {}
    data['path'] = os.path.join(dir_path, dataset)

    # Iterate through each file in the directory
    for img in sorted(os.listdir(os.path.join(dir_path, dataset))):
        # Check if the file ends with .jpg or .png
        # Leave extentions to handle multiple data types
        if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg"):
            # Add the file to the list
            data['imlist'].append(img)

    # Iterate over all text files in the directory
    for filename in sorted(os.listdir(os.path.join(dir_path, dataset, "groundtruth"))):
        if filename.endswith('.txt'):
            # Determine the category based on the filename
            parts = filename.split('_')
            query_name = '_'.join(parts[:-1])  # Extract query name
            category = parts[-1][:-4]

            # Read the content of the text file
            with open(os.path.join(os.path.join(dir_path, dataset, "groundtruth"), filename), 'r') as file:
                lines = file.readlines()

            # Process each line in the text file
            for line in lines:
                parts = line.split()

                if query_name not in query_info:
                    query_info[query_name] = {'query': None, 'bbx': None,
                                              'ok': [], 'good': [], 'junk': []}

                # Check if the line indicates a query
                if category == 'query':
                    query_info[query_name][category] = parts[0][5:] + '.jpg'
                    query_info[query_name]['bbx'] = list(map(float, parts[1:]))
                    data['qimlist'].append(parts[0][5:] + '.jpg')
                    shutil.copy(os.path.join(dir_path, dataset, parts[0][5:] + '.jpg'), os.path.join(dir_path, dataset, "queries"))

                # Populate data dictionary based on category
                if category in ['ok', 'good', 'junk']:
                    query_info[query_name][category].append(parts[0] + '.jpg')

    # Populate 'gnd' based on query info
    for query_name, info in query_info.items():
        data['gnd'].append(info)

    # Save the result as json
    with open(os.path.join(dir_path, dataset, f'gnd_{dataset}.json'), 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return data
