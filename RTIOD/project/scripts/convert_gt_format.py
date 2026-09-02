import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Convert COCO GT to dictionary format
    converted = {}
    
    # Map image IDs to file names/UIDs
    id_to_uid = {}
    for img in data['images']:
        # Extract UID from something like frames/20210214/clip_21_2239/image_0015.jpg -> frames_20210214_clip_21_2239_image_0015
        uid = img['file_name'].replace('/', '_').replace('.jpg', '')
        id_to_uid[img['id']] = uid
        converted[uid] = {'boxes': [], 'labels': []}

    for ann in data['annotations']:
        uid = id_to_uid[ann['image_id']]
        box = ann['bbox']
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        xyxy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        converted[uid]['boxes'].append(xyxy)
        converted[uid]['labels'].append(ann['category_id'])

    with open(args.output_json, 'w') as f:
        json.dump(converted, f, indent=4)
        
    print(f"Converted {len(data['annotations'])} annotations across {len(converted)} images.")

if __name__ == '__main__':
    main()
