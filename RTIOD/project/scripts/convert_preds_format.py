import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json')
    parser.add_argument('output_json')
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    # Convert list of predictions to dictionary by image_id
    converted = {}
    for pred in data:
        img_id = pred['image_id']
        if img_id not in converted:
            converted[img_id] = {'boxes': [], 'labels': [], 'scores': []}
        
        # the evaluator expects labels to be 1-indexed (COCO format)
        converted[img_id]['boxes'].append(pred['bbox'])
        converted[img_id]['labels'].append(pred['category_id']) 
        converted[img_id]['scores'].append(pred['score'])

    with open(args.output_json, 'w') as f:
        json.dump(converted, f, indent=4)
        
    print(f"Converted {len(data)} predictions across {len(converted)} images.")

if __name__ == '__main__':
    main()
