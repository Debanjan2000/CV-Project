import json
import os

def map_predictions():
    valid_json_path = 'data/Valid.json'
    predictions_path = 'submissions/predictions_v12m.json'
    mapped_predictions_path = 'submissions/predictions_v12m_mapped.json'
    
    print(f"Loading {valid_json_path}...")
    with open(valid_json_path, 'r') as f:
        valid_data = json.load(f)
        
    # Create mapping from uid (frames_...) to numerical id (2020...)
    uid_to_id = {}
    for img in valid_data['images']:
        filename = img['file_name']
        # The YOLO images use flattened names with underscores instead of folders
        uid = os.path.splitext(filename.replace('/', '_'))[0]
        uid_to_id[uid] = str(img['id'])
        
    print(f"Loading {predictions_path}...")
    with open(predictions_path, 'r') as f:
        preds = json.load(f)
        
    # Swap the keys
    mapped_preds = {}
    for uid, data in preds.items():
        if uid in uid_to_id:
            mapped_preds[uid_to_id[uid]] = data
        else:
            print(f"Warning: UID {uid} not found in Valid.json")
            
    with open(mapped_predictions_path, 'w') as f:
        json.dump(mapped_preds, f, indent=4)
        
    print(f"Successfully mapped {len(mapped_preds)} predictions!")
    print(f"Saved to {mapped_predictions_path}")

if __name__ == '__main__':
    map_predictions()
