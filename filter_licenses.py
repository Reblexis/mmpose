import json
import os
from copy import deepcopy

def filter_dataset_by_license(input_path, output_path):
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist!")
        return
    
    try:
        # Load the JSON file
        print(f"\nProcessing {os.path.basename(input_path)}...")
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Create a new dataset with the same structure
        filtered_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': data['categories'],
            'images': [],
            'annotations': []
        }
        
        # Get images with licenses 4-7
        valid_image_ids = set()
        for image in data['images']:
            if 4 <= image['license'] <= 7:
                valid_image_ids.add(image['id'])
                filtered_data['images'].append(image)
        
        # Get annotations for the filtered images
        for ann in data['annotations']:
            if ann['image_id'] in valid_image_ids:
                filtered_data['annotations'].append(ann)
        
        # Print statistics
        print(f"Original images: {len(data['images'])}")
        print(f"Filtered images: {len(filtered_data['images'])}")
        print(f"Original annotations: {len(data['annotations'])}")
        print(f"Filtered annotations: {len(filtered_data['annotations'])}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the filtered dataset
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f)
        print(f"Saved filtered dataset to {output_path}")
            
    except json.JSONDecodeError:
        print(f"Error: {input_path} is not a valid JSON file!")
    except KeyError as e:
        print(f"Error: Missing required key in JSON structure: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Process train dataset
    train_input = "data/coco/annotations/coco_wholebody_train_v1.0.json"
    train_output = "data/coco/annotations/coco_wholebody_train_v1.0_filtered_4-7.json"
    filter_dataset_by_license(train_input, train_output)
    
    # Process val dataset
    val_input = "data/coco/annotations/coco_wholebody_val_v1.0.json"
    val_output = "data/coco/annotations/coco_wholebody_val_v1.0_filtered_4-7.json"
    filter_dataset_by_license(val_input, val_output) 