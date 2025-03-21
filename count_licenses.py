import json
import os

def count_license_images(json_path):
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} does not exist!")
        return
    
    try:
        # Load the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get total number of images
        total_images = len(data['images'])
        
        # Create a set to store unique image IDs that match our license criteria
        target_images = set()
        
        # Count images with licenses 4-7
        for image in data['images']:
            if 4 <= image['license'] <= 7:
                target_images.add(image['id'])
        
        # Print results
        print(f"\nResults for {os.path.basename(json_path)}:")
        print(f"Total number of images: {total_images}")
        print(f"Total images with licenses 4-7: {len(target_images)}")
        print(f"Percentage with licenses 4-7: {(len(target_images)/total_images)*100:.1f}%")
        
        # Count individual licenses
        license_counts = {i: 0 for i in range(4, 8)}
        for image in data['images']:
            if image['license'] in license_counts:
                license_counts[image['license']] += 1
        
        # Print breakdown
        print("\nBreakdown by license:")
        for license_id, count in license_counts.items():
            percentage = (count/total_images)*100
            print(f"License {license_id}: {count} images ({percentage:.1f}%)")
            
    except json.JSONDecodeError:
        print(f"Error: {json_path} is not a valid JSON file!")
    except KeyError as e:
        print(f"Error: Missing required key in JSON structure: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

if __name__ == "__main__":
    json_path = "data/coco/annotations/coco_wholebody_train_v1.0.json"
    count_license_images(json_path) 