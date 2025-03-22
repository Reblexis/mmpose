import os
import json
import cv2
import numpy as np
import argparse

def load_predictions(pred_dir):
    """Load all prediction JSON files."""
    predictions = {}
    for filename in os.listdir(pred_dir):
        if filename.endswith('.json'):
            image_name = os.path.splitext(filename)[0]
            with open(os.path.join(pred_dir, filename), 'r') as f:
                predictions[image_name] = json.load(f)
    return predictions

def draw_keypoints(image, keypoints, keypoint_scores, confidence_threshold=0.3):
    """Draw keypoints and connections on the image."""
    # Convert to numpy array if needed
    keypoints = np.array(keypoints)
    keypoint_scores = np.array(keypoint_scores)
    
    # Colors for visualization
    colors = {
        'high_confidence': (0, 255, 0),    # Green
        'medium_confidence': (0, 255, 255), # Yellow
        'low_confidence': (0, 0, 255)       # Red
    }
    
    # Draw each keypoint
    for kpt, score in zip(keypoints[0], keypoint_scores[0]):  # Assuming first person
        x, y = int(kpt[0]), int(kpt[1])
        if score < confidence_threshold:
            continue
            
        # Determine color based on confidence
        if score > 0.7:
            color = colors['high_confidence']
        elif score > 0.5:
            color = colors['medium_confidence']
        else:
            color = colors['low_confidence']
            
        # Draw the keypoint
        cv2.circle(image, (x, y), 2, color, -1)
        cv2.circle(image, (x, y), 4, color, 1)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='View pose predictions interactively')
    parser.add_argument('image_dir', help='Directory containing input images')
    parser.add_argument('pred_dir', help='Directory containing prediction JSONs')
    args = parser.parse_args()
    
    # Load predictions
    predictions = load_predictions(args.pred_dir)
    if not predictions:
        print("No predictions found!")
        return
        
    # Get sorted list of image names
    image_names = sorted(predictions.keys())
    current_idx = 0
    
    print("\nControls:")
    print("'d' - Next image")
    print("'a' - Previous image")
    print("'q' - Quit")
    print("'s' - Save current visualization")
    
    while True:
        # Load current image and its predictions
        image_name = image_names[current_idx]
        image_path = os.path.join(args.image_dir, image_name + '.png')  # Try PNG first
        if not os.path.exists(image_path):
            image_path = os.path.join(args.image_dir, image_name + '.jpg')  # Try JPG
            if not os.path.exists(image_path):
                print(f"Image not found: {image_name}")
                continue
                
        # Read and process image
        image = cv2.imread(image_path)
        pred = predictions[image_name]
        
        # Draw keypoints
        if 'keypoints' in pred and 'keypoint_scores' in pred:
            image = draw_keypoints(image, pred['keypoints'], pred['keypoint_scores'])
        
        # Show image with info
        info_text = f"Image {current_idx + 1}/{len(image_names)}: {image_name}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Predictions Viewer', image)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('d'):  # Next image
            current_idx = (current_idx + 1) % len(image_names)
        elif key == ord('a'):  # Previous image
            current_idx = (current_idx - 1) % len(image_names)
        elif key == ord('s'):  # Save current visualization
            save_path = os.path.join('visualizations', f'{image_name}_vis.jpg')
            os.makedirs('visualizations', exist_ok=True)
            cv2.imwrite(save_path, image)
            print(f"Saved visualization to {save_path}")
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 