import os
import json
import argparse
import numpy as np
from mmpose.apis.inferencers import MMPoseInferencer

def is_valid_image_file(filename):
    """Check if file is a valid image file (PNG or JPG/JPEG)."""
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    return os.path.splitext(filename.lower())[1] in valid_extensions

def process_predictions(predictions):
    """Convert predictions to a serializable format."""
    result = {}
    
    # Handle different prediction formats
    if isinstance(predictions, dict):
        if 'predictions' in predictions:
            pred_instances = predictions['predictions'][0]
        else:
            print("No predictions found in result")
            return result
    else:
        pred_instances = predictions[0]
        
    # Get keypoints and scores
    if hasattr(pred_instances, 'pred_instances'):
        instances = pred_instances.pred_instances
        
        # Convert keypoints and scores to regular lists
        if hasattr(instances, 'keypoints'):
            keypoints = instances.keypoints
            # Handle both torch tensors and numpy arrays
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu().numpy()
            if isinstance(keypoints, np.ndarray):
                keypoints = keypoints.tolist()
            result['keypoints'] = keypoints
            print(f"Found {len(keypoints)} keypoints")
        
        if hasattr(instances, 'keypoint_scores'):
            keypoint_scores = instances.keypoint_scores
            # Handle both torch tensors and numpy arrays
            if hasattr(keypoint_scores, 'cpu'):
                keypoint_scores = keypoint_scores.cpu().numpy()
            if isinstance(keypoint_scores, np.ndarray):
                keypoint_scores = keypoint_scores.tolist()
            result['keypoint_scores'] = keypoint_scores
            
            # Calculate average score, handling nested lists
            if keypoint_scores:
                if isinstance(keypoint_scores[0], list):
                    # Handle nested lists
                    scores = [score for sublist in keypoint_scores for score in sublist]
                else:
                    scores = keypoint_scores
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"Average confidence score: {avg_score:.3f}")
        
        # Add bbox if available
        if hasattr(instances, 'bboxes'):
            bboxes = instances.bboxes
            # Handle both torch tensors and numpy arrays
            if hasattr(bboxes, 'cpu'):
                bboxes = bboxes.cpu().numpy()
            if isinstance(bboxes, np.ndarray):
                bboxes = bboxes.tolist()
            result['bboxes'] = bboxes
            print(f"Found {len(bboxes)} bounding boxes")
    else:
        print("No pred_instances found in prediction")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run pose estimation on a folder of images')
    parser.add_argument('input_dir', help='Directory containing input images (PNG or JPG)')
    parser.add_argument('output_dir', help='Directory to save JSON files')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    inferencer = MMPoseInferencer(
        pose2d='configs/face_2d_keypoint/rtmpose/coco_wholebody_face/rtmpose-t_8xb32-60e_coco-wholebody-face-256x256.py',
        pose2d_weights='work_dirs/rtmpose-t_8xb32-60e_coco-wholebody-face-256x256/best_NME_epoch_56.pth',
        device=args.device
    )
    print("Model initialized successfully")
    
    # Get list of valid image files
    all_files = os.listdir(args.input_dir)
    image_files = [f for f in all_files if is_valid_image_file(f)]
    
    if not image_files:
        print(f"No PNG or JPG files found in {args.input_dir}")
        return
    
    skipped_files = [f for f in all_files if os.path.isfile(os.path.join(args.input_dir, f)) and not is_valid_image_file(f)]
    if skipped_files:
        print(f"Skipping {len(skipped_files)} non-PNG/JPG files")
    
    total_images = len(image_files)
    processed_count = 0
    skipped_count = 0
    print(f"\nStarting processing of {total_images} images...")
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        # Check if JSON already exists
        json_path = os.path.join(args.output_dir, os.path.splitext(img_file)[0] + '.json')
        if os.path.exists(json_path):
            print(f"\nSkipping {img_file} - JSON already exists")
            skipped_count += 1
            continue
            
        img_path = os.path.join(args.input_dir, img_file)
        print(f"\nProcessing image {i}/{total_images}: {img_file}")
        
        # Run inference
        predictions = next(inferencer(img_path, return_datasample=True))
        
        # Process results
        result = process_predictions(predictions)
        
        # If no keypoints were found, skip this image
        if not result.get('keypoints'):
            print(f"WARNING: No keypoints detected in {img_file}, skipping")
            continue
            
        # Save individual JSON file
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        processed_count += 1
        print(f"Progress: {i}/{total_images} ({(i/total_images)*100:.1f}%)")
        print(f"Saved results to {json_path}")
    
    print(f"\nProcessing complete:")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already existed): {skipped_count}")
    print(f"Failed/no keypoints: {total_images - processed_count - skipped_count}")
    print(f"Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main() 