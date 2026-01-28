"""
Convert BDD100K lane annotations to UFLD v2 format.

This script generates:
1. Segmentation mask images (PNG) where pixel values indicate lane IDs (1-4)
2. Train/val list files in the format: "image_path label_path bin_labels"
3. Cache file (bdd100k_anno_cache.json) with pre-computed lane coordinates

BDD100K format:
- Labels: JSON with {frames: [{objects: [...]}]} structure
- Lane objects have category starting with 'lane/' and poly2d data as [[x, y, type], ...]

UFLD v2 format:
- Labels: PNG segmentation masks with lane IDs as pixel values
- Cache: JSON with lane coordinates for each image
"""

import json
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# === Configuration ===
BDD_ROOT = Path('~/capstone/bdd100k').expanduser()
OUTPUT_ROOT = Path('~/capstone/Ultra-Fast-Lane-Detection-v2/data/bdd100k').expanduser()
SPLITS = ['train', 'val']

# BDD100K image dimensions
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# Lane drawing settings
LANE_THICKNESS = 16

# TuSimple-compatible row anchors (56 points from y=160 to y=710)
ROW_ANCHORS = np.array([160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                        270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370,
                        380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480,
                        490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
                        600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])


def calc_k(points):
    """Calculate the direction (angle) of a lane."""
    if len(points) < 2:
        return -10
    
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    length = np.sqrt((xs[0] - xs[-1])**2 + (ys[0] - ys[-1])**2)
    if length < 50:
        return -10
    
    try:
        p = np.polyfit(xs, ys, deg=1)
        rad = np.arctan(p[0])
        return rad
    except:
        return -10


def draw_lane(mask, points, lane_id):
    """Draw a lane on the segmentation mask."""
    if len(points) < 2:
        return
    
    pts = [(int(p[0]), int(p[1])) for p in points]
    for i in range(len(pts) - 1):
        cv2.line(mask, pts[i], pts[i + 1], lane_id, thickness=LANE_THICKNESS)


def interpolate_lane_to_anchors(points, row_anchors):
    """
    Interpolate lane points to get x-coordinates at each row anchor.
    Returns array of shape (num_anchors, 2) with [y, x] coordinates.
    """
    if len(points) < 2:
        result = np.zeros((len(row_anchors), 2))
        result[:, 0] = row_anchors
        result[:, 1] = -99999
        return result
    
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    
    # Sort by y
    sort_idx = np.argsort(ys)
    xs = xs[sort_idx]
    ys = ys[sort_idx]
    
    result = np.zeros((len(row_anchors), 2))
    result[:, 0] = row_anchors
    result[:, 1] = -99999  # Default invalid
    
    try:
        # Interpolate x at each row anchor
        x_interp = np.interp(row_anchors, ys, xs, left=-99999, right=-99999)
        
        # Only keep valid interpolations (within original y range)
        for i, (y_anchor, x_val) in enumerate(zip(row_anchors, x_interp)):
            if ys.min() <= y_anchor <= ys.max():
                if 0 <= x_val < IMG_WIDTH:
                    result[i, 1] = x_val
    except:
        pass
    
    return result


def organize_lanes(lanes, img_width):
    """
    Organize lanes into 4 fixed positions using direction-based sorting.
    Returns tuple of (organized_points, organized_raw, bin_labels)
    """
    if not lanes:
        empty_points = np.zeros((4, len(ROW_ANCHORS), 2))
        empty_points[:, :, 0] = np.tile(ROW_ANCHORS, (4, 1))
        empty_points[:, :, 1] = -99999
        return empty_points, [[], [], [], []], [0, 0, 0, 0]
    
    # Calculate direction for each lane
    lane_directions = []
    for lane_pts in lanes:
        k = calc_k(lane_pts)
        lane_directions.append((k, lane_pts))
    
    # Separate left (negative direction) and right (positive direction) lanes
    left_lanes = [(k, pts) for k, pts in lane_directions if k < 0 and k != -10]
    right_lanes = [(k, pts) for k, pts in lane_directions if k > 0 and k != -10]
    
    # Sort: left lanes by angle descending, right lanes by angle ascending
    left_lanes.sort(key=lambda x: x[0], reverse=True)
    right_lanes.sort(key=lambda x: x[0])
    
    # Initialize results
    all_points = np.zeros((4, len(ROW_ANCHORS), 2))
    all_points[:, :, 0] = np.tile(ROW_ANCHORS, (4, 1))
    all_points[:, :, 1] = -99999
    
    organized_raw = [[], [], [], []]
    bin_labels = [0, 0, 0, 0]
    
    # Assign left lanes (positions 0 and 1)
    if len(left_lanes) >= 2:
        organized_raw[0] = left_lanes[1][1]  # outer left
        organized_raw[1] = left_lanes[0][1]  # inner left
        all_points[0] = interpolate_lane_to_anchors(left_lanes[1][1], ROW_ANCHORS)
        all_points[1] = interpolate_lane_to_anchors(left_lanes[0][1], ROW_ANCHORS)
        bin_labels[0] = 1
        bin_labels[1] = 1
    elif len(left_lanes) == 1:
        organized_raw[1] = left_lanes[0][1]  # single left goes to inner
        all_points[1] = interpolate_lane_to_anchors(left_lanes[0][1], ROW_ANCHORS)
        bin_labels[1] = 1
    
    # Assign right lanes (positions 2 and 3)
    if len(right_lanes) >= 2:
        organized_raw[2] = right_lanes[0][1]  # inner right
        organized_raw[3] = right_lanes[1][1]  # outer right
        all_points[2] = interpolate_lane_to_anchors(right_lanes[0][1], ROW_ANCHORS)
        all_points[3] = interpolate_lane_to_anchors(right_lanes[1][1], ROW_ANCHORS)
        bin_labels[2] = 1
        bin_labels[3] = 1
    elif len(right_lanes) == 1:
        organized_raw[2] = right_lanes[0][1]  # single right goes to inner
        all_points[2] = interpolate_lane_to_anchors(right_lanes[0][1], ROW_ANCHORS)
        bin_labels[2] = 1
    
    return all_points, organized_raw, bin_labels


def process_split(split):
    """Process one split (train or val)"""
    image_dir = BDD_ROOT / 'images' / '100k' / split
    label_dir = BDD_ROOT / 'labels' / '100k' / split
    output_label_dir = OUTPUT_ROOT / 'labels' / split
    output_list_file = OUTPUT_ROOT / 'lists' / f'{split}_gt.txt'
    
    print(f"\n{split.upper()} Split Processing:")
    print(f"  Image dir: {image_dir} (exists: {image_dir.exists()})")
    print(f"  Label dir: {label_dir} (exists: {label_dir.exists()})")
    
    # Create output directories
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_list_file.parent.mkdir(parents=True, exist_ok=True)
    
    list_entries = []
    cache_dict = {}  # For the annotation cache
    processed = 0
    skipped_no_label = 0
    skipped_no_lanes = 0
    
    image_paths = sorted(image_dir.glob('*.jpg'))
    print(f"  Found {len(image_paths)} images")
    
    for img_path in tqdm(image_paths, desc=f"Processing {split}"):
        img_name = img_path.name
        json_path = label_dir / img_name.replace('.jpg', '.json')
        output_png_path = output_label_dir / img_name.replace('.jpg', '.png')
        
        if not json_path.exists():
            skipped_no_label += 1
            continue
        
        try:
            with open(json_path) as f:
                label_data = json.load(f)
            
            # Get objects from first frame
            objects = label_data.get('frames', [{}])[0].get('objects', [])
            
            # Extract lane polylines
            lanes = []
            for obj in objects:
                category = obj.get('category', '')
                if category.startswith('lane/') and 'poly2d' in obj:
                    lanes.append(obj['poly2d'])
            
            if not lanes:
                skipped_no_lanes += 1
                continue
            
            # Organize lanes into 4 positions
            all_points, organized_raw, bin_labels = organize_lanes(lanes, IMG_WIDTH)
            
            # Create segmentation mask
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            
            for lane_id, lane_pts in enumerate(organized_raw, start=1):
                if lane_pts:
                    draw_lane(mask, lane_pts, lane_id)
            
            # Only save if we have at least one lane
            if mask.max() > 0:
                cv2.imwrite(str(output_png_path), mask)
                
                # Relative paths for list file
                rel_img = f'images/{split}/{img_name}'
                rel_label = f'labels/{split}/{img_name.replace(".jpg", ".png")}'
                
                # Add to list with bin labels
                bin_str = ' '.join(map(str, bin_labels))
                list_entries.append(f'{rel_img} {rel_label} {bin_str}')
                
                # Add to cache (key is relative image path)
                cache_dict[rel_img] = all_points.tolist()
                
                processed += 1
            else:
                skipped_no_lanes += 1
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    # Write list file
    with open(output_list_file, 'w') as f:
        f.write('\n'.join(list_entries))
    
    print(f"\n{split} Results:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (no label file): {skipped_no_label}")
    print(f"  Skipped (no lanes): {skipped_no_lanes}")
    
    return cache_dict


def create_symlinks():
    """Create symlinks for images in the output directory"""
    for split in SPLITS:
        src_image_dir = BDD_ROOT / 'images' / '100k' / split
        dst_image_dir = OUTPUT_ROOT / 'images' / split
        
        dst_image_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if dst_image_dir.is_symlink():
            dst_image_dir.unlink()
        elif dst_image_dir.exists():
            import shutil
            shutil.rmtree(dst_image_dir)
        
        dst_image_dir.symlink_to(src_image_dir)
        print(f"Created symlink: {dst_image_dir} -> {src_image_dir}")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=np.RankWarning)
    
    print("BDD100K to UFLD v2 Conversion")
    print("=" * 50)
    print(f"BDD_ROOT: {BDD_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    
    # Create image symlinks
    print("\nCreating image symlinks...")
    create_symlinks()
    
    # Process each split and collect cache data
    all_cache = {}
    for split in SPLITS:
        cache_dict = process_split(split)
        all_cache.update(cache_dict)
    
    # Write combined cache file
    cache_path = OUTPUT_ROOT / 'bdd100k_anno_cache.json'
    print(f"\nWriting annotation cache to {cache_path}...")
    with open(cache_path, 'w') as f:
        json.dump(all_cache, f)
    print(f"Cache contains {len(all_cache)} entries")
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"\nTo train:")
    print(f"  python train.py configs/bdd100k_res18.py")
