"""
Simple script to visualize object point clouds.
Loads point clouds from data/PointCloud/object/ and displays them using viser.
"""

import os
import sys
import argparse
import time
import viser
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.affordance import has_affordance, is_no_grasp_zone


def load_object_pc(dataset_name, object_name):
    """
    Load object point cloud from saved file.
    
    :param dataset_name: str, e.g., 'contactdb', 'ycb', 'Chem'
    :param object_name: str, e.g., 'apple', 'beaker'
    :return: torch.Tensor, (N, 3) or (N, 6) point cloud [xyz, (optional) normals]
    """
    pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{dataset_name}/{object_name}.pt')
    if not os.path.exists(pc_path):
        raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
    
    pc_data = torch.load(pc_path, map_location='cpu')
    if isinstance(pc_data, torch.Tensor):
        return pc_data
    else:
        raise ValueError(f"Unexpected data format in {pc_path}")


def list_available_objects():
    """List all available object point clouds."""
    pc_dir = os.path.join(ROOT_DIR, 'data/PointCloud/object')
    if not os.path.exists(pc_dir):
        print(f"Point cloud directory not found: {pc_dir}")
        return []
    
    objects = []
    for dataset_name in os.listdir(pc_dir):
        dataset_path = os.path.join(pc_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.pt'):
                object_name = file_name[:-3]  # remove .pt
                objects.append((dataset_name, object_name))
    
    return sorted(objects)


def main(dataset_name=None, object_name=None, show_normals=False):
    """
    Visualize object point cloud.
    
    :param dataset_name: str, dataset name (e.g., 'contactdb', 'ycb', 'Chem')
    :param object_name: str, object name (e.g., 'apple', 'beaker')
    :param show_normals: bool, whether to visualize normal vectors
    """
    # If not specified, list available objects
    if dataset_name is None or object_name is None:
        objects = list_available_objects()
        if len(objects) == 0:
            print("No object point clouds found!")
            return
        
        print("Available objects:")
        for i, (ds, obj) in enumerate(objects):
            print(f"  {i}: {ds}+{obj}")
        
        if dataset_name is None or object_name is None:
            print("\nUsage: python vis_object_pc.py --dataset <dataset> --object <object>")
            print(f"Example: python vis_object_pc.py --dataset Chem --object beaker")
            return
    
    # Load point cloud
    try:
        pc_data = load_object_pc(dataset_name, object_name)
        print(f"Loaded point cloud: {dataset_name}+{object_name}")
        print(f"  Shape: {pc_data.shape}")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return
    
    # Extract xyz and normals
    if pc_data.shape[1] >= 6:
        pc_xyz = pc_data[:, :3]
        pc_normals = pc_data[:, 3:6]
        has_normals = True
    elif pc_data.shape[1] >= 3:
        pc_xyz = pc_data[:, :3]
        pc_normals = None
        has_normals = False
    else:
        print(f"Unexpected point cloud shape: {pc_data.shape}")
        return
    
    print(f"  Point cloud: {pc_xyz.shape[0]} points")
    if has_normals:
        print(f"  Has normals: Yes")
    else:
        print(f"  Has normals: No")
    
    # Create viser server
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    
    # Add point cloud: blue = graspable, red = no-grasp (affordance) when defined; else height colormap
    if pc_xyz.shape[0] > 0:
        object_key = f"{dataset_name}+{object_name}"
        if has_affordance(object_key):
            no_grasp = is_no_grasp_zone(object_key, pc_xyz)
            if isinstance(no_grasp, torch.Tensor):
                no_grasp = no_grasp.cpu().numpy()
            # graspable = blue [0, 0, 1], no-grasp (affordance) = red [1, 0, 0]
            pc_colors_normalized = np.zeros((pc_xyz.shape[0], 3))
            pc_colors_normalized[~no_grasp] = [0.0, 0.0, 1.0]
            pc_colors_normalized[no_grasp] = [1.0, 0.0, 0.0]
            n_red = no_grasp.sum()
            n_blue = pc_xyz.shape[0] - n_red
            server.scene.add_point_cloud(
                'object_pc',
                pc_xyz.numpy(),
                point_size=0.001,
                point_shape="circle",
                colors=pc_colors_normalized
            )
            print(f"\nPoint cloud displayed with affordance coloring.")
            print(f"  Blue (graspable): {n_blue} points")
            print(f"  Red (no-grasp):   {n_red} points")
            print(f"  Open http://127.0.0.1:8080 in your browser to view.")
        else:
            # No affordance: height-based colormap
            z_coords = pc_xyz[:, 2].numpy()
            z_min, z_max = z_coords.min(), z_coords.max()
            if z_max > z_min:
                z_normalized = (z_coords - z_min) / (z_max - z_min)
            else:
                z_normalized = np.zeros_like(z_coords)
            z_norm = z_normalized
            pc_colors = np.zeros((len(z_norm), 3))
            mask1 = z_norm < 0.25
            t1 = z_norm[mask1] / 0.25
            pc_colors[mask1] = np.column_stack([np.zeros_like(t1), t1 * 255, np.full_like(t1, 255)])
            mask2 = (z_norm >= 0.25) & (z_norm < 0.5)
            t2 = (z_norm[mask2] - 0.25) / 0.25
            pc_colors[mask2] = np.column_stack([np.zeros_like(t2), np.full_like(t2, 255), (1 - t2) * 255])
            mask3 = (z_norm >= 0.5) & (z_norm < 0.75)
            t3 = (z_norm[mask3] - 0.5) / 0.25
            pc_colors[mask3] = np.column_stack([t3 * 255, np.full_like(t3, 255), np.zeros_like(t3)])
            mask4 = z_norm >= 0.75
            t4 = (z_norm[mask4] - 0.75) / 0.25
            pc_colors[mask4] = np.column_stack([np.full_like(t4, 255), (1 - t4) * 255, np.zeros_like(t4)])
            pc_colors_normalized = pc_colors / 255.0
            server.scene.add_point_cloud(
                'object_pc',
                pc_xyz.numpy(),
                point_size=0.001,
                point_shape="circle",
                colors=pc_colors_normalized
            )
            print(f"\nPoint cloud displayed with height-based coloring (no affordance for this object).")
            print(f"  Color scheme: Blue (low) -> Cyan -> Green -> Yellow -> Red (high)")
            print(f"  Z range: [{z_min:.4f}, {z_max:.4f}]")
            print(f"  Open http://127.0.0.1:8080 in your browser to view.")
    
    # Add normal vectors if available and requested
    if has_normals and show_normals and pc_normals is not None:
        # Sample normals for visualization (show every Nth normal to avoid clutter)
        sample_rate = max(1, pc_xyz.shape[0] // 100)  # Show at most 100 normals
        sampled_indices = np.arange(0, pc_xyz.shape[0], sample_rate)
        
        sampled_xyz = pc_xyz[sampled_indices].numpy()
        sampled_normals = pc_normals[sampled_indices].numpy()
        
        # Create line segments for normals (from point to point + normal*scale)
        normal_scale = 0.01  # Scale factor for normal visualization
        normal_endpoints = sampled_xyz + sampled_normals * normal_scale
        
        # viser doesn't have direct line drawing, so we'll use a workaround
        # Create small cylinders/spheres at endpoints
        for i, (start, end) in enumerate(zip(sampled_xyz, normal_endpoints)):
            # Create a simple line representation using a small cylinder
            # For simplicity, we'll just add points at endpoints
            pass  # Normal visualization can be added if needed
    
    # Keep server running
    print("\nPress Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize object point clouds')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (e.g., contactdb, ycb, Chem)')
    parser.add_argument('--object', type=str, default=None,
                        help='Object name (e.g., apple, beaker)')
    parser.add_argument('--show_normals', action='store_true',
                        help='Show normal vectors (if available)')
    parser.add_argument('--list', action='store_true',
                        help='List all available objects and exit')
    
    args = parser.parse_args()
    
    if args.list:
        objects = list_available_objects()
        if len(objects) == 0:
            print("No object point clouds found!")
        else:
            print("Available objects:")
            for ds, obj in objects:
                print(f"  {ds}+{obj}")
    else:
        main(args.dataset, args.object, args.show_normals)

