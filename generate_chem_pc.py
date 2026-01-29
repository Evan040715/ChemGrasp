c#!/usr/bin/env python3
"""
临时脚本：为 Chem 数据集中的特定物体生成点云
不修改原 generate_pc.py 文件
"""
import os
import sys
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

def generate_chem_object_pc(object_names, num_points=512):
    """为 Chem 数据集中的物体生成点云"""
    dataset_type = 'Chem'
    input_dir = os.path.join(ROOT_DIR, 'data/data_urdf/object', dataset_type)
    output_dir = os.path.join(ROOT_DIR, 'data/PointCloud/object', dataset_type)
    os.makedirs(output_dir, exist_ok=True)
    
    for object_name in object_names:
        print(f'Processing {dataset_type}/{object_name}...')
        mesh_path = os.path.join(input_dir, object_name, f'{object_name}.stl')
        
        if not os.path.exists(mesh_path):
            print(f'Warning: {mesh_path} not found, skipping...')
            continue
            
        mesh = trimesh.load_mesh(mesh_path)
        object_pc, face_indices = mesh.sample(num_points, return_index=True)
        object_pc = torch.tensor(object_pc, dtype=torch.float32)
        normals = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)
        object_pc_normals = torch.cat([object_pc, normals], dim=-1)
        
        output_path = os.path.join(output_dir, f'{object_name}.pt')
        torch.save(object_pc_normals, output_path)
        print(f'  Saved to {output_path}')
        print(f'  Point cloud shape: {object_pc_normals.shape}')
    
    print("\nGenerating object point cloud finished.")

if __name__ == '__main__':
    # 生成这两个物体的点云
    objects = ['beaker_250ml', 'conical_beaker_250ml']
    generate_chem_object_pc(objects, num_points=512)
