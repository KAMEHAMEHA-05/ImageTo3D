import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from collections import defaultdict
import trimesh


device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
model = VGGT.from_pretrained("<saved_model_path>").to(device)
image_names = [r"<sample_image_path>"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

def three(path):
    image_names = [path]  
    images = load_and_preprocess_images(image_names).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    world_points = predictions['world_points'].squeeze().cpu().numpy()
    image_tensor = predictions['images'].squeeze()
    conf_tensor = predictions['depth_conf'].squeeze().cpu().numpy()
    
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    has_alpha = image_tensor.shape[0] == 4
    rgb_tensor = image_tensor[:3, :, :]
    alpha_np = image_tensor[3, :, :].cpu().numpy() if has_alpha else np.ones_like(conf_tensor)
    
    image_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1) if image_np.max() <= 1.0 else image_np / 255.0
    
    conf_thresh, alpha_thresh = 0.1, 0.05
    z_near, z_far = 0.01, 5.0
    
    valid_conf = conf_tensor > conf_thresh
    valid_alpha = alpha_np > alpha_thresh
    valid_z = np.logical_and(world_points[..., 2] > z_near, world_points[..., 2] < z_far)
    non_white = np.linalg.norm(image_np - np.array([1.0, 1.0, 1.0]), axis=-1) > (1.0 - 0.95)
    final_mask = valid_conf & valid_alpha & non_white & valid_z
    
    ys, xs = np.where(final_mask)
    pixels = np.stack((xs, ys), axis=-1)
    colors = image_np[ys, xs]
    points_3d = world_points[ys, xs]
    
    center = points_3d.mean(axis=0)
    points_3d_centered = points_3d - center
    
    tri = Delaunay(pixels)
    max_edge_length = 10.0
    valid_triangles = []
    for triangle in tri.simplices:
        pts = pixels[triangle]
        if np.max([np.linalg.norm(pts[i] - pts[(i + 1) % 3]) for i in range(3)]) < max_edge_length:
            valid_triangles.append(triangle)
    valid_triangles = np.array(valid_triangles)

    mirror_offset = 0.25
    points_mirror = points_3d_centered.copy()
    points_mirror[:, 2] *= -1
    points_mirror[:, 2] += mirror_offset
    
    mirror_triangles = valid_triangles + len(points_3d_centered)
    mirror_triangles = mirror_triangles[:, [0, 2, 1]]
    
    combined_vertices = np.vstack([points_3d_centered, points_mirror])
    combined_colors = np.vstack([colors, colors])
    
    def find_boundary_edges(triangles):
        edge_count = defaultdict(int)
        for tri in triangles:
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                edge_count[edge] += 1
        return [edge for edge, count in edge_count.items() if count == 1]
    
    front_boundary = find_boundary_edges(valid_triangles)
    
    avg_color = colors.mean(axis=0)
    
    def make_shades(base_color, num=10):
        shades = []
        for i in range(num):
            factor = i / (num - 1)
            shade = (1 - factor) * base_color * 0.6 + factor * base_color
            shades.append(np.clip(shade, 0, 1))
        return np.array(shades)
    
    avg_color_shades = make_shades(avg_color, num=10)
    
    side_vertices = []
    side_triangles = []
    side_colors = []
    
    offset = len(combined_vertices)
    color_index = 0
    
    for edge in front_boundary:
        v0, v1 = edge
        v0_m, v1_m = v0 + len(points_3d_centered), v1 + len(points_3d_centered)

        z_vals = [combined_vertices[v0][2], combined_vertices[v1][2],
                  combined_vertices[v1_m][2], combined_vertices[v0][2],
                  combined_vertices[v1_m][2], combined_vertices[v0_m][2]]
        
        z_min, z_max = min(z_vals), max(z_vals)
        z_range = z_max - z_min if z_max != z_min else 1.0
        
        pts = [combined_vertices[v0], combined_vertices[v1], combined_vertices[v1_m],
               combined_vertices[v0], combined_vertices[v1_m], combined_vertices[v0_m]]
        
        cols = []
        for pt in pts:
            z_norm = (pt[2] - z_min) / z_range
            shade = (1 - z_norm) * avg_color * 0.6 + z_norm * avg_color  
            cols.append(np.clip(shade, 0, 1))
        
        tri1 = [offset, offset + 1, offset + 2]
        tri2 = [offset + 3, offset + 4, offset + 5]
        
        side_vertices.extend(pts)
        side_colors.extend(cols)
        side_triangles.append(tri1)
        side_triangles.append(tri2)
        
        offset += 6
        color_index += 1
    
    final_vertices = np.vstack([combined_vertices, np.array(side_vertices)])
    final_colors = np.vstack([combined_colors, np.array(side_colors)])
    final_triangles = np.vstack([valid_triangles, mirror_triangles, np.array(side_triangles)])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(final_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(final_triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(final_colors)
    mesh.compute_vertex_normals()
    
    mesh = trimesh.load('sealed_surface_mesh_colored.ply')

    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        raise ValueError("The mesh does not contain vertex colors.")
    if mesh.visual.vertex_colors.max() <= 1.0:
        mesh.visual.vertex_colors = (mesh.visual.vertex_colors * 255).astype(np.uint8)
    
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.fix_normals()
    mesh.rezero()
    
    mesh.export('final_mesh.glb')

#Example Usage
three("<path_to_an_image>")