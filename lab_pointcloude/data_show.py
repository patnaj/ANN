import open3d as o3d
import numpy as np
import pye57

# alternative app (CloudCompare) to open .e57 files
# https://www.danielgm.net/cc/


def show_pointcloud(points, colors=None, lines=None, line_colors=None):
    if 'o3d' in globals():
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,0:3].cpu().numpy())
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        elif points.size(1) >= 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:,3:6].cpu().numpy())  
        if lines is not None:
            lcd = o3d.geometry.LineSet();
            lcd.points = o3d.utility.Vector3dVector(points[:,0:3].cpu().numpy())   
            lcd.lines = o3d.utility.Vector2iVector(lines[:,0:2].numpy())  
            if line_colors is not None:
                lcd.colors = o3d.utility.Vector3dVector(line_colors[:,:].cpu().numpy())
            o3d.visualization.RenderOption.line_width=12.0
            o3d.visualization.draw_geometries([pcd,lcd])
        else:          
            o3d.visualization.draw_geometries([pcd])


e57 = pye57.E57("<scan file>.e57")

# all the header information can be printed using:
for line in e57.get_header(0):
    print(line)


# read scan at index 0
data = e57.read_scan(0, colors=True)

for i, d in data.items():
    print(i)
    
import torch    
# dx = torch.Tensor
dx = torch.cat([
    torch.from_numpy( data['cartesianX']).unsqueeze(1),
    torch.from_numpy( data['cartesianY']).unsqueeze(1),
    torch.from_numpy( data['cartesianZ']).unsqueeze(1),
    torch.from_numpy( data['colorRed']).unsqueeze(1)/255,
    torch.from_numpy( data['colorGreen']).unsqueeze(1)/255,
    torch.from_numpy( data['colorBlue']).unsqueeze(1)/255 
    ], dim=1)

# po = dx[100000:800000]

po = dx[(torch.rand(100000)*dx.size(0)).long(),:]
show_pointcloud(po[:,:3], po[:,3:])


