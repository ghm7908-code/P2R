# Point2Roof 可视化功能使用说明

## 添加的可视化功能

本项目已添加以下可视化功能：

### 1. OBJ 格式线框模型导出
- 真值模型：`_gt.obj`
- 预测模型：`_pred.obj`

### 2. 3D 对比可视化图
- 单视角对比图：`_comparison.png`
- 多视角对比图：`_multi_view.png`

### 3. PLY 格式带颜色模型
- 可直接在 MeshLab 等软件中打开

---

## 使用方法

### 基础测试
```bash
python test.py --data_path <数据集路径> --test_tag full_run_v2
```

### 可视化单个样本
```bash
python test.py \
    --data_path <数据集路径> \
    --test_tag full_run_v2 \
    --visualize \
    --vis_sample_id sample_001
```

### 可视化所有样本
```bash
python test.py \
    --data_path <数据集路径> \
    --test_tag full_run_v2 \
    --vis_all
```

---

## 输出目录结构

```
output/full_run_v2/test/
├── log.txt                    # 测试日志
├── visualization/            # 可视化结果目录
│   ├── sample_001_gt.obj     # 真值线框模型
│   ├── sample_001_pred.obj   # 预测线框模型
│   ├── sample_001_comparison.png  # 3D对比图
│   └── sample_001_multi_view.png  # 多视角图
```

---

## OBJ 文件格式

导出的 OBJ 文件符合标准格式：

```
# 顶点
v x1 y1 z1
v x2 y2 z2
...

# 边 (索引从1开始)
l 1 2
l 2 3
...
```

---

## 使用 MeshLab 打开

1. 下载安装 [MeshLab](https://www.meshlab.net/)
2. 打开 `*.obj` 文件即可查看 3D 线框模型

---

## 使用 Python 可视化 (独立脚本)

```python
import numpy as np
from test_util import save_wireframe_obj, visualize_3d_comparison

# 加载数据
vertices = np.loadtxt('vertices.txt')
edges = np.loadtxt('edges.txt', dtype=np.int32)

# 保存 OBJ
save_wireframe_obj(vertices, edges, 'model.obj')

# 生成对比图
visualize_3d_comparison(
    point_cloud=None,
    gt_vertices=vertices,
    gt_edges=edges,
    pred_vertices=vertices,
    pred_edges=edges,
    output_path='comparison.png'
)
```

---

## 依赖安装

确保安装了以下依赖：
```bash
pip install matplotlib trimesh scipy numpy
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--visualize` | flag | False | 启用可视化功能 |
| `--vis_sample_id` | str | sample_001 | 要可视化的样本ID |
| `--vis_all` | flag | False | 可视化所有样本 |

---

## 注意事项

1. 可视化会占用额外内存，请根据需要选择可视化样本数量
2. 对比图中的点云是归一化后的坐标
3. OBJ 文件中的顶点坐标已转换回原始尺度
4. 预测边是基于预测点的全连接组合（实际使用时需根据模型输出调整）
