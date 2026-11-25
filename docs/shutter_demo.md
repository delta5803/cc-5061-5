# 快门 / 运动模糊演示系统使用说明

本演示用一个**自制的、运动路径完全已知**的合成场景，区分两种常被混为一谈的
物理效应，并提供可编辑参数的图形界面与可复现的命令行工具：

- **运动拖影（motion smear）**——由曝光窗长度 E 决定，每行在曝光窗内对真实
  场景做时间积分；
- **逐行畸变 / 卷帘畸变（rolling skew）**——由读出时长 R 与读出方向决定，
  各行曝光中心在时间轴上错开，几何形状被逐行剪切。

所有样片均来自真实的时间曝光积分（逐时间样本重渲染解析场景再加权平均），
**不使用任何后期模糊滤镜**。仓库内同时保留高斯模糊基线，测试用数值证明它
无法复现曝光积分结果（`test_gaussian_post_filter_is_not_the_integrated_result`）。

## 1. 图形入口（主要使用方式）

启动本地运动分析播放器：

```bash
.venv/bin/python webapp/server.py          # 默认 http://127.0.0.1:8000
.venv/bin/python webapp/server.py 8011     # 也可指定端口
```

浏览器打开 `http://127.0.0.1:8000` 后：

1. 场景选择 **dial**（带 12 根旋转细辐条 + 60 刻度时间标尺的转盘，以及一条
   做“匀速漂移 + 正弦振荡”非匀速运动的前景窄条）；
2. 直接调节：快门角度 / 曝光时长、时间样本数、**读出时长 R**、**读出方向**、
   **曝光扫描方向**（与读出方向相互独立）、**曝光轮廓**（box / trapezoid /
   gaussian）；
3. 预览区实时显示曝光积分图像与采样路径叠加，可逐帧步进、播放、导出单帧或
   mp4/gif 片段；
4. 左侧“样片资源”面板列出已入库资产并做 sha256 哈希核验，点击即载入对应
   转盘场景。

界面的每一次参数修改都会调用后端的 `render_rolling` 真实积分实现
（`webapp/server.py` → `shutter_blur/rolling.py`），**不存在固定/预制数据
冒充路径**；参数非法时由 `ShutterKernel` 校验并以 `ShutterError` JSON
返回，前端在面板内显示错误。

## 2. 命令行（仅作辅助）

```bash
# 生成完整资产包（清单 + 源轨迹 + 高采样参考 + 四种帧）
.venv/bin/python -m scripts.shutter_cli author --name dial_demo

# 仅依据 manifest 参数重渲染，并与保存的高采样参考比对 RMSE
.venv/bin/python -m scripts.shutter_cli rebuild \
    outputs/shutter_demo/assets/dial_demo

# 列出已入库资产
.venv/bin/python -m scripts.shutter_cli list

# 实测：固定场景改变曝光窗 vs 固定曝光重排读出时序
.venv/bin/python -m scripts.shutter_cli measure

# 渲染核验图（四种快门变体、曝光轮廓与时间权重、源轨迹时间标尺）
.venv/bin/python scripts/render_demo_figures.py
```

## 3. 资产盘上布局（真实存在，可核验）

`outputs/shutter_demo/assets/dial_demo/`：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 场景与内核全部参数、每变体的样本时刻与**归一化时间权重**（权重和为 1） |
| `trajectory.npz` | 源轨迹：240 个时间点、12 根辐条世界角、窄条中心 x 与速度 x、E/R 时间标尺 |
| `reference.npz` | 96 样本高采样参考帧（固定感光点观测模型） |
| `frames/clean.png` | 瞬时（零曝光）帧 |
| `frames/global.png` | 全局快门 E=180°, R=0 |
| `frames/rolling.png` | 卷帘 E=180°, R=0.9dt, trapezoid 轮廓 |
| `frames/short.png` | 短曝光 E=45°, R=0.9dt |

核验图位于 `outputs/shutter_demo/figures/`：`dial_variants.png`（拖影与
逐行畸变并排对比）、`exposure_profiles.png`（曝光轮廓与样本权重）、
`source_trajectory.png`（带曝光窗标注的源轨迹时间标尺）。实测数据表写入
`outputs/shutter_demo/exposure_vs_readout.{json,csv}`。

## 4. 重建与验证流程

```bash
.venv/bin/python tests/test_rolling.py     # 卷帘/轮廓/资产生命周期，14 项
.venv/bin/python tests/test_shutter.py     # 原有全局快门回归，13 项
.venv/bin/python scripts/run_benchmark.py  # 旧快门拖影基准，须运行到底（exit 0）
.venv/bin/python -m scripts.shutter_cli author --name dial_demo
.venv/bin/python -m scripts.shutter_cli rebuild \
    outputs/shutter_demo/assets/dial_demo  # 每变体 RMSE 须 < 5e-3
.venv/bin/python scripts/render_demo_figures.py
```

重建只读取 `manifest.json` 中的参数重新生成世界，因此通过即证明来源与参数
足以完整复现样片；当前三个变体的重建 RMSE 约为 0.0004–0.0009。

## 5. 实测结论（数据决定，而非断言）

`measure` 在同一转盘场景上做两组扫描，结果保存在
`exposure_vs_readout.json`，四项判据全部成立：

- 曝光窗 30°→360°（R 固定为 0）：拖影 footprint 215 → 10748 像素，单调增长；
- 读出占比 0→0.95dt（E 固定 180°）：逐行偏移斜率 0 → −1.21 px/行，拖影
  footprint 仅 6717 → 7656（近似不变）；
- 读出方向反转：斜率由 −1.15 翻转为 +1.26（符号反转），拖影 footprint
  基本不变（7586 vs 7566）。

结论：**改变曝光窗用于控制运动拖影；重排读出时序用于控制逐行畸变**。
偏移量通过刻度环第 60 次谐波的逐行傅里叶相位相对全局快门参考帧测得，
为亚像素精度，能穿透对称辐条图案。

## 6. 代码结构

- `shutter_blur/kernel.py`：`ShutterKernel`——E / dt / R 三条独立时间轴、
  读出与曝光方向、三种曝光轮廓及归一化时间权重；
- `shutter_blur/scene.py`：分层解析场景、面积覆盖抗锯齿、Porter-Duff over
  合成、条带热路径 `render_strip`；
- `shutter_blur/demo_scenes.py`：dial 场景（转盘细辐条、刻度标尺、振荡窄条）；
- `shutter_blur/rolling.py`：`render_rolling`（UI 与 CLI 共用的真实积分入口）
  与 `rolling_ground_truth`（同观测模型的高采样参考）；
- `shutter_blur/integrators.py`：naive / center_depth / visibility /
  time_split 四种旧积分器与高斯模糊反例；
- `shutter_blur/assets.py`：资产创建、载入与重建核验；
- `scripts/`：资产 CLI、实测脚本、核验图脚本、旧基准脚本。
