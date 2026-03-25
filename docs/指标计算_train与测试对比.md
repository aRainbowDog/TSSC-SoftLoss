# 指标计算：`train.py` 验证 与 `metric.py` / `test.py` + `utils.py` 对比

本文记录两套代码在 **MAE / MSE / PSNR**（及测试端 **SSIM**）上的实现差异，便于写论文或统一口径时查阅。

> **说明**：`metric.py` 与 `test.py` 在「数据预处理 + 调用 `calculate_metrics`」部分逻辑一致；差异主要存在于 **`train.py` 验证循环** 与 **`utils.calculate_metrics`** 之间。

---

## 1. 代码位置

| 角色 | 文件与位置 |
|------|------------|
| 训练时验证（完整验证） | `train.py`：`(epoch + 1) % args.val_interval == 0` 时的循环内，约 `gt_np` / `pred_np` 与 `val_mae` 等 |
| 测试/独立评测 | `metric.py` 或 `test.py`：uint8 转换后调用 `calculate_metrics(pred_video, gt_video)` |
| 指标定义 | `utils.py`：`calculate_metrics(generated_video, gt_video)`（约 1341 行起） |

---

## 2. 数值动态范围

### `train.py`（验证）

- 假设张量范围为 **[-1, 1]**。
- 计算前映射到 **[0, 255]**（浮点）：
  - `(video * 0.5 + 0.5) * 255`
- **PSNR** 使用 `data_range=255`。
- **不做** min–max 逐视频拉伸。

### `metric.py` / `test.py` + `utils.calculate_metrics`

- 先将 **[-1, 1]** 映射为 **uint8**，约 **[0, 255]**（含 `clamp`）。
- 进入 `calculate_metrics` 后，对 **`generated_video` 与 `gt_video` 分别** 做整段视频的 min–max：
  - `(x - x.min()) / (x.max() - x.min())`
- 实际参与 **MSE / MAE / PSNR / SSIM** 的数值约在 **[0, 1]**。
- **PSNR / SSIM** 使用 `data_range=1`（与归一化后一致）。  
  - 注意：`utils.py` 中注释写「data_range=255」与代码不符，以 **`data_range=1`** 为准。

**结论**：训练验证指标在 **固定线性标度 0–255** 上；测试脚本在 **各自 min–max 到 [0,1]** 上，二者 **数值不可直接横向对比**，除非改其中一套实现。

---

## 3. 聚合方式：2D 逐帧 vs 高维一把平均

### `train.py`

- `gt_np` / `pred_np` 形状为 **`(B, F, C, H, W)`**。
- **MAE**：`np.mean(np.abs(pred_np - gt_np))` —— 对 **batch × 时间 × 通道 × 空间** 全部元素取平均。
- **MSE**：`mean_squared_error(reshape(-1), ...)` —— 同上，**全局拉平** 后一个 MSE。
- **PSNR**：对 **整块 5D 数组** 用 skimage 计算（与全局 MSE 一致）。
- 每个 batch 的贡献再 **`/ len(loader_val)`** 累加。

语义：**每个验证 batch 一个「全局像素级」标量，再对 batch 平均**（非逐帧 2D 再平均）。

### `utils.calculate_metrics`

- 输入为 **`[F, H, W, C]`**（单样本 clip）。
- 对 **每一帧** ` [H, W, C]` 计算 MSE、MAE、PSNR、SSIM（**2D 多通道**；SSIM 使用 `channel_axis=-1`）。
- 再在 **时间维 F** 上取平均：`total / num_frames`。

语义：**逐帧 2D 指标，再对帧平均**。

### `metric.py` / `test.py` 外层

- 每个 loader batch 通常 **只取第 0 个样本**（`x_test[0]`）算一套 `avg_*`，再对 **测试样本数** 做 `total_* / sample_count`。

---

## 4. 是否包含首尾帧、通道与模型权重

| 项目 | `train.py` 验证 | `metric.py` / `test.py` |
|------|-----------------|-------------------------|
| 推理输入 | 完整 `video_val`，解码后首尾来自 GT latent | 仅首尾保留真实像素，中间为生成（与训练任务一致） |
| 指标比较对象 | `video_val` vs `decoded`（**首尾在 decoded 中与 GT 一致**） | `x_test` vs `decoded_x`；通道上曾把 R/B 置为与 G 相同（与 `metric.py` 中 157–160 行一致） |
| 模型 | **`get_raw_model(model)`**（非 EMA） | 默认 **`checkpoint["ema"]`** |

---

## 5. 测试端独有的指标

- **`SSIM`**：仅在 `calculate_metrics`（测试/metric 路径）中计算；`train.py` 验证 **没有** SSIM。

---

## 6. 若需统一口径的建议（可选）

1. **动态范围**：统一采用「**[-1,1]→[0,255] 线性映射、不做 min–max**」或统一「**min–max 到 [0,1] + data_range=1**」，两套脚本一致即可。
2. **聚合**：统一为「**逐帧 2D 再对 F 平均**」或「**整段 5D 全局 mean**」，并与论文描述一致。



---

