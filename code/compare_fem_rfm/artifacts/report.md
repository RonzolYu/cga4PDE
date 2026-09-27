# FEM / RFM / CGA 五算例等自由度对比实验报告

## 摘要

本实验在统一的可训练全局系数（DOF）口径下比较连续 Lagrange FEM P1/P2/P3、冻结随机 ReLU³ 特征的 RFM，以及冻结 pool-small 原子轨迹的 CGA。RFM 使用配置中登记的 seeds=201, 202，表中给出成功运行的中位数及四分位区间。FEM、RFM 与 CGA 终点由本项目评价器重算；缺少系数快照的 CGA 中间 dyadic 点保留只读归档评价值并显式标注。

- C1 在五种变体共同可比的最大 DOF=16 时，自然误差最小的是 FEM P3；RFM median/CGA=0.99，FEM P3/CGA=0.97。
- C2 在五种变体共同可比的最大 DOF=16 时，自然误差最小的是 FEM P3；RFM median/CGA=0.87，FEM P3/CGA=0.85。
- C3 在五种变体共同可比的最大 DOF=16 时，自然误差最小的是 CGA；RFM median/CGA=1.49，FEM P3/CGA=1.43。
- C4 在五种变体共同可比的最大 DOF=16 时，自然误差最小的是 RFM median；RFM median/CGA=0.88，FEM P3/CGA=4.20。
- C5 在五种变体共同可比的最大 DOF=16 时，自然误差最小的是 CGA；RFM median/CGA=2.20，FEM P3/CGA=2.67。

## 公平性与实现

- PDE、制造解、载荷、能量和误差定义由同一 `ProblemSpec` 路径提供。精确解只用于载荷构造与最终评价。
- FEM 使用 `[0,1]` 或 `[0,1]^2` 上的拟一致结构化网格；二维网格固定剖分为三角形，连续 P1/P2/P3 共用网格族。纯 p 模型用一个全局零均值约束，因此 DOF=`dim(Vh)-1`。
- RFM 对每个种子一次生成最大特征池，小宽度严格取嵌套前缀；特征按自然度量归一化，纯 p 情形先中心化。
- CGA 不重新选原子。归档未保存中间前缀的系数快照，因此中间 dyadic 点采用带来源标记的归档评价值；冻结终点诊断模型由共同评价器重算。原始历史另存为 `data/cga_archive_history.csv`，不会被覆盖。
- 公共表格只做被实际点包围的 log-log 插值，绝不外推。C4 的 CGA 公共表截止 128，完整轨迹终止于 141。

## 主要科学问题

### 1. 自适应字典选择的收益

- C1: 在 DOF=16，RFM median/CGA 自然误差比为 0.99；该中位数来自登记种子 201, 202，不代表更大规模的随机总体。
- C2: 在 DOF=16，RFM median/CGA 自然误差比为 0.87；该中位数来自登记种子 201, 202，不代表更大规模的随机总体。
- C3: 在 DOF=16，RFM median/CGA 自然误差比为 1.49；该中位数来自登记种子 201, 202，不代表更大规模的随机总体。
- C4: 在 DOF=16，RFM median/CGA 自然误差比为 0.88；该中位数来自登记种子 201, 202，不代表更大规模的随机总体。
- C5: 在 DOF=16，RFM median/CGA 自然误差比为 2.20；该中位数来自登记种子 201, 202，不代表更大规模的随机总体。

### 2. 高阶 FEM 是否追上 CGA？

- C1: FEM P3/CGA 末端误差比 0.97，P3≤CGA 的 crossover 首次出现在 DOF=8。
- C2: FEM P3/CGA 末端误差比 0.85，P3≤CGA 的 crossover 首次出现在 DOF=8。
- C3: FEM P3/CGA 末端误差比 1.43，P3≤CGA 的 crossover 未在合法公共网格发生。
- C4: FEM P3/CGA 末端误差比 4.20，P3≤CGA 的 crossover 首次出现在 DOF=8。
- C5: FEM P3/CGA 末端误差比 2.67，P3≤CGA 的 crossover 未在合法公共网格发生。

### 3. 收敛阶与平台

`artifacts/tables/order_summary.csv` 同时给出最后原始局部阶、最后有效下降阶和预注册窗口拟合阶。理论参考线使用代码中固定的 CGA Hessian--WOGA、FEM 标准 H1 和 RFM Monte-Carlo 斜率，只做竖直平移，没有从数据拟合理论斜率。末段上升或局部阶≤0.15 被标成平台诊断，不被解释为算法本征阶。

### 4. 数值审计与限制

- problem hash 一致：True；除明确标注的 CGA 中间归档点外，终点共同评价器一致：True；C4 冻结终点正确：True。
- RFM 注册种子 201, 202 完整：True；求解器未通过残差门槛的记录数：0。
- 末端评价规则加密的最大相对变化：自然误差 1.176e-01，V-map 1.939e-02，能量差 3.516e+00。能量差接近积分误差底时显著更敏感。
- 二维 RFM 训练使用与最终评价相互独立的确定性复合 Gauss 规则；CGA 中间点沿用带来源标记的归档评价规则。本文比较的是有限 DOF 窗口，不能据此宣称连续字典的无条件渐近定理。

## 文件索引

- `data/*_raw.csv`：新计算原始记录；`data/cga_evaluated.csv`：冻结 CGA 的归档中间点与共同重评终点。
- `data/common_grid.csv`：只用于表格的合法插值记录。
- `artifacts/figures/`：误差、能量、V-map 与局部阶图。
- `artifacts/manifest.json` 与 `artifacts/validation.json`：环境、输入哈希和验收检查。
