> Historical audit of the pre-2026-10-01 package. Its counts, paths, and conclusions are not current revision evidence.

# P01：主文基线图表与逐状态记录的闭合核查

## 结论

现有 `paper/sisc_cga/figures/baselines/` 的12张图和 `generated/baseline_terminal.tex`
使用的是313行逐seed记录所对应的数据。其统计口径为：C1–C4各显示宽度均10/10；
C5 N=128为7/10，N=256为4/10。C3 N=256的10/10是真实的当前图表样本数。
旧 `generated/unified_audit_summary.tex` 中C3的7/10、C5的9/10和8/10没有与这组
图表所用逐状态记录建立可核验映射，因此不能解释为本组图表的“另一种指标有效数”。
应从主文撤去该旧表及617状态统计，使用本目录生成的简表与补充说明。

## 可重复核查

从项目根目录运行：

```sh
python3 paper/sisc_cga/tools/rebuild_baseline_evidence.py
```

依赖为Python和`pypdf`，不运行PDE求解。脚本读取：

- `result/experiments/rfm_multiseed_raw.csv`：313个互异案例/宽度/seed组合；304成功、9失败。
- `result/experiments/rfm_multiseed_summary.csv`：32组中位数和四分位数。
- `result/baselines/baseline_actual_points.csv`：现有各曲线的数据点。
- `result/baselines/baseline_terminal.csv`：端点误差、FEM插值值、比值和计数。
- `config/plots.json`：显示宽度、终点和案例映射。

结果写入同目录的`rfm_state_audit.csv`、`rfm_recomputed_summary.csv`、
`baseline_endpoint_verification.csv`、`baseline_figure_verification.csv`和
`baseline_evidence_verification.json`；紧凑表写入`generated/rfm_fixed_denominators.tex`。
`tools/generate_rfm_fixed_denominators.py`现为此完整核查脚本的兼容入口。

## 指标参与规则

逐状态表分别记录energy gap、natural error、V-error是否适用，记录值是否有限且非负，
是否参与该指标的summary，以及是否进入主文图。C1–C3的V-error不适用，其NA不能
作为H1或能量统计的共同排除条件。这里的“有限且非负”仅是字段数值检查，不能等同于
独立残差阈值或独立求积精度认证。失败理由保留原始`solver_success=False`及原始消息；
不根据`CONVERGENCE`字样擅自把失败标志改为成功。

所有32组逐seed中位数、Q1、Q3与summary一致；所有主文RFM数据点与summary一致。
其中C5 N=512仅3个seed，未入主文统计；C4 N=256的10个状态超出显示范围。因而主文
显示的宽度共有300个规定seed尝试，其中291成功、9失败。

## 端点复算

| 案例 | N | RFM样本 | RFM/CGA（重算） | FEM P3/CGA（重算） |
|---|---:|---:|---:|---:|
| C1 | 256 | 10 | 1.4860295935 | 4.2418820347 |
| C2 | 256 | 10 | 1.5248008608 | 4.3525328134 |
| C3 | 256 | 10 | 1.7661939550 | 5.6380673964 |
| C4 | 128 | 10 | 5.7649850506 | 1.8823077561 |
| C5 | 256 | 4 | 5.0060101725 | 16.1155838399 |

这些比值与现有端点表取两位小数后的值一致。各端点中位数和IQR也逐一一致，故没有改动
该表数值。此处核对的是已有FEM插值数值对应的比值，没有重新求解FEM问题。

## 图形来源与旧manifest限制

主文PDF与原`tex/figures/baselines/`同名PDF的SHA相同。旧sidecar记录的输入CSV
SHA为`19cfea…`，当前CSV为`6603e6…`，旧输入哈希已不能证明当前CSV的字节一致性。
因此本次没有仅依靠旧manifest认定图形来源，而是直接读取12张PDF的矢量路径：
以对数坐标的仿射变换核对RFM中位数的全部点和IQR的上下边界，再用同一坐标变换
核对CGA、FEM P1、FEM P3的可见数据点。最大坐标差为约`9.75e-7` PDF点，
低于设置的`1e-4`容差。这证实现有图上数值与当前数据记录一致，无需重绘或重跑实验。
当前输入文件及图形输出SHA已经写入新的核查JSON/CSV，不修改或虚构旧manifest。

## 补充材料与残留限制

本目标使用 `paper/sisc_cga/supplement/S3_failures_and_statistics.md` 定义图表的样本。
根目录旧S3将617状态批次称作主文依据，与当前图表事实不一致，不应作为本目标的S3引用。
617状态的其他独立评价结果可以另存补充材料，但不能以缺乏逐状态映射的汇总替换当前
可复算曲线，也不能把其7/9/8计数解释为当前图表的额外过滤。

本次核查没有审计求解器实现、独立验证solver_success判定规则，或重新验证617状态
批次的每一条记录。它闭合的是当前基线图、表、逐seed指标和分母口径；它没有把
可复算统计提升为新的PDE精度证书。新的补充说明及审计文件仍需随下一次仓库更新发布。
