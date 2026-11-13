# 年龄时期队列的可估对比与约束敏感性

本项目保存虚构数据共享及产品运营的合成表格与 Python APC（年龄-时期-队列）分析工具。
数据为加性加权最小二乘汇总格：`y = intercept + A_age + P_period + C_cohort`，类别水平
**显式编码**（不预先丢弃参考组列）。工具的核心目标是把数据真正能决定的线性对比与
依赖识别假设才能确定的趋势分开，而不是用一条漂亮的拟合线掩盖不可识别问题。

> 所有地区、机构、产品与人数均为合成。报告不会把任何约束选出的队列趋势解释为因果影响。

## 字段与单位

age 与 period 均整数年，cohort 必须等于 `period - age`；y 是合成汇总均值，weight 为
已知正的相对精度权。contrasts 用显式参数键（如 `A:20`、`P:2021`、`C:2000`、`intercept`）
定义线性组合。

material-index.json 记录全部输入内容哈希；fixtures 保存原始资料与配置，计算输出放在
`.workspace-output`（已被 git 忽略），原始资料不会被覆盖。

## 不可识别问题与本工具的立场

任取 t，把 `A_age += t*age`、`P_period -= t*period`、`C_cohort += t*cohort`，因
`age - period + cohort = 0`，所有拟合值保持不变，但三条曲线的线性趋势都改变。因此
线性趋势本身不可由数据识别。此外显式全指示编码下，连通设计的数值零空间为 4 维
（常数重定基、线性漂移、两个对角边界方向）；断连或单水平因子会进一步扩大零空间。
只检查经典一条线性漂移会漏掉这些额外退化。

工具据此：

- 用加权设计矩阵的 **SVD 数值零空间**（保留全部奇异值）定义可估性：对比 l 只有与
  零空间每个基向量正交（`l'Z = 0`）才标记为可估，绝不用加约束后的满秩矩阵冒充数据识别。
- 对登记的识别约束，用 KKT 求解并区分三类：等价重参数化（只张成零空间、SSE 与拟合
  不变）、额外模型假设（删除了数据支持的可拟合方向、SSE 改变）、矛盾或欠识别。
- 输出各曲线二阶（均）差分（对漂移不变，且逐一核对与完整零空间正交）、可估对比的
  点估计与区间、不可估线性趋势的明确标注，以及拟合格。
- 区间仅在“独立同方差加权误差”假设下给出；残差自由度不足（`n - rank <= 0`）时
  不估计 sigma²，不输出任何区间，避免假精度。

## 安装与运行

Python 3.11 以上；分析功能依赖 numpy（已在 pyproject.toml 显式登记），其余只用标准库。
建议以可编辑方式安装，这样 Python 隔离模式（`-I`，防止 fixtures 目录中的同名模块被
导入）下也能运行：

    pip install -e .
    python -I -m src.apc_cli fixtures/apc.csv fixtures/apc.json fixtures/contrasts.json
    # 或安装后的入口
    apc-estimable fixtures/apc.csv fixtures/apc.json fixtures/contrasts.json

仅读取（不依赖 numpy）：

    python -m src.apc_reader fixtures/apc.csv

输出（默认写入 `.workspace-output/`）：

- `apc_report.json`：完整结构化报告（奇异值、零空间诊断、对比可估性、约束求解与跨约束核验）
- `apc_fitted.csv`：未约束最小范数解的逐格拟合值与残差
- `apc_summary.txt`：人类可读结论，含不可估趋势与非因果解释边界

测试与编译检查：

    python -I -m unittest discover -s tests -v
    python -m compileall -q src tests

测试覆盖：不齐整网格与缺格、断连分量、单队列/单年龄退化、错误 cohort、非正权重、
重复格、矛盾/欠识别/删方向约束、漂移不变性、完整零空间可估性、自由度不足不输出区间、
二阶均差、权重对拟合的作用，以及 t 分位数与标准 t 表的对照。

## 固定容器

    docker build -t apc-estimable:local .
    docker run --rm apc-estimable:local

容器基础镜像以完整 digest 固定，默认运行隔离模式测试；numpy 依赖在镜像构建时按
pyproject.toml 显式安装。
