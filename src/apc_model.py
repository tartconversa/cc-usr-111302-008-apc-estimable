"""年龄-时期-队列（APC）加性模型的可估性分析。

模型（显式类别水平编码，无参考组丢弃）::

    y_i = intercept + A[age_i] + P[period_i] + C[cohort_i] + e_i

其中 cohort 必须满足 cohort = period - age。加权最小二乘使用已知正权重
w_i（相对精度），最小化 sum_i w_i (y_i - x_i'beta)^2。

本模块严格区分：

* 由 *原始未约束* 设计矩阵的数值零空间定义的可估性，而不是给数据加约束后
  得到的满秩矩阵；后者只是研究者选定的一组解，不能冒充由数据识别。
* 只在零空间上施加约束（等价重参数化：拟合值、残差、可估对比不变）与
  实际删除可拟合方向的约束（额外模型假设：残差改变）。
* 线性漂移不可估（在零空间内）与二阶差分/曲率可估（与完整零空间正交）。

数值上保留奇异值与每个对比对零空间的正交残差，使结论可复核。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .apc_reader import load


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class APCDataError(ValueError):
    """输入表格不满足 APC 数据契约（cohort 不一致、缺字段、权重非正等）。"""


class ConstraintError(ValueError):
    """识别约束的规则字符串无法解析或引用了不存在的参数。"""


# ---------------------------------------------------------------------------
# 数据载入与校验
# ---------------------------------------------------------------------------


@dataclass
class APCTable:
    """校验后的合成汇总格。"""

    age: np.ndarray
    period: np.ndarray
    cohort: np.ndarray
    y: np.ndarray
    weight: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def age_levels(self) -> list[int]:
        return sorted(set(self.age.tolist()))

    @property
    def period_levels(self) -> list[int]:
        return sorted(set(self.period.tolist()))

    @property
    def cohort_levels(self) -> list[int]:
        return sorted(set(self.cohort.tolist()))


def _as_int(value, field_name: str, row: int) -> int:
    if isinstance(value, bool):
        raise APCDataError(f"第{row}行 {field_name} 不是整数")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise APCDataError(f"第{row}行 {field_name}={value!r} 不是整数年")
    return int(text)


def _as_float(value, field_name: str, row: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise APCDataError(f"第{row}行 {field_name}={value!r} 不是数值") from exc


def read_table(path) -> APCTable:
    """读取并严格校验合成汇总格。

    只接受包含 age/period/cohort/y/weight 五列的记录列表；逐行检查
    cohort == period - age、整数年龄/年份/队列、有限响应与严格正权重、
    重复格等。读取器本身只负责解析，校验集中在这里。
    """
    rows = load(path)
    if isinstance(rows, dict):
        raise APCDataError("汇总格必须是记录列表，而不是单个对象")
    required = {"age", "period", "cohort", "y", "weight"}
    ages: list[int] = []
    periods: list[int] = []
    cohorts: list[int] = []
    ys: list[float] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for index, raw in enumerate(rows):
        row_no = index + 1
        if not isinstance(raw, dict) or not required.issubset(raw):
            missing = required - set(raw) if isinstance(raw, dict) else required
            raise APCDataError(f"第{row_no}行缺少字段: {sorted(missing)}")
        age = _as_int(raw["age"], "age", row_no)
        period = _as_int(raw["period"], "period", row_no)
        cohort = _as_int(raw["cohort"], "cohort", row_no)
        if cohort != period - age:
            raise APCDataError(
                f"第{row_no}行 cohort 不一致: age={age}, period={period}, "
                f"cohort={cohort}，应为 {period - age}"
            )
        y = _as_float(raw["y"], "y", row_no)
        w = _as_float(raw["weight"], "weight", row_no)
        if not math.isfinite(y):
            raise APCDataError(f"第{row_no}行 y 必须有限")
        if not math.isfinite(w) or w <= 0.0:
            raise APCDataError(f"第{row_no}行 weight 必须为已知正数，得到 {w}")
        key = (age, period)
        if key in seen:
            raise APCDataError(f"第{row_no}行 (age={age}, period={period}) 重复")
        seen.add(key)
        ages.append(age)
        periods.append(period)
        cohorts.append(cohort)
        ys.append(y)
        weights.append(w)
    if not rows:
        raise APCDataError("汇总格为空")
    return APCTable(
        age=np.array(ages, dtype=int),
        period=np.array(periods, dtype=int),
        cohort=np.array(cohorts, dtype=int),
        y=np.array(ys, dtype=float),
        weight=np.array(weights, dtype=float),
    )


# ---------------------------------------------------------------------------
# 设计矩阵与零空间
# ---------------------------------------------------------------------------


@dataclass
class APCDesign:
    """显式编码设计矩阵 X 及其数值零空间分析。"""

    table: APCTable
    columns: list[str]
    X: np.ndarray
    Xw: np.ndarray
    sqrt_w: np.ndarray
    singular_values: np.ndarray
    rank: int
    null_basis: np.ndarray  # 列是 Xw beta = 0 的零空间基（也是 X beta = 0）
    tol: float
    age_levels: list[int]
    period_levels: list[int]
    cohort_levels: list[int]
    missing_cells: list[dict] = field(default_factory=list)
    components: list[dict] = field(default_factory=list)
    n_components: int = 1
    canonical_checks: dict = field(default_factory=dict)
    extra_nullity: int = 0
    extra_nullity_reasons: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.table.n

    @property
    def p(self) -> int:
        return self.X.shape[1]

    @property
    def nullity(self) -> int:
        return self.p - self.rank

    @property
    def residual_df(self) -> int:
        return self.n - self.rank

    def column_index(self, name: str) -> int:
        try:
            return self.columns.index(name)
        except ValueError as exc:
            raise ConstraintError(f"参数 {name!r} 不存在；可用参数见 design.columns") from exc


def build_design(table: APCTable, rank_tol: float = 1e-10) -> APCDesign:
    """构建设计矩阵、加权 SVD 零空间并做结构诊断。"""
    age_levels = table.age_levels
    period_levels = table.period_levels
    cohort_levels = table.cohort_levels
    columns = ["intercept"]
    columns += [f"A:{a}" for a in age_levels]
    columns += [f"P:{p}" for p in period_levels]
    columns += [f"C:{c}" for c in cohort_levels]
    n, p = table.n, len(columns)
    X = np.zeros((n, p), dtype=float)
    X[:, 0] = 1.0
    age_idx = {a: i for i, a in enumerate(age_levels)}
    period_idx = {v: i for i, v in enumerate(period_levels)}
    cohort_idx = {c: i for i, c in enumerate(cohort_levels)}
    base_a, base_p, base_c = 1, 1 + len(age_levels), 1 + len(age_levels) + len(period_levels)
    for i in range(n):
        X[i, base_a + age_idx[int(table.age[i])]] = 1.0
        X[i, base_p + period_idx[int(table.period[i])]] = 1.0
        X[i, base_c + cohort_idx[int(table.cohort[i])]] = 1.0
    sqrt_w = np.sqrt(table.weight)
    Xw = X * sqrt_w[:, None]
    # SVD：加权设计的奇异值与零空间。Xw v = 0 等价于 X v = 0（w > 0）。
    _, singular_values, vt = np.linalg.svd(Xw, full_matrices=True)
    threshold = rank_tol * float(singular_values[0])
    rank = int(np.sum(singular_values > threshold))
    null_basis = vt[rank:].T  # p × nullity

    design = APCDesign(
        table=table,
        columns=columns,
        X=X,
        Xw=Xw,
        sqrt_w=sqrt_w,
        singular_values=singular_values,
        rank=rank,
        null_basis=null_basis,
        tol=rank_tol,
        age_levels=age_levels,
        period_levels=period_levels,
        cohort_levels=cohort_levels,
    )
    design.missing_cells = _missing_cells(table, age_levels, period_levels)
    design.components, design.n_components = _connected_components(table)
    design.canonical_checks = _canonical_null_checks(design)
    design.extra_nullity, design.extra_nullity_reasons = _extra_nullity(design)
    return design


def _missing_cells(table: APCTable, age_levels, period_levels) -> list[dict]:
    """报告 age×period 完整笛卡尔网格上的缺格。"""
    observed = set(zip(table.age.tolist(), table.period.tolist()))
    missing = [
        {"age": a, "period": p, "expected_cohort": p - a}
        for p in period_levels
        for a in age_levels
        if (a, p) not in observed
    ]
    return missing


def _connected_components(table: APCTable) -> tuple[list[dict], int]:
    """以 (年龄水平, 时期水平, 队列水平) 为顶点、每条观测为超边的连通分量。

    断连（例如两块数据不共享任何年龄/时期/队列水平）会产生超出经典
    单一线性漂移的额外秩亏。返回并查集得到的分量及成员。
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, p, c in zip(table.age, table.period, table.cohort):
        va, vp, vc = f"A:{int(a)}", f"P:{int(p)}", f"C:{int(c)}"
        find(va)
        find(vp)
        find(vc)
        union(va, vp)
        union(vp, vc)

    groups: dict[str, dict] = {}
    for vertex in list(parent):
        root = find(vertex)
        groups.setdefault(root, {"factor_vertices": []})["factor_vertices"].append(vertex)
    components = []
    for component in groups.values():
        vertices = sorted(component["factor_vertices"])
        component["factor_vertices"] = vertices
        component["n_vertices"] = len(vertices)
        components.append(component)
    components.sort(key=lambda d: d["factor_vertices"][0])
    return components, len(components)


def _canonical_null_checks(design: APCDesign) -> dict:
    """核验两类经典零方向：常数重定基与线性漂移。

    显式编码下每行有 3 个激活的类别指示（A/P/C 各一），故常数重定基
    方向为 intercept=-3、所有类别系数 +1（每行 -3+1+1+1=0）。
    漂移向量 v(t)：A 加 t*age、P 减 t*period、C 加 t*cohort，截距 0。
    由 age - period + cohort = 0，对每条观测 X v = 0，故所有拟合值不变。
    两者是否落在 *数值* 零空间内也一并核验（断连/单格水平时结构会退化）。
    """
    p = design.p
    constant = np.zeros(p)
    constant[0] = -3.0  # 每行三个类别项：-3 + 1 + 1 + 1 = 0
    constant[1:] = 1.0

    drift = np.zeros(p)
    drift[1 : 1 + len(design.age_levels)] = np.array(design.age_levels, dtype=float)
    base_p = 1 + len(design.age_levels)
    base_c = base_p + len(design.period_levels)
    drift[base_p:base_c] = -np.array(design.period_levels, dtype=float)
    drift[base_c:] = np.array(design.cohort_levels, dtype=float)

    def report(vec: np.ndarray) -> dict:
        residual = design.X @ vec
        max_abs = float(np.max(np.abs(residual))) if len(residual) else 0.0
        Z = design.null_basis
        projection = Z @ (Z.T @ vec) if Z.shape[1] else np.zeros_like(vec)
        projection_gap = float(np.linalg.norm(vec - projection))
        return {
            "max_abs_Xv": max_abs,
            "in_null_space": bool(max_abs <= design.tol * np.linalg.norm(vec, ord=np.inf) + 1e-12),
            "distance_to_numerical_nullspace": projection_gap,
        }

    return {
        "constant_releveling": report(constant),
        "linear_drift": report(drift),
        "drift_orthogonal_to_constant": float(abs(constant @ drift)),
    }


def _extra_nullity(design: APCDesign) -> tuple[int, list[str]]:
    """区分 APC 结构性零空间与额外退化。

    连通且三因子均至少两个水平时，显式“截距+全部指示”编码的标准零空间
    维数为 4：常数重定基、线性漂移，以及两个对角边界方向（最早/最晚队列
    各只与一个年龄-时期组合相连）。参考组编码因预先丢弃三列只呈现 2 维，
    但那是编码选择；这里不丢列，数值零空间就是 4 维。超出该结构的维数
    才是需要警告的额外退化（断连、单水平因子等）。
    """
    reasons: list[str] = []
    single = []
    if len(design.age_levels) < 2:
        single.append("age")
    if len(design.period_levels) < 2:
        single.append("period")
    if len(design.cohort_levels) < 2:
        single.append("cohort")
    if single:
        reasons.append(
            "存在只有单一水平的因子: " + ", ".join(single) +
            "；该因子与截距/其他因子共线，标准 4 维零空间结构不完整"
        )
    # 单格水平是边界零方向的结构性来源（信息性诊断，不单独计入额外维数）。
    table = design.table
    singleton_levels: dict[str, list[int]] = {}
    for factor, values in (
        ("A", table.age),
        ("P", table.period),
        ("C", table.cohort),
    ):
        unique, counts = np.unique(values, return_counts=True)
        singletons = [int(v) for v, c in zip(unique, counts) if c == 1]
        if singletons:
            singleton_levels[factor] = singletons
    if singleton_levels:
        rendered = "; ".join(
            f"因子 {factor} 的单格水平 {levels}"
            for factor, levels in singleton_levels.items()
        )
        reasons.append(
            "存在只在单个观测格出现的水平（" + rendered +
            "）：其系数可被同行其他项完全吸收，对应边界零方向；"
            "只检查经典线性漂移会漏掉这些退化"
        )
    if design.n_components > 1:
        reasons.append(
            f"超图断连为 {design.n_components} 个互不共享水平的分量："
            "额外分量各自带常数/漂移/边界零方向，产生远超单一漂移的秩亏"
        )
    if design.missing_cells:
        reasons.append(
            f"age×period 网格存在 {len(design.missing_cells)} 个缺格；"
            "缺格本身不必然提高零空间维数，但会削弱连通性"
        )

    connected_non_degenerate = design.n_components == 1 and not single
    if connected_non_degenerate:
        baseline = 4
        extra = design.nullity - baseline
        if extra > 0 and not reasons:
            reasons.append("数值零空间维数超过连通设计的标准 4 维，存在未被经典结构涵盖的退化")
    else:
        # 退化/断连情形不存在统一简单基准：直接以数值零空间为准，不做减法冒充。
        baseline = None
        extra = design.nullity
        reasons.append("结构退化或断连：不套用标准基准，可估性一律以完整数值零空间正交检验为准")
    return max(0, extra), reasons


# ---------------------------------------------------------------------------
# 加权最小二乘（未约束，最小范数解）
# ---------------------------------------------------------------------------


@dataclass
class WLSFit:
    beta: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    sse: float
    sigma2: float | None
    residual_df: int


def wls_min_norm(design: APCDesign, y: np.ndarray) -> WLSFit:
    """未约束 WLS 的最小范数解。

    用加权设计的 SVD 伪逆求解，自动跳过零奇异值；该解代表同一等价类
    （相同拟合值）中范数最小的一组系数，不代表约束识别了参数。
    """
    Xw, sqrt_w = design.Xw, design.sqrt_w
    u, s, vt = np.linalg.svd(Xw, full_matrices=False)
    rank = design.rank
    y_w = sqrt_w * y
    beta = (vt[:rank].T * (1.0 / s[:rank])) @ (u[:, :rank].T @ y_w)
    fitted = design.X @ beta
    residual = y - fitted
    sse = float(design.table.weight @ residual**2)
    df = design.residual_df
    sigma2 = sse / df if df > 0 else None
    return WLSFit(beta, fitted, residual, sse, sigma2, df)


# ---------------------------------------------------------------------------
# 对比（线性组合）可估性
# ---------------------------------------------------------------------------


@dataclass
class ContrastResult:
    key: str
    l: np.ndarray
    estimable: bool
    null_projection_norm: float
    orthogonality_residual: float
    estimate: float | None = None
    std_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reason: str = ""


def build_contrast_vector(design: APCDesign, coefficients: dict[str, float]) -> np.ndarray:
    """把 {参数键: 系数} 展开到完整设计列序；未知键直接报错。"""
    vec = np.zeros(design.p)
    for name, value in coefficients.items():
        vec[design.column_index(name)] += float(value)
    if np.count_nonzero(vec) == 0:
        raise ConstraintError("对比系数全为 0，无法定义线性组合")
    return vec


def evaluate_contrast(
    design: APCDesign,
    fit: WLSFit,
    key: str,
    coefficients: dict[str, float],
    alpha: float,
) -> ContrastResult:
    """以 *原始未约束* 零空间判定可估性，并在可估时给点估计与区间。

    可估 ⇔ l 与零空间每个基向量正交（l'Z = 0）。只检查经典一条线性漂移
    会漏掉断连/单水平造成的额外退化，因此这里投影到完整数值零空间。
    """
    l = build_contrast_vector(design, coefficients)
    Z = design.null_basis
    projection = Z @ (Z.T @ l)
    proj_norm = float(np.linalg.norm(projection))
    ortho_resid = float(np.max(np.abs(Z.T @ l))) if Z.shape[1] else 0.0
    result = ContrastResult(
        key=key,
        l=l,
        estimable=bool(proj_norm <= design.tol * max(1.0, np.linalg.norm(l))),
        null_projection_norm=proj_norm,
        orthogonality_residual=ortho_resid,
    )
    if not result.estimable:
        result.reason = (
            "l 在零空间上的投影非零：该线性组合含不可识别的常数/漂移"
            "（或断连产生的额外退化）方向，数据无法确定其值"
        )
        return result
    result.estimate = float(l @ fit.beta)
    if fit.sigma2 is None:
        result.reason = "可估，但残差自由度为 0，无法估计误差方差，按要求不输出区间"
        return result
    # Var(l'beta) = sigma2 * l'(X'WX)^+ l；可估对比与伪逆选择无关。
    # 经济 SVD: Xw = U S Vt，(Xw'Xw)^+ = V_r S_r^{-2} V_r'。
    Xw = design.Xw
    _, s, vt = np.linalg.svd(Xw, full_matrices=False)
    rank = design.rank
    v_r = vt[:rank].T
    q = v_r @ ((v_r.T @ l) / (s[:rank] ** 2))
    var = fit.sigma2 * float(l @ q)
    if var < 0.0:
        var = 0.0
    se = math.sqrt(var)
    result.std_error = se
    t_crit = t_quantile(1.0 - alpha / 2.0, fit.residual_df)
    result.ci_low = result.estimate - t_crit * se
    result.ci_high = result.estimate + t_crit * se
    result.reason = (
        f"可估；区间基于独立同方差加权误差假设，t 分位数 df={fit.residual_df}"
    )
    return result


# ---------------------------------------------------------------------------
# 识别约束
# ---------------------------------------------------------------------------


_RULE_SUM = re.compile(
    r"^sum\(\s*(A|P|C)\s*\)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_RULE_WEIGHTED_SUM = re.compile(
    r"^sum\(\s*\(\s*(age|period|cohort)\s*-\s*([+-]?\d+)\s*\)\s*\*\s*(A|P|C)\s*\)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
_RULE_COLUMN = re.compile(
    r"^(intercept|(?:A|P|C):-?\d+)\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)

_FACTOR_AXIS = {"A": "age", "P": "period", "C": "cohort"}
_FACTOR_PREFIX = {"age": "A", "period": "P", "cohort": "C"}


def parse_rule(design: APCDesign, rule: str) -> tuple[np.ndarray, float]:
    """把单条约束规则解析为 (r, rhs)，表示 r' beta = rhs。

    支持：
      sum(A)=0 / sum(P)=... / sum(C)=...
      sum((period-2021)*P)=0 / sum((cohort-2000)*C)=0 / sum((age-..)*A)=..
      intercept=0、A:20=0 这样的单列约束
    不做任何 eval：配置文本是外来数据，按白名单语法严格解析。
    """
    text = rule.strip()
    r = np.zeros(design.p)
    match = _RULE_SUM.match(text)
    if match:
        factor, rhs = match.groups()
        levels = _factor_levels(design, factor)
        for level in levels:
            r[design.column_index(f"{factor}:{level}")] = 1.0
        return r, float(rhs)
    match = _RULE_WEIGHTED_SUM.match(text)
    if match:
        axis, center, factor, rhs = match.groups()
        if _FACTOR_AXIS[factor] != axis:
            raise ConstraintError(
                f"规则 {rule!r} 中因子 {factor} 与轴 {axis} 不匹配"
            )
        for level in _factor_levels(design, factor):
            r[design.column_index(f"{factor}:{level}")] = int(level) - int(center)
        return r, float(rhs)
    match = _RULE_COLUMN.match(text)
    if match:
        name, rhs = match.groups()
        r[design.column_index(name)] = 1.0
        return r, float(rhs)
    raise ConstraintError(f"无法解析约束规则: {rule!r}")


def _factor_levels(design: APCDesign, factor: str) -> list[int]:
    if factor == "A":
        return design.age_levels
    if factor == "P":
        return design.period_levels
    return design.cohort_levels


def parse_constraint_set(design: APCDesign, rules: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not rules:
        raise ConstraintError("约束集为空：至少需要一条识别规则")
    rows = [parse_rule(design, rule) for rule in rules]
    R = np.vstack([r for r, _ in rows])
    rhs = np.array([value for _, value in rows])
    return R, rhs


@dataclass
class ConstraintSolution:
    key: str
    rules: list[str]
    status: str  # equivalent_reparameterization | extra_assumption | contradictory | underidentified
    beta: np.ndarray | None
    fitted: np.ndarray | None
    residual: np.ndarray | None
    sse: float | None
    sse_delta: float | None
    constraint_residual: float
    rn_rank: int
    rn_singular_values: list[float]
    spans_null_space: bool
    note: str


def solve_constrained(
    design: APCDesign,
    y: np.ndarray,
    baseline: WLSFit,
    key: str,
    rules: list[str],
    tol: float,
) -> ConstraintSolution:
    """在约束 R beta = rhs 下求 WLS 解，并判定约束的性质。

    判定逻辑（全部基于原始未约束设计矩阵）：

    * R Z 行秩 < nullity：约束没有张成零空间，解不唯一（欠识别）。
    * 约束残差（rhs 在可满足子空间上的投影差）大：约束矛盾、无解。
    * 有解且 SSE 与未约束解相同：约束只作用于零空间，是等价重参数化，
      拟合值与一切可估对比不变。
    * 有解但 SSE 变大：约束删除了数据支持的可拟合方向，属于额外模型假设，
      不能称为等价重参数化，更不能把它选出的队列趋势解释为因果。
    """
    R, rhs = parse_constraint_set(design, rules)
    Z = design.null_basis
    RN = R @ Z
    rn_sv = np.linalg.svd(RN, compute_uv=False)
    rn_rank = int(np.sum(rn_sv > tol * (rn_sv[0] if len(rn_sv) else 1.0)))
    spans = rn_rank == design.nullity

    # KKT 法方程（G 半正定）：
    # [X'WX  R'] [beta] = [X'Wy]
    # [R     0 ] [lam ]   [ rhs  ]
    XtWX = design.Xw.T @ design.Xw
    XtWy = design.Xw.T @ (design.sqrt_w * y)
    k = R.shape[0]
    kkt = np.block([
        [XtWX, R.T],
        [R, np.zeros((k, k))],
    ])
    target = np.concatenate([XtWy, rhs])
    sol, _, kkt_rank, _ = np.linalg.lstsq(kkt, target, rcond=None)
    beta = sol[: design.p]
    constraint_residual = float(np.max(np.abs(R @ beta - rhs))) if k else 0.0
    # 可行性容差同时按 rhs 尺度与“R*beta”自身尺度归一，避免年份大数值（如 2021）
    # 或大系数导致绝对残差看起来很大却只是浮点误差。
    scale = max(
        1.0,
        float(np.max(np.abs(rhs))) if k else 1.0,
        float(np.max(np.abs(R @ beta))) if k else 1.0,
    )
    feasible = constraint_residual <= max(tol, 1e-12) * scale * 1e3

    singular = [float(v) for v in rn_sv]
    if not feasible:
        return ConstraintSolution(
            key=key, rules=rules, status="contradictory", beta=None, fitted=None,
            residual=None, sse=None, sse_delta=None,
            constraint_residual=constraint_residual, rn_rank=rn_rank,
            rn_singular_values=singular, spans_null_space=spans,
            note="约束 R beta = rhs 无解（相互矛盾或与观测结构冲突），不存在满足全部规则的系数",
        )

    fitted = design.X @ beta
    residual = y - fitted
    sse = float(design.table.weight @ residual**2)
    sse_delta = sse - baseline.sse
    # 同时按 SSE 尺度与 y 的加权平方和归一，精确加性数据下 SSE≈0 也不会误判。
    y_energy = float(design.table.weight @ y**2)
    fit_scale = max(1.0, abs(baseline.sse), y_energy)
    same_fit = abs(sse_delta) <= max(tol, 1e-12) * fit_scale * 1e3

    if not spans:
        status = "underidentified"
        note = (
            f"约束矩阵在零空间上的秩为 {rn_rank}，小于零空间维数 {design.nullity}；"
            "仍有无穷多组系数满足约束，参数未被唯一识别"
            + ("，且当前解与未约束拟合一致" if same_fit else "，且约束改变了拟合")
        )
    elif same_fit:
        status = "equivalent_reparameterization"
        note = (
            "约束恰好张成零空间且不改变 SSE：只是同一等价类内的重参数化，"
            "拟合值、残差与所有可估对比与未约束解完全相同；"
            "被约束固定的线性趋势本身仍不可估"
        )
    else:
        status = "extra_assumption"
        note = (
            f"约束改变了拟合（SSE 增加 {sse_delta:.6g}）：它删除了数据支持的可拟合方向，"
            "属于额外模型假设而非等价重参数化；由此产生的队列/时期趋势不得解释为因果影响"
        )
    return ConstraintSolution(
        key=key, rules=rules, status=status, beta=beta, fitted=fitted,
        residual=residual, sse=sse, sse_delta=sse_delta,
        constraint_residual=constraint_residual, rn_rank=rn_rank,
        rn_singular_values=singular, spans_null_space=spans, note=note,
    )


# ---------------------------------------------------------------------------
# 曲线二阶差分（曲率，可估）
# ---------------------------------------------------------------------------


def second_differences(levels: list[int], values: np.ndarray) -> list[dict]:
    """等间隔水平的二阶差分；不齐整网格改用二阶均差（除以间距乘积）。

    线性漂移给曲线加一条直线，任何二阶（均）差都对直线为 0，因此曲率
    不受漂移影响；但可估性仍以完整零空间正交校验为准，不只依赖此性质。
    """
    out = []
    for i in range(1, len(levels) - 1):
        x0, x1, x2 = float(levels[i - 1]), float(levels[i]), float(levels[i + 1])
        v0, v1, v2 = float(values[i - 1]), float(values[i]), float(values[i + 1])
        h0, h1 = x1 - x0, x2 - x1
        raw = v2 - 2.0 * v1 + v0
        # 二阶均差 f[x0,x1,x2]：对任何线性函数恒为 0（含不等间距），
        # 等间距时退化为 raw / h^2。
        divided = (
            (v2 - v1) / h1 - (v1 - v0) / h0
        ) / (0.5 * (h0 + h1))
        out.append({
            "level": levels[i],
            "left_spacing": h0,
            "right_spacing": h1,
            "equally_spaced": h0 == h1,
            "raw_second_difference": raw,
            "second_divided_difference": divided,
        })
    return out


def curvature_orthogonal_to_nullspace(
    design: APCDesign, factor: str, levels: list[int]
) -> dict:
    """把一条曲线的全部二阶差分作为对比，逐一验证与完整零空间正交。"""
    prefix = _FACTOR_PREFIX[factor]
    checks = []
    for i in range(1, len(levels) - 1):
        l = np.zeros(design.p)
        x0, x1, x2 = levels[i - 1], levels[i], levels[i + 1]
        h0, h1 = x1 - x0, x2 - x1
        half_sum = 0.5 * (h0 + h1)
        # 与 second_differences 的二阶均差同一组系数
        l[design.column_index(f"{prefix}:{x0}")] = 1.0 / (h0 * half_sum)
        l[design.column_index(f"{prefix}:{x1}")] = -(1.0 / h1 + 1.0 / h0) / half_sum
        l[design.column_index(f"{prefix}:{x2}")] = 1.0 / (h1 * half_sum)
        Z = design.null_basis
        ortho = float(np.max(np.abs(Z.T @ l))) if Z.shape[1] else 0.0
        checks.append({
            "level": x1,
            "max_abs_nullspace_inner_product": ortho,
            "orthogonal": bool(ortho <= design.tol * max(1.0, np.linalg.norm(l))),
        })
    return {"factor": factor, "checks": checks}


# ---------------------------------------------------------------------------
# 纯标准库 t 分布（避免引入 scipy）
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    # Lentz 连分式（Numerical Recipes），用于正则化不完全贝塔函数。
    tiny, epsilon, max_iter = 1e-30, 1e-15, 400
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betainc_regularized(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_bt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_bt) * _betacf(b, a, 1.0 - x) / b


def t_sf_abs(t: float, df: int) -> float:
    """双侧生存概率 P(|T_df| >= t)，t >= 0。"""
    x = df / (df + t * t)
    return _betainc_regularized(df / 2.0, 0.5, x)


def t_quantile(p: float, df: int) -> float:
    """Student-t 的 p 分位数（0 < p < 1），对 df 用二分法求逆。"""
    if df <= 0:
        raise ValueError("t 分位数需要正自由度")
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0,1) 内")
    target = 2.0 * (1.0 - p)  # 双侧尾概率
    lo, hi = 0.0, 1.0
    while t_sf_abs(hi, df) > target:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_sf_abs(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# 端到端分析与报告
# ---------------------------------------------------------------------------


def _factor_curve(beta: np.ndarray, design: APCDesign, factor: str) -> dict:
    prefix = _FACTOR_PREFIX[factor]
    levels = _factor_levels(design, prefix)
    values = np.array([beta[design.column_index(f"{prefix}:{v}")] for v in levels])
    diffs = second_differences(levels, values)
    # 把每个二阶（均）差对完整零空间的可估性直接标注在曲线上。
    # 漂移不变只保证它对经典线性漂移正交；断连/稀疏网格下它仍可能落在
    # 额外零方向上，此时数值随约束变化，必须标为不可估。
    ortho = curvature_orthogonal_to_nullspace(design, factor, levels)
    for diff, check in zip(diffs, ortho["checks"]):
        diff["estimable"] = check["orthogonal"]
        diff["max_abs_nullspace_inner_product"] = check[
            "max_abs_nullspace_inner_product"
        ]
        if not diff["estimable"]:
            diff["estimability_note"] = (
                "该曲率对比与完整数值零空间不正交：漂移不变但仍不可估，"
                "其数值会随识别约束变化"
            )
    return {
        "factor": factor,
        "levels": levels,
        "coefficients": [float(v) for v in values],
        "second_differences": diffs,
    }


def analyze(data_path, config_path, contrasts_path) -> dict:
    """跑完整 APC 可估性分析，返回可 JSON 序列化的报告字典。"""
    config = load(config_path)
    contrast_doc = load(contrasts_path)
    rank_tol = float(config.get("rank_relative_tolerance", 1e-10))
    alpha = float(config.get("interval_alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise ConstraintError("interval_alpha 必须在 (0,1) 内")

    table = read_table(data_path)
    design = build_design(table, rank_tol=rank_tol)
    fit = wls_min_norm(design, table.y)

    # 若 SSE 相对 y 的能量小到机器精度量级，残差只是浮点舍入而非抽样误差：
    # df 虽然为正，形式上可算区间，但 sigma^2 没有统计含义，必须明确警告，
    # 不能让 1e-13 宽度的区间被读成“估计极精确”。
    y_energy = float(table.weight @ table.y**2)
    residual_is_machine_noise = (
        fit.residual_df > 0 and fit.sse <= 1e-20 * max(1.0, y_energy)
    )

    contrasts = [
        evaluate_contrast(design, fit, q["key"], q["coefficients"], alpha)
        for q in contrast_doc["queries"]
    ]
    constraint_solutions = [
        solve_constrained(design, table.y, fit, cs["key"], cs["rules"], rank_tol)
        for cs in config["constraint_sets"]
    ]

    # 跨约束核验：等价类相同 ⇔ 拟合值与可估对比一致。
    cross_check = _cross_constraint_check(design, fit, contrasts, constraint_solutions)

    curves = [_factor_curve(fit.beta, design, f) for f in ("age", "period", "cohort")]
    curvature_checks = [
        curvature_orthogonal_to_nullspace(design, f, _factor_levels(design, _FACTOR_PREFIX[f]))
        for f in ("age", "period", "cohort")
    ]
    report = {
        "data": {
            "n_cells": table.n,
            "age_levels": design.age_levels,
            "period_levels": design.period_levels,
            "cohort_levels": design.cohort_levels,
            "missing_cells": design.missing_cells,
            "n_missing_cells": len(design.missing_cells),
            "connected_components": design.components,
            "n_components": design.n_components,
        },
        "design": {
            "parameter_coding": config.get("parameter_coding"),
            "columns": design.columns,
            "n_parameters": design.p,
            "rank": design.rank,
            "nullity": design.nullity,
            "residual_df": design.residual_df,
            "rank_relative_tolerance": rank_tol,
            "singular_values": [float(v) for v in design.singular_values],
            "canonical_null_directions": design.canonical_checks,
            "extra_nullity": design.extra_nullity,
            "extra_nullity_reasons": design.extra_nullity_reasons,
        },
        "unconstrained_wls": {
            "method": "SVD 最小范数解（代表同一拟合等价类，不构成识别）",
            "sse": fit.sse,
            "sigma2": fit.sigma2,
            "residual_df": fit.residual_df,
            "intervals_emitted": fit.residual_df > 0 and not residual_is_machine_noise,
            "residual_is_machine_precision_noise": residual_is_machine_noise,
            "interval_note": (
                "残差自由度不足，按要求不输出任何基于误差方差的区间"
                if fit.residual_df <= 0
                else (
                    "残差相对响应能量处于机器精度量级，sigma^2 只反映浮点舍入而非"
                    "抽样误差；下列区间在数值上可算但无统计精度含义，不可解读为高置信"
                    if residual_is_machine_noise
                    else f"独立同方差加权误差假设下估计 sigma^2，alpha={alpha}"
                )
            ),
            "fitted": _fitted_rows(table, fit),
            "curves": curves,
            "curvature_nullspace_checks": curvature_checks,
        },
        "contrasts": [
            _contrast_payload(c, alpha, residual_is_machine_noise) for c in contrasts
        ],
        "constraint_sets": [
            _constraint_payload(sol, design) for sol in constraint_solutions
        ],
        "cross_constraint_check": cross_check,
        "interpretation_guardrails": [
            "可估性由原始未约束设计矩阵的完整数值零空间定义，加约束后的满秩矩阵不改变这一结论",
            "线性漂移（A+t*age, P-t*period, C+t*cohort）保持所有拟合值不变，因此线性趋势本身不可估",
            "等间隔曲线的二阶差分对该漂移不变，但仍已逐一核对其与完整零空间正交",
            "约束选出的队列/时期趋势只是识别假设下的一组解，不得解释为代际或时期的因果影响",
            "区间仅在独立同方差加权误差假设下成立；残差自由度为 0 时不输出假精度",
        ],
    }
    return report


def _fitted_rows(table: APCTable, fit: WLSFit) -> list[dict]:
    return [
        {
            "age": int(table.age[i]),
            "period": int(table.period[i]),
            "cohort": int(table.cohort[i]),
            "y": float(table.y[i]),
            "weight": float(table.weight[i]),
            "fitted": float(fit.fitted[i]),
            "residual": float(fit.residual[i]),
        }
        for i in range(table.n)
    ]


def _contrast_payload(result: ContrastResult, alpha: float, machine_noise: bool) -> dict:
    payload = {
        "key": result.key,
        "estimable": result.estimable,
        "null_projection_norm": result.null_projection_norm,
        "max_abs_nullspace_inner_product": result.orthogonality_residual,
        "estimate": result.estimate,
        "std_error": result.std_error,
        "ci_level": 1.0 - alpha,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "interval_has_statistical_meaning": result.ci_low is not None and not machine_noise,
        "reason": result.reason,
    }
    if machine_noise and result.ci_low is not None:
        payload["interval_warning"] = (
            "残差为机器精度噪声，此区间仅形式存在，不代表可实现的统计精度"
        )
    if not result.estimable:
        payload["note"] = "不可估线性趋势：不同识别约束会给出不同数值，数据本身无法确定"
    return payload


def _constraint_payload(sol: ConstraintSolution, design: APCDesign) -> dict:
    payload = {
        "key": sol.key,
        "rules": sol.rules,
        "status": sol.status,
        "spans_null_space": sol.spans_null_space,
        "R_nullspace_rank": sol.rn_rank,
        "R_nullspace_singular_values": sol.rn_singular_values,
        "constraint_residual": sol.constraint_residual,
        "sse": sol.sse,
        "sse_delta_vs_unconstrained": sol.sse_delta,
        "note": sol.note,
    }
    if sol.beta is not None:
        payload["coefficients"] = {
            name: float(value) for name, value in zip(design.columns, sol.beta)
        }
        payload["curves"] = [
            _factor_curve(sol.beta, design, f) for f in ("age", "period", "cohort")
        ]
    return payload


def _cross_constraint_check(
    design: APCDesign,
    baseline: WLSFit,
    contrasts: list[ContrastResult],
    solutions: list[ConstraintSolution],
) -> dict:
    """核验同一等价类的约束给出相同预测与相同可估对比。"""
    tol = design.tol
    y_scale = max(1.0, float(np.max(np.abs(baseline.fitted))))
    fit_tol = max(tol, 1e-12) * y_scale * 1e3
    fit_diffs = []
    for sol in solutions:
        if sol.fitted is None:
            fit_diffs.append({"key": sol.key, "max_abs_fitted_diff": None,
                              "same_equivalence_class": False})
            continue
        diff = float(np.max(np.abs(sol.fitted - baseline.fitted)))
        fit_diffs.append({
            "key": sol.key,
            "max_abs_fitted_diff": diff,
            "same_equivalence_class": diff <= fit_tol,
        })
    contrast_diffs = []
    for result in contrasts:
        if not result.estimable:
            continue
        entry = {"key": result.key, "values": {}}
        for sol in solutions:
            if sol.beta is not None:
                entry["values"][sol.key] = float(result.l @ sol.beta)
        values = list(entry["values"].values())
        entry["max_abs_diff_across_constraints"] = (
            float(max(values) - min(values)) if values else None
        )
        contrast_diffs.append(entry)
    return {
        "fitted_value_comparison": fit_diffs,
        "estimable_contrast_comparison": contrast_diffs,
        "summary": (
            "若各约束仅选择同一等价类，则预测与可估对比必须数值相同；"
            "任何 SSE/拟合差异都意味着引入了额外模型假设"
        ),
    }
