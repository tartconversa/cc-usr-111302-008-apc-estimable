"""APC 可估性分析的数学性质与边界测试。

覆盖：不齐整网格、单队列、错误 cohort、矛盾/欠识别/删方向约束、
漂移不变性、完整零空间可估性（不只查一条漂移）、残差自由度不足时
不输出区间、二阶差分、权重作用、对比正交残差与奇异值保留。
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.apc_model import (
    APCDataError,
    ConstraintError,
    analyze,
    build_design,
    build_contrast_vector,
    curvature_orthogonal_to_nullspace,
    evaluate_contrast,
    parse_rule,
    read_table,
    second_differences,
    solve_constrained,
    t_quantile,
    wls_min_norm,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CSV = ROOT / "fixtures" / "apc.csv"
FIXTURE_CFG = ROOT / "fixtures" / "apc.json"
FIXTURE_CON = ROOT / "fixtures" / "contrasts.json"


def write_csv(rows, header="age,period,cohort,y,weight\n"):
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    )
    handle.write(header)
    for row in rows:
        handle.write(row + "\n")
    handle.close()
    return handle.name


def grid_rows(ages, periods, y_func=None, weights=None):
    rows = []
    for a in ages:
        for p in periods:
            y = y_func(a, p) if y_func else 1.0
            w = weights(a, p) if weights else 1.0
            rows.append(f"{a},{p},{p - a},{y},{w}")
    return rows


def design_of(rows):
    return build_design(read_table(write_csv(rows)))


def drift_vector(design):
    """任务给定的漂移：A+t*age、P-t*period、C+t*cohort。"""
    p = design.p
    drift = np.zeros(p)
    na = len(design.age_levels)
    bp = 1 + na
    bc = bp + len(design.period_levels)
    drift[1:bp] = np.array(design.age_levels, dtype=float)
    drift[bp:bc] = -np.array(design.period_levels, dtype=float)
    drift[bc:] = np.array(design.cohort_levels, dtype=float)
    return drift


class FixtureStructureTests(unittest.TestCase):
    def setUp(self):
        self.table = read_table(FIXTURE_CSV)
        self.design = build_design(self.table)
        self.fit = wls_min_norm(self.design, self.table.y)

    def test_explicit_coding_shape_and_full_singular_values_retained(self):
        # 1 截距 + 3 年龄 + 3 时期 + 5 队列 = 12 列；9 个观测
        self.assertEqual(self.design.p, 12)
        self.assertEqual(self.design.n, 9)
        self.assertEqual(len(self.design.singular_values), min(9, 12))
        self.assertEqual(len(self.design.singular_values), 9)
        self.assertEqual(self.design.rank + self.design.nullity, self.design.p)

    def test_connected_design_has_structural_nullity_four(self):
        # 连通非退化设计：显式全指示编码零空间为 4 维（常数+漂移+两边界方向）
        self.assertEqual(self.design.n_components, 1)
        self.assertEqual(self.design.nullity, 4)
        self.assertEqual(self.design.extra_nullity, 0)

    def test_canonical_constant_and_drift_are_numerical_null_vectors(self):
        checks = self.design.canonical_checks
        self.assertTrue(checks["constant_releveling"]["in_null_space"])
        self.assertTrue(checks["linear_drift"]["in_null_space"])
        self.assertLess(checks["constant_releveling"]["max_abs_Xv"], 1e-10)
        self.assertLess(checks["linear_drift"]["max_abs_Xv"], 1e-10)

    def test_drift_leaves_all_fitted_values_unchanged(self):
        drift = drift_vector(self.design)
        for t in (-7.0, -0.3, 2.25):
            moved = self.design.X @ (self.fit.beta + t * drift)
            self.assertTrue(np.allclose(moved, self.fit.fitted, atol=1e-10))

    def test_wls_reproduces_data_when_residual_space_saturated(self):
        # fixture 秩 8、9 格，残差自由度 1；数据本身近似精确加性
        self.assertEqual(self.design.residual_df, 1)
        self.assertLess(self.fit.sse, 1e-20)


class EstimabilityTests(unittest.TestCase):
    def setUp(self):
        self.table = read_table(FIXTURE_CSV)
        self.design = build_design(self.table)
        self.fit = wls_min_norm(self.design, self.table.y)

    def test_curvature_contrast_estimable_and_invariant_under_drift(self):
        result = evaluate_contrast(
            self.design, self.fit, "age-curvature",
            {"A:20": 1, "A:21": -2, "A:22": 1}, 0.05,
        )
        self.assertTrue(result.estimable)
        self.assertLess(result.null_projection_norm, 1e-9)
        self.assertAlmostEqual(result.estimate, -0.6, places=8)
        # 显式验证：曲率对比与漂移内积为 0
        l = result.l
        self.assertAlmostEqual(float(l @ drift_vector(self.design)), 0.0, places=10)

    def test_linear_trend_contrast_not_estimable(self):
        result = evaluate_contrast(
            self.design, self.fit, "age-trend",
            {"A:20": -1, "A:22": 1}, 0.05,
        )
        self.assertFalse(result.estimable)
        self.assertGreater(result.null_projection_norm, 0.1)
        self.assertIsNone(result.estimate)
        self.assertIsNone(result.ci_low)
        self.assertGreater(result.orthogonality_residual, 0.0)

    def test_estimability_uses_full_nullspace_not_just_classic_drift(self):
        # 构造一个与经典漂移正交、但落在边界零方向上的对比：
        # 边缘队列 C:1998 只出现在单格，单列对比必然不可估，
        # 即使它与漂移向量内积为 0。
        l = build_contrast_vector(self.design, {"C:1998": 1.0})
        self.assertAlmostEqual(float(l @ drift_vector(self.design)), 1998.0, places=8)
        # 改用与漂移正交的组合（减去漂移投影后仍在零空间内）
        Z = self.design.null_basis
        # 取零空间基中与漂移正交的一个向量作为对比
        drift = drift_vector(self.design)
        drift = drift / np.linalg.norm(drift)
        for j in range(Z.shape[1]):
            z = Z[:, j] - drift * (drift @ Z[:, j])
            if np.linalg.norm(z) > 1e-8:
                coefficients = {
                    name: float(value)
                    for name, value in zip(self.design.columns, z)
                    if abs(value) > 1e-12
                }
                result = evaluate_contrast(
                    self.design, self.fit, "extra-null", coefficients, 0.05
                )
                self.assertFalse(
                    result.estimable,
                    "与经典漂移正交但位于额外零方向的对比必须判为不可估",
                )
                self.assertAlmostEqual(z @ drift_vector(self.design), 0.0, places=8)
                break
        else:  # pragma: no cover - 防御性
            self.fail("零空间中未找到与漂移正交的额外方向")

    def test_interval_emitted_with_residual_df_and_uses_t_distribution(self):
        result = evaluate_contrast(
            self.design, self.fit, "age-curvature",
            {"A:20": 1, "A:21": -2, "A:22": 1}, 0.05,
        )
        self.assertIsNotNone(result.ci_low)
        self.assertLess(result.ci_low, result.estimate)
        self.assertLess(result.estimate, result.ci_high)
        self.assertAlmostEqual(t_quantile(0.975, 1), 12.7062, places=3)
        self.assertAlmostEqual(t_quantile(0.975, 10), 2.2281, places=3)

    def test_no_interval_when_residual_df_zero(self):
        # 饱和设计：n == rank，残差自由度 0，禁止输出假精度
        rows = grid_rows([30, 31], [2020, 2021],
                         y_func=lambda a, p: 1.0 + 0.1 * a + 0.2 * p)
        table = read_table(write_csv(rows))
        design = build_design(table)
        fit = wls_min_norm(design, table.y)
        self.assertEqual(design.residual_df, 0)
        self.assertIsNone(fit.sigma2)
        result = evaluate_contrast(
            design, fit, "any-estimable",
            {"A:30": -1, "A:31": 1, "P:2020": 1, "P:2021": -1}, 0.05,
        )
        # 该对比在单队列 2x2 结构下也可能不可估；无论如何不能有区间
        self.assertIsNone(result.ci_low)
        self.assertIsNone(result.ci_high)
        self.assertIsNone(result.std_error)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.table = read_table(FIXTURE_CSV)
        self.design = build_design(self.table)
        self.fit = wls_min_norm(self.design, self.table.y)
        self.config = json.loads(FIXTURE_CFG.read_text(encoding="utf-8"))

    def test_registered_sets_are_equivalent_reparameterizations(self):
        for cs in self.config["constraint_sets"]:
            sol = solve_constrained(
                self.design, self.table.y, self.fit,
                cs["key"], cs["rules"], 1e-10,
            )
            self.assertEqual(sol.status, "equivalent_reparameterization")
            self.assertTrue(sol.spans_null_space)
            self.assertEqual(sol.rn_rank, self.design.nullity)
            self.assertLess(abs(sol.sse_delta), 1e-8)
            self.assertTrue(
                np.allclose(sol.fitted, self.fit.fitted, atol=1e-8)
            )

    def test_equivalent_constraints_agree_on_estimable_contrast(self):
        curvature = {"A:20": 1, "A:21": -2, "A:22": 1}
        values = []
        for cs in self.config["constraint_sets"]:
            sol = solve_constrained(
                self.design, self.table.y, self.fit,
                cs["key"], cs["rules"], 1e-10,
            )
            l = build_contrast_vector(self.design, curvature)
            values.append(float(l @ sol.beta))
        self.assertTrue(np.allclose(values, [-0.6, -0.6], atol=1e-8))

    def test_equivalent_constraints_disagree_on_nonestimable_trend(self):
        # 同一不可估线性趋势在两套约束下数值不同——这正是不可识别的证据
        l = build_contrast_vector(self.design, {"A:20": -1, "A:22": 1})
        values = []
        for cs in self.config["constraint_sets"]:
            sol = solve_constrained(
                self.design, self.table.y, self.fit,
                cs["key"], cs["rules"], 1e-10,
            )
            values.append(float(l @ sol.beta))
        self.assertGreater(abs(values[0] - values[1]), 0.1)

    def test_constraint_deleting_fit_direction_is_extra_assumption(self):
        rules = self.config["constraint_sets"][0]["rules"] + ["A:20=0.3"]
        sol = solve_constrained(
            self.design, self.table.y, self.fit, "extra", rules, 1e-10,
        )
        self.assertEqual(sol.status, "extra_assumption")
        self.assertGreater(sol.sse_delta, 1e-6)
        self.assertFalse(np.allclose(sol.fitted, self.fit.fitted, atol=1e-8))

    def test_contradictory_constraints_detected(self):
        sol = solve_constrained(
            self.design, self.table.y, self.fit,
            "contradiction", ["intercept=0", "intercept=1"], 1e-10,
        )
        self.assertEqual(sol.status, "contradictory")
        self.assertIsNone(sol.beta)
        self.assertGreater(sol.constraint_residual, 0.1)

    def test_underidentified_when_constraints_do_not_span_nullspace(self):
        sol = solve_constrained(
            self.design, self.table.y, self.fit,
            "under", ["sum(A)=0", "sum(P)=0"], 1e-10,
        )
        self.assertEqual(sol.status, "underidentified")
        self.assertLess(sol.rn_rank, self.design.nullity)
        self.assertFalse(sol.spans_null_space)

    def test_empty_rule_set_rejected(self):
        with self.assertRaises(ConstraintError):
            solve_constrained(
                self.design, self.table.y, self.fit, "empty", [], 1e-10
            )

    def test_contradiction_takes_precedence_over_underidentification(self):
        sol = solve_constrained(
            self.design, self.table.y, self.fit,
            "mixed", ["sum(A)=0", "intercept=0", "intercept=1"], 1e-10,
        )
        self.assertEqual(sol.status, "contradictory")

    def test_rule_parser_rejects_unknown_syntax_and_parameters(self):
        with self.assertRaises(ConstraintError):
            parse_rule(self.design, "eval(__import__('os'))=0")
        with self.assertRaises(ConstraintError):
            parse_rule(self.design, "A:99=0")
        with self.assertRaises(ConstraintError):
            parse_rule(self.design, "sum((age-20)*P)=0")

    def test_rule_parser_accepts_documented_forms(self):
        r, rhs = parse_rule(self.design, "sum(C)=0")
        self.assertEqual(rhs, 0.0)
        self.assertEqual(int(r.sum()), len(self.design.cohort_levels))
        r, rhs = parse_rule(self.design, "sum((cohort-2000)*C)=0")
        cohort_cols = [
            self.design.column_index(f"C:{c}") for c in self.design.cohort_levels
        ]
        self.assertTrue(np.allclose(
            r[cohort_cols],
            np.array(self.design.cohort_levels, dtype=float) - 2000.0,
        ))
        r, rhs = parse_rule(self.design, "intercept=0")
        self.assertEqual(r[0], 1.0)

    def test_zero_contrast_rejected(self):
        # 空系数展开后为零向量，必须拒绝
        with self.assertRaises(ConstraintError):
            build_contrast_vector(self.design, {})


class GridAndDegeneracyTests(unittest.TestCase):
    def test_irregular_grid_reports_missing_cells_but_stays_connected(self):
        rows = [
            "20,2020,2000,1,1", "20,2021,2001,1,1",
            "21,2021,2000,1,1", "21,2022,2001,1,1",
            "22,2022,2000,1,1",
        ]
        design = design_of(rows)
        self.assertEqual(design.n_components, 1)
        # 3x3 笛卡尔网格应有 9 格，观测 5 格，缺 4 格
        self.assertEqual(len(design.missing_cells), 4)
        self.assertEqual(design.nullity, 4)

    def test_disconnected_blocks_inflate_nullspace_beyond_classic(self):
        rows = grid_rows([20, 21], [2020, 2021]) + grid_rows([40, 41], [2030, 2031])
        design = design_of(rows)
        self.assertEqual(design.n_components, 2)
        self.assertGreater(design.nullity, 4)
        # 经典常数与漂移仍在零空间，但远不是全部
        self.assertTrue(design.canonical_checks["linear_drift"]["in_null_space"])

    def test_single_cohort_design_is_rank_deficient_and_trends_unestimable(self):
        rows = ["20,2020,2000,1,1", "21,2021,2000,2,1"]
        table = read_table(write_csv(rows))
        design = build_design(table)
        self.assertEqual(design.cohort_levels, [2000])
        self.assertLess(design.rank, design.p)
        self.assertGreater(design.nullity, 0)
        # 单队列下年龄趋势与时期趋势完全共线，任何纯线性趋势都不可估
        fit = wls_min_norm(design, table.y)
        result = evaluate_contrast(
            design, fit, "age-trend", {"A:20": -1, "A:21": 1}, 0.05
        )
        self.assertFalse(result.estimable)
        self.assertIsNone(result.ci_low)

    def test_single_age_level_flagged(self):
        rows = ["30,2020,1990,1,1", "30,2021,1991,1,1", "30,2022,1992,1,1"]
        design = design_of(rows)
        self.assertTrue(
            any("单一水平" in reason for reason in design.extra_nullity_reasons)
        )

    def test_wrong_cohort_rejected(self):
        path = write_csv(["20,2020,1999,1,1"])
        with self.assertRaises(APCDataError):
            read_table(path)

    def test_non_positive_weight_rejected(self):
        for bad in ("0", "-1.5"):
            path = write_csv([f"20,2020,2000,1,{bad}"])
            with self.assertRaises(APCDataError):
                read_table(path)

    def test_duplicate_cell_rejected(self):
        path = write_csv([
            "20,2020,2000,1,1",
            "20,2020,2000,2,1",
        ])
        with self.assertRaises(APCDataError):
            read_table(path)

    def test_weights_actually_change_the_wls_fit(self):
        base = grid_rows(
            [20, 21, 22], [2020, 2021, 2022],
            y_func=lambda a, p: float(a + p),
        )
        # 让一个格成为强影响离群点
        base[-1] = "22,2022,2000,99,1"
        weighted = list(base)
        weighted[-1] = "22,2022,2000,99,100"
        t1 = read_table(write_csv(base))
        t2 = read_table(write_csv(weighted))
        f1 = wls_min_norm(build_design(t1), t1.y)
        f2 = wls_min_norm(build_design(t2), t2.y)
        self.assertGreater(abs(f1.fitted[-1] - f2.fitted[-1]), 1e-4)


class CurvatureTests(unittest.TestCase):
    def test_second_difference_zero_on_linear_curve(self):
        levels = [10, 11, 12]
        values = np.array([2.0, 4.0, 6.0])
        diffs = second_differences(levels, values)
        self.assertEqual(len(diffs), 1)
        self.assertAlmostEqual(diffs[0]["raw_second_difference"], 0.0)
        self.assertTrue(diffs[0]["equally_spaced"])

    def test_divided_difference_on_irregular_spacing(self):
        # 线性函数在不齐整间距下：raw 非零，但二阶均差为 0
        levels = [0, 1, 3]
        values = np.array([0.0, 2.0, 6.0])  # v = 2x
        diffs = second_differences(levels, values)
        self.assertFalse(diffs[0]["equally_spaced"])
        self.assertNotAlmostEqual(diffs[0]["raw_second_difference"], 0.0)
        self.assertAlmostEqual(diffs[0]["second_divided_difference"], 0.0)

    def test_all_curvature_contrasts_orthogonal_to_full_nullspace(self):
        design = build_design(read_table(FIXTURE_CSV))
        for factor in ("age", "period", "cohort"):
            prefix = {"age": "A", "period": "P", "cohort": "C"}[factor]
            levels = {
                "age": design.age_levels,
                "period": design.period_levels,
                "cohort": design.cohort_levels,
            }[factor]
            if len(levels) < 3:
                continue
            check = curvature_orthogonal_to_nullspace(design, factor, levels)
            self.assertTrue(all(item["orthogonal"] for item in check["checks"]))

    def test_drift_invariance_does_not_guarantee_estimability_on_sparse_grid(self):
        # 漂移对任何二阶均差为 0，但稀疏/断连网格上曲率仍可能落在额外零方向。
        rows = [
            "20,2020,2000,1,1", "20,2022,2002,2,1", "20,2025,2005,3,1",
            "21,2020,1999,2,1", "21,2022,2001,3,1", "21,2025,2004,4,1",
            "23,2020,1997,3,1", "23,2022,1999,4,1", "23,2025,2002,5,1",
        ]
        design = design_of(rows)
        self.assertGreaterEqual(design.nullity, 5)
        check = curvature_orthogonal_to_nullspace(
            design, "age", design.age_levels
        )
        # 关键断言：不能因为“二阶差分对漂移不变”就自动盖章可估
        self.assertFalse(all(c["orthogonal"] for c in check["checks"]))
        # 曲线报告必须显式标注该曲率不可估
        table = read_table(write_csv(rows))
        fit = wls_min_norm(design, table.y)
        from src.apc_model import _factor_curve
        curve = _factor_curve(fit.beta, design, "age")
        flags = [d["estimable"] for d in curve["second_differences"]]
        self.assertIn(False, flags)


class EndToEndTests(unittest.TestCase):
    def test_full_report_on_fixtures_is_consistent(self):
        report = analyze(FIXTURE_CSV, FIXTURE_CFG, FIXTURE_CON)
        self.assertEqual(report["design"]["nullity"], 4)
        contrast_by_key = {c["key"]: c for c in report["contrasts"]}
        self.assertTrue(contrast_by_key["age-curvature"]["estimable"])
        self.assertFalse(contrast_by_key["age-trend"]["estimable"])
        for sol in report["constraint_sets"]:
            self.assertEqual(sol["status"], "equivalent_reparameterization")
        cross = report["cross_constraint_check"]
        self.assertTrue(all(
            item["same_equivalence_class"]
            for item in cross["fitted_value_comparison"]
        ))
        # fixture 是精确加性合成数据：残差为机器精度噪声，必须显式警告
        self.assertTrue(
            report["unconstrained_wls"]["residual_is_machine_precision_noise"]
        )
        self.assertFalse(
            contrast_by_key["age-curvature"]["interval_has_statistical_meaning"]
        )
        # JSON 可序列化（无 NaN/Inf 泄漏 numpy 标量）
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
        self.assertIn("interpretation_guardrails", encoded)

    def test_noisy_data_produce_statistically_meaningful_intervals(self):
        rows = grid_rows(
            [30, 31, 32, 33], [2010, 2011, 2012],
            y_func=lambda a, p: 1.0 + 0.3 * a + 0.5 * p + ((a * p) % 5) * 0.4,
        )
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "noisy.csv"
            data.write_text(
                "age,period,cohort,y,weight\n" + "\n".join(rows) + "\n",
                encoding="utf-8",
            )
            contrasts = Path(directory) / "contrasts.json"
            contrasts.write_text(json.dumps({"queries": [
                {"key": "age-curvature",
                 "coefficients": {"A:30": 1, "A:31": -2, "A:32": 1}},
                {"key": "age-trend",
                 "coefficients": {"A:30": -1, "A:33": 1}},
            ]}), encoding="utf-8")
            report = analyze(data, FIXTURE_CFG, contrasts)
            wls = report["unconstrained_wls"]
            self.assertFalse(wls["residual_is_machine_precision_noise"])
            self.assertTrue(wls["intervals_emitted"])
            curvature = next(
                c for c in report["contrasts"] if c["key"] == "age-curvature"
            )
            self.assertTrue(curvature["interval_has_statistical_meaning"])
            self.assertLess(curvature["ci_low"], curvature["ci_high"])

    def test_report_marks_extra_assumption_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "cfg.json"
            cfg = json.loads(FIXTURE_CFG.read_text(encoding="utf-8"))
            cfg["constraint_sets"].append({
                "key": "forces-fit",
                "rules": cfg["constraint_sets"][0]["rules"] + ["A:20=0.3"],
            })
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            report = analyze(FIXTURE_CSV, cfg_path, FIXTURE_CON)
            statuses = {s["key"]: s["status"] for s in report["constraint_sets"]}
            self.assertEqual(statuses["forces-fit"], "extra_assumption")

    def test_report_with_contradictory_constraint_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "cfg.json"
            cfg = json.loads(FIXTURE_CFG.read_text(encoding="utf-8"))
            cfg["constraint_sets"] = [{
                "key": "impossible",
                "rules": ["intercept=0", "intercept=1"],
            }]
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            report = analyze(FIXTURE_CSV, cfg_path, FIXTURE_CON)
            self.assertEqual(
                report["constraint_sets"][0]["status"], "contradictory"
            )
            json.dumps(report, allow_nan=False)  # 仍可序列化


if __name__ == "__main__":
    unittest.main()
