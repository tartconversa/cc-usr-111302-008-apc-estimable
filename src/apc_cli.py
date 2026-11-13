"""APC 可估性分析的命令行入口。

输出写入独立目录（默认 .workspace-output，已被 git 忽略），不覆盖 fixtures：

* apc_report.json —— 完整结构化报告（奇异值、零空间诊断、可估性、约束求解）
* apc_fitted.csv —— 未约束最小范数解的逐格拟合值与残差
* apc_summary.txt —— 人类可读结论，明确标注不可估趋势与非因果解释边界
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .apc_model import analyze


def _write_fitted_csv(path: Path, report: dict) -> None:
    rows = report["unconstrained_wls"]["fitted"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["age", "period", "cohort", "y", "weight", "fitted", "residual"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_number(value, digits: int = 6) -> str:
    if value is None:
        return "不输出"
    return f"{value:.{digits}g}"


def _write_summary(path: Path, report: dict) -> None:
    lines: list[str] = []
    design = report["design"]
    wls = report["unconstrained_wls"]
    lines.append("年龄-时期-队列（APC）可估性分析摘要")
    lines.append("=" * 48)
    lines.append(
        f"格数 n={report['data']['n_cells']}，参数 p={design['n_parameters']}，"
        f"秩 rank={design['rank']}，零空间维数 nullity={design['nullity']}，"
        f"残差自由度 df={design['residual_df']}"
    )
    lines.append(
        f"缺格 {report['data']['n_missing_cells']} 个，"
        f"连通分量 {report['data']['n_components']} 个，"
        f"额外零空间维数 {design['extra_nullity']}"
    )
    for reason in design["extra_nullity_reasons"]:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append("加权设计奇异值（保留全部，含零奇异值）:")
    lines.append("  " + ", ".join(f"{v:.6g}" for v in design["singular_values"]))
    canonical = design["canonical_null_directions"]
    lines.append(
        "经典零方向核验: 常数重定基 max|Xv|="
        f"{canonical['constant_releveling']['max_abs_Xv']:.3g}, "
        "线性漂移 max|Xv|="
        f"{canonical['linear_drift']['max_abs_Xv']:.3g}"
    )
    lines.append("")
    lines.append(f"未约束 WLS: SSE={wls['sse']:.6g}, sigma^2={_format_number(wls['sigma2'])}")
    lines.append(f"区间: {wls['interval_note']}")
    if wls.get("residual_is_machine_precision_noise"):
        lines.append("  警告: 残差仅为浮点舍入噪声，形式区间不具备统计精度含义")
    lines.append("")
    lines.append("用户查询对比（可估性基于原始未约束设计矩阵）:")
    for item in report["contrasts"]:
        lines.append(f"  [{item['key']}]")
        lines.append(
            f"    可估={item['estimable']}；零空间投影范数="
            f"{item['null_projection_norm']:.3g}；正交残差="
            f"{item['max_abs_nullspace_inner_product']:.3g}"
        )
        if item["estimable"]:
            ci = (
                f"，{item['ci_level']:.0%} CI=({_format_number(item['ci_low'])}, "
                f"{_format_number(item['ci_high'])})"
                if item["ci_low"] is not None
                else "，自由度不足不输出区间"
            )
            lines.append(f"    估计={_format_number(item['estimate'])}，SE={_format_number(item['std_error'])}{ci}")
        else:
            lines.append("    不可估线性趋势：约束一变数值就变，数据无法确定；不报告点估计与区间")
        lines.append(f"    说明: {item['reason']}")
    lines.append("")
    lines.append("曲线二阶差分（对线性漂移不变；已逐一核对完整零空间正交性）:")
    for curve in wls["curves"]:
        lines.append(f"  {curve['factor']} 曲线:")
        for diff in curve["second_differences"]:
            tag = "等间隔" if diff["equally_spaced"] else "不齐整(二阶均差)"
            est = "可估" if diff["estimable"] else "不可估(额外零方向)"
            lines.append(
                f"    水平 {diff['level']} [{tag}/{est}]: "
                f"raw={diff['raw_second_difference']:.6g}, "
                f"divided={diff['second_divided_difference']:.6g}, "
                f"零空间内积={diff['max_abs_nullspace_inner_product']:.3g}"
            )
    lines.append("")
    lines.append("登记的识别约束:")
    for sol in report["constraint_sets"]:
        lines.append(f"  [{sol['key']}] 状态={sol['status']}")
        for rule in sol["rules"]:
            lines.append(f"    规则: {rule}")
        lines.append(
            f"    RZ 秩={sol['R_nullspace_rank']}/{design['nullity']}，"
            f"约束残差={sol['constraint_residual']:.3g}，"
            f"SSE={_format_number(sol['sse'])}，"
            f"ΔSSE={_format_number(sol['sse_delta_vs_unconstrained'])}"
        )
        lines.append(f"    判定: {sol['note']}")
    lines.append("")
    lines.append("跨约束核验:")
    for item in report["cross_constraint_check"]["fitted_value_comparison"]:
        lines.append(
            f"  {item['key']}: 拟合最大差={_format_number(item['max_abs_fitted_diff'])}，"
            f"同一等价类={item['same_equivalence_class']}"
        )
    for item in report["cross_constraint_check"]["estimable_contrast_comparison"]:
        values = ", ".join(f"{k}={v:.6g}" for k, v in item["values"].items())
        lines.append(
            f"  可估对比 {item['key']}: {values}；跨约束最大差="
            f"{_format_number(item['max_abs_diff_across_constraints'])}"
        )
    lines.append("")
    lines.append("解释边界:")
    for guard in report["interpretation_guardrails"]:
        lines.append(f"  ! {guard}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="APC 加性模型的可估对比与约束敏感性分析（不做因果解释）"
    )
    parser.add_argument("data", help="合成汇总格 CSV/JSON 路径")
    parser.add_argument("config", help="识别约束与方差配置 JSON 路径")
    parser.add_argument("contrasts", help="用户对比查询 JSON 路径")
    parser.add_argument(
        "-o", "--output-dir", default=".workspace-output", help="输出目录（默认 .workspace-output）"
    )
    args = parser.parse_args(argv)

    report = analyze(args.data, args.config, args.contrasts)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "apc_report.json"
    fitted_path = out_dir / "apc_fitted.csv"
    summary_path = out_dir / "apc_summary.txt"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_fitted_csv(fitted_path, report)
    _write_summary(summary_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "fitted_csv": str(fitted_path),
                "summary": str(summary_path),
                "rank": report["design"]["rank"],
                "nullity": report["design"]["nullity"],
                "residual_df": report["design"]["residual_df"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
