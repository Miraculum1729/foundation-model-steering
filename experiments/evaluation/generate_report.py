#!/usr/bin/env python3
"""Phase 6: 综合报告与可视化 Dashboard。汇总指标并生成含图表的 summary.html。"""
from pathlib import Path

def main():
    root = Path("/mnt/hbnas/home/pfp/hiv2026/dplm/exp_results")
    out_dir = root / "dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 相对路径：从 dashboard/ 到各子目录
    rel = lambda subpath: f"../{subpath}"

    parts = ["# HIV-1 PR DPLM 验证报告\n"]
    for name, p in [("Perplexity", root / "perplexity/perplexity_summary.txt"),
                    ("AAR", root / "aar/aar_summary.txt"),
                    ("Diversity", root / "diversity/diversity_summary.txt")]:
        if p.exists():
            parts.append("## " + name + "\n\n```\n" + p.read_text().strip() + "\n```\n")
    report = "\n".join(parts)
    (out_dir / "final_report.md").write_text(report)

    # Dashboard HTML：嵌入所有关键图表
    charts = [
        ("Diversity: Hamming", "diversity/hamming_distances.png"),
        ("Diversity: Entropy by site", "diversity/entropy_by_site.png"),
        ("AAR distribution", "aar/aar_distribution.png"),
        ("Mutation: key sites KL", "mutation/key_sites_comparison.png"),
        ("Embeddings: t-SNE", "embeddings/tsne_plot.png"),
        ("Embeddings: PCA", "embeddings/pca_plot.png"),
    ]
    body = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>HIV-1 PR DPLM Dashboard</title>",
        "<style>body{font-family:sans-serif;margin:1rem;} h1{color:#333;} .chart{margin:1.5rem 0;} .chart img{max-width:100%;}</style></head><body>",
        "<h1>HIV-1 PR DPLM Potts 引导生成与验证 — 可视化 Dashboard</h1>",
        "<p><a href='final_report.md'>综合报告 (Markdown)</a></p>",
    ]
    for title, subpath in charts:
        p = root / subpath.replace("/", "/")
        if p.exists():
            body.append(f"<div class='chart'><h2>{title}</h2><img src='{rel(subpath)}' alt='{title}' /></div>")
    body.append("</body></html>")
    (out_dir / "summary.html").write_text("\n".join(body))
    print("Phase 6 done. See", out_dir, "- summary.html 已嵌入图表")
    print("  建议在浏览器中打开: file://" + str((out_dir / "summary.html").resolve()))

if __name__ == "__main__":
    main()
