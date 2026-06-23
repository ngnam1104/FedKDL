"""
preprocess_related_work.py
==========================
Tiền xử lý file .docs/related_work.md:
  1. Chỉ giữ lại 41 bài báo có trong danh sách chính thức.
  2. Tạo lại file Markdown mới với 4 nhóm phân loại chi tiết,
     mỗi nhóm có bảng tóm tắt + mô tả chi tiết từng bài.

Chạy từ thư mục gốc của repo:
    python scripts/preprocess_related_work.py
"""

from __future__ import annotations
import io
import re
import sys
from pathlib import Path

# Fix UnicodeEncodeError tren Windows console (cp1252 khong ho tro tieng Viet)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE  = REPO_ROOT / ".docs" / "related_work.md"
OUTPUT_FILE = REPO_ROOT / ".docs" / "related_work_cleaned.md"

# ─────────────────────────────────────────────────────────────────────────────
#  41 BÀI BÁO CHÍNH THỨC – keyword khớp (case-insensitive, substring match)
#  Mỗi entry: (slug_for_matching, group_id, role_note)
# ─────────────────────────────────────────────────────────────────────────────
OFFICIAL_PAPERS = [
    # ── Nhóm 1: HFL / Topology (9 bài) ──────────────────────────────────────
    ("AUV-Edge-Cloud Hierarchical Federated Learning|Client-Edge-Cloud|HierFAVG|Lumin Liu",
                                                                        1, "HierFAVG – Co so kien truc 3 tang"),
    ("Federated Learning for Internet of Things.*Comprehensive Survey", 1, "Tổng quan rào cản IoT"),
    ("Efficient Asynchronous Federated Learning for AUV Swarm",        1, "Tối ưu độ trễ & năng lượng bầy đàn AUV"),
    ("Clustered.*Federated.*Multitask.*Non-IID.*Enhanced Privacy",     1, "CFMTL – EMD clustering (rất quan trọng)"),
    ("Mobility-aware.*Decentralized.*Federated.*Autonomous Underwater", 1, "Khắc phục tính di động AUV"),
    ("Topology-aware Federated Learning in Edge Computing",             1, "Phân loại các hình thái Topology"),
    ("Optimization of Model Aggregation for Federated Learning.*Edge", 1, "Toán học chọn trạm tổng hợp trung gian"),
    ("Energy-Aware Clustered Federated Learning.*Underwater",          1, "Lập lịch năng lượng + tổng hợp trung vị"),
    ("Energy-Efficient Hierarchical Federated Anomaly Detection.*Omeke|Selective Cooperative Aggregation",
                                                                        1, "Đối thủ trực tiếp / Baseline HFL-Selective + Top-K"),

    # ── Nhóm 2: Non-IID (8 bài) ──────────────────────────────────────────────
    ("Communication-Efficient Learning of Deep Networks.*Decentralized Data|FedAvg",
                                                                        2, "FedAvg – Thuật toán gốc (McMahan 2017)"),
    ("Federated Multi-Task Learning",                                   2, "MOCHA – Góc nhìn Multi-task"),
    ("FEDERATED OPTIMIZATION IN HETEROGENEOUS NETWORKS|FedProx",       2, "FedProx – Proximal term điều chuẩn"),
    ("SCAFFOLD.*Stochastic Controlled Averaging",                      2, "SCAFFOLD – Biến kiểm soát phương sai"),
    ("Model-Contrastive Federated Learning|MOON",                      2, "MOON – Contrastive learning mô hình"),
    ("FedProto.*Federated Prototype",                                  2, "FedProto – Giao tiếp bằng nguyên mẫu"),
    ("Towards Personalized Federated Learning",                        2, "Survey cá nhân hóa FL"),
    ("FedSiKD.*Similarity.*Knowledge.*Distillation",                   2, "K-Means trên μ,σ để gom cụm"),

    # ── Nhóm 3: Nén truyền thông (16 bài) ────────────────────────────────────
    ("Distilling the Knowledge in a Neural Network",                   3, "Nền tảng KD – Soft labels (Hinton)"),
    ("FedMD.*Heterogenous.*Model Distillation",                        3, "KD qua dữ liệu proxy"),
    ("Similarity-Preserving Knowledge Distillation",                   3, "Cơ sở LoRA-Projection KD"),
    ("Deep Gradient Compression.*Communication Bandwidth",             3, "DGC – Top-K nén gradient (bị phá ảnh 2D)"),
    ("Ensemble Distillation.*Robust Model Fusion.*Federated|FedDF",    3, "FedDF – Chưng cất tập hợp"),
    ("LORA.*LOW-RANK ADAPTATION.*LARGE LANGUAGE MODELS|LoRA: LOW-RANK",3, "Nền tảng LoRA (Hu et al. 2021)"),
    ("FedKD.*Communication.*Efficient.*Knowledge Distil",             3, "SVD nén gradient"),
    ("Adaptive Model Pruning.*Personalization.*Federated.*Wireless",  3, "Tối ưu KKT cho pruning"),
    ("Federated Fine-tuning.*Large Language.*Heterogeneous Tasks|FlexLoRA",
                                                                        3, "FlexLoRA – SVD tổng hợp LoRA rank khác nhau"),
    ("IMPROVING LORA.*PRIVACY-PRESERVING.*FEDERATED|FFA-LoRA",        3, "FFA-LoRA – Đóng băng A, train B"),
    ("Communication-Aware Knowledge Distillation.*Federated LLM|AdaLD",3,"AdaLD – Top-K logit + h=Ax qua vô tuyến"),
    ("FedDT.*Communication-Efficient.*Knowledge Distillation.*Ternary",3, "INT8/Ternary lượng tử hóa"),
    ("Federated Low-Rank Adaptation.*Foundation Models.*Survey",       3, "Mâu thuẫn toán học FedAvg trên LoRA"),
    ("FEDQLORA.*FEDERATED QUANTIZATION-AWARE LORA",                    3, "Bù đắp lỗi lượng tử hóa"),
    ("HAFLQ.*Heterogeneous.*Adaptive.*Federated.*LoRA",               3, "Nén thích ứng"),
    ("ILoRA.*Federated Learning.*Low-Rank Adaptation",                 3, "Phân rã QR đồng bộ không gian LoRA"),

    # ── Nhóm 4: YOLO / IoUT (8 bài) ──────────────────────────────────────────
    ("Federated Learning for IoUT.*Concepts.*Applications",            4, "Bối cảnh chung IoUT"),
    ("UltraFlwr.*Efficient.*Federated.*Surgical.*Object Detection",    4, "Chia tách module YOLO"),
    ("Underwater Federated Learning.*Autonomous Underwater Vehicle.*Swarm|Empowering.*AUV.*Swarm",
                                                                        4, "Cắt tỉa & lượng tử hóa AUV"),
    ("YOLOV11.*OVERVIEW.*KEY ARCHITECTURAL|YOLOv11",                  4, "Kiến trúc YOLOv11, C3k2"),
    ("Federated Learning.*Internet of Underwater Things.*Lightweight Distillation",
                                                                        4, "Chưng cất dưới nước, bù nhiễu ảnh"),
    ("FEDEXCHANGE.*Bridging.*Domain Gap.*Federated Object Detection",  4, "Vượt rào cản domain ảnh"),
    ("Knowledge.*Distillation.*Object Detection.*Resource-Constrained",4, "KD cho Object Detection tại Edge AI"),
    ("YOLOv12.*Attention-Centric.*Real-Time Object",                  4, "Teacher Oracle của FedKDL"),
]

GROUP_META = {
    1: {
        "title": "Nhóm 1: Kiến trúc Mạng & Học liên kết phân cấp (HFL / Topology)",
        "purpose": (
            "Nhóm này chứng minh cấu trúc 3 tầng và luật kết nối $D_{joint}$ của FedKDL là sự "
            "tiến hóa tất yếu từ các kiến trúc HFL hiện có. Các bài báo trải dài từ nền tảng lý "
            "thuyết (HierFAVG), qua khảo sát IoT, tới các đặc thù AUV dưới nước (năng lượng, di "
            "chuyển, topology) và một đối thủ trực tiếp làm baseline so sánh."
        ),
    },
    2: {
        "title": "Nhóm 2: Xử lý Dữ liệu phi đồng nhất (Non-IID)",
        "purpose": (
            "Nhóm này chỉ ra sự hạn chế của các phương pháp xử lý Non-IID nặng nề tại biên "
            "(FedProx, SCAFFOLD, MOON…), làm nổi bật hiệu quả của cơ chế EMD Clustering trong "
            "FedKDL. FedSiKD là điểm so sánh gần nhất về phân cụm dựa trên thống kê dữ liệu."
        ),
    },
    3: {
        "title": "Nhóm 3: Nén truyền thông (LoRA, INT8, KD, SVD)",
        "purpose": (
            "Day la 'vu khi hang nang' bao ve truc tiep cho phuong trinh SVD-LoRA Aggregation "
            "và cơ chế LoRA-Projection KD. Nhóm xây dựng luận điểm từ nền tảng (Hinton KD, LoRA) "
            "qua các biến thể FL (FFA-LoRA, FlexLoRA, FedKD) đến ứng dụng thực tế (AdaLD, "
            "FedQLoRA, FedDT)."
        ),
    },
    4: {
        "title": "Nhóm 4: Ứng dụng Thị giác (YOLO) & IoUT",
        "purpose": (
            "Nhóm này chứng minh độ khó của bài toán: YOLOv12 quá nặng để chạy trực tiếp trên "
            "AUV dưới nước, cần giải pháp tách bờ (Gateway KD). Bao gồm bối cảnh IoUT, kiến trúc "
            "YOLO, các phương pháp KD/nén dành riêng cho thị giác dưới nước và Teacher Oracle."
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  PARSE: trích xuất các entry từ bảng Markdown
# ─────────────────────────────────────────────────────────────────────────────

def parse_table_rows(text: str) -> list[dict]:
    """
    Phân tích bảng markdown (pipe-separated) và trả về list dict
    với keys: raw_cell0, raw_cell1(title+author), cell2, cell3, cell4, cell5.
    """
    rows = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break          # kết thúc bảng
            continue
        if stripped.startswith("| :---") or stripped.startswith("|---"):
            in_table = True
            continue
        if stripped.startswith("| | Tác giả"):
            in_table = True
            continue
        in_table = True
        # tách cells
        cells = [c.strip() for c in stripped.split("|")]
        cells = [c for c in cells if c != ""]   # bỏ rìa trống
        if len(cells) < 5:
            continue
        rows.append({
            "author_title": cells[0],
            "problem":      cells[1] if len(cells) > 1 else "",
            "method":       cells[2] if len(cells) > 2 else "",
            "category":     cells[3] if len(cells) > 3 else "",
            "result":       cells[4] if len(cells) > 4 else "",
        })
    return rows


def parse_detail_sections(text: str) -> dict[str, str]:
    """
    Trả về dict mapping: keyword_fragment -> full_section_text
    Mỗi section được bắt đầu bằng một heading (####, ###, **[...)
    """
    # Tìm tất cả sections chi tiết (bắt đầu với #### hoặc **[...)
    pattern = re.compile(
        r'((?:#{2,4}\s+\[.+?\]|#{2,4}\s+.+?|\*\*\[.+?\]\*\*).+?)(?=(?:#{2,4}\s|\*\*\[|\Z))',
        re.DOTALL
    )
    sections = {}
    for m in pattern.finditer(text):
        block = m.group(0).strip()
        # lấy dòng đầu làm key
        first_line = block.splitlines()[0]
        sections[first_line] = block
    return sections


# ─────────────────────────────────────────────────────────────────────────────
#  MATCH: khớp mỗi OFFICIAL_PAPER với rows và detail sections
# ─────────────────────────────────────────────────────────────────────────────

def find_row(rows: list[dict], pattern: str) -> dict | None:
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for r in rows:
        if regex.search(r["author_title"]):
            return r
    return None


def find_detail(sections: dict[str, str], pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for key, block in sections.items():
        if regex.search(key) or regex.search(block[:300]):
            return block
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER: tạo Markdown output
# ─────────────────────────────────────────────────────────────────────────────

TABLE_HEADER = (
    "| # | Tác giả & Tên bài báo | Vấn đề mục tiêu | Phương pháp cốt lõi | Vai trò trong FedKDL | Đóng góp / Kết quả chính |\n"
    "|---|---|---|---|---|---|\n"
)


def render_group(
    group_id: int,
    papers_in_group: list[tuple],
    rows: list[dict],
    sections: dict[str, str],
) -> str:
    meta = GROUP_META[group_id]
    lines = []
    lines.append(f"\n---\n\n## {meta['title']}\n")
    lines.append(f"> **Mục đích lập luận:** {meta['purpose']}\n")

    # ── Bảng tóm tắt ─────────────────────────────────────────────────────────
    lines.append("\n### Bảng tóm tắt\n")
    lines.append(TABLE_HEADER)
    idx = 1
    for (slug, gid, role) in papers_in_group:
        if gid != group_id:
            continue
        row = find_row(rows, slug)
        if row:
            at   = row["author_title"].replace("\n", " ").replace("\r", "")
            prob = row["problem"][:120].replace("\n", " ") + ("…" if len(row["problem"]) > 120 else "")
            meth = row["method"][:120].replace("\n", " ") + ("…" if len(row["method"]) > 120 else "")
            res  = row["result"][:120].replace("\n", " ") + ("…" if len(row["result"]) > 120 else "")
        else:
            at   = f"*(Chưa tìm thấy: {slug[:60]})*"
            prob = meth = res = "—"
        lines.append(f"| {idx} | {at} | {prob} | {meth} | {role} | {res} |\n")
        idx += 1

    # ── Mô tả chi tiết từng bài ───────────────────────────────────────────────
    lines.append("\n### Mô tả chi tiết\n")
    for (slug, gid, role) in papers_in_group:
        if gid != group_id:
            continue
        detail = find_detail(sections, slug)
        if detail:
            lines.append(f"\n{detail}\n")
        else:
            lines.append(f"\n> *(Chưa có mô tả chi tiết cho: `{slug[:80]}`)*\n")

    return "".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_FILE.exists():
        print(f"[LỖI] Không tìm thấy file đầu vào: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    text = INPUT_FILE.read_text(encoding="utf-8")

    print(f"[1] Đọc file: {INPUT_FILE} ({len(text):,} bytes)")
    rows    = parse_table_rows(text)
    print(f"    → Tìm thấy {len(rows)} hàng trong bảng tóm tắt.")

    sections = parse_detail_sections(text)
    print(f"    → Tìm thấy {len(sections)} section chi tiết.")

    # Báo cáo match
    matched_table   = 0
    matched_detail  = 0
    unmatched       = []
    for (slug, gid, role) in OFFICIAL_PAPERS:
        r = find_row(rows, slug)
        d = find_detail(sections, slug)
        if r:
            matched_table += 1
        if d:
            matched_detail += 1
        if not r and not d:
            unmatched.append(slug[:80])

    print(f"\n[2] Kết quả khớp:")
    print(f"    Bảng tóm tắt : {matched_table}/{len(OFFICIAL_PAPERS)} bài")
    print(f"    Chi tiết      : {matched_detail}/{len(OFFICIAL_PAPERS)} bài")
    if unmatched:
        print(f"\n[!] Chưa tìm thấy ({len(unmatched)} bài):")
        for u in unmatched:
            print(f"    • {u}")

    # Xây dựng output
    output_parts = [
        "# Related Work — FedKDL\n\n",
        "> File được tạo tự động bởi `scripts/preprocess_related_work.py`.\n",
        "> Chỉ chứa 41 tài liệu chính thức, được phân thành 4 nhóm lập luận.\n\n",
        "---\n\n",
        "## Sơ đồ chiến thuật: 4 nhóm lập luận cốt lõi\n\n",
        "| Nhóm | Tên | Số bài | Vai trò trong luận văn |\n",
        "|---|---|:---:|---|\n",
        "| 1 | Kiến trúc Mạng & HFL / Topology | 9 | Chứng minh cấu trúc 3 tầng & $D_{joint}$ |\n",
        "| 2 | Non-IID | 8 | Đặt vấn đề, chỉ ra hạn chế baseline |\n",
        "| 3 | Nén truyền thông (LoRA, KD, SVD) | 16 | Bảo vệ SVD-LoRA Agg + LoRA-Proj KD |\n",
        "| 4 | YOLO & IoUT | 8 | Chứng minh độ khó, cần Gateway KD |\n",
    ]

    for gid in [1, 2, 3, 4]:
        output_parts.append(render_group(gid, OFFICIAL_PAPERS, rows, sections))

    output_text = "".join(output_parts)
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")

    print(f"\n[3] Đã lưu file kết quả: {OUTPUT_FILE}")
    print(f"    Kích thước: {len(output_text):,} bytes")
    print("\nHoàn thành! ✅")


if __name__ == "__main__":
    main()
