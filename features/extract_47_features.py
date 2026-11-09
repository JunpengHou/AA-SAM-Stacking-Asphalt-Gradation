#!/usr/bin/env python3
"""从二维实例标签 TIFF 批量计算稿件附录的 47 项特征。Python 3.10+。
安装: python -m pip install numpy scipy pillow openpyxl
运行: python extract_47_features.py --input masks --output features47.xlsx
每个 TIFF 是一块已经裁好的视场。默认整图为 150 × 150 mm，零为背景。
标签 ID 是实例身份：同 ID 即使分离也算一个实例；接触的不同 ID 分开计数。
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from openpyxl import Workbook
from openpyxl.cell import Cell
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# 以区间下限命名。分档等效圆直径 d；统一左闭右开，末档覆盖所有 d >= 16。
LOWER = (0, 2.36, 4.75, 9.5, 13.2, 16)
TAGS = tuple(f"{v:g}" for v in LOWER)
COARSE_MM = 4.75
EXTENSIONS = {".tif", ".tiff"}


def feature_dictionary():
    """元组字段为 特征名、类型、单位、定义。该顺序也是模型输入表的顺序。"""
    bins = [f"[{v:g}, {LOWER[i+1]:g}) mm" if i+1 < len(LOWER) else "[16, +∞) mm"
            for i, v in enumerate(LOWER)]
    rows = []
    def group(prefix, suffix, kind, unit, meaning):
        rows.extend((prefix+t+suffix, kind, unit, f"{b}，{meaning}")
                    for t, b in zip(TAGS, bins))
    group("N", "", "数量", "个", "实例数 Ni")
    rows += [("Nc","数量","个","d≥4.75 mm 的实例数"),
             ("Nf","数量","个","d<4.75 mm 的实例数"),
             ("N","数量","个","纳入分档的实例总数")]
    group("Rn", "", "数量", "比例", "Ni/N")
    rows.append(("Nedge","数量","个","接触图像四边任意一边的实例总数，按 ID 去重"))
    rows.append(("A","面积","mm²","纳入分档实例的可见投影面积之和"))
    group("A", "", "面积", "mm²", "该档实例投影面积之和 Ai")
    group("A", "_avg", "面积", "mm²", "该档平均实例投影面积 Ai/Ni")
    rows.append(("Aavg","面积","mm²","A/N"))
    group("Ra", "", "面积", "比例", "Ai/A")
    rows += [
        ("Ac","面积","mm²","粗集料可见投影总面积"), ("Ac_avg","面积","mm²","Ac/Nc"),
        ("Af","面积","mm²","细集料可见投影总面积"), ("Af_avg","面积","mm²","Af/Nf"),
        ("Racf","面积","比值","Ac/Af"), ("Ra","面积","比例","A/整幅视场面积"),
        ("Ras","面积","比值","A/(整幅视场面积−A)，分母为剩余投影区域"),
        ("Aedge","面积","mm²","触边实例在视场内的平均投影面积"),
        ("Redge","面积","比例","所有触边实例可见投影总面积/整幅视场面积"),
        ("Davg","位置","mm","纳入统计的全部不同实例对之间的质心欧氏距离均值"),
        ("Dmin","位置","mm","逐实例寻找最近其他实例的质心，取这些最近邻距离的均值")]
    return rows


DICTIONARY = feature_dictionary()
HEADERS = [row[0] for row in DICTIONARY]
assert len(HEADERS) == len(set(HEADERS)) == 47


def ratio(x, y):
    return float(x / y) if y else 0.0


def centroid_distances(points):
    n = len(points)
    if n < 2:
        return 0., 0.
    # 分块以限制距离矩阵占用，计算精确的所有实例对。自身不参与。
    block = max(1, min(n, 2_000_000 // n))
    total = nearest = 0.
    for start in range(0, n, block):
        stop = min(start + block, n)
        distances = cdist(points[start:stop], points)
        idx = np.arange(stop - start)
        distances[idx, start + idx] = 0
        total += distances.sum()
        distances[idx, start + idx] = np.inf
        nearest += distances.min(axis=1).sum()
    return float(total / (n*(n-1))), float(nearest / n)


def read_mask(path):
    with Image.open(path) as im:
        if im.n_frames != 1 or im.mode not in {"I", "L", "I;16", "I;16B", "I;16L", "I;16N"}:
            raise ValueError(f"需要单页二维整数实例标签，实际 mode={im.mode}, 页数={im.n_frames}")
        mask = np.asarray(im)
    if mask.ndim != 2 or mask.dtype.kind not in "iu" or mask.min() < 0:
        raise ValueError("标签必须是二维非负整数数组")
    return mask


def measure(mask, width_mm, height_mm, particles):
    h, w = mask.shape
    sx, sy = width_mm / w, height_mm / h
    pixel_area = sx * sy
    ys, xs = np.nonzero(mask)
    ids, inverse, pixels = np.unique(mask[ys, xs], return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    count = len(ids)
    cx = np.bincount(inverse, weights=xs, minlength=count) / pixels
    cy = np.bincount(inverse, weights=ys, minlength=count) / pixels
    centers = np.column_stack(((cx + 0.5)*sx, (cy + 0.5)*sy))
    edge_ids = np.unique(np.concatenate((mask[0], mask[-1], mask[:,0], mask[:,-1])))
    edge = np.isin(ids, edge_ids)
    ne = int(edge.sum())
    edge_pixels = int(pixels[edge].sum())
    keep = ~edge if particles == "interior" else np.ones(count, dtype=bool)
    px = pixels[keep]
    areas = px * pixel_area
    diameter = np.sqrt(4 * areas / np.pi)
    # 只对浮点舍入量级的差异吸附到分界；不按显示精度四舍五入再分档。
    for cut in LOWER[1:]:
        diameter[np.abs(diameter - cut) <= 8*np.spacing(float(cut))] = cut
    slot = np.searchsorted(LOWER, diameter, side="right") - 1
    numbers = np.bincount(slot, minlength=6)
    sums = np.bincount(slot, weights=areas, minlength=6)
    n = len(px)
    coarse = diameter >= COARSE_MM
    nc = int(coarse.sum())
    ac = float(areas[coarse].sum())
    af = float(areas[~coarse].sum())
    used_pixels = int(px.sum())
    a = used_pixels * pixel_area
    result = dict(Nc=nc, Nf=n-nc, N=n, Nedge=ne, A=a, Aavg=ratio(a,n),
                  Ac=ac, Ac_avg=ratio(ac,nc), Af=af, Af_avg=ratio(af,n-nc),
                  Racf=ratio(ac,af), Ra=ratio(used_pixels,w*h),
                  Ras=ratio(used_pixels,w*h-used_pixels),
                  Aedge=ratio(edge_pixels*pixel_area,ne), Redge=ratio(edge_pixels,w*h))
    for i, t in enumerate(TAGS):
        result["N"+t] = int(numbers[i])
        result["Rn"+t] = ratio(numbers[i],n)
        result["A"+t] = float(sums[i])
        result["A"+t+"_avg"] = ratio(sums[i],numbers[i])
        result["Ra"+t] = ratio(sums[i],a)
    result["Davg"], result["Dmin"] = centroid_distances(centers[keep])
    assert sum(numbers) == n and np.isclose(sums.sum(), a) and np.isclose(ac+af,a)
    qc = [w,h,sx,sy,count,n,ne,int(np.sum(diameter>19)),int((numbers==0).sum()),
          "空掩膜" if count==0 else ("无纳入实例" if n==0 else "正常")]
    return [result[k] for k in HEADERS], qc


def add_sheet(book, name, headers, widths):
    ws = book.create_sheet(name)
    ws.append(headers)
    for c in ws[1]:
        c.font = Font(color="FFFFFF", bold=True)
        c.fill = PatternFill("solid", fgColor="27536B")
    for i, width in enumerate(widths,1):
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = "B2"
    return ws


def push(sheet, row):
    # 文件名等文本强制按字面量保存；无需逐行扫描已有工作表。
    values = []
    for value in row:
        if isinstance(value, str):
            cell = Cell(sheet, value=value)
            cell.data_type = "s"
            values.append(cell)
        else:
            values.append(value)
    sheet.append(values)


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--input", type=Path, required=True, help="实例标签 TIFF 或其所在文件夹")
    cli.add_argument("--output", type=Path, default=Path("features47.xlsx"))
    cli.add_argument("--width-mm", type=float, default=150, help="单张掩膜覆盖的物理宽度")
    cli.add_argument("--height-mm", type=float, default=150, help="单张掩膜覆盖的物理高度")
    cli.add_argument("--particles", choices=["all","interior"], default="all",
                     help="all 纳入全部实例；interior 只用非触边实例计算主体指标。三项 edge 指标始终统计原图")
    cli.add_argument("--recursive", action="store_true", help="递归读取子文件夹，用相对路径区分同名文件")
    cli.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出工作簿")
    args = cli.parse_args()
    if not all(math.isfinite(v) and v>0 for v in [args.width_mm,args.height_mm]):
        cli.error("物理尺寸必须是有限正数")
    out = args.output
    if out.suffix.lower() != ".xlsx" or (out.exists() and not args.overwrite):
        cli.error("输出须为 .xlsx；覆盖已有文件需加 --overwrite")
    src = args.input
    if src.is_file() and src.suffix.lower() in EXTENSIONS:
        root, files = src.parent, [src]
    elif src.is_dir():
        root = src
        files = sorted(p for p in src.glob("**/*" if args.recursive else "*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    else:
        cli.error("找不到输入或输入并非 TIFF")
    if not files:
        cli.error("目录中没有 .tif/.tiff 文件")
    book = Workbook()
    book.remove(book.active)
    fs = add_sheet(book,"Features47",["image_id",*HEADERS],[44]+[15]*47)
    qs = add_sheet(book,"QC",["image_id","width_px","height_px","mm_per_px_x","mm_per_px_y","原始实例数","纳入数","触边数","纳入且d>19数","空粒径档数","状态"],[44]+[19]*10)
    ds = add_sheet(book,"Dictionary",["特征名","类别","单位","定义"],[20,14,12,72])
    for row in DICTIONARY:
        ds.append(row)
    settings = add_sheet(book,"Settings",["项目","使用口径"],[24,110])
    for row in [
        ("物理视场",f"{args.width_mm:g} × {args.height_mm:g} mm，按每张图实际宽高像素标定"),
        ("图像单位","每张 TIFF 对应一条记录；视场由用户指定，不自动切图。"),
        ("标签","零为背景，每个不同正整数 ID 为一个实例，同 ID 不拆连通域。"),
        ("粒径","d = sqrt(4 × 可见投影面积_mm² / π)，按区间下限命名，末档包括 >19 mm。"),
        ("粗细分界","d≥4.75 mm 为粗，d<4.75 mm 为细。"),
        ("边界策略",args.particles+"；接触任意一条外边即为触边，按可见像素面积计量。"),
        ("空间距离","基于物理坐标质心的欧氏距离，自身排除；每个实例权重相同。"),
        ("未定义值","空档均值、零分母比值以及实例不足两个时的距离均赋 0。合法空掩膜保留。"),
        ("比例","Rn、Ra 类比例输出小数，不乘 100。Racf、Ras 可以超过 1。"),
        ("失败处理","读取失败的文件列于 Errors，不以零特征冒充空掩膜；有失败时进程返回 1。")]:
        settings.append(row)
    errors = add_sheet(book,"Errors",["image_id","原因"],[44,110])
    failed = 0
    for path in files:
        key = path.relative_to(root).as_posix()
        try:
            vals, qc = measure(read_mask(path),args.width_mm,args.height_mm,args.particles)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            push(errors,[key,str(exc)])
            failed += 1
            print(f"跳过 {key}: {exc}",file=sys.stderr)
            continue
        push(fs,[key,*vals])
        push(qs,[key,*qc])
    for sheet in [fs,qs,ds,errors]:
        sheet.auto_filter.ref = sheet.dimensions
    out.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.NamedTemporaryFile(dir=out.parent,suffix=".xlsx",delete=False) as f:
        temp = Path(f.name)
    try:
        book.save(temp)
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{out}: 成功 {len(files)-failed} 张，失败 {failed} 张，模型特征 47 列。")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
