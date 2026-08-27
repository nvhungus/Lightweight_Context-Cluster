"""
Convert HBCC Vietnamese LaTeX paper to IEEE A4 Word format.
Usage: python scripts/convert_vi_to_docx.py
Output: Report_LaTeX/HBCC_vi_A4.docx

Layout:
  Section 1 (1-col, continuous): Title · Authors · Abstract · Keywords
  Section 2 (2-col): Body (Sections I–V + References)
  Pipeline figure (Hình 1) wrapped in a 1-col mini-section so it spans both columns.

After opening in Word:
  - Replace each [EQ N: ...] placeholder with Insert → Equation
  - Bold the best-result cells in each table
  - Check the empty switching paragraphs around Hình 1; delete if unwanted
"""
from pathlib import Path
import zipfile, os, tempfile
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT   = Path(__file__).parent.parent / 'Report_LaTeX'
IMAGES = ROOT / 'images'
OUTPUT = ROOT / 'HBCC_vi_A4.docx'

# ── Convert template from ISO Strict → OOXML Transitional namespaces ─────────
_NS_MAP = [
    (b'http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument',
     b'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument'),
    (b'http://purl.oclc.org/ooxml/officeDocument/relationships',
     b'http://schemas.openxmlformats.org/officeDocument/2006/relationships'),
    (b'http://purl.oclc.org/ooxml/wordprocessingml/main',
     b'http://schemas.openxmlformats.org/wordprocessingml/2006/main'),
    (b'http://purl.oclc.org/ooxml/officeDocument/math',
     b'http://schemas.openxmlformats.org/officeDocument/2006/math'),
    (b'http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing',
     b'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'),
    (b'http://purl.oclc.org/ooxml/drawingml/2006/main',
     b'http://schemas.openxmlformats.org/drawingml/2006/main'),
    (b' w:conformance="strict"', b''),
]

_tmp_dir   = tempfile.mkdtemp()
_fixed_tpl = os.path.join(_tmp_dir, 'template_fixed.docx')
with zipfile.ZipFile(str(ROOT / 'conference-template-a4.docx'), 'r') as zin, \
     zipfile.ZipFile(_fixed_tpl, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith('.xml') or item.filename.endswith('.rels'):
            for old, new in _NS_MAP:
                data = data.replace(old, new)
        zout.writestr(item, data)

doc = Document(_fixed_tpl)

# ── Clear body content, keep sectPr (page/margin settings) ───────────────────
_body = doc.element.body
for child in list(_body):
    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
    if tag != 'sectPr':
        _body.remove(child)

# ── Set body sectPr to 2 columns (the default for the main body) ─────────────
_body_sectPr = _body.find(qn('w:sectPr'))
for t in _body_sectPr.findall(qn('w:type')):   # remove 'continuous' from final section
    _body_sectPr.remove(t)
_cols = _body_sectPr.find(qn('w:cols'))
if _cols is None:
    _cols = OxmlElement('w:cols')
    _body_sectPr.append(_cols)
_cols.set(qn('w:num'), '2')          # 2 columns
# w:space is already '36pt' from the template

# ── Helpers ───────────────────────────────────────────────────────────────────
STYLE_FALLBACK = {
    'papertitle': 'Title', 'Author': 'Normal', 'Affiliation': 'Normal',
    'Abstract': 'Normal', 'Keywords': 'Normal',
    'Heading1': 'Heading 1', 'Heading2': 'Heading 2',
    'Heading3': 'Heading 3', 'Heading4': 'Heading 4',
    'BodyText': 'Normal', 'bulletlist': 'List Bullet',
    'equation': 'Normal', 'figurecaption': 'Caption',
    'references': 'Normal', 'tablehead': 'Normal',
}

def _style(name):
    try:
        doc.styles[name]; return name
    except KeyError:
        return STYLE_FALLBACK.get(name, 'Normal')

def para(text='', style='BodyText', bold=False, italic=False, center=False):
    s = _style(style)
    pr = doc.add_paragraph(style=s)
    if text:
        run = pr.add_run(text)
        run.bold = bold
        run.italic = italic
    if center:
        pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return pr

def heading(level, text):
    return para(text, style=['Heading1','Heading2','Heading3','Heading4'][level-1])

def equation(label, latex):
    pr = doc.add_paragraph(style=_style('equation'))
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pr.add_run(f'[EQ {label}:  {latex}  ]').italic = True
    return pr

def bullet(text):
    return para(text, style='bulletlist')

def tbl_caption(text):
    return para(text, style='tablehead')

def make_table(headers, rows):
    tbl = doc.add_table(rows=1, cols=len(headers))
    try: tbl.style = 'Table Grid'
    except Exception: pass
    for i, h in enumerate(headers):
        tbl.rows[0].cells[i].text = h
        tbl.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for rv in rows:
        row = tbl.add_row()
        for i, v in enumerate(rv):
            row.cells[i].text = str(v)
    return tbl

# ── Section-layout helpers ────────────────────────────────────────────────────
def _make_col_sectPr(num_cols):
    """Return a minimal sectPr element for a continuous section break."""
    sp = OxmlElement('w:sectPr')
    typ = OxmlElement('w:type')
    typ.set(qn('w:val'), 'continuous')
    sp.append(typ)
    cols = OxmlElement('w:cols')
    if num_cols > 1:
        cols.set(qn('w:num'), str(num_cols))
    cols.set(qn('w:space'), '36pt')
    sp.append(cols)
    return sp

def embed_section_end(para_obj, num_cols):
    """Embed a sectPr in para_obj's pPr to end a section of num_cols columns."""
    pPr = para_obj._p.get_or_add_pPr()
    pPr.append(_make_col_sectPr(num_cols))
    return para_obj

def col_switch(num_cols):
    """Insert an empty paragraph that acts as a column-count transition point."""
    pr = doc.add_paragraph(style=_style('BodyText'))
    pPr = pr._p.get_or_add_pPr()
    sp = OxmlElement('w:spacing')
    sp.set(qn('w:before'), '0'); sp.set(qn('w:after'), '0')
    pPr.append(sp)
    pPr.append(_make_col_sectPr(num_cols))
    return pr

# Figure: full-width (spanning) vs column-width
def figure(fname, caption, width_in=3.1, full_width=False):
    """Insert figure and caption. full_width=True wraps in 1-col section.

    Sectioning around a spanning figure:
      col_switch(2) ends the preceding 2-col section  ← before figure
      col_switch(1) ends the 1-col figure section     ← after caption
    The remainder of the document falls into Section N+1, which is 2-col
    (defined by the body's final sectPr).
    """
    if full_width:
        col_switch(2)          # ends the preceding 2-col section
    path = IMAGES / fname
    pr = doc.add_paragraph(style=_style('BodyText'))
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        pr.add_run().add_picture(str(path), width=Inches(width_in))
    else:
        pr.add_run(f'[Figure not found: {fname}]').italic = True
    cap = doc.add_paragraph(caption, style=_style('figurecaption'))
    if full_width:
        col_switch(1)          # ends the 1-col figure section → next section is 2-col
    return pr

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 (1-col): TITLE · AUTHORS · ABSTRACT · KEYWORDS
# ═════════════════════════════════════════════════════════════════════════════
para('HBCC: Kiến trúc lai cục bộ–phân cụm cho phân loại ảnh nhẹ',
     style='papertitle')

para('Nguyễn Việt Hùng, Nguyễn Bá Nam, Lê Hoàng Thái*', style='Author')
para('Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, '
     'Đại học Quốc gia Thành phố Hồ Chí Minh, Thành phố Hồ Chí Minh, Việt Nam',
     style='Affiliation')
para('{23122032, 23122043}@student.hcmus.edu.vn  |  lhthai@fit.hcmus.edu.vn',
     style='Affiliation')
para('*Tác giả liên hệ', style='Affiliation')

para(
    'Tóm tắt — Phân loại ảnh trên thiết bị biên đòi hỏi sự cân bằng giữa độ chính xác '
    'và chi phí tính toán. Các CNN nhẹ giảm chi phí nhưng có thể hạn chế khả năng biểu '
    'diễn, trong khi Context Cluster (CoC) tổng hợp ngữ cảnh bằng phân cụm nhưng bỏ qua '
    'kết cấu cục bộ và có nguy cơ suy biến token trên ảnh nhỏ. Bài báo đề xuất HBCC '
    '(Hybrid Block Context Cluster), kiến trúc lai bốn tầng sử dụng LBPConv và DWConv '
    'ở độ phân giải cao, sau đó chuyển sang phân cụm thuần. Thiết kế stem, độ sâu từng '
    'tầng và fold partitioning được điều chỉnh để hạn chế suy biến token; chưng cất tri '
    'thức từ ResNet-18 được dùng để hỗ trợ huấn luyện. Trên CIFAR-10, HBCC-Medium+KD đạt '
    '95,36% top-1 với 1,76M tham số và 138,8M MACs. Trên Food-101 ở độ phân giải 224×224, '
    'biến thể HBCC-2.5M huấn luyện từ đầu đạt 82,53% top-1 và 95,33% top-5; mô hình chỉ '
    'kém ResNet-18 lần lượt 0,38% và 0,16% trong khi sử dụng ít tham số hơn 4,38 lần. '
    'Trên Tiny-ImageNet-200 (200 lớp), HBCC-Medium+KD đạt 58,90%, vượt cả ResNet-18 giáo '
    'viên (58,58%) với 6,26 lần ít tham số hơn. Phân tích toán tử cho thấy sự phân mảnh '
    'kernel vẫn là nút thắt khiến mức giảm MACs chưa chuyển hóa hoàn toàn thành cải thiện '
    'độ trễ thực đo.',
    style='Abstract')

kw = para(
    'Từ khóa — phân loại ảnh, Context Cluster, kiến trúc lai cục bộ–phân cụm, mạng nơ-ron '
    'nhẹ, chưng cất tri thức, suy biến token, CIFAR-10, Food-101, Tiny-ImageNet-200.',
    style='Keywords')
# ← End of 1-col header section: embed sectPr{1-col, continuous} into keywords para
embed_section_end(kw, 1)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 (2-col): BODY — SECTIONS I–V + REFERENCES
# ═════════════════════════════════════════════════════════════════════════════

# ── I. GIỚI THIỆU ─────────────────────────────────────────────────────────────
heading(1, 'I. Giới thiệu')

para(
    'Phân loại ảnh trên thiết bị biên đòi hỏi sự cân bằng giữa độ chính xác và chi phí '
    'tính toán. Các mô hình CNN sâu [1] và Vision Transformer [2] có khả năng biểu diễn '
    'mạnh nhưng thường vượt quá giới hạn tài nguyên của nền tảng nhúng. MobileNetV2 [3] '
    'và ShuffleNetV2 [4] giảm chi phí tính toán, song việc tinh giản kiến trúc có thể '
    'hạn chế khả năng biểu diễn đặc trưng.')

para(
    'Context Cluster (CoC) [5] cung cấp một hướng tiếp cận khác bằng cách biểu diễn ảnh '
    'như tập hợp điểm và tổng hợp ngữ cảnh thông qua phân cụm. Tuy nhiên, phân cụm thuần '
    'túy bỏ qua kết cấu cục bộ ở độ phân giải cao. Trên ảnh nhỏ, việc giảm độ phân giải '
    'quá nhanh còn có thể gây suy biến token, khiến số điểm trong mỗi cụm không đủ để '
    'phép tổng hợp ngữ cảnh phát huy tác dụng.')

para(
    'Nghiên cứu này hướng tới xây dựng một kiến trúc nhẹ kết hợp xử lý cục bộ với phân '
    'cụm ngữ cảnh, đồng thời khảo sát cách phân bổ hai cơ chế theo độ phân giải. Nghiên '
    'cứu đánh giá cả ảnh nhỏ (32×32) và ảnh độ phân giải chuẩn (224×224), với ngân sách '
    'không quá 3M tham số cho HBCC, thông qua hai câu hỏi:')

para(
    '(CQ1) Liệu kiến trúc lai kết hợp xử lý cục bộ và phân cụm ngữ cảnh có thể cải thiện '
    'đồng thời độ chính xác và hiệu quả tính toán so với cả CNN nhẹ thuần túy lẫn mô hình '
    'thuần phân cụm?')

para(
    '(CQ2) Chiến lược phân bổ nhánh cục bộ, fold và độ rộng theo tầng có còn hiệu quả khi '
    'chuyển từ ảnh nhỏ 32×32 sang đầu vào 224×224 hay không?')

para(
    'Để trả lời các câu hỏi trên, bài báo đề xuất HBCC (Hybrid Block Context Cluster), '
    'kiến trúc bốn tầng sử dụng khối lai ở các tầng có độ phân giải cao và phân cụm thuần '
    'ở các tầng sau. Thiết kế stem và độ sâu từng tầng được điều chỉnh để hạn chế suy biến '
    'token; chưng cất tri thức được sử dụng trong huấn luyện mà không làm tăng chi phí suy '
    'luận của mô hình học sinh. Đối với Food-101, kiến trúc được mở rộng có kiểm soát lên '
    '2,57M tham số, dùng stem stride 4 và các vùng phân cụm đồng nhất 7×7 để bảo toàn cấu '
    'trúc không gian trên đầu vào 224×224.')

para(
    'Bố cục: Phần II tổng quan các nghiên cứu liên quan; Phần III trình bày phương pháp '
    'đề xuất; Phần IV mô tả thiết lập, kết quả và phân tích thực nghiệm; Phần V thảo luận '
    'câu hỏi nghiên cứu, các hạn chế và kết luận.',
    italic=True)

# ── II. CÁC NGHIÊN CỨU LIÊN QUAN ──────────────────────────────────────────────
heading(1, 'II. Các nghiên cứu liên quan')

heading(2, 'A. Mạng CNN và residual learning')

para(
    'ResNet [1] giới thiệu residual connection y = F(x) + x, cho phép gradient truyền '
    'thẳng qua lớp, ngăn vanishing gradient và mở ra kỷ nguyên huấn luyện mạng 50–152 lớp. '
    'Standard 3×3 convolution với C_in kênh vào, C_out kênh ra, kernel size k có chi phí '
    'tỷ lệ với C_in · k² · C_out, làm ResNet-18 cần 556,7M MACs dù ảnh chỉ 32×32.')

para(
    'MobileNetV2 [3] thay standard conv bằng depthwise separable conv, giảm tổng chi phí '
    'về O(C_in(k² + C_out)) — tiết kiệm xấp xỉ 1/k² lần. ShuffleNetV2 [4] chỉ ra rằng '
    'FLOPs không tương quan trực tiếp với tốc độ thực đo và đề xuất bốn nguyên tắc: '
    '(i) cân bằng số channel đầu vào/ra; (ii) tránh group convolution; (iii) tránh phân '
    'nhánh kiến trúc; (iv) giảm element-wise operations.')

heading(2, 'B. Vision Transformer và MetaFormer')

para(
    'ViT [2] chia ảnh thành n patch và áp dụng multi-head self-attention với độ phức tạp '
    'O(n²d_k). Trên ảnh 32×32 với patch 2×2, n = 256 token; chi phí O(256²) là quá lớn '
    'cho bài toán lightweight. Trong khi đó, phân cụm với O(n·K), K = 4 có chi phí trộn '
    'token lý thuyết thấp hơn xấp xỉ 64× so với single-head attention. Các kiến trúc '
    'MetaFormer cho thấy cấu trúc hai khối (trộn token + MLP) của Transformer quan trọng '
    'không kém toán tử cụ thể.')
equation('1', 'Attn(Q,K,V) = softmax( QKᵀ / √d_k ) · V')

heading(2, 'C. Context Cluster')

para(
    'CoC [5] biểu diễn ảnh như tập hợp điểm 5D (R,G,B,x,y), tích hợp tọa độ không gian '
    'vào đặc trưng. Cluster centers được tạo tự động qua AdaptiveAvgPool (không tham số '
    'học); điểm được gán cứng vào cluster gần nhất qua cosine similarity với chi phí '
    'O(N·M·D). Hạn chế chính: pure clustering ở mọi tầng bỏ qua thông tin tần số cao '
    '(kết cấu vi mô, cạnh), làm CoC-baseline chỉ đạt 88,34% trên CIFAR-10. Ngoài ra, '
    'suy biến token xảy ra khi bản đồ đặc trưng quá nhỏ so với số tâm cụm (phân tích '
    'chi tiết ở Mục III-G).')

heading(2, 'D. Local Binary Pattern')

para('LBP [6] mã hóa kết cấu tại pixel (x_c, y_c):')
equation('2',
    'LBP_{P,R}(x_c,y_c) = Σ_{p=0}^{P-1} s(g_p − g_c)·2^p,   '
    's(u) = 1 nếu u ≥ 0,  0 nếu u < 0')
para(
    'trong đó g_c là cường độ pixel trung tâm, g_p là pixel thứ p trên vòng tròn bán kính '
    'R với P điểm mẫu đều nhau. LBPConv áp dụng công thức này trên từng channel với bộ '
    'lọc cố định (không học tham số tích chập), phù hợp để trích xuất kết cấu vi mô trên '
    'các bản đồ đặc trưng có độ phân giải cao.')

heading(2, 'E. Chưng cất tri thức')

para(
    'Hinton và cộng sự [7] chỉ ra rằng một mô hình nhỏ (học sinh) học tốt hơn khi bắt '
    'chước phân phối xác suất "mềm" của một mô hình lớn (giáo viên). Hàm mất mát KD tổ '
    'hợp cross-entropy với nhãn cứng và KL-divergence với đầu ra giáo viên đã làm mềm bởi '
    'nhiệt độ T (chi tiết ở Mục IV-A-4). Phân phối mềm truyền thêm thông tin về quan hệ '
    'liên lớp, đặc biệt hữu ích trong bài toán phân loại nhiều lớp.')

para(
    'Nhìn chung, CNN nhẹ khai thác tốt tính cục bộ nhưng còn đánh đổi độ chính xác; '
    'self-attention nắm bắt quan hệ toàn cục nhưng có chi phí bậc hai theo số token; còn '
    'CoC giảm độ phức tạp song bỏ qua kết cấu vi mô và có nguy cơ suy biến token trên '
    'ảnh nhỏ. Khoảng trống đó là cơ sở cho phương pháp được đề xuất trong phần tiếp theo.')

# ── III. PHƯƠNG PHÁP ĐỀ XUẤT ──────────────────────────────────────────────────
heading(1, 'III. Phương pháp đề xuất')

heading(2, 'A. Tổng quan pipeline')

para(
    'Hình 1 trình bày toàn bộ pipeline. Input ảnh B×3×32×32 được bổ sung tọa độ '
    '(Mục III-B) và đưa qua Stem convolution (stride 1), tạo feature map 32×32×C₁. '
    'Bốn tầng liên tiếp thực hiện phân tích đặc trưng ở độ phân giải giảm dần; giữa '
    'các tầng là PointReducer (conv 1×1, stride 2) để downsample và tăng số channel. '
    'Cuối cùng, Global Average Pooling (GAP) và lớp Linear cho ra vector phân loại '
    'C-chiều (C = 10 hoặc 200).')

# full_width=True: wraps the pipeline figure in a 1-col section so it spans both columns
figure('fig_hbcc_pipeline.png',
       'Hình 1. Pipeline 4-tầng của HBCC. Màu xanh: Hybrid stage (nhánh cục bộ + cluster). '
       'Màu đỏ: Cluster-only stage. Độ phân giải spatial và số channel Cᵢ ghi chú tại '
       'mỗi tầng.',
       width_in=6.5, full_width=True)

heading(2, 'B. Bổ sung tọa độ (x, y)')

para(
    'Cosine similarity trên RGB thuần túy không phân biệt hai điểm cùng màu ở vị trí '
    'khác nhau. Chúng tôi bổ sung hai kênh tọa độ chuẩn hóa về [0,1]:')
equation('3',
    'X = [I ; P_x ; P_y] ∈ ℝ^{B×5×H₀×W₀},  '
    'P_x[i,j] = j/(W₀−1),  P_y[i,j] = i/(H₀−1)')
para('Bổ sung tọa độ giúp cluster centers nhận biết vị trí không gian mà không cần '
     'positional encoding tường minh.')

heading(2, 'C. Cơ chế Context Cluster')

para('Cho feature tensor X ∈ ℝ^{B×C×H×W}, ContextClusterOp thực hiện sáu bước '
     '(trên mỗi head độc lập rồi nối lại):')

para('Bước 1 — Chiếu đôi:', bold=True)
equation('4', 'F = Conv_{1×1}(X),   V = Conv_{1×1}(X)')
para('F dùng để so sánh similarity; V là giá trị cần aggregate.')

para('Bước 2 — Đề xuất cluster centers:', bold=True)
equation('5', 'F_c = AvgPool_{Kh×Kw}(F),   V_c = AvgPool_{Kh×Kw}(V)')
para('Với (K_h, K_w) = (2,2), mỗi tile cho K = 4 cluster centers. AvgPool được chọn '
     '(thay vì MaxPool) vì tâm cụm phải là centroid của toàn bộ vùng, không phải cực trị.')

para('Bước 3 — Tính similarity và gán cứng:', bold=True)
equation('6', 's_{ij} = σ( α · (f_{ci}·f_j) / (‖f_{ci}‖·‖f_j‖) + β )')
equation('7', 'm_{ij} = 1[ i = argmax_k s_{kj} ]')
para('với α, β ∈ ℝ là tham số học được. Hard assignment (m_{ij}) đảm bảo mỗi điểm chỉ '
     'đóng góp vào một cluster.')

para('Bước 4 — Aggregate (trung bình có trọng số):', bold=True)
equation('8',
    'g_i = ( v_{ci} + Σ_j m_{ij} s_{ij} v_j ) / ( 1 + Σ_j m_{ij} s_{ij} )')
para('Mẫu số 1 + Σs_{ij} chuẩn hóa bao gồm tự-trọng số của center, đảm bảo giá trị '
     'đầu ra ổn định kể cả khi cluster rỗng.')

para('Bước 5 — Dispatch:', bold=True)
equation('9', 'ṽ_j = Σ_i g_i · m_{ij} · s_{ij}')
para('Thông tin tổng hợp được phân phối ngược về mỗi điểm theo trọng số.')

para('Bước 6 — Chiếu đầu ra:  Y = Conv_{1×1}(Ṽ).')

heading(2, 'D. Fold partitioning')

para('Ở resolution cao (32×32, 16×16), phân cụm toàn cục tạo cluster quá lớn, làm mờ '
     'chi tiết cục bộ (global over-smoothing). Chúng tôi chia feature map thành '
     'F_h × F_w tile không chồng lấp:')
equation('10', '(B, C, F_h·h, F_w·w)  →  (B·F_h·F_w, C, h, w)')
para('CoC áp dụng độc lập trong từng tile rồi reshape ngược lại. Fold thu nhỏ theo tầng: '
     '4×4 (Tầng 1, 32×32), 2×2 (Tầng 2, 16×16), 1×1 (Tầng 3–4, toàn cục).')

heading(2, 'E. Hybrid Cluster Block')

para('Phân cụm ngữ cảnh và tích chập cục bộ có thế mạnh bổ sung nhau: phân cụm nắm bắt '
     'quan hệ vùng toàn cục nhưng bỏ qua kết cấu vi mô, trong khi tích chập cục bộ nắm '
     'tốt thông tin tần số cao nhưng bị giới hạn bởi receptive field nhỏ.')

figure('fig_hybrid_block.png',
       'Hình 2. Cấu trúc Hybrid Cluster Block. Nhánh xanh: CoC '
       '(C_c = C(1−ρ) channel). Nhánh cam: Local branch (C_l = Cρ channel). '
       'Fuse qua channel shuffle và 1×1 conv. Layer-scale γ khởi tạo 10⁻⁵.',
       width_in=3.1)

para('Với local ratio ρ ∈ [0,1]: nhánh cục bộ xử lý C_l = ⌊Cρ⌉ channel; nhánh cluster '
     'xử lý C_c = C − C_l channel. Tại Tầng 1–2, ρ = 0,5 để phân bổ đều băng thông; '
     'tại Tầng 3–4, ρ = 0 để chuyển hoàn toàn sang phân cụm.')

para('Token mixing:', bold=True)
equation('11', '[x_c ; x_l] = split( BN(x), [C_c, C_l] )')
equation('12', 'y = Fuse_{1×1}( Shuffle[ CoC(x_c) ; Local(x_l) ] )')
equation('13', 'x ← x + DropPath( γ₁ ⊙ y )')

para('MLP:', bold=True)
equation('14', 'x ← x + DropPath( γ₂ ⊙ MLP( BN(x) ) )')
para('trong đó γ₁, γ₂ ∈ ℝ^C là layer-scale khởi tạo tại 10⁻⁵. Channel Shuffle [4] '
     'trao đổi thông tin giữa hai nhóm kênh trước khi fuse bằng 1×1 convolution. '
     'Khi ρ = 0 (Tầng 3–4), nhánh cục bộ, Shuffle, và Fuse bị bỏ qua; block trở thành '
     'CoC block thuần túy.')

heading(2, 'F. Nhánh cục bộ')

para('LBPConv (Tầng 1, 32×32).', bold=True)
para('Ở Tầng 1, bản đồ đặc trưng vẫn ở độ phân giải đầy đủ 32×32, nơi thông tin kết cấu '
     'vi mô còn phong phú nhất. Chúng tôi chọn LBPConv thay vì tích chập học tham số '
     '(3×3 conv) vì ba lý do:')
para('(i) Chi phí gần bằng không: bộ lọc LBP cố định chỉ gồm phép trừ và ngưỡng nhị '
     'phân, hầu như không đóng góp vào tổng MACs.', italic=True)
para('(ii) Không cạnh tranh tham số với nhánh cluster: LBPConv không có tham số tích chập '
     'học được, nên hai nhánh không tranh giành dung lượng tham số.', italic=True)
para('(iii) Tương thích tự nhiên với AvgPool: LBPConv tạo ra bản đồ kích hoạt nhị phân; '
     'AvgPool cho ra histogram mềm phản ánh phân phối mẫu kết cấu trong vùng.', italic=True)

para('DWConv (Tầng 2, 16×16).', bold=True)
para('Ở Tầng 2 (16×16), sau một lần downsample, mô hình cần thêm receptive field rộng hơn '
     'để liên kết các vùng lân cận. Depthwise convolution 3×3 xử lý từng channel độc lập '
     'với chi phí O(k²C_in) thay vì O(k²C_in·C_out) — giảm chi phí xấp xỉ 1/C_out lần.')

para('Pure Cluster (Tầng 3–4, 8×8, 4×4).', bold=True)
para('Ở các tầng sau, đặc trưng đã đủ trừu tượng hóa — quan hệ ngữ nghĩa chiếm ưu thế. '
     'Phân cụm toàn cục (fold (1,1)) khai thác các mối quan hệ này hiệu quả hơn tích chập '
     'cục bộ. Chuyển hoàn toàn sang cluster cũng loại bỏ chi phí nhánh cục bộ và bước '
     'channel shuffle (ρ = 0).')

heading(2, 'G. Phân tích suy biến token và lựa chọn stem_stride')

para('Số điểm trên mỗi cụm trong một vùng fold là:')
equation('15', 'r = ( H/F_h · W/F_w ) / ( K_h · K_w )')
para('Để phân cụm có ý nghĩa, r ≥ 4 là cần thiết. Với stem_stride = 2 trên ảnh 32×32:')
bullet('Tầng 4: 2×2 = 4 điểm, proposals (2,2) tạo 4 tâm → r = 1 — mỗi cụm chứa đúng '
       'một điểm, toán tử phân cụm là hằng đồng nhất, không học được.')
bullet('Tầng 3: 4×4 = 16 điểm, fold(1,1), proposals(2,2) → r = 4 — chỉ đủ ngưỡng '
       'tối thiểu.')
para('Giải pháp: giảm stem_stride xuống 1, khiến sau stem còn 32×32 = 1024 điểm. '
     'Kết hợp với fold theo từng tầng, r = 16 đạt được nhất quán:')
bullet('Tầng 1: fold(4,4) trên 32×32: 8×8 = 64 điểm/vùng, 4 tâm → r = 16.')
bullet('Tầng 2: fold(2,2) trên 16×16: 8×8 = 64 điểm/vùng, 4 tâm → r = 16.')
bullet('Tầng 3: fold(1,1) trên 8×8: 64 điểm, 4 tâm → r = 16.')
bullet('Tầng 4: fold(1,1), proposals(1,1) trên 4×4: 16 điểm, 1 tâm → r = 16.')
para('Để bù lại chi phí tính toán tăng (diện tích feature map gấp 4× ở mỗi tầng), số '
     'khối mỗi tầng được thu hẹp từ [2,2,4,2] xuống [1,1,2,1]. Cấu hình cuối (Bảng I) '
     'giữ tổng chi phí ở mức hợp lý trong khi đảm bảo toán tử phân cụm hoạt động đúng.')

heading(2, 'H. Cấu hình tầng')

para('Bảng I liệt kê cấu hình dùng cho CIFAR-10 và Tiny-ImageNet-200. MLP expansion '
     'ratio: 3,0. Proposal size (K_h, K_w) = (2,2) cho tất cả tầng, ngoại trừ Tầng 4 '
     'dùng (1,1) để global context.')

tbl_caption('Bảng I. Cấu hình tầng HBCC-Small (S) và HBCC-Medium (M) cho đầu vào 32×32. '
            'd: số block; C: embed dim; ρ: local ratio.')
make_table(
    headers=['Tầng', 'Res.', 'Chế độ', 'Cục bộ', 'ρ', 'Fold', 'd_{S/M}', 'C_S / C_M'],
    rows=[
        ('1', '32²', 'Hybrid',  'LBPConv', '0,5', '4×4', '1/1', '48/64'),
        ('2', '16²', 'Hybrid',  'DWConv',  '0,5', '2×2', '1/1', '80/96'),
        ('3', '8²',  'Cluster', '—',       '0,0', '1×1', '2/2', '160/192'),
        ('4', '4²',  'Cluster', '—',       '0,0', '1×1', '1/1', '224/256'),
    ])

heading(2, 'I. Biến thể HBCC-2.5M cho Food-101')

para('Việc áp dụng trực tiếp cấu hình ảnh nhỏ cho đầu vào 224×224 làm số token ở tầng '
     'đầu tăng mạnh và khiến chi phí phân cụm không còn phù hợp. Vì vậy, biến thể Food-101 '
     'sử dụng stem 7×7, stride 4, padding 3 để tạo feature map 56×56, sau đó giảm kích '
     'thước bằng các point reducer 3×3, stride 2. Bốn fold lần lượt là 8×8, 4×4, 2×2 và '
     '1×1; do đó mỗi tile ở mọi tầng đều chứa vùng không gian 7×7.')
para('Mỗi tile dùng proposal 2×2 (bốn tâm), cosine similarity và hard assignment ở cả '
     'bốn tầng. Tọa độ (x,y) được ghép vào đầu vào, GroupNorm thay BatchNorm để giảm '
     'phụ thuộc vào thống kê batch. Tầng 1–2 là hybrid; Tầng 3 là cluster thuần; Tầng 4 '
     'đưa trở lại 25% kênh DWConv nhằm bù thông tin cục bộ ở biểu diễn ngữ nghĩa cuối. '
     'Cấu hình cuối gồm bốn tầng (C = 48/80/160/256, d = 2/2/3/2, fold 8²/4²/2²/1²) '
     'có tổng 2.566.391 tham số, trong đó 2.562.935 tham số có thể học.')

# ── IV. THÍ NGHIỆM ────────────────────────────────────────────────────────────
heading(1, 'IV. Thí nghiệm')

heading(2, 'A. Thiết lập thực nghiệm')

heading(3, '1. Bộ dữ liệu và phân chia dữ liệu')

para('Chúng tôi đánh giá trên ba bộ dữ liệu. CIFAR-10 [8] gồm 10 lớp và 60.000 ảnh '
     '32×32 (45.000 train / 5.000 val / 10.000 test). Food-101 [9] gồm 101.000 ảnh thuộc '
     '101 lớp món ăn. Để thiết lập tập xác thực mà không sử dụng thông tin từ tập kiểm '
     'tra, chúng tôi lấy mẫu phân tầng 75 ảnh từ mỗi lớp của tập huấn luyện chính thức, '
     'tạo ra 68.175 ảnh train / 7.575 val / 25.250 test, không có phần tử trùng lặp. '
     'Tiny-ImageNet-200 [10] gồm 200 lớp với 100.000 ảnh 64×64 '
     '(90.000 train / 10.000 val / 10.000 test); ảnh được resize về 32×32 để tương thích '
     'với các cấu hình ảnh nhỏ. Toàn bộ phân chia dữ liệu cố định với random seed 42.')

heading(3, '2. Công thức huấn luyện cho ảnh nhỏ')

tbl_caption('Bảng II. Recipe dùng chung trên CIFAR-10 và Tiny-ImageNet-200.')
make_table(
    headers=['Siêu tham số', 'Giá trị'],
    rows=[
        ('Optimizer',                'AdamW'),
        ('Learning rate',            '1×10⁻³'),
        ('Weight decay',             '0,05'),
        ('LR schedule',              'Cosine annealing'),
        ('Warmup epochs',            '5'),
        ('Total epochs (CE)',         '300'),
        ('Total epochs (KD)',         '250'),
        ('Label smoothing',           '0,1'),
        ('RandAugment [11]',          'N = 2, M = 9'),
        ('MixUp [12] α',              '0,2'),
        ('CutMix [13] α, p',          '1,0; 0,5'),
        ('Random Erasing [14] p',     '0,25'),
        ('AMP (mixed precision)',      'Có'),
        ('Drop path (Small / Med.)', '0,05 / 0,08'),
    ])

para('Trên CIFAR-10 và Tiny-ImageNet-200, tất cả mô hình — HBCC-Small, HBCC-Medium và '
     'toàn bộ baseline (ResNet-18, MobileNetV2, ShuffleNetV2, CoC-baseline) — được huấn '
     'luyện với cùng recipe trong Bảng II. Drop path là siêu tham số duy nhất chỉ áp dụng '
     'cho HBCC (0,05 với Small, 0,08 với Medium); các baseline không dùng drop path.')

heading(3, '3. Giao thức huấn luyện Food-101')

para('Tất cả ảnh được đưa về 224×224. Trong huấn luyện, chúng tôi sử dụng cắt ảnh ngẫu '
     'nhiên có thay đổi kích thước, lật ngang, RandAugment, xóa ngẫu nhiên và chuẩn hóa '
     'ImageNet. Khi xác thực và kiểm tra, cạnh ngắn được đổi kích thước thành 249 pixel '
     'bằng nội suy bicubic, sau đó lấy vùng trung tâm 224×224 và chuẩn hóa.')
para('Cả hai mô hình được khởi tạo ngẫu nhiên và huấn luyện trong 200 epoch (5 epoch '
     'warmup, lịch cosine, huấn luyện chính xác hỗn hợp; lô 32 train / 64 eval; hai '
     'phiên 100 epoch nối liên tục; ngừng MixUp và CutMix ở giai đoạn cuối). HBCC-2.5M '
     'dùng AdamW (lr = 8×10⁻⁴, wd 0,03, chuẩn gradient 1,0) với mức tăng cường vừa phải '
     '(RandAugment (2,7), xóa ngẫu nhiên p = 0,05, MixUp α = 0,10). ResNet-18 dùng '
     'SGD–Nesterov (lr = 1,25×10⁻², wd 10⁻⁴) với chính quy hóa mạnh hơn vì có tham số '
     'lớn hơn 4,38×.')

heading(3, '4. Chưng cất tri thức')

para('Mô hình giáo viên và học sinh.', bold=True)
para('Trong khung chưng cất tri thức [7], chúng tôi sử dụng ResNet-18 làm giáo viên và '
     'HBCC (Small hoặc Medium) làm học sinh. ResNet-18 được huấn luyện từ đầu trên cùng '
     'tập train, đạt 96,48% trên CIFAR-10 và 58,58% trên Tiny-ImageNet-200. Với 11,17–'
     '11,27M tham số và 556,7M MACs, ResNet-18 có khả năng biểu diễn cao hơn nhiều so '
     'với HBCC (1,27–1,76M tham số) nhưng quá nặng để triển khai trực tiếp trên thiết '
     'bị biên. Thực nghiệm Food-101 không sử dụng chưng cất tri thức; ResNet-18 đóng vai '
     'trò mô hình đối chứng độc lập.')
para('Ở mỗi bước huấn luyện, hàm mất mát kết hợp cross-entropy nhãn cứng và KL-divergence '
     'với phân phối mềm của giáo viên (nhiệt độ T = 4,0, trọng số α = 0,5); hệ số T² bù '
     'lại việc gradient bị thu nhỏ khi làm mềm phân phối. Số epoch giảm từ 300 xuống 250.')

heading(3, '5. Đo lường hiệu năng')

para('Tham số và MACs được đo tự động trước khi huấn luyện. Vì công cụ đo không tính đầy '
     'đủ một số phép tính trong bước phân cụm (gán điểm, sigmoid, tổng hợp), các giá trị '
     'MACs báo cáo cho HBCC là giới hạn dưới.')

para('Độ trễ được đo trên GPU Tesla T4 với batch size 1, thực hiện 30 lần chạy khởi '
     'động và 100 lần đo chính thức; GPU được đồng bộ hóa sau mỗi lần đo để đảm bảo '
     'thời gian ghi lại phản ánh chi phí thực thi thực sự. Kết quả đại diện cho độ trễ '
     'suy luận đơn mẫu.')

para('Bộ nhớ GPU đỉnh được ghi lại trong suốt quá trình suy luận. Hồ sơ toán tử được thu '
     'thập bằng cách theo dõi từng phép tính GPU riêng lẻ để xác định các nút thắt cổ '
     'chai hiệu năng.')

heading(3, '6. Mô hình đối chứng')

bullet('ResNet-18 [1]: thay conv 7×7 stride-2 bằng conv 3×3 stride-1, bỏ max-pool đầu '
       'cho các thí nghiệm 32×32 và đóng vai giáo viên KD. Trên Food-101, giữ nguyên '
       'stem 7×7 stride-2 và max-pool chuẩn, thay lớp phân loại cuối bằng 101 đầu ra.')
bullet('MobileNetV2 [3]: stem conv 3×3 stride-1, bỏ max-pool.')
bullet('ShuffleNetV2-x1.0 [4]: chỉnh tương tự, stride đầu giảm xuống 1.')
bullet('CoC-baseline: CoC phân cụm toàn cục (ρ = 0) ở mọi tầng, không có nhánh cục bộ '
       '— đối chứng kiến trúc trực tiếp để cô lập đóng góp của nhánh cục bộ.')

heading(2, 'B. Kết quả phân loại')

heading(3, '1. CIFAR-10')

para('Bảng III so sánh sáu mô hình trên CIFAR-10. HBCC-Medium+KD đạt 95,36% — cao nhất '
     'trong nhóm mô hình nhẹ, chỉ thấp hơn ResNet-18 (−1,12%) trong khi sử dụng 6,34× '
     'ít tham số (1,76M so với 11,17M) và 4,01× ít MACs (138,8M so với 556,7M). '
     'HBCC-Small+KD vs. ShuffleNetV2 (cùng ngân sách tham số): +3,25% top-1. '
     'HBCC-Medium+KD vs. CoC-baseline: +7,02% top-1 và nhanh hơn (8,09 ms vs. 11,71 ms).')

tbl_caption('Bảng III. So sánh trên CIFAR-10. In đậm: tốt nhất trong nhóm nhẹ. '
            '+KD: huấn luyện với chưng cất tri thức.')
make_table(
    headers=['Mô hình', 'Top-1(%)', 'Params(M)', 'MACs(M)', 'Trễ(ms)', 'Mem(MB)'],
    rows=[
        ('ResNet-18',         '96,48', '11,17', '556,66', ' 3,03', '326,4'),
        ('HBCC-Medium+KD',    '95,36', ' 1,76', '138,81', ' 8,09', '481,4'),
        ('HBCC-Small+KD',     '94,93', ' 1,27', ' 92,34', ' 8,41', '363,6'),
        ('ShuffleNetV2-x1.0', '91,68', ' 1,26', ' 46,13', ' 6,26', ' 94,8'),
        ('MobileNetV2',       '91,04', ' 2,24', ' 25,57', ' 4,76', '123,4'),
        ('CoC-baseline',      '88,34', ' 2,40', ' 34,37', '11,71', ' 71,8'),
    ])

heading(3, '2. Tiny-ImageNet-200')

para('Với 200 lớp ngữ nghĩa mịn, Tiny-ImageNet-200 là bộ dữ liệu thách thức nhất trong '
     'ba bộ và cũng là nơi HBCC thể hiện kết quả nổi bật nhất (Bảng IV).')

para('Học sinh vượt qua giáo viên.', bold=True)
para('HBCC-Medium+KD đạt 58,90% top-1 và 81,41% top-5 — vượt qua ResNet-18 (+0,32% top-1, '
     '+1,01% top-5) trong khi chỉ dùng 6,26× ít tham số và 4,01× ít MACs. Đây là kết '
     'quả đáng chú ý nhất: ở bài toán 200 lớp, phân phối mềm của giáo viên mang đủ thông '
     'tin liên lớp để học sinh không chỉ bắt kịp mà còn vượt qua giáo viên trên tập kiểm '
     'tra. HBCC-Small+KD (57,56%) cũng vượt ShuffleNetV2 (55,68%) tới +1,88%.')

tbl_caption('Bảng IV. So sánh trên Tiny-ImageNet-200. In đậm: tốt nhất tổng thể '
            '(kể cả ResNet-18). Mũi tên CE→KD: cùng kiến trúc.')
make_table(
    headers=['Mô hình', 'Top-1(%)', 'Top-5(%)', 'Params(M)', 'MACs(M)', 'Trễ(ms)', 'Nhớ(MB)'],
    rows=[
        ('ResNet-18',         '58,58', '80,40', '11,27', '556,76', ' 3,13', '326,7'),
        ('HBCC-Medium+KD',    '58,90', '81,41', ' 1,80', '138,86', ' 7,57', '481,6'),
        ('HBCC-Small+KD',     '57,56', '80,76', ' 1,32', ' 92,38', ' 7,50', '363,7'),
        ('HBCC-Medium (CE)',  '56,80', '79,13', ' 1,80', '138,86', ' 8,20', '481,6'),
        ('HBCC-Small (CE)',   '56,15', '79,14', ' 1,32', ' 92,38', ' 7,77', '363,7'),
        ('ShuffleNetV2-x1.0', '55,68', '79,32', ' 1,46', ' 46,32', ' 6,04', ' 95,6'),
        ('MobileNetV2',       '49,97', '75,35', ' 2,48', ' 25,80', ' 4,66', '124,3'),
        ('CoC-baseline',      '49,30', '73,89', ' 2,46', ' 34,43', '11,52', ' 72,0'),
    ])

para('CE models đã mạnh trước khi thêm KD.', bold=True)
para('Ngay cả không có KD, HBCC-Medium-CE (56,80%) và HBCC-Small-CE (56,15%) đều vượt '
     'ShuffleNetV2 (55,68%), phản ánh sự phù hợp tự nhiên của kiến trúc lai với bài toán '
     'phân loại mịn nhiều lớp.')

para('Tại sao CoC-baseline và MobileNetV2 kém.', bold=True)
para('CoC-baseline chỉ đạt 49,30% top-1 — thấp nhất toàn bảng; không có nhánh cục bộ, '
     'phân cụm thuần không thể phân biệt 200 lớp ngữ nghĩa mịn. Khoảng cách với '
     'HBCC-Medium+KD (+9,60%) lớn hơn so với CIFAR-10 (+7,02%), xác nhận lợi ích của '
     'nhánh cục bộ tăng theo độ phức tạp phân loại. Hình 3 trực quan hóa Pareto frontier.')

heading(3, '3. Food-101')

para('Bảng V trình bày kết quả sau 200 epoch huấn luyện. Trạng thái có top-1 cao nhất '
     'trên tập xác thực được chọn trước khi đánh giá một lần trên tập kiểm tra '
     '(epoch 199 cho HBCC, epoch 191 cho ResNet-18).')

tbl_caption('Bảng V. So sánh trên Food-101. In đậm: tốt nhất.')
make_table(
    headers=['Mô hình', 'Tham số (M)', 'Top-1(%)', 'Top-5(%)'],
    rows=[
        ('ResNet-18', '11,228', '82,91', '95,49'),
        ('HBCC-2.5M',  ' 2,566', '82,53', '95,33'),
    ])

para('Hiệu quả tham số.', bold=True)
para('HBCC đạt 82,53% top-1 và 95,33% top-5, thấp hơn ResNet-18 lần lượt 0,38 và 0,16 '
     'điểm phần trăm. Tuy nhiên, số tham số của HBCC chỉ là 2,566M so với 11,228M của '
     'ResNet-18 — giảm 77,14% số tham số, tương đương 4,38 lần. Kết quả cho thấy HBCC '
     'nằm tại một điểm đánh đổi thuận lợi: mức suy giảm top-1 nhỏ hơn nửa điểm phần '
     'trăm đi kèm với mức giảm đáng kể về số lượng trọng số.')
para('Trong phạm vi thí nghiệm Food-101, số phép nhân–cộng, độ trễ và bộ nhớ cực đại '
     'chưa được đo cho cả hai mô hình trên cùng phần cứng và cùng quy trình thực thi.')

heading(2, 'C. Hiệu quả của chưng cất tri thức')

para('Bảng VI chỉ so sánh CE và KD trên Tiny-ImageNet-200; kiến trúc, phân chia dữ liệu '
     'và siêu tham số của từng cặp được giữ nguyên; vì vậy Δ đo trực tiếp mức tăng do KD. '
     'Biến thể Medium tăng 2,10 điểm và Small tăng 1,41 điểm. Mức tăng đủ để '
     'HBCC-Medium+KD vượt ResNet-18 0,32 điểm top-1.')

tbl_caption('Bảng VI. Mức tăng top-1 do KD trên Tiny-ImageNet-200 '
            '(Δ = KD − CE, cùng kiến trúc).')
make_table(
    headers=['Mô hình', 'CE(%)', 'KD(%)', 'Δ'],
    rows=[
        ('HBCC-Medium', '56,80', '58,90', '+2,10'),
        ('HBCC-Small',  '56,15', '57,56', '+1,41'),
    ])

heading(2, 'D. Phân tích hiệu năng')

heading(3, '1. Phân tích Pareto')

para('Hình 3–5 biểu diễn accuracy vs. số tham số và accuracy vs. latency trên hai bộ dữ '
     'liệu ảnh nhỏ. Cả HBCC-Small+KD và HBCC-Medium+KD đều nằm ở vùng Pareto-dominant '
     'trên trục tham số: accuracy cao hơn với tham số ít hơn so với mọi baseline nhẹ. '
     'Trên Tiny-ImageNet-200 (Hình 3), HBCC-Medium+KD vượt qua cả ResNet-18 về accuracy '
     'trong khi vẫn giữ 1,80M tham số — điểm duy nhất nằm phía trên-trái so với ResNet-18 '
     'trên cả hai trục đồng thời.')
para('Hình 5 trình bày trục tham số trên Food-101: ResNet-18 cao hơn 0,38 điểm phần trăm '
     'top-1 nhưng sử dụng 4,38× tham số nhiều hơn.')

figure('pareto_tin.png',
       'Hình 3. Pareto frontier trên Tiny-ImageNet-200. Mũi tên: mức tăng do KD. '
       'HBCC-Medium+KD (★) vượt ResNet-18 trên cả hai trục với 6,26× ít tham số hơn.',
       width_in=3.1)

figure('pareto_cifar10.png',
       'Hình 4. Pareto frontier trên CIFAR-10. HBCC+KD nằm ở vùng Pareto-dominant so '
       'với các baseline nhẹ trên cả trục tham số lẫn trục latency.',
       width_in=3.1)

figure('pareto_food101.png',
       'Hình 5. Phân bổ tham số trên Food-101 (chỉ trục tham số). '
       'HBCC-2.5M (★) đạt 82,53% với 4,38× ít tham số hơn ResNet-18.',
       width_in=3.1)

heading(3, '2. Phân tích latency và bộ nhớ đỉnh')

para('Latency: ít MACs hơn nhưng vẫn chậm hơn.', bold=True)
para('Mặc dù HBCC-Medium dùng ít MACs hơn ResNet-18 (138,9M vs. 556,8M), latency batch-1 '
     'lại cao hơn (≈8 ms vs. ≈3 ms). Nguyên nhân là operator fragmentation: ResNet-18 '
     'dành ≈94% thời gian CUDA trong một kernel duy nhất (cudnn_convolution) được cuDNN '
     'tối ưu hóa kỹ, trong khi HBCC phân tán thời gian trên nhiều kernel nhỏ nối tiếp '
     '(conv 1×1, bmm, vector norm, scatter/gather clustering). HBCC vẫn nhanh hơn CoC '
     'thuần 30–35% (8 ms vs. 11 ms), cho thấy thiết kế lai cải thiện đáng kể.')

para('Bộ nhớ đỉnh: scatter buffer là nút cổ chai.', bold=True)
para('HBCC-Medium đạt 481,6 MB — cao hơn ResNet-18 (326,7 MB) tới 47% dù ít hơn 6,26× '
     'tham số. Bước scatter/gather phải đồng thời giữ tensor tương đồng (N×K) và tổng '
     'đặc trưng (K×C) cho mỗi tầng; với stem_stride = 1 các buffer này lớn gấp 4× so '
     'với stride 2. CoC-baseline (72,0 MB) tránh chi phí này nhờ stride 2 và chiều nhúng '
     'hẹp hơn. Giảm stem_stride hoặc thu hẹp chiều nhúng Tầng 1 là hai đòn bẩy trực '
     'tiếp để cắt giảm cả latency lẫn bộ nhớ.')

# ── V. KẾT LUẬN ───────────────────────────────────────────────────────────────
heading(1, 'V. Kết luận')

para('HBCC là kiến trúc lai bốn tầng kết hợp Context Cluster với xử lý cục bộ: LBPConv '
     'ở Tầng 1, DWConv ở Tầng 2, cluster thuần ở Tầng 3–4; stem stride 1 và fold giảm '
     'dần khắc phục suy biến token. HBCC-Medium+KD đạt 95,36% trên CIFAR-10 và 58,90% '
     'trên Tiny-ImageNet-200 — vượt ResNet-18 giáo viên 0,32 điểm với 6,26× ít tham số '
     'hơn, đồng thời nới rộng khoảng cách so với CoC-baseline từ 7,02 lên 9,60 điểm '
     'phần trăm. Trên Food-101, HBCC-2.5M đạt 82,53% top-1 với 4,38× ít tham số hơn '
     'ResNet-18, cho thấy kiến trúc lai có thể mở rộng lên đầu vào 224×224.')

para('Các giới hạn chính: biên nhỏ trên Tiny-ImageNet (0,32 pp) và Food-101 (0,38 pp) '
     'cần được xác nhận bằng nhiều seed; kết quả Food-101 phản ánh hiệu năng tổng thể '
     'của kiến trúc và công thức huấn luyện; phân mảnh toán tử vẫn giữ latency và bộ '
     'nhớ cao hơn các baseline tích chập. Ba hướng này là ưu tiên cho nghiên cứu tiếp theo.')

# ── TÀI LIỆU THAM KHẢO ────────────────────────────────────────────────────────
heading(1, 'Tài liệu tham khảo')

for ref in [
    '[1] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image '
    'recognition," in IEEE CVPR, 2016, pp. 770–778.',

    '[2] A. Dosovitskiy et al., "An image is worth 16×16 words: Transformers for '
    'image recognition at scale," in ICLR, 2021.',

    '[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: '
    'Inverted residuals and linear bottlenecks," in IEEE CVPR, 2018, pp. 4510–4520.',

    '[4] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun, "ShuffleNet V2: Practical guidelines '
    'for efficient CNN architecture design," in ECCV, 2018, pp. 122–138.',

    '[5] X. Ma et al., "Image as set of points," in ICLR, 2023.',

    '[6] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and '
    'rotation invariant texture classification with local binary patterns," IEEE Trans. '
    'Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971–987, 2002.',

    '[7] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural '
    'network," arXiv:1503.02531, 2015.',

    '[8] A. Krizhevsky, "Learning multiple layers of features from tiny images," '
    'Univ. Toronto, Tech. Rep., 2009.',

    '[9] L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 — Mining discriminative '
    'components with random forests," in ECCV, 2014, pp. 446–461.',

    '[10] Y. Le and X. Yang, "Tiny ImageNet visual recognition challenge," Stanford '
    'Univ., CS231N, Tech. Rep., 2015.',

    '[11] E. D. Cubuk, B. Zoph, J. Shlens, and Q. V. Le, "RandAugment: Practical '
    'automated data augmentation with a reduced search space," in NeurIPS, 2020.',

    '[12] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond '
    'empirical risk minimization," in ICLR, 2018.',

    '[13] S. Yun et al., "CutMix: Regularization strategy to train strong classifiers '
    'with localizable features," in IEEE/CVF ICCV, 2019, pp. 6023–6032.',

    '[14] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random erasing data '
    'augmentation," in AAAI, 2020, pp. 13001–13008.',
]:
    para(ref, style='references')

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(str(OUTPUT))
print(f'Done. Saved → {OUTPUT}')