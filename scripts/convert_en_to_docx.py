"""
Convert HBCC English LaTeX paper to IEEE A4 Word format.
Usage: python scripts/convert_en_to_docx.py
Output: Report_LaTeX/HBCC_en_A4.docx

Layout:
  Section 1 (1-col, continuous): Title · Authors · Abstract · Keywords
  Section 2 (2-col): Body (Sections I–V + References)
  Pipeline figure (Fig. 1) wrapped in a 1-col mini-section so it spans both columns.

After opening in Word:
  - Replace each [EQ N: ...] placeholder with Insert → Equation
  - Bold the best-result cells in each table
  - Check the empty switching paragraphs around Fig. 1; delete if unwanted
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
OUTPUT = ROOT / 'HBCC_en_A4.docx'

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
for t in _body_sectPr.findall(qn('w:type')):
    _body_sectPr.remove(t)
_cols = _body_sectPr.find(qn('w:cols'))
if _cols is None:
    _cols = OxmlElement('w:cols')
    _body_sectPr.append(_cols)
_cols.set(qn('w:num'), '2')

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
    pPr = para_obj._p.get_or_add_pPr()
    pPr.append(_make_col_sectPr(num_cols))
    return para_obj

def col_switch(num_cols):
    pr = doc.add_paragraph(style=_style('BodyText'))
    pPr = pr._p.get_or_add_pPr()
    sp = OxmlElement('w:spacing')
    sp.set(qn('w:before'), '0'); sp.set(qn('w:after'), '0')
    pPr.append(sp)
    pPr.append(_make_col_sectPr(num_cols))
    return pr

def figure(fname, caption, width_in=3.1, full_width=False):
    if full_width:
        col_switch(2)
    path = IMAGES / fname
    pr = doc.add_paragraph(style=_style('BodyText'))
    pr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        pr.add_run().add_picture(str(path), width=Inches(width_in))
    else:
        pr.add_run(f'[Figure not found: {fname}]').italic = True
    cap = doc.add_paragraph(caption, style=_style('figurecaption'))
    if full_width:
        col_switch(1)
    return pr

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 (1-col): TITLE · AUTHORS · ABSTRACT · KEYWORDS
# ═════════════════════════════════════════════════════════════════════════════
para('HBCC: A Hybrid Local–Cluster Architecture for Lightweight Image Classification',
     style='papertitle')

para('Nguyen Viet Hung, Nguyen Ba Nam, Le Hoang Thai*', style='Author')
para('Faculty of Information Technology, University of Science, '
     'Vietnam National University Ho Chi Minh City, Ho Chi Minh City, Vietnam',
     style='Affiliation')
para('{23122032, 23122043}@student.hcmus.edu.vn  |  lhthai@fit.hcmus.edu.vn',
     style='Affiliation')
para('*Corresponding author', style='Affiliation')

para(
    'Abstract — Edge-device image classification demands a careful balance between '
    'accuracy and computational cost. Lightweight CNNs reduce cost but may sacrifice '
    'representational capacity, while Context Cluster (CoC) aggregates global context '
    'through clustering yet ignores local texture and risks token degeneracy on small '
    'images. This paper proposes HBCC (Hybrid Block Context Cluster), a four-stage hybrid '
    'architecture that applies LBPConv and DWConv at high-resolution stages and transitions '
    'to pure clustering thereafter. Stem stride, block depth, and stage-shrinking fold '
    'partitioning are co-designed to eliminate token degeneracy; knowledge distillation from '
    'a frozen ResNet-18 teacher is used to boost student accuracy at no additional inference '
    'cost. On CIFAR-10, HBCC-Medium+KD achieves 95.36% top-1 with 1.76M parameters and '
    '138.8M MACs. On Food-101 at 224×224 resolution, HBCC-2.5M trained from scratch '
    'achieves 82.53% top-1 and 95.33% top-5, trailing ResNet-18 by only 0.38% and 0.16% '
    'while using 4.38× fewer parameters. On Tiny-ImageNet-200 (200 classes), '
    'HBCC-Medium+KD attains 58.90%, exceeding the ResNet-18 teacher (58.58%) with 6.26× '
    'fewer parameters. Operator profiling reveals that kernel fragmentation in clustering ops '
    'remains the bottleneck preventing the MAC reduction from fully translating into measured '
    'latency gains.',
    style='Abstract')

kw = para(
    'Keywords — Image classification, Context Cluster, hybrid local–cluster '
    'architecture, lightweight neural networks, knowledge distillation.',
    style='Keywords')
embed_section_end(kw, 1)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 (2-col): BODY — SECTIONS I–V + REFERENCES
# ═════════════════════════════════════════════════════════════════════════════

# ── I. INTRODUCTION ────────────────────────────────────────────────────────────
heading(1, 'I. Introduction')

para(
    'Edge-device image classification requires a careful balance between accuracy and '
    'computational cost. Deep CNNs [1] and Vision Transformers [2] achieve strong '
    'representational power but routinely exceed the resource budgets of embedded platforms. '
    'MobileNetV2 [3] and ShuffleNetV2 [4] reduce computational cost through architectural '
    'streamlining, yet aggressive compression can limit the network\'s capacity to '
    'distinguish between visually similar categories.')

para(
    'Context Cluster (CoC) [5] offers a complementary perspective: it represents an image '
    'as a set of points and synthesizes global context through similarity-based clustering. '
    'However, pure clustering ignores local texture at high resolutions. On small images, '
    'aggressively reducing spatial resolution also risks token degeneracy — the state '
    'in which the number of active feature points per cluster is too small for the '
    'aggregation operator to carry meaningful information.')

para(
    'This work targets a lightweight architecture that combines local processing with context '
    'clustering, and studies how each mechanism should be allocated across resolution stages. '
    'We evaluate on small images (32×32) and standard-resolution (224×224) inputs '
    'within a strict parameter budget, addressing two research questions:')

para(
    '(RQ1) Can a hybrid architecture combining local processing and context clustering '
    'improve both accuracy and computational efficiency relative to both pure lightweight '
    'CNNs and pure clustering models?')

para(
    '(RQ2) Does the stage-wise local/cluster allocation strategy remain effective when '
    'scaling from small-image (32×32) to standard-resolution (224×224) input?')

para(
    'To answer these questions, we propose HBCC (Hybrid Block Context Cluster), a '
    'four-stage architecture that uses hybrid blocks at high-resolution stages and pure '
    'clustering blocks at later stages. Stem design and per-stage block depth are carefully '
    'chosen to prevent token degeneracy; knowledge distillation is applied during training '
    'without increasing student inference cost. For Food-101, a controlled extension to '
    '2.57M parameters uses a 7×7 stem with stride 4 and uniform 7×7 fold tiles '
    'to preserve spatial structure on 224×224 input.')

para(
    'Paper structure: Section II surveys related work; Section III presents the proposed '
    'method; Section IV describes setup, results, and analysis; Section V states findings, '
    'limitations, and conclusions.',
    italic=True)

# ── II. RELATED WORK ──────────────────────────────────────────────────────────
heading(1, 'II. Related Work')

heading(2, 'A. CNN and Residual Learning')

para(
    'ResNet [1] introduced the residual connection y = F(x) + x, enabling gradient flow '
    'through arbitrarily deep networks and opening the door to 50–152-layer training. '
    'Standard 3×3 convolution with C_in input channels, C_out output channels, and '
    'kernel size k incurs cost proportional to C_in · k² · C_out, which '
    'causes ResNet-18 to require 556.7M MACs even on 32×32 images.')

para(
    'MobileNetV2 [3] replaces standard convolution with depthwise separable convolution, '
    'reducing total cost to O(C_in(k² + C_out)) — a 1/k² saving. '
    'ShuffleNetV2 [4] argues that FLOPs are poorly correlated with measured throughput and '
    'proposes four practical guidelines: balance input/output channel widths, avoid group '
    'convolution, avoid excessive branching, and minimize element-wise operations.')

heading(2, 'B. Vision Transformers and MetaFormers')

para(
    'ViT [2] divides an image into n patches and applies multi-head self-attention with '
    'complexity O(n² d_k). On a 32×32 image with 2×2 patches, n = 256 '
    'tokens; the O(256²) cost is prohibitive for lightweight models. By contrast, '
    'clustering with complexity O(n·K) and K = 4 centers is theoretically 64× '
    'cheaper than single-head attention. The MetaFormer abstraction demonstrates that the '
    'two-block structure (token mixer + MLP) is at least as important as the specific '
    'choice of mixer operator.')
equation('1', 'Attn(Q,K,V) = softmax( QKᵀ / √d_k ) · V')

heading(2, 'C. Context Cluster')

para(
    'CoC [5] represents an image as a set of 5D points (R,G,B,x,y), embedding spatial '
    'coordinates directly into the feature space. Cluster centers are proposed '
    'parameter-free via AdaptiveAvgPool; each point is hard-assigned to its nearest center '
    'by cosine similarity at cost O(N·M·D). The main limitation of pure '
    'clustering is that it discards high-frequency information (edges, fine texture), '
    'causing the CoC baseline to reach only 88.34% on CIFAR-10. Furthermore, token '
    'degeneracy occurs when the feature map is too small relative to the number of cluster '
    'centers (analyzed in detail in Section III-G).')

heading(2, 'D. Local Binary Patterns')

para('LBP [6] encodes local texture at pixel (x_c, y_c):')
equation('2',
    'LBP_{P,R}(x_c,y_c) = Σ_{p=0}^{P-1} s(g_p − g_c)·2^p,   '
    's(u) = 1 if u ≥ 0,  0 if u < 0')
para(
    'where g_c is the intensity of the center pixel and g_p is the p-th neighbor on a '
    'circle of radius R with P equally spaced sampling points. The operation involves only '
    'subtractions and binary thresholding — near-zero MAC cost. LBPConv applies this '
    'formula channel-wise using a fixed, non-learned filter, making it ideal for capturing '
    'fine-grained texture at high-resolution feature maps without competing with the '
    'clustering branch for trainable parameters.')

heading(2, 'E. Knowledge Distillation')

para(
    'Hinton et al. [7] showed that a small student network learns more effectively when '
    'it mimics the soft probability distribution produced by a large teacher network. The '
    'combined loss function blends cross-entropy on hard labels with KL-divergence on '
    'temperature-softened teacher outputs (see Section IV-A-4 for details). Soft '
    'distributions carry inter-class relationship signals that are entirely absent from '
    'one-hot labels, making KD particularly effective for fine-grained multi-class problems.')

para(
    'In summary: lightweight CNNs exploit locality well but trade accuracy; self-attention '
    'captures global relations at quadratic token cost; CoC reduces complexity but discards '
    'local texture and risks token degeneracy on small images. No prior work has clearly '
    'characterized how to combine local processing with context clustering, or how to '
    'allocate these mechanisms across resolution stages under a strict parameter budget. '
    'This gap motivates the method proposed next.')

# ── III. PROPOSED METHOD ───────────────────────────────────────────────────────
heading(1, 'III. Proposed Method')

heading(2, 'A. Pipeline Overview')

para(
    'Figure 1 illustrates the complete pipeline. Input image B×3×32×32 is '
    'augmented with spatial coordinates (Section III-B) and passed through a Stem '
    'convolution (stride 1), producing a 32×32×C₁ feature map. Four '
    'successive stages perform feature extraction at decreasing resolutions; between stages, '
    'a PointReducer (1×1 convolution, stride 2) downsamples and expands channels. '
    'A Global Average Pooling (GAP) layer followed by a Linear classifier produces the '
    'C-dimensional output vector (C = 10, 101, or 200).')

figure('fig_hbcc_pipeline.png',
       'Fig. 1. The four-stage HBCC pipeline. Blue: Hybrid stage (local branch + cluster '
       'branch in parallel). Red: Pure-cluster stage. Spatial resolution and channel count '
       'Cᵢ are annotated at each stage.',
       width_in=6.5, full_width=True)

heading(2, 'B. Spatial Coordinate Augmentation')

para(
    'Cosine similarity computed on RGB features alone cannot distinguish two points with '
    'identical color at different spatial positions. We therefore append two normalized '
    'coordinate channels:')
equation('3',
    'X = [I ; P_x ; P_y] ∈ ℝ^{B×5×H₀×W₀},  '
    'P_x[i,j] = j/(W₀−1),  P_y[i,j] = i/(H₀−1)')
para('This allows cluster centers to encode spatial position without requiring any explicit '
     'positional encoding.')

heading(2, 'C. Context Cluster Mechanism')

para('Given a feature tensor X ∈ ℝ^{B×C×H×W}, the '
     'ContextClusterOp executes six steps (applied per head, then concatenated):')

para('Step 1 — Dual projection:', bold=True)
equation('4', 'F = Conv_{1×1}(X),   V = Conv_{1×1}(X)')
para('F carries the similarity representation; V carries the values to be aggregated.')

para('Step 2 — Cluster center proposal:', bold=True)
equation('5', 'F_c = AvgPool_{K_h×K_w}(F),   V_c = AvgPool_{K_h×K_w}(V)')
para('With (K_h, K_w) = (2,2), each fold tile yields K = 4 cluster centers. AvgPool is '
     'chosen because cluster centers must represent the centroid of their region, not an '
     'extremum; AvgPool over LBP activation maps also produces a soft histogram of texture '
     'patterns, which is far more informative than MaxPool.')

para('Step 3 — Similarity and hard assignment:', bold=True)
equation('6', 's_{ij} = σ( α · (f_{c_i}·f_j) / (‖f_{c_i}‖‖f_j‖) + β )')
equation('7', 'm_{ij} = 1[ i = argmax_k s_{kj} ]')
para('where α, β ∈ ℝ are learnable scalars. Hard assignment (m_{ij}) '
     'ensures each point contributes to exactly one cluster.')

para('Step 4 — Weighted aggregation:', bold=True)
equation('8',
    'g_i = ( v_{c_i} + Σ_j m_{ij} s_{ij} v_j ) / ( 1 + Σ_j m_{ij} s_{ij} )')
para('The denominator 1 + Σs_{ij} normalizes including the center\'s self-weight, '
     'ensuring a stable output even when a cluster is empty.')

para('Step 5 — Dispatch:', bold=True)
equation('9', 'ṽ_j = Σ_i g_i · m_{ij} · s_{ij}')
para('Aggregated features are redistributed back to each point by similarity weight.')

para('Step 6 — Output projection: Y = Conv_{1×1}(Ṽ).')

heading(2, 'D. Fold Partitioning')

para('At high resolutions (32×32, 16×16), global clustering creates oversized '
     'clusters that smooth over local detail. We partition the feature map into '
     'F_h × F_w non-overlapping tiles:')
equation('10', '(B, C, F_h·h, F_w·w)  ⟶  (B·F_h·F_w, C, h, w)')
para('CoC is applied independently within each tile, then reshaped back. The fold count '
     'shrinks progressively: 4×4 (Stage 1, 32×32), 2×2 (Stage 2, 16×16), '
     '1×1 (Stages 3–4, global).')

heading(2, 'E. Hybrid Cluster Block')

para('Context clustering and local convolution are complementary: clustering captures global '
     'inter-region relationships but misses fine texture, while local convolution excels at '
     'high-frequency detail but is bounded by its receptive field. The Hybrid Cluster Block '
     'exploits both simultaneously by splitting the input channels into two parallel branches '
     'and fusing their outputs.')

figure('fig_hybrid_block.png',
       'Fig. 2. Hybrid Cluster Block structure. Blue branch: CoC (C_c = C(1−ρ) '
       'channels). Orange branch: local operator (C_l = Cρ channels). Branches are '
       'fused via Channel Shuffle and a 1×1 convolution. Layer-scale γ '
       'initialized at 10⁻⁵.',
       width_in=3.1)

para('Given local ratio ρ ∈ [0,1]: the local branch processes '
     'C_l = ⌊Cρ⌈ channels; the cluster branch processes C_c = C − C_l '
     'channels. Stages 1–2 use ρ = 0.5 to split bandwidth equally; '
     'Stages 3–4 use ρ = 0 to commit fully to clustering.')

para('Token mixing:', bold=True)
equation('11', '[x_c ; x_l] = split( BN(x), [C_c, C_l] )')
equation('12', 'y = Fuse_{1×1}( Shuffle[ CoC(x_c) ; Local(x_l) ] )')
equation('13', 'x ← x + DropPath( γ₁ ⊙ y )')

para('MLP:', bold=True)
equation('14', 'x ← x + DropPath( γ₂ ⊙ MLP( BN(x) ) )')
para('where γ₁, γ₂ ∈ ℝ^C are layer-scale vectors initialized '
     'at 10⁻⁵. Channel Shuffle [4] interleaves features from both branches before '
     'the fuse convolution, enabling faster convergence. When ρ = 0 (Stages 3–4), '
     'the local branch, Channel Shuffle, and Fuse are all bypassed; the block reduces to a '
     'standard CoC block.')

heading(2, 'F. Local Branch Design')

para('LBPConv (Stage 1, 32×32).', bold=True)
para('At Stage 1, feature maps retain full 32×32 resolution, where fine-grained '
     'texture (edges, contours, surface patterns) is most abundant. Three reasons favor '
     'LBPConv over a learned 3×3 convolution:')
para('(i) Near-zero cost. LBPConv applies fixed filters consisting only of subtractions and '
     'binary thresholding — no multiply-accumulate operations — so it contributes '
     'almost nothing to the total MAC count.', italic=True)
para('(ii) No parameter competition. Because LBPConv has no learned convolution weights and '
     'processes each channel independently with the same fixed binary filter, the two '
     'branches do not compete for parameter capacity.', italic=True)
para('(iii) Natural compatibility with AvgPool. LBPConv produces near-binary activation maps. '
     'When cluster centers are proposed via AvgPool over these maps, the result is a soft '
     'histogram — the fraction of active bits per tile — which captures the '
     'distribution of texture patterns in each region.', italic=True)

para('DWConv (Stage 2, 16×16).', bold=True)
para('At Stage 2, after one downsampling step, the model benefits from a wider spatial '
     'receptive field to link nearby regions before transitioning to pure clustering. '
     'Depthwise 3×3 convolution processes each channel independently at cost '
     'O(k² C_in) instead of O(k² C_in C_out) for standard convolution — '
     'a 1/C_out reduction.')

para('Pure Cluster (Stages 3–4, 8×8, 4×4).', bold=True)
para('At later stages, features have become sufficiently abstract that semantic inter-region '
     'relationships dominate. Global clustering with fold (1,1) exploits these relationships '
     'more effectively than any local convolution. Removing the local branch and Channel '
     'Shuffle eliminates their associated costs (ρ = 0).')

heading(2, 'G. Token Degeneracy and Stem Stride')

para('The number of feature points per cluster within a fold tile is:')
equation('15', 'r = ( H/F_h · W/F_w ) / ( K_h · K_w )')
para('For clustering to be meaningful, r ≥ 4 is necessary. With stem_stride=2 on '
     '32×32 input:')
bullet('Stage 4: 2×2 = 4 points, proposals(2,2) yield 4 centers ⇒ r = 1 — '
       'each cluster contains exactly one point, making the clustering operator the identity '
       'function and preventing any learning.')
bullet('Stage 3: 4×4 = 16 points, fold(1,1), proposals(2,2) ⇒ r = 4 — '
       'barely sufficient.')
para('Setting stem_stride=1 retains 32×32 = 1024 points after the stem. Combined with '
     'stage-wise fold shrinking, r = 16 is achieved consistently throughout the pipeline:')
bullet('Stage 1: fold(4,4) on 32×32: 8×8 = 64 points/tile, 4 centers ⇒ r = 16.')
bullet('Stage 2: fold(2,2) on 16×16: 8×8 = 64 points/tile, 4 centers ⇒ r = 16.')
bullet('Stage 3: fold(1,1) on 8×8: 64 points, 4 centers ⇒ r = 16.')
bullet('Stage 4: fold(1,1), proposals(1,1) on 4×4: 16 points, 1 center ⇒ r = 16.')
para('Because stem_stride=1 quadruples the feature map area at every stage, the per-stage '
     'block count is reduced from [2,2,4,2] to [1,1,2,1] to keep total cost reasonable '
     'while ensuring the clustering operator functions correctly throughout.')

heading(2, 'H. Stage Configuration')

para('Table I lists the full configuration. MLP expansion ratio is 3.0 throughout. Proposal '
     'size (K_h, K_w) = (2,2) at all stages (4 cluster centers per tile), except Stage 4 '
     'which uses (1,1) for global context.')

tbl_caption('Table I. Stage configuration for HBCC-Small (S) and HBCC-Medium (M). '
            'd: blocks per stage; C: embedding dimension; ρ: local ratio.')
make_table(
    headers=['Stage', 'Res.', 'Mode', 'Local op', 'ρ', 'Fold', 'd_{S/M}', 'C_S / C_M'],
    rows=[
        ('1', '32²', 'Hybrid',  'LBPConv', '0.5', '4×4', '1/1', '48/64'),
        ('2', '16²', 'Hybrid',  'DWConv',  '0.5', '2×2', '1/1', '80/96'),
        ('3', '8²',  'Cluster', '—',  '0.0', '1×1', '2/2', '160/192'),
        ('4', '4²',  'Cluster', '—',  '0.0', '1×1', '1/1', '224/256'),
    ])

heading(2, 'I. HBCC-2.5M Variant for Food-101')

para('Applying the small-image configuration directly to 224×224 input would raise the '
     'Stage-1 token count to 224² = 50,176, making clustering prohibitively expensive. '
     'The Food-101 variant uses a 7×7 stem convolution with stride 4 and padding 3, '
     'producing a 56×56 feature map, which subsequent 3×3 stride-2 point reducers '
     'progressively halve. Fold sizes are set to 8×8, 4×4, 2×2, and 1×1 '
     'across four stages, so every fold tile spans a 7×7 spatial region at any '
     'resolution — keeping the local clustering problem scale-invariant as depth '
     'increases.')
para('Each tile uses 2×2 proposals (four centers), cosine similarity, and hard '
     'assignment at all stages. Spatial coordinates (x,y) are appended to the input; '
     'GroupNorm replaces BatchNorm to reduce batch statistics dependence. Stages 1–2 '
     'are hybrid; Stage 3 is pure cluster; Stage 4 returns a 25% DWConv local branch to '
     'recover fine spatial detail in the final semantic representation. The four-stage '
     'configuration (C = 48/80/160/256, d = 2/2/3/2, folds 8²/4²/2²/1²) '
     'contains 2,566,391 parameters (2,562,935 learnable; the remainder are fixed LBP '
     'filter weights).')

# ── IV. EXPERIMENTS ────────────────────────────────────────────────────────────
heading(1, 'IV. Experiments')

heading(2, 'A. Experimental Setup')

heading(3, '1. Datasets and Data Splits')

para('We evaluate on three datasets spanning two resolution regimes. CIFAR-10 [8] (10 '
     'classes, 60,000 images at 32×32) is split into 45,000 training / 5,000 '
     'validation / 10,000 test images. Food-101 [9] (101 classes, 101,000 images) follows '
     'the official train/test split. To obtain a validation set without touching the test '
     'data, we stratified-sample 75 images per class from the official training set; the '
     'remaining 675 images per class are used for optimization. This yields 68,175 training / '
     '7,575 validation / 25,250 test images with no overlap. Tiny-ImageNet-200 [10] (200 '
     'classes, 100,000 images at 64×64; 90,000 train / 10,000 validation / 10,000 test) '
     'has all images resized to 32×32 to maintain uniform resolution for the small-image '
     'configurations. All splits are fixed with random seed 42.')

heading(3, '2. Training Recipe for Small-Image Datasets')

tbl_caption('Table II. Training recipe shared by all models on CIFAR-10 and Tiny-ImageNet-200.')
make_table(
    headers=['Hyperparameter', 'Value'],
    rows=[
        ('Optimizer',                'AdamW'),
        ('Learning rate',            '1×10⁻³'),
        ('Weight decay',             '0.05'),
        ('LR schedule',              'Cosine annealing'),
        ('Warmup epochs',            '5'),
        ('Total epochs (CE)',         '300'),
        ('Total epochs (KD)',         '250'),
        ('Label smoothing',           '0.1'),
        ('RandAugment [11]',          'N = 2, M = 9'),
        ('MixUp [12] α',         '0.2'),
        ('CutMix [13] α, p',     '1.0; 0.5'),
        ('Random Erasing [14] p',     '0.25'),
        ('AMP (mixed precision)',      'Yes'),
        ('Drop path (Small / Med.)',  '0.05 / 0.08'),
    ])

para('On CIFAR-10 and Tiny-ImageNet-200, all models — HBCC-Small, HBCC-Medium, and '
     'all four baselines (ResNet-18, MobileNetV2, ShuffleNetV2, CoC-baseline) — are '
     'trained with the identical recipe in Table II, ensuring a fair comparison. Drop path '
     'is the only HBCC-specific hyperparameter (0.05 for Small, 0.08 for Medium); baselines '
     'do not use drop path.')

heading(3, '3. Food-101 Training Protocol')

para('All images are resized to 224×224. Both models are initialized randomly and '
     'trained for 200 epochs (5-epoch warmup, cosine schedule, mixed-precision; batch 32 '
     'train / 64 eval; two 100-epoch sessions chained with full state restore; MixUp and '
     'CutMix disabled in the final phase). HBCC-2.5M uses AdamW (lr = 8×10⁻⁴, '
     'wd 0.03, gradient clip 1.0) with moderate augmentation (RandAugment (2,7), random '
     'erasing p = 0.05, MixUp α = 0.10, CutMix α = 0.50). ResNet-18 uses '
     'SGD–Nesterov (lr = 1.25×10⁻², wd 10⁻⁴) with stronger '
     'regularization, since it is 4.38× larger.')

heading(3, '4. Knowledge Distillation')

para('Teacher and student.', bold=True)
para('We use ResNet-18 as the teacher and HBCC (Small or Medium) as the student, following '
     'the Hinton et al. [7] framework. ResNet-18 is trained from scratch on the same '
     'training set, achieving 96.48% on CIFAR-10 and 58.58% on Tiny-ImageNet-200. With '
     '11.17–11.27M parameters and 556.7M MACs, ResNet-18 far exceeds the '
     'representational capacity of HBCC (1.27–1.76M parameters) but is too heavy for '
     'direct edge deployment. The Food-101 experiment uses no knowledge distillation; '
     'ResNet-18 serves as an independent architectural baseline trained from random '
     'initialization, enabling a fair cross-entropy-only comparison of the two architectures.')
para('At each training step, the combined loss mixes cross-entropy on hard labels with '
     'KL-divergence on the teacher\'s temperature-softened probabilities (T = 4.0, '
     'α = 0.5); the T² factor restores gradient scale. The teacher is kept frozen; '
     'the epoch count is reduced from 300 to 250 because each step incurs an additional '
     'teacher forward pass.')

heading(3, '5. Performance Measurement')

para('Parameters and MACs are measured automatically before training. Because the profiling '
     'tool does not fully account for certain operations in the clustering step (point '
     'assignment, sigmoid, aggregation), the reported MACs for HBCC are a lower bound.')

para('Latency is measured on a Tesla T4 GPU at batch size 1 with 30 warmup runs followed '
     'by 100 timed iterations; the GPU is synchronized after each iteration to ensure that '
     'recorded times reflect true execution cost. Results represent single-sample inference '
     'latency.')

para('Peak GPU memory is recorded throughout inference. Operator profiles are collected by '
     'tracing individual GPU operations to identify performance bottlenecks.')

heading(3, '6. Baselines')

bullet('ResNet-18 [1]: for small-image experiments, the initial 7×7 stride-2 '
       'convolution is replaced by 3×3 stride-1 and the first max-pool is removed; '
       'also serves as the KD teacher. On Food-101, the standard 7×7 stride-2 stem '
       'and max-pool are retained; the final classification layer is replaced with 101 '
       'outputs and no pretrained weights are loaded.')
bullet('MobileNetV2 [3]: stem 3×3 stride-1, max-pool removed (small-image only).')
bullet('ShuffleNetV2-x1.0 [4]: analogous stem adjustment, first stride reduced to 1 '
       '(small-image only).')
bullet('CoC-baseline: pure global clustering (ρ = 0) at every stage, no local branch '
       '— the direct architectural counterpart of HBCC isolating the contribution of '
       'the local branch (small-image only).')

heading(2, 'B. Classification Results')

heading(3, '1. CIFAR-10')

para('Table III compares six models on CIFAR-10. HBCC-Medium+KD achieves 95.36% — '
     'the highest accuracy among all lightweight models, trailing ResNet-18 by only '
     '−1.12 pp while using 6.34× fewer parameters and 4.01× fewer MACs. '
     'HBCC-Small+KD gains +3.25% over ShuffleNetV2 at the same parameter budget; '
     'HBCC-Medium+KD gains +7.02% over CoC-baseline with fewer parameters and lower '
     'latency (8.09 ms vs. 11.71 ms), confirming the dual benefit of the local branch.')

tbl_caption('Table III. Results on CIFAR-10. Bold: best among lightweight models. '
            '+KD: trained with knowledge distillation.')
make_table(
    headers=['Model', 'Top-1(%)', 'Params(M)', 'MACs(M)', 'Lat.(ms)', 'Mem(MB)'],
    rows=[
        ('ResNet-18',         '96.48', '11.17', '556.66', ' 3.03', '326.4'),
        ('HBCC-Medium+KD',    '95.36', ' 1.76', '138.81', ' 8.09', '481.4'),
        ('HBCC-Small+KD',     '94.93', ' 1.27', ' 92.34', ' 8.41', '363.6'),
        ('ShuffleNetV2-x1.0', '91.68', ' 1.26', ' 46.13', ' 6.26', ' 94.8'),
        ('MobileNetV2',       '91.04', ' 2.24', ' 25.57', ' 4.76', '123.4'),
        ('CoC-baseline',      '88.34', ' 2.40', ' 34.37', '11.71', ' 71.8'),
    ])

heading(3, '2. Food-101')

para('Table IV reports results after 200 training epochs. The checkpoint with the highest '
     'validation top-1 is selected before a single evaluation on the test set. HBCC\'s '
     'best state occurs at epoch 199 (the final epoch, indicating the model had not yet '
     'converged); ResNet-18\'s best state occurs at epoch 191.')

tbl_caption('Table IV. Results on Food-101. Bold: best.')
make_table(
    headers=['Model', 'Params (M)', 'Top-1 (%)', 'Top-5 (%)'],
    rows=[
        ('ResNet-18', '11.228', '82.91', '95.49'),
        ('HBCC-2.5M',  ' 2.566', '82.53', '95.33'),
    ])

para('Parameter efficiency.', bold=True)
para('HBCC achieves 82.53% top-1 and 95.33% top-5, trailing ResNet-18 by 0.38% and 0.16% '
     'respectively. HBCC uses only 2.566M parameters vs. ResNet-18\'s 11.228M — a '
     '77.14% reduction (4.38× fewer). This places HBCC at a favorable '
     'accuracy-vs-size trade-off: a sub-half-point top-1 gap paired with a fourfold '
     'reduction in model weight. The gap is likely a lower bound: HBCC\'s best validation '
     'accuracy appeared at the final epoch, suggesting continued improvement with additional '
     'training budget.')
para('MACs, latency, and peak memory were not measured for Food-101 under the same hardware '
     'conditions as the small-image benchmarks; these quantities are therefore not reported '
     'and should not be extrapolated from the 32×32 results.')

heading(3, '3. Tiny-ImageNet-200')

para('With 200 fine-grained semantic classes, Tiny-ImageNet-200 is the most challenging of '
     'the three datasets and also where HBCC achieves its most remarkable results (Table V).')

para('Student surpasses teacher.', bold=True)
para('HBCC-Medium+KD reaches 58.90% top-1 and 81.41% top-5 — surpassing ResNet-18 '
     'by +0.32% top-1 and +1.01% top-5 while using only 6.26× fewer parameters (1.80M '
     'vs. 11.27M) and 4.01× fewer MACs (138.9M vs. 556.8M). This is the most '
     'significant result: on a 200-class problem, the teacher\'s soft distributions carry '
     'sufficient inter-class relationship information for the student to surpass the teacher '
     'on the test set. HBCC-Small+KD (57.56%) also exceeds ShuffleNetV2 (55.68%) by '
     '+1.88% with fewer parameters (1.32M vs. 1.46M).')

tbl_caption('Table V. Results on Tiny-ImageNet-200. Bold: best overall (including ResNet-18). '
            'Arrows CE→KD indicate same-architecture pairs.')
make_table(
    headers=['Model', 'Top-1(%)', 'Top-5(%)', 'Params(M)', 'MACs(M)', 'Lat.(ms)', 'Mem(MB)'],
    rows=[
        ('ResNet-18',         '58.58', '80.40', '11.27', '556.76', ' 3.13', '326.7'),
        ('HBCC-Medium+KD',    '58.90', '81.41', ' 1.80', '138.86', ' 7.57', '481.6'),
        ('HBCC-Small+KD',     '57.56', '80.76', ' 1.32', ' 92.38', ' 7.50', '363.7'),
        ('HBCC-Medium (CE)',  '56.80', '79.13', ' 1.80', '138.86', ' 8.20', '481.6'),
        ('HBCC-Small (CE)',   '56.15', '79.14', ' 1.32', ' 92.38', ' 7.77', '363.7'),
        ('ShuffleNetV2-x1.0', '55.68', '79.32', ' 1.46', ' 46.32', ' 6.04', ' 95.6'),
        ('MobileNetV2',       '49.97', '75.35', ' 2.48', ' 25.80', ' 4.66', '124.3'),
        ('CoC-baseline',      '49.30', '73.89', ' 2.46', ' 34.43', '11.52', ' 72.0'),
    ])

para('CE models are strong before KD.', bold=True)
para('Even without knowledge distillation, HBCC-Medium-CE (56.80%) and HBCC-Small-CE '
     '(56.15%) both exceed ShuffleNetV2 (55.68%), reflecting a natural fit between the '
     'hybrid local–cluster architecture and fine-grained multi-class classification. '
     'Tiny-ImageNet-200 is the only dataset where all four HBCC variants (CE and KD) '
     'outperform every lightweight baseline.')

para('Why CoC-baseline and MobileNetV2 underperform.', bold=True)
para('CoC-baseline (49.30%) ranks lowest: without a local branch, pure context clustering '
     'cannot capture texture cues needed for 200 fine-grained classes. The +9.60% gap vs. '
     'HBCC-Medium+KD exceeds the CIFAR-10 gap (+7.02%), confirming that the local branch '
     'grows more valuable with classification complexity. Figure 3 visualizes the Pareto '
     'frontier.')

heading(2, 'C. Knowledge Distillation Effectiveness')

para('Table VI compares CE and KD performance on Tiny-ImageNet-200, the dataset where both '
     'training regimes share the same architecture, enabling a clean isolated measurement '
     'of distillation gain (Δ). HBCC-Medium improves by +2.10 points and HBCC-Small '
     'by +1.41 points. The Medium gain is sufficient for HBCC-Medium+KD to surpass the '
     'ResNet-18 teacher itself on the test set.')

tbl_caption('Table VI. Top-1 accuracy gain from KD (Δ = KD − CE, same '
            'architecture) on Tiny-ImageNet-200.')
make_table(
    headers=['Model', 'CE (%)', 'KD (%)', 'Δ'],
    rows=[
        ('HBCC-Medium', '56.80', '58.90', '+2.10'),
        ('HBCC-Small',  '56.15', '57.56', '+1.41'),
    ])

heading(2, 'D. Performance Analysis')

heading(3, '1. Pareto Analysis')

para('Figures 3–5 plot accuracy vs. parameter count and accuracy vs. latency on the '
     'two small-image datasets. Both HBCC-Small+KD and HBCC-Medium+KD occupy the '
     'Pareto-dominant region on the parameter axis: higher accuracy at lower parameter '
     'counts than every lightweight baseline on both benchmarks. On Tiny-ImageNet-200 '
     '(Figure 3), HBCC-Medium+KD exceeds ResNet-18 in accuracy while maintaining only '
     '1.80M parameters — the single point simultaneously above and to the left of '
     'ResNet-18 on both axes.')
para('Figure 5 shows the parameter-only Pareto for Food-101: HBCC-2.5M and ResNet-18 '
     'represent two different trade-off points, with ResNet-18 0.38 percentage points '
     'higher in top-1 accuracy and HBCC using 4.38× fewer parameters.')

figure('pareto_tin.png',
       'Fig. 3. Pareto frontier on Tiny-ImageNet-200. Arrows: KD gains. '
       'HBCC-Medium+KD (★) surpasses ResNet-18 on both axes with 6.26× fewer parameters.',
       width_in=3.1)

figure('pareto_cifar10.png',
       'Fig. 4. Pareto frontier on CIFAR-10. HBCC+KD occupies the Pareto-dominant '
       'region on both the parameter and latency axes relative to all lightweight baselines.',
       width_in=3.1)

figure('pareto_food101.png',
       'Fig. 5. Parameter efficiency on Food-101 (parameter axis only). '
       'HBCC-2.5M (★) reaches 82.53% with 4.38× fewer parameters than ResNet-18.',
       width_in=3.1)

heading(3, '2. Latency and Peak Memory Analysis')

para('Latency: Fewer MACs, Yet Slower.', bold=True)
para('Although HBCC-Medium uses fewer MACs than ResNet-18 (138.9M vs. 556.8M), its '
     'batch-1 latency is higher (≈8 ms vs. ≈3 ms). The cause is operator '
     'fragmentation: ResNet-18 spends ≈94% of its CUDA time in a single heavily '
     'optimized kernel (cudnn_convolution), while HBCC distributes time across many small '
     'sequential kernels — 1×1 convolutions, batch matrix multiplication, vector '
     'normalization, scatter/gather for clustering assignment. HBCC nevertheless runs '
     '30–35% faster than pure CoC (8 ms vs. 11 ms), demonstrating that the hybrid '
     'design materially reduces latency relative to global-only clustering.')

para('Peak Memory: Scatter Buffers Dominate.', bold=True)
para('HBCC-Medium reaches 481.6 MB — 47% above ResNet-18 (326.7 MB) despite 6.26× '
     'fewer parameters. The root cause mirrors the latency bottleneck: scatter/gather '
     'aggregation must simultaneously hold a similarity tensor (N×K) and a feature sum '
     '(K×C) per stage; with stem_stride=1 these buffers are 4× larger than with '
     'stride 2. CoC-baseline (72.0 MB) avoids this via stem_stride=2 and narrower '
     'embeddings. Reducing stem_stride or Stage-1 embedding width are the most direct '
     'levers to cut both latency and memory.')

heading(2, 'E. Discussion on Suitability and Application Scope')

para('Across the three benchmarks, HBCC\'s performance varies in a way that directly '
     'reflects the interaction between its hybrid design and the nature of each '
     'classification task.')

para('CIFAR-10: when CNN inductive bias dominates.', bold=True)
para('With 10 classes and 32×32 images, decision boundaries in CIFAR-10 are closely '
     'tied to local texture. ResNet-18 benefits from a strong inductive bias toward '
     'translation equivariance and local neighbourhood structure — a natural fit for '
     'this setting — yielding a 1.12% advantage. This is not a fundamental limitation '
     'of hybrid architectures in general; rather, it reflects the fact that when the class '
     'space is simple enough, local features alone suffice to discriminate, and context '
     'clustering provides no additional discriminative signal. Notably, this 1.12% gap is '
     'produced by a model with over 6× more parameters — a trade-off that is '
     'entirely acceptable under strict resource constraints.')

para('Tiny-ImageNet-200: where context clustering excels.', bold=True)
para('With 200 fine-grained semantic categories, the model must relate distant image regions '
     'to distinguish visually similar classes. This is precisely where context clustering '
     'has a natural advantage: its point-assignment and aggregation mechanism captures '
     'inter-region relationships that local convolution misses. HBCC-Medium+KD surpasses '
     'the ResNet-18 teacher by 0.32% top-1 with 6.26× fewer parameters — the '
     'only dataset in our experiments where the student outperforms the teacher. This '
     'confirms that as class-space complexity grows, HBCC\'s contextual advantage fully '
     'compensates for the local inductive bias of CNNs.')

para('Food-101: scalability and balance.', bold=True)
para('With 101 classes at 224×224 resolution, Food-101 lies between the two extremes: '
     'more complex than CIFAR-10, yet food categories still exhibit strong local cues '
     '(colour, surface texture). At 224 resolution, feature maps are large enough for '
     'clustering to operate effectively and avoid token degeneracy, while the hybrid design '
     'preserves local texture information in the early stages. The gap with ResNet-18 '
     'narrows to just 0.38% top-1 with 4.38× fewer parameters, confirming that HBCC '
     'scales to standard-resolution inputs without sacrificing its parameter efficiency.')

para('When HBCC is the right choice.', bold=True)
para('Synthesising across all three datasets, HBCC is most appropriate when: '
     '(i) the class space is large (≽100 categories), so that inter-class semantic '
     'relationships are complex and local features alone are insufficient; '
     '(ii) input resolution is high enough to provide each cluster with an adequate number '
     'of tokens; and '
     '(iii) the parameter budget is tightly constrained while competitive accuracy is still '
     'required. '
     'Conversely, for few-class tasks with no resource constraints, a standard lightweight '
     'CNN remains more efficient, thanks to its inductive bias that naturally matches the '
     'regular grid structure of images.')

para('In terms of practical deployment, HBCC targets edge scenarios requiring fine-grained '
     'multi-class recognition: food identification, plant disease classification, wildlife '
     'species monitoring — tasks that demand fine semantic discrimination while running '
     'on constrained hardware. The results on Tiny-ImageNet-200 and Food-101 highlight the '
     'promise of this direction; the CIFAR-10 gap simultaneously clarifies where further '
     'architectural refinement is needed to close the distance with pure CNNs on simpler '
     'classification problems.')

# ── V. CONCLUSION ──────────────────────────────────────────────────────────────
heading(1, 'V. Conclusion')

para('HBCC is a four-stage hybrid architecture combining Context Cluster with progressive '
     'local processing (LBPConv at Stage 1, DWConv at Stage 2, clustering at Stages 3–4), '
     'co-designed with stem_stride=1 and stage-shrinking folds to prevent token degeneracy. '
     'On small images, HBCC-Medium+KD reaches 95.36% (CIFAR-10) and 58.90% '
     '(Tiny-ImageNet-200) — surpassing the ResNet-18 teacher (58.58%) with 6.26× '
     'fewer parameters, and widening the gap over CoC-baseline from 7.02 to 9.60 pp as '
     'task complexity increases. On Food-101 (224×224), HBCC-2.5M achieves 82.53% '
     'top-1 with 4.38× fewer parameters than ResNet-18, demonstrating that the hybrid '
     'design scales to standard-resolution classification.')

para('Limitations: narrow margins should be confirmed with multiple seeds; Food-101 results '
     'reflect architecture-plus-recipe performance; and operator fragmentation keeps latency '
     'and memory above convolutional baselines with comparable MACs. These three directions '
     'are the priorities for future work.')

# ── REFERENCES ─────────────────────────────────────────────────────────────────
heading(1, 'References')

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

    '[6] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution '
    'gray-scale and rotation invariant texture classification with local binary patterns," '
    'IEEE Trans. Pattern Anal. Mach. Intell., vol. 24, no. 7, pp. 971–987, 2002.',

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
