import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Minimum Distance Problem", layout="wide")

# --- CSS للعربية ---
st.markdown("""
<style>
    .main { direction: rtl; }
    h1, h2, h3, p, div { text-align: right; }
    .stMetric { text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- العنوان ---
st.title("📏 برنامج محاكاة: أقرب نقطة على المنحنى")
st.markdown("""
المسألة: **أوجد النقطة على المنحنى $y = x^2$ التي تكون أقرب ما يمكن للنقطة $(0, y_0)$.**
""")
st.divider()

# --- القائمة الجانبية (المدخلات) ---
with st.sidebar:
    st.header("إعدادات المسألة")
    
    # النقطة الثابتة (في السؤال هي 0,1)
    # جعلتها متغيرة لكي يستطيع المعلم تغيير الرقم 1 إلى أي رقم آخر للشرح
    target_y = st.number_input("إحداثي ص للنقطة الثابتة (0, y)", value=1.0, step=0.5)
    
    st.divider()
    st.info("حرك النقطة على المنحنى:")
    
    # النقطة المتحركة x
    x_val = st.slider("موقع النقطة x", -2.0, 2.0, 1.5, 0.05)

# --- الحسابات الرياضية ---
# الدالة الأساسية: y = x^2
def curve_func(x):
    return x**2

# دالة المسافة (أو مربع المسافة لتسهيل الاشتقاق)
# D^2 = (x - 0)^2 + (y - y0)^2
# D^2 = x^2 + (x^2 - y0)^2
def distance_sq_func(x, y0):
    return x**2 + (x**2 - y0)**2

def distance_func(x, y0):
    return np.sqrt(distance_sq_func(x, y0))

# --- الحل الرياضي (Optimization) ---
# نشتق دالة مربع المسافة بالنسبة لـ x
# f(x) = x^2 + (x^2 - y0)^2
# f'(x) = 2x + 2(x^2 - y0)(2x)
# f'(x) = 2x [ 1 + 2(x^2 - y0) ]
# f'(x) = 2x [ 1 + 2x^2 - 2y0 ]
# النقاط الحرجة: إما x=0 أو القوس = 0
# 2x^2 = 2y0 - 1  =>  x^2 = y0 - 0.5

optimal_points = [] # قائمة لتخزين الحلول
if target_y <= 0.5:
    # إذا كانت النقطة قريبة جداً من الرأس، يكون الرأس هو الأقرب
    optimal_x = 0.0
    optimal_points.append(0.0)
else:
    # وإلا يوجد حلان (يمين ويسار)
    val = target_y - 0.5
    opt_x_positive = np.sqrt(val)
    opt_x_negative = -np.sqrt(val)
    optimal_x = opt_x_positive # نختار الموجب للعرض الرقمي
    optimal_points = [opt_x_negative, opt_x_positive]

min_dist = distance_func(optimal_x, target_y)
current_dist = distance_func(x_val, target_y)

# --- عرض الأرقام ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("موقع x الحالي", f"{x_val:.2f}")
col2.metric("المسافة الحالية", f"{current_dist:.2f}")
# عرض الحلول المثلى
solutions_text = " , ".join([f"{p:.2f}" for p in optimal_points])
col3.metric("قيم x المثلى (الحل)", solutions_text, delta_color="off")
col4.metric("أقل مسافة ممكنة", f"{min_dist:.2f}", delta_color="off")

st.divider()

# --- الرسومات البيانية ---
c1, c2 = st.columns([1, 1])

# الرسم الأول: الهندسة (المنحنى والنقطة)
with c1:
    st.subheader("1. التمثيل الهندسي (Geometry)")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    
    # رسم المنحنى y=x^2
    x_range = np.linspace(-2.5, 2.5, 200)
    ax1.plot(x_range, x_range**2, label='$y=x^2$', color='#2c3e50', linewidth=2)
    
    # رسم النقطة الثابتة (0, y0)
    ax1.scatter([0], [target_y], color='black', s=100, zorder=5, label=f'Fixed (0, {target_y})')
    
    # رسم النقطة المتحركة
    curr_y = curve_func(x_val)
    ax1.scatter([x_val], [curr_y], color='red', s=100, zorder=5, label='Moving Point')
    
    # رسم خط المسافة بينهما
    ax1.plot([0, x_val], [target_y, curr_y], color='red', linestyle='--', linewidth=2)
    
    # رسم الحلول المثلى (نقاط خضراء)
    for opt in optimal_points:
        opt_y = curve_func(opt)
        ax1.scatter([opt], [opt_y], color='#27ae60', s=80, zorder=4, marker='X')

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-0.5, 3.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

# الرسم الثاني: دالة المسافة (Optimization)
with c2:
    st.subheader("2. دالة تقليل المسافة (Minimizing Distance)")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    
    # رسم منحنى دالة المسافة D(x)
    dist_y_vals = distance_func(x_range, target_y)
    ax2.plot(x_range, dist_y_vals, label='Distance $D(x)$', color='#e67e22', linewidth=2)
    
    # النقطة الحالية على منحنى المسافة
    ax2.scatter([x_val], [current_dist], color='red', s=100, zorder=5)
    
    # النقاط الصغرى (Minima)
    for opt in optimal_points:
        d_opt = distance_func(opt, target_y)
        ax2.scatter([opt], [d_opt], color='green', zorder=5)
        ax2.text(opt, d_opt + 0.1, f'min', ha='center', color='green')

    ax2.set_xlabel('x coordinate')
    ax2.set_ylabel('Distance')
    ax2.set_ylim(0, max(dist_y_vals))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

# --- المعادلات الديناميكية ---
st.divider()
st.subheader("شرح الخطوات رياضياً (ديناميكي):")

# تنسيق الأرقام
y0_str = f"{int(target_y)}" if target_y.is_integer() else f"{target_y:.1f}"

# المعادلة 1: قانون المسافة
st.markdown("##### 1. نكتب قانون المسافة بين نقطة عشوائية $(x, x^2)$ والنقطة الثابتة $(0, y_0)$:")
st.latex(rf"D = \sqrt{{(x - 0)^2 + (x^2 - {y0_str})^2}}")

# المعادلة 2: التبسيط
st.markdown("##### 2. للسهولة، نقلل مربع المسافة $f(x) = D^2$ (لأن الجذر لا يغير موقع القيم القصوى):")
st.latex(rf"f(x) = x^2 + (x^4 - 2({y0_str})x^2 + {y0_str}^2)")
term_x2 = 1 - 2*target_y
term_x2_str = f"{int(term_x2)}" if term_x2.is_integer() else f"{term_x2:.1f}"
st.latex(rf"f(x) = x^4 + ({term_x2_str})x^2 + {float(target_y)**2:.2f}")

# المعادلة 3: المشتقة
st.markdown("##### 3. نوجد المشتقة ونساويها بالصفر:")
diff_term = 2 * term_x2
diff_term_str = f"{int(diff_term)}" if diff_term.is_integer() else f"{diff_term:.1f}"

st.latex(rf"f'(x) = 4x^3 + ({diff_term_str})x = 0")
st.markdown("نأخذ $x$ عامل مشترك:")
st.latex(rf"x(4x^2 + {diff_term_str}) = 0")

st.info("""
لاحظ هنا: 
* إما $x = 0$ (وهي نقطة عظمى محلية في هذا الرسم لأنها "سنام" المسافة).
* أو نحل القوس لإيجاد القيم الصغرى (أقصر مسافة).
""")
