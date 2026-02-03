import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# --- Page Setup / إعداد الصفحة ---
st.set_page_config(page_title="Optimization: Min Distance", layout="wide")

# --- Custom CSS / تنسيقات خاصة ---
# We force LTR for the whole page to ensure equations are correct.
# We use a specific class for Arabic text if needed to align right, but standard bilingual is usually LTR.
st.markdown("""
<style>
    /* Force Left-to-Right for correct Equation rendering */
    .main { direction: ltr; }
    
    /* Style for bilingual headers */
    h1, h2, h3 { font-family: sans-serif; }
    
    /* Make metrics stand out */
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Title / العنوان ---
st.title("📏 Minimizing Distance / تطبيقات القيم الصغرى: أقل مسافة")
st.markdown("""
**Problem:** Find the point on the curve $y = x^2$ that is closest to the point $(0, y_0)$.
<br>
**المسألة:** أوجد النقطة على المنحنى $y = x^2$ التي تكون أقرب ما يمكن للنقطة $(0, y_0)$.
""", unsafe_allow_html=True)
st.divider()

# --- Sidebar (Inputs) / القائمة الجانبية ---
with st.sidebar:
    st.header("Settings / الإعدادات")
    
    # Target Point / النقطة الثابتة
    target_y = st.number_input(
        "Fixed Point Y (0, y) / إحداثي ص للنقطة الثابتة", 
        value=1.0, step=0.5
    )
    
    st.divider()
    st.info("Move the slider to change X / حرك المؤشر لتغيير قيمة س")
    
    # Moving Point X / النقطة المتحركة
    x_val = st.slider(
        "Point X position / موقع النقطة س", 
        -2.0, 2.0, 1.5, 0.05
    )

# --- Math Logic / العمليات الحسابية ---
def curve_func(x):
    return x**2

def distance_func(x, y0):
    # Distance formula: sqrt((x-0)^2 + (y-y0)^2)
    return np.sqrt(x**2 + (x**2 - y0)**2)

# Finding the optimal solution mathematically
# Deriving D^2 = x^2 + (x^2 - y0)^2
# Resulting critical points logic:
optimal_points = []
if target_y <= 0.5:
    optimal_x = 0.0
    optimal_points.append(0.0)
else:
    val = target_y - 0.5
    opt_x_positive = np.sqrt(val)
    opt_x_negative = -np.sqrt(val)
    optimal_x = opt_x_positive
    optimal_points = [opt_x_negative, opt_x_positive]

min_dist = distance_func(optimal_x, target_y)
current_dist = distance_func(x_val, target_y)

# --- Metrics / عرض النتائج ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current X / س الحالية", f"{x_val:.2f}")
col2.metric("Distance / المسافة", f"{current_dist:.2f}")

# Formatting optimal solution text
solutions_text = " , ".join([f"±{abs(p):.2f}" for p in optimal_points]) if len(optimal_points) > 1 else f"{optimal_points[0]:.2f}"
col3.metric("Optimal X / الحل الأمثل", solutions_text, delta_color="off")
col4.metric("Min Distance / أقل مسافة", f"{min_dist:.2f}", delta_color="off")

st.divider()

# --- Visualization / الرسومات البيانية ---
c1, c2 = st.columns([1, 1])

# Plot 1: Geometry
with c1:
    st.subheader("1. Geometric Representation / التمثيل الهندسي")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    
    x_range = np.linspace(-2.5, 2.5, 200)
    ax1.plot(x_range, x_range**2, label='$y=x^2$', color='#2c3e50', linewidth=2)
    
    # Fixed Point
    ax1.scatter([0], [target_y], color='black', s=100, zorder=5, label=f'Fixed (0, {target_y})')
    
    # Moving Point
    curr_y = curve_func(x_val)
    ax1.scatter([x_val], [curr_y], color='red', s=100, zorder=5, label='Moving / متحركة')
    
    # Connection Line
    ax1.plot([0, x_val], [target_y, curr_y], color='red', linestyle='--', linewidth=2)
    
    # Optimal Points
    for opt in optimal_points:
        opt_y = curve_func(opt)
        ax1.scatter([opt], [opt_y], color='#27ae60', s=80, zorder=4, marker='X')

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_title(f"Distance: {current_dist:.2f}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    st.pyplot(fig1)

# Plot 2: Optimization Function
with c2:
    st.subheader("2. Optimization Function / دالة الهدف")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    
    dist_y_vals = distance_func(x_range, target_y)
    ax2.plot(x_range, dist_y_vals, label='Distance $D(x)$', color='#e67e22', linewidth=2)
    
    # Current Point
    ax2.scatter([x_val], [current_dist], color='red', s=100, zorder=5)
    
    # Minima
    for opt in optimal_points:
        d_opt = distance_func(opt, target_y)
        ax2.scatter([opt], [d_opt], color='green', zorder=5)
        ax2.text(opt, d_opt + 0.1, 'min', ha='center', color='green', fontweight='bold')

    ax2.set_xlabel('x')
    ax2.set_ylabel('Distance')
    ax2.set_ylim(0, max(dist_y_vals))
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

# --- Equations / المعادلات ---
st.divider()
st.header("Mathematical Derivation / الاشتقاق الرياضي")

# Formatting helpers
y0_fmt = f"{int(target_y)}" if target_y.is_integer() else f"{target_y:.1f}"

st.markdown("##### 1. Distance Formula / قانون المسافة")
st.latex(rf"D(x) = \sqrt{{(x - 0)^2 + (x^2 - {y0_fmt})^2}}")

st.markdown("##### 2. Simplify by minimizing $D^2$ / التبسيط بتربيع الدالة")
st.markdown("We minimize $f(x) = D^2$ to avoid the square root derivative:")
# Equation logic
term_x2 = 1 - 2*target_y
term_c = target_y**2
term_x2_str = f"{int(term_x2)}" if term_x2.is_integer() else f"{term_x2:.1f}"
term_c_str = f"{int(term_c)}" if term_c.is_integer() else f"{term_c:.2f}"

# Handling signs for clean display
sign_str = "+" if term_x2 >= 0 else "" # value already has sign if negative

st.latex(rf"f(x) = x^4 {sign_str} {term_x2_str} x^2 + {term_c_str}")

st.markdown("##### 3. Derivative / المشتقة الأولى")
st.markdown("Find $f'(x)$ and set it to 0:")

diff_term = 2 * term_x2
diff_term_str = f"{int(diff_term)}" if diff_term.is_integer() else f"{diff_term:.1f}"
sign_diff = "+" if diff_term >= 0 else ""

st.latex(rf"f'(x) = 4x^3 {sign_diff} {diff_term_str} x = 0")
st.latex(rf"x(4x^2 {sign_diff} {diff_term_str}) = 0")

st.markdown("##### 4. Critical Points / النقاط الحرجة")
st.markdown(f"""
Solutions are $x = 0$ or roots of $(4x^2 {sign_diff} {diff_term_str} = 0)$.
""")

if target_y > 0.5:
    res = np.sqrt(target_y - 0.5)
    st.success(f"Since $y_0 > 0.5$, we have minimums at: $x = \pm {res:.2f}$")
else:
    st.warning(f"Since $y_0 \leq 0.5$, the only minimum is at vertex: $x = 0$")
