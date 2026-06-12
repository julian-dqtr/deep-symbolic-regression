"""
feynman_ground_truth.py
=======================
Official Feynman SR benchmark ground truth expressions and variable metadata.
Source: Udrescu & Tegmark (2020) — AI Feynman: A Physics-Inspired Method for
        Symbolic Regression. Science Advances.
        https://github.com/SJ001/AI-Feynman

Variables are named as in the original paper (m_0, v, c, etc.) and mapped to
PMLB column order (x0, x1, x2, ...) based on the variable listing in the table.

Difficulty classification:
  Easy   — ≤3 variables, polynomial or simple ratio structure
  Medium — 3-5 variables, one non-linear function (sqrt, trig, exp)
  Hard   — 5+ variables or deeply nested non-linear structure
"""

from typing import Dict

# ---------------------------------------------------------------------------
# Ground truth — corrected from official AI Feynman table
# Keys match PMLB dataset names.
# 'expr' uses original variable names from the paper.
# 'vars' lists variables in PMLB column order (= x0, x1, x2, ...).
# 'difficulty' is our classification for analysis.
# ---------------------------------------------------------------------------

FEYNMAN_GROUND_TRUTH: Dict[str, Dict] = {
    # --- Series I ---
    "feynman_I_6_2": {
        "expr": "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        "vars": ["sigma", "theta"],
        "difficulty": "Medium",
    },
    "feynman_I_6_2a": {
        "expr": "exp(-theta**2/2)/sqrt(2*pi)",
        "vars": ["theta"],
        "difficulty": "Easy",
    },
    "feynman_I_6_2b": {
        "expr": "exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        "vars": ["sigma", "theta", "theta1"],
        "difficulty": "Medium",
    },
    "feynman_I_8_14": {
        "expr": "sqrt((x2-x1)**2+(y2-y1)**2)",
        "vars": ["x1", "x2", "y1", "y2"],
        "difficulty": "Easy",
    },
    "feynman_I_9_18": {
        "expr": "G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)",
        "vars": ["m1", "m2", "G", "x1", "x2", "y1", "y2", "z1", "z2"],
        "difficulty": "Hard",
    },
    "feynman_I_10_7": {
        "expr": "m_0/sqrt(1-v**2/c**2)",
        "vars": ["m_0", "v", "c"],
        "difficulty": "Medium",
    },
    "feynman_I_11_19": {
        "expr": "x1*y1+x2*y2+x3*y3",
        "vars": ["x1", "x2", "x3", "y1", "y2", "y3"],
        "difficulty": "Easy",
    },
    "feynman_I_12_1": {
        "expr": "mu*Nn",
        "vars": ["mu", "Nn"],
        "difficulty": "Easy",
    },
    "feynman_I_12_2": {
        "expr": "q1*q2*r/(4*pi*epsilon*r**3)",
        "vars": ["q1", "q2", "epsilon", "r"],
        "difficulty": "Medium",
    },
    "feynman_I_12_4": {
        "expr": "q1*r/(4*pi*epsilon*r**3)",
        "vars": ["q1", "epsilon", "r"],
        "difficulty": "Medium",
    },
    "feynman_I_12_5": {
        "expr": "q2*Ef",
        "vars": ["q2", "Ef"],
        "difficulty": "Easy",
    },
    "feynman_I_12_11": {
        "expr": "q*(Ef+B*v*sin(theta))",
        "vars": ["q", "Ef", "B", "v", "theta"],
        "difficulty": "Medium",
    },
    "feynman_I_13_4": {
        "expr": "1/2*m*(v**2+u**2+w**2)",
        "vars": ["m", "v", "u", "w"],
        "difficulty": "Easy",
    },
    "feynman_I_13_12": {
        "expr": "G*m1*m2*(1/r2-1/r1)",
        "vars": ["m1", "m2", "r1", "r2", "G"],
        "difficulty": "Medium",
    },
    "feynman_I_14_3": {
        "expr": "m*g*z",
        "vars": ["m", "g", "z"],
        "difficulty": "Easy",
    },
    "feynman_I_14_4": {
        "expr": "1/2*k_spring*x**2",
        "vars": ["k_spring", "x"],
        "difficulty": "Easy",
    },
    "feynman_I_15_3x": {
        "expr": "(x-u*t)/sqrt(1-u**2/c**2)",
        "vars": ["x", "u", "c", "t"],
        "difficulty": "Hard",
    },
    "feynman_I_15_3t": {
        "expr": "(t-u*x/c**2)/sqrt(1-u**2/c**2)",
        "vars": ["x", "c", "u", "t"],
        "difficulty": "Hard",
    },
    "feynman_I_15_10": {
        "expr": "m_0*v/sqrt(1-v**2/c**2)",
        "vars": ["m_0", "v", "c"],
        "difficulty": "Hard",
    },
    "feynman_I_16_6": {
        "expr": "(u+v)/(1+u*v/c**2)",
        "vars": ["c", "v", "u"],
        "difficulty": "Medium",
    },
    "feynman_I_18_4": {
        "expr": "(m1*r1+m2*r2)/(m1+m2)",
        "vars": ["m1", "m2", "r1", "r2"],
        "difficulty": "Easy",
    },
    "feynman_I_18_12": {
        "expr": "r*F*sin(theta)",
        "vars": ["r", "F", "theta"],
        "difficulty": "Easy",
    },
    "feynman_I_18_14": {
        "expr": "m*r*v*sin(theta)",
        "vars": ["m", "r", "v", "theta"],
        "difficulty": "Easy",
    },
    "feynman_I_24_6": {
        "expr": "1/2*m*(omega**2+omega_0**2)*1/2*x**2",
        "vars": ["m", "omega", "omega_0", "x"],
        "difficulty": "Medium",
    },
    "feynman_I_25_13": {
        "expr": "q/C",
        "vars": ["q", "C"],
        "difficulty": "Easy",
    },
    "feynman_I_26_2": {
        "expr": "arcsin(n*sin(theta2))",
        "vars": ["n", "theta2"],
        "difficulty": "Medium",
    },
    "feynman_I_27_6": {
        "expr": "1/(1/d1+n/d2)",
        "vars": ["d1", "d2", "n"],
        "difficulty": "Medium",
    },
    "feynman_I_29_4": {
        "expr": "omega/c",
        "vars": ["omega", "c"],
        "difficulty": "Easy",
    },
    "feynman_I_29_16": {
        "expr": "sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))",
        "vars": ["x1", "x2", "theta1", "theta2"],
        "difficulty": "Medium",
    },
    "feynman_I_30_3": {
        "expr": "Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
        "vars": ["Int_0", "theta", "n"],
        "difficulty": "Hard",
    },
    "feynman_I_30_5": {
        "expr": "arcsin(lambd/(n*d))",
        "vars": ["lambd", "d", "n"],
        "difficulty": "Medium",
    },
    "feynman_I_32_5": {
        "expr": "q**2*a**2/(6*pi*epsilon*c**3)",
        "vars": ["q", "a", "epsilon", "c"],
        "difficulty": "Medium",
    },
    "feynman_I_32_17": {
        "expr": "(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
        "vars": ["epsilon", "c", "Ef", "r", "omega", "omega_0"],
        "difficulty": "Hard",
    },
    "feynman_I_34_1": {
        "expr": "omega_0/(1-v/c)",
        "vars": ["c", "v", "omega_0"],
        "difficulty": "Easy",
    },
    "feynman_I_34_8": {
        "expr": "q*v*B/p",
        "vars": ["q", "v", "B", "p"],
        "difficulty": "Easy",
    },
    "feynman_I_34_14": {
        "expr": "(1+v/c)/sqrt(1-v**2/c**2)*omega_0",
        "vars": ["c", "v", "omega_0"],
        "difficulty": "Hard",
    },
    "feynman_I_34_27": {
        "expr": "(h/(2*pi))*omega",
        "vars": ["omega", "h"],
        "difficulty": "Easy",
    },
    "feynman_I_37_4": {
        "expr": "I1+I2+2*sqrt(I1*I2)*cos(delta)",
        "vars": ["I1", "I2", "delta"],
        "difficulty": "Medium",
    },
    "feynman_I_38_12": {
        "expr": "4*pi*epsilon*(h/(2*pi))**2/(m*q**2)",
        "vars": ["m", "q", "h", "epsilon"],
        "difficulty": "Medium",
    },
    "feynman_I_39_1": {
        "expr": "3/2*pr*V",
        "vars": ["pr", "V"],
        "difficulty": "Easy",
    },
    "feynman_I_39_11": {
        "expr": "1/(gamma-1)*pr*V",
        "vars": ["gamma", "pr", "V"],
        "difficulty": "Easy",
    },
    "feynman_I_39_22": {
        "expr": "n*kb*T/V",
        "vars": ["n", "T", "V", "kb"],
        "difficulty": "Easy",
    },
    "feynman_I_40_1": {
        "expr": "n_0*exp(-m*g*x/(kb*T))",
        "vars": ["n_0", "m", "x", "T", "g", "kb"],
        "difficulty": "Medium",
    },
    "feynman_I_41_16": {
        "expr": "h/(2*pi)*omega**3/(pi**2*c**2*(exp((h/(2*pi))*omega/(kb*T))-1))",
        "vars": ["omega", "T", "h", "kb", "c"],
        "difficulty": "Hard",
    },
    "feynman_I_43_16": {
        "expr": "mu_drift*q*Volt/d",
        "vars": ["mu_drift", "q", "Volt", "d"],
        "difficulty": "Easy",
    },
    "feynman_I_43_31": {
        "expr": "mob*kb*T",
        "vars": ["mob", "T", "kb"],
        "difficulty": "Easy",
    },
    "feynman_I_43_43": {
        "expr": "1/(gamma-1)*kb*v/A",
        "vars": ["gamma", "kb", "A", "v"],
        "difficulty": "Medium",
    },
    "feynman_I_44_4": {
        "expr": "n*kb*T*ln(V2/V1)",
        "vars": ["n", "kb", "T", "V1", "V2"],
        "difficulty": "Medium",
    },
    "feynman_I_47_23": {
        "expr": "sqrt(gamma*pr/rho)",
        "vars": ["gamma", "pr", "rho"],
        "difficulty": "Easy",
    },
    "feynman_I_48_2": {
        "expr": "m*c**2/sqrt(1-v**2/c**2)",
        "vars": ["m", "v", "c"],
        "difficulty": "Hard",
    },
    "feynman_I_50_26": {
        "expr": "x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
        "vars": ["x1", "omega", "t", "alpha"],
        "difficulty": "Medium",
    },
    # --- Series II ---
    "feynman_II_2_42": {
        "expr": "kappa*(T2-T1)*A/d",
        "vars": ["kappa", "T1", "T2", "A", "d"],
        "difficulty": "Easy",
    },
    "feynman_II_3_24": {
        "expr": "Pwr/(4*pi*r**2)",
        "vars": ["Pwr", "r"],
        "difficulty": "Easy",
    },
    "feynman_II_4_23": {
        "expr": "q/(4*pi*epsilon*r)",
        "vars": ["q", "epsilon", "r"],
        "difficulty": "Easy",
    },
    "feynman_II_6_11": {
        "expr": "1/(4*pi*epsilon)*p_d*cos(theta)/r**2",
        "vars": ["epsilon", "p_d", "theta", "r"],
        "difficulty": "Medium",
    },
    "feynman_II_6_15a": {
        "expr": "p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
        "vars": ["epsilon", "p_d", "r", "x", "y", "z"],
        "difficulty": "Hard",
    },
    "feynman_II_6_15b": {
        "expr": "p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
        "vars": ["epsilon", "p_d", "theta", "r"],
        "difficulty": "Medium",
    },
    "feynman_II_8_7": {
        "expr": "3/5*q**2/(4*pi*epsilon*d)",
        "vars": ["q", "epsilon", "d"],
        "difficulty": "Easy",
    },
    "feynman_II_8_31": {
        "expr": "epsilon*Ef**2/2",
        "vars": ["epsilon", "Ef"],
        "difficulty": "Easy",
    },
    "feynman_II_10_9": {
        "expr": "sigma_den/epsilon*1/(1+chi)",
        "vars": ["sigma_den", "epsilon", "chi"],
        "difficulty": "Easy",
    },
    "feynman_II_11_3": {
        "expr": "q*Ef/(m*(omega_0**2-omega**2))",
        "vars": ["q", "Ef", "m", "omega_0", "omega"],
        "difficulty": "Medium",
    },
    "feynman_II_11_17": {
        "expr": "n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
        "vars": ["n_0", "kb", "T", "theta", "p_d", "Ef"],
        "difficulty": "Medium",
    },
    "feynman_II_11_20": {
        "expr": "n_rho*p_d**2*Ef/(3*kb*T)",
        "vars": ["n_rho", "p_d", "Ef", "kb", "T"],
        "difficulty": "Medium",
    },
    "feynman_II_11_27": {
        "expr": "n*alpha/(1-(n*alpha/3))*epsilon*Ef",
        "vars": ["n", "alpha", "epsilon", "Ef"],
        "difficulty": "Hard",
    },
    "feynman_II_11_28": {
        "expr": "1+n*alpha/(1-(n*alpha/3))",
        "vars": ["n", "alpha"],
        "difficulty": "Medium",
    },
    "feynman_II_13_17": {
        "expr": "1/(4*pi*epsilon*c**2)*2*I/r",
        "vars": ["epsilon", "c", "I", "r"],
        "difficulty": "Easy",
    },
    "feynman_II_13_23": {
        "expr": "rho_c_0/sqrt(1-v**2/c**2)",
        "vars": ["rho_c_0", "v", "c"],
        "difficulty": "Medium",
    },
    "feynman_II_13_34": {
        "expr": "rho_c_0*v/sqrt(1-v**2/c**2)",
        "vars": ["rho_c_0", "v", "c"],
        "difficulty": "Hard",
    },
    "feynman_II_15_4": {
        "expr": "-mom*B*cos(theta)",
        "vars": ["mom", "B", "theta"],
        "difficulty": "Easy",
    },
    "feynman_II_15_5": {
        "expr": "-p_d*Ef*cos(theta)",
        "vars": ["p_d", "Ef", "theta"],
        "difficulty": "Easy",
    },
    "feynman_II_21_32": {
        "expr": "q/(4*pi*epsilon*r*(1-v/c))",
        "vars": ["q", "epsilon", "r", "v", "c"],
        "difficulty": "Medium",
    },
    "feynman_II_24_17": {
        "expr": "sqrt(omega**2/c**2-pi**2/d**2)",
        "vars": ["omega", "c", "d"],
        "difficulty": "Medium",
    },
    "feynman_II_27_16": {
        "expr": "epsilon*c*Ef**2",
        "vars": ["epsilon", "c", "Ef"],
        "difficulty": "Easy",
    },
    "feynman_II_27_18": {
        "expr": "epsilon*Ef**2",
        "vars": ["epsilon", "Ef"],
        "difficulty": "Easy",
    },
    "feynman_II_34_2": {
        "expr": "q*v*r/2",
        "vars": ["q", "v", "r"],
        "difficulty": "Easy",
    },
    "feynman_II_34_2a": {
        "expr": "q*v/(2*pi*r)",
        "vars": ["q", "v", "r"],
        "difficulty": "Easy",
    },
    "feynman_II_34_11": {
        "expr": "g_*q*B/(2*m)",
        "vars": ["g_", "q", "B", "m"],
        "difficulty": "Easy",
    },
    "feynman_II_34_29a": {
        "expr": "q*h/(4*pi*m)",
        "vars": ["q", "h", "m"],
        "difficulty": "Easy",
    },
    "feynman_II_34_29b": {
        "expr": "g_*mom*B*Jz/(h/(2*pi))",
        "vars": ["g_", "h", "Jz", "mom", "B"],
        "difficulty": "Medium",
    },
    "feynman_II_35_18": {
        "expr": "n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
        "vars": ["n_0", "kb", "T", "mom", "B"],
        "difficulty": "Hard",
    },
    "feynman_II_35_21": {
        "expr": "n_rho*mom*tanh(mom*B/(kb*T))",
        "vars": ["n_rho", "mom", "B", "kb", "T"],
        "difficulty": "Hard",
    },
    "feynman_II_36_38": {
        "expr": "mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M",
        "vars": ["mom", "H", "kb", "T", "alpha", "epsilon", "c", "M"],
        "difficulty": "Hard",
    },
    "feynman_II_37_1": {
        "expr": "mom*(1+chi)*B",
        "vars": ["mom", "B", "chi"],
        "difficulty": "Easy",
    },
    "feynman_II_38_3": {
        "expr": "Y*A*x/d",
        "vars": ["Y", "A", "d", "x"],
        "difficulty": "Easy",
    },
    "feynman_II_38_14": {
        "expr": "Y/(2*(1+sigma))",
        "vars": ["Y", "sigma"],
        "difficulty": "Easy",
    },
    # --- Series III ---
    "feynman_III_4_32": {
        "expr": "1/(exp((h/(2*pi))*omega/(kb*T))-1)",
        "vars": ["h", "omega", "kb", "T"],
        "difficulty": "Hard",
    },
    "feynman_III_4_33": {
        "expr": "(h/(2*pi))*omega/(exp((h/(2*pi))*omega/(kb*T))-1)",
        "vars": ["h", "omega", "kb", "T"],
        "difficulty": "Hard",
    },
    "feynman_III_7_38": {
        "expr": "2*mom*B/(h/(2*pi))",
        "vars": ["mom", "B", "h"],
        "difficulty": "Easy",
    },
    "feynman_III_8_54": {
        "expr": "sin(E_n*t/(h/(2*pi)))**2",
        "vars": ["E_n", "t", "h"],
        "difficulty": "Medium",
    },
    "feynman_III_9_52": {
        "expr": "(p_d*Ef*t/(h/(2*pi)))*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
        "vars": ["p_d", "Ef", "t", "h", "omega", "omega_0"],
        "difficulty": "Hard",
    },
    "feynman_III_10_19": {
        "expr": "mom*sqrt(Bx**2+By**2+Bz**2)",
        "vars": ["mom", "Bx", "By", "Bz"],
        "difficulty": "Easy",
    },
    "feynman_III_12_43": {
        "expr": "n*(h/(2*pi))",
        "vars": ["n", "h"],
        "difficulty": "Easy",
    },
    "feynman_III_13_18": {
        "expr": "2*E_n*d**2*k/(h/(2*pi))",
        "vars": ["E_n", "d", "k", "h"],
        "difficulty": "Easy",
    },
    "feynman_III_14_14": {
        "expr": "I_0*(exp(q*Volt/(kb*T))-1)",
        "vars": ["I_0", "q", "Volt", "kb", "T"],
        "difficulty": "Hard",
    },
    "feynman_III_15_12": {
        "expr": "2*U*(1-cos(k*d))",
        "vars": ["U", "k", "d"],
        "difficulty": "Medium",
    },
    "feynman_III_15_14": {
        "expr": "(h/(2*pi))**2/(2*E_n*d**2)",
        "vars": ["h", "E_n", "d"],
        "difficulty": "Medium",
    },
    "feynman_III_15_27": {
        "expr": "2*pi*alpha/(n*d)",
        "vars": ["alpha", "n", "d"],
        "difficulty": "Easy",
    },
    "feynman_III_17_37": {
        "expr": "beta*(1+alpha*cos(theta))",
        "vars": ["beta", "alpha", "theta"],
        "difficulty": "Easy",
    },
    "feynman_III_19_51": {
        "expr": "-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)",
        "vars": ["m", "q", "h", "n", "epsilon"],
        "difficulty": "Hard",
    },
    "feynman_III_21_20": {
        "expr": "-rho_c_0*q*A_vec/m",
        "vars": ["rho_c_0", "q", "A_vec", "m"],
        "difficulty": "Easy",
    },
}

# Quality thresholds (NMSE-based) — shared with evaluate_expressions.py
NMSE_PERFECT = 0.001
NMSE_GOOD    = 0.05
DIFF_ORDER   = ["Easy", "Medium", "Hard", "Unknown"]


def classify_quality(nmse: float) -> str:
    if nmse < NMSE_PERFECT:
        return "Perfect"
    if nmse < NMSE_GOOD:
        return "Good"
    return "Poor"


def get_expr(task_name: str) -> str:
    """Return the ground-truth expression for a task, or 'unknown'."""
    return FEYNMAN_GROUND_TRUTH.get(task_name, {}).get("expr", "unknown")


def get_difficulty(task_name: str) -> str:
    """Return difficulty tier for a task."""
    return FEYNMAN_GROUND_TRUTH.get(task_name, {}).get("difficulty", "Unknown")


def get_vars(task_name: str):
    """Return the ordered list of original variable names."""
    return FEYNMAN_GROUND_TRUTH.get(task_name, {}).get("vars", [])
