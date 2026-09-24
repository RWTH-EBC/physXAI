import math
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import pandas as pd

class CasadiConstrainedFit:
    def __init__(self, degree=3):
        self.degree = degree
        self.opti = ca.Opti()
        self.weights_val = None
        self.term_names = [] # Zum Speichern der Namen (z.B. "x0*x1^2")
        
    def fit(self, X, y, monotonic_features=[], convex_features=[]):
        if X.ndim == 1: X = X[:, None]
        N, n_features = X.shape
        
        # --- 1. Symbolisches Modell ---
        w_sym_list = []
        x_sym = ca.MX.sym('x', n_features)
        y_sym = 0
        self.term_names = []
        
        # Iteriere über alle Polynom-Grade
        for d in range(self.degree + 1):
            for indices in combinations_with_replacement(range(n_features), d):
                w_i = self.opti.variable()
                
                # Term bauen und Namen speichern
                term = 1
                name_parts = []
                for idx in indices:
                    term *= x_sym[idx]
                    name_parts.append(f"x{idx}")
                
                term_name = "*".join(name_parts) if name_parts else "Bias"
                self.term_names.append(term_name)
                
                y_sym += w_i * term
                w_sym_list.append(w_i)
        
        w_vec = ca.vertcat(*w_sym_list)

        # --- 2. Ableitungen (Wichtig: gradient statt jacobian) ---
        grad_sym = ca.gradient(y_sym, x_sym)     
        hess_sym, _ = ca.hessian(y_sym, x_sym)   
        
        f_pred = ca.Function('f_pred', [x_sym, w_vec], [y_sym])
        f_grad = ca.Function('f_grad', [x_sym, w_vec], [grad_sym])
        f_hess = ca.Function('f_hess', [x_sym, w_vec], [ca.diag(hess_sym)]) 

        # --- 3. Zielfunktion ---
        y_pred_all = f_pred.map(N)(X.T, ca.repmat(w_vec, 1, N))
        error = y_pred_all - y.reshape(1, N)
        obj = ca.mtimes(error, error.T)
        self.opti.minimize(obj)

        # --- 4. Constraints ---
        if monotonic_features:
            grad_all = f_grad.map(N)(X.T, ca.repmat(w_vec, 1, N))
            for feat_idx in monotonic_features:
                self.opti.subject_to(grad_all[feat_idx, :].T >= 0)

        if convex_features:
            hess_all = f_hess.map(N)(X.T, ca.repmat(w_vec, 1, N))
            for feat_idx in convex_features:
                self.opti.subject_to(hess_all[feat_idx, :].T >= 0)

        # --- 5. Lösen ---
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'expand': True}
        self.opti.solver('ipopt', opts)
        
        try:
            sol = self.opti.solve()
            self.weights_val = sol.value(w_vec)
        except:
            self.weights_val = self.opti.debug.value(w_vec)

        self.predict_fn = f_pred
        self.grad_fn = f_grad

    def predict(self, X):
        if X.ndim == 1: X = X[:, None]
        N = X.shape[0]
        res = self.predict_fn.map(N)(X.T, ca.repmat(self.weights_val, 1, N))
        return res.full().flatten()

    def predict_derivative(self, X, feature_idx):
        if X.ndim == 1: X = X[:, None]
        N = X.shape[0]
        grad = self.grad_fn.map(N)(X.T, ca.repmat(self.weights_val, 1, N))
        return grad.full()[feature_idx, :].flatten()
    
    def inspect_weights(self):
        """Zeigt die gelernten Gewichte tabellarisch an."""
        df = pd.DataFrame({
            'Term': self.term_names,
            'Weight': self.weights_val
        })
        # Filtere kleine Gewichte raus für Übersichtlichkeit
        print("\n--- Gelernte Gewichte (Top Beiträge) ---")
        print(df[df['Weight'].abs() > 1e-4])
        print("----------------------------------------\n")

# --- ANWENDUNG ---

np.random.seed(42)
N = 100

x1 = np.linspace(0, 3, N) 
x2 = np.random.uniform(0, 3, N)

X_train = np.column_stack([x1, x2])

# Wahre Funktion
y_true = x1 ** 3 + x2 * x1 ** 2 
y_noise = y_true 





# Fitting
model = CasadiConstrainedFit(degree=3) 
model.fit(X_train, y_noise, monotonic_features=[], convex_features=[])

# Gewichte prüfen - schauen Sie, ob x0*x0 (also x1^2) ein hohes Gewicht hat!
model.inspect_weights()

# Plots
y_pred = model.predict(X_train)
dy_dx1 = model.predict_derivative(X_train, feature_idx=0)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Regression
# Wir sortieren für den Plot, sonst sieht es bei zufälligem x2 wüst aus
sort_idx = np.argsort(x1)
ax[0].scatter(x1, y_noise, color='gray', alpha=0.3, label='Daten')
ax[0].plot(x1[sort_idx], y_pred[sort_idx], 'r-', linewidth=2, label='CasADi Fit')
ax[0].set_title('Fit (y vs x1)')
ax[0].legend()

# Plot 2: Ableitung
# Da x2 zufällig ist, wackelt die Ableitung dy/dx1 leicht (wegen Interaktionstermen),
# aber der Trend (lineares Ansteigen, da Funktion quadratisch) muss sichtbar sein.
ax[1].plot(x1[sort_idx], dy_dx1[sort_idx], 'b.', markersize=2, label="dy/dx1 (Punktwolke)")
# Trendlinie der Ableitung
ax[1].plot(x1[sort_idx], np.poly1d(np.polyfit(x1, dy_dx1, 2))(x1[sort_idx]), 'b-', alpha=0.3)
ax[1].axhline(0, color='red', linestyle='--', label="Monotonie-Grenze")
ax[1].set_title('Ableitung nach x1')
ax[1].legend()

plt.show()