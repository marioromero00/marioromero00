from scipy.stats import ks_2samp

def detect_drift(reference_df, new_df, feature_cols, threshold=0.05):

    #drift_detected = False: usado para testear el caso base, sin seleccion de drfi
    p_values = {}
    
    for col in feature_cols:
        stat, p_value = ks_2samp(
            reference_df[col].dropna(),
            new_df[col].dropna()
        )
        p_values[col] = p_value
        
        if p_value < threshold:
            print(f" Drift detectado en variable {col} (p-value={p_value:.4f})")
            drift_detected = True
        else:
            print(f"No hay drift en {col} (p-value={p_value:.4f})")
            
    return drift_detected, p_values
