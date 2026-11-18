1. Print what pandas actually sees so you can verify
print("Columns found:", df.columns.tolist())

# 2. If pandas lumped everything into 1 column because of tabs/spaces, reload properly:
# df = pd.read_csv('your_file_name.csv', sep=r'\t|\s{2,}', engine='python')

# 3. Automatically grab whichever column has "precip" in its name
precip_col = [col for col in df.columns if 'precip' in col.lower()][0]

# 4. Clean and convert to 0 and 1
precip_numeric = pd.to_numeric(
    df[precip_col].astype(str).str.replace(' in', '', regex=False).str.strip(),
    errors='coerce'
)

df['Precip_Binary'] = (precip_numeric > 0).astype(int)
print(df[[precip_col, 'Precip_Binary']].head())

# %%
precip_numeric = pd.to_numeric(
    df.iloc[:, -1].astype(str).str.replace(' in', '', regex=False).str.strip(),
    errors='coerce'
)

df['Precip_Binary'] = (precip_numeric > 0).astype(int)

# %%
print("Columns:", df.columns.tolist())
print(df.head(3))

# %%
# 1. Split the single jammed column on commas into 8 separate columns
raw_col_name = df.columns[0]
col_names = [c.strip() for c in raw_col_name.split(',')]

clean_df = df[raw_col_name].astype(str).str.split(',', expand=True)
clean_df.columns = col_names

# 2. Clean and convert every column to numeric values
clean_df['Month']       = pd.to_datetime(clean_df['Date'], errors='coerce').dt.month
clean_df['Temp_High']   = pd.to_numeric(clean_df['Temp High (°F)'], errors='coerce')
clean_df['Temp_Low']    = pd.to_numeric(clean_df['Temp Low (°F)'], errors='coerce')

dew_split               = clean_df['Dew Pt High/Low'].str.split('/', expand=True)
clean_df['Dew_Pt_High'] = pd.to_numeric(dew_split[0].str.replace('°F', '', regex=False).str.strip(), errors='coerce')
clean_df['Dew_Pt_Low']  = pd.to_numeric(dew_split[1].str.replace('°F', '', regex=False).str.strip(), errors='coerce')

clean_df['Humidity']    = pd.to_numeric(clean_df['Humidity'].str.replace('%', '', regex=False).str.strip(), errors='coerce')
clean_df['Max_Wind']    = pd.to_numeric(clean_df['Max Wind'].str.replace('mph', '', regex=False).str.strip(), errors='coerce')
clean_df['Pressure']    = pd.to_numeric(clean_df['Pressure'].str.replace('in', '', regex=False).str.strip(), errors='coerce')

# 3. Properly create the binary target variable (0 = 0 in, 1 = > 0 in)
precip_numeric = pd.to_numeric(clean_df['Precip.'].str.replace('in', '', regex=False).str.strip(), errors='coerce')
clean_df['Precip_Binary'] = (precip_numeric > 0).astype(int)

# Verify the target counts (you should now see both 0s and 1s!)
print("Rainfall class counts:\n", clean_df['Precip_Binary'].value_counts(), "\n")

# 4. Prepare features (X) and target (y), dropping any malformed/truncated rows
feature_cols = ['Month', 'Temp_High', 'Temp_Low', 'Dew_Pt_High', 'Dew_Pt_Low', 'Humidity', 'Max_Wind', 'Pressure']
model_df = clean_df[feature_cols + ['Precip_Binary']].dropna()

X = model_df[feature_cols]
y = model_df['Precip_Binary']

# 5. Train/Test Split & Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = rf_model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Rain (0)', 'Rain (1)']))

# 7. Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# %%
# 1. Train a class-balanced Random Forest
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Penalizes missed rain events more heavily
    random_state=42
)
rf_balanced.fit(X_train, y_train)

# Default 0.50 threshold with class_weight='balanced'
y_pred_balanced = rf_balanced.predict(X_test)
print("=== 1. class_weight='balanced' (Default 0.50 Threshold) ===")
print(classification_report(y_test, y_pred_balanced, target_names=['No Rain (0)', 'Rain (1)']))

# 2. Adjust the probability threshold using predict_proba()
# Column 1 [:, 1] contains the predicted probability of Rain (1)
rain_probs = rf_balanced.predict_proba(X_test)[:, 1]

print("=== 2. Threshold Comparison for Rain (1) ===")
print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 48)

for thresh in [0.50, 0.45, 0.40, 0.35, 0.30]:
    y_pred_custom = (rain_probs >= thresh).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_custom, pos_label=1, average='binary')
    print(f"{thresh:<12.2f} {prec:<12.2%} {rec:<12.2%} {f1:<12.2f}")

# %%
best_threshold = 0.35
final_predictions = (rf_balanced.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)

# %%
# 1. Get predicted probabilities for the Rain (1) class
rain_probs = rf_balanced.predict_proba(X_test)[:, 1]

# 2. Calculate Precision, Recall, and Thresholds across all possible cutoffs
precisions, recalls, thresholds = precision_recall_curve(y_test, rain_probs)
pr_auc = average_precision_score(y_test, rain_probs)

# 3. Find the optimal threshold that maximizes the F1-score
# Note: precisions and recalls have 1 extra element at the end, so slice [:-1] to align with thresholds
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]
best_precision = precisions[best_idx]
best_recall    = recalls[best_idx]
best_f1        = f1_scores[best_idx]

# Find the closest index to the default 0.50 threshold for visual comparison
default_idx = np.argmin(np.abs(thresholds - 0.50))

# 4. Plot the Precision-Recall Curve
plt.figure(figsize=(9, 6))
plt.plot(recalls, precisions, color='#1f77b4', lw=2.5, label=f'Random Forest (PR-AUC = {pr_auc:.3f})')

# Mark the Optimal F1 Threshold
plt.scatter(
    best_recall, best_precision, 
    color='#d62728', s=120, zorder=5, 
    label=f'Optimal Threshold = {best_threshold:.2f} (F1 = {best_f1:.2f})'
)

# Mark the Default 0.50 Threshold
plt.scatter(
    recalls[default_idx], precisions[default_idx], 
    color='#ff7f0e', marker='s', s=100, zorder=5, 
    label=f'Default Threshold = 0.50 (F1 = {f1_scores[default_idx]:.2f})'
)

# Add dashed guide lines and annotation for the optimal point
plt.axvline(best_recall, color='#d62728', linestyle='--', alpha=0.4)
plt.axhline(best_precision, color='#d62728', linestyle='--', alpha=0.4)
plt.annotate(
    f'Cutoff: {best_threshold:.2f}\nRecall: {best_recall:.1%}\nPrecision: {best_precision:.1%}',
    xy=(best_recall, best_precision),
    xytext=(best_recall - 0.22, best_precision - 0.14),
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
    fontsize=10, fontweight='bold', color='#8c1d1d',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', edgecolor='#d62728', alpha=0.9)
)

# Baseline horizontal line (proportion of rainy days in test set)
baseline = y_test.mean()
plt.axhline(baseline, color='gray', linestyle=':', lw=1.5, label=f'No-Skill Baseline ({baseline:.1%} Rain)')

plt.title('Precision-Recall Curve for Rainfall Prediction', fontsize=14, fontweight='bold')
plt.xlabel('Recall (Proportion of Actual Rainy Days Caught)', fontsize=11)
plt.ylabel('Precision (Accuracy When Predicting Rain)', fontsize=11)
plt.xlim([0.0, 1.02])
plt.ylim([0.0, 1.05])
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.show()

# 5. Print the final classification report using the optimal threshold
print(f"=== Classification Report at Optimal Threshold ({best_threshold:.2f}) ===")
y_pred_optimal = (rain_probs >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=['No Rain (0)', 'Rain (1)']))
