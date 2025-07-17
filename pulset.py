
import os, re, json, joblib, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from collections import Counter

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# %% [markdown]
# ## Config
# Set these to match your dataset. If the path doesn't exist, you'll be prompted to upload.

# %%
DATA_PATH = '/content/chat_dataset.csv'  # change if needed
TEXT_COL = None     # e.g., 'message'; leave None to auto-detect
LABEL_COL = None    # e.g., 'sentiment'; leave None to auto-detect
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
ALPHA = 0.1  # NB smoothing

SHOW_PLOTS = True
SHOW_EXAMPLE_PREDICTIONS = True
SAVE_MODEL_PATH = '/content/sentiment_model.pkl'  # set None to skip saving

# %% [markdown]
# ## Utility: Upload file if not found

# %%
try:
    from google.colab import files  # only works in Colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not os.path.exists(DATA_PATH):
    if IN_COLAB:
        print(f"File not found at {DATA_PATH}. Please upload a CSV…")
        uploaded = files.upload()
        # take first uploaded file
        if uploaded:
            DATA_PATH = list(uploaded.keys())[0]
            print(f"Using uploaded file: {DATA_PATH}")
    else:
        raise FileNotFoundError(f"CSV not found at {DATA_PATH} and not running in Colab for upload.")

# %% [markdown]
# ## Load data & basic exploration

# %%
def auto_detect_columns(df, text_col=None, label_col=None):
    """Try to guess text & label column names if not provided."""
    if text_col is None:
        for c in df.columns:
            if c.lower() in {'message','text','review','comment','content','body'}:
                text_col = c; break
    if label_col is None:
        for c in df.columns:
            if c.lower() in {'sentiment','label','target','polarity','class'}:
                label_col = c; break
    if text_col is None or label_col is None:
        raise ValueError("Could not auto-detect TEXT_COL and/or LABEL_COL. Please set them explicitly.")
    return text_col, label_col


def load_and_explore_data(path, text_col=None, label_col=None):
    df = pd.read_csv(path)
    text_col, label_col = auto_detect_columns(df, text_col, label_col)

    print("\n" + "="*70)
    print("DATASET EXPLORATION")
    print("="*70)
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Text column: {text_col} | Label column: {label_col}")

    # Class distribution
    print("\nClass distribution:")
    counts = df[label_col].value_counts(dropna=False)
    for k,v in counts.items():
        pct = v/len(df)*100
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Missing
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Length stats (char & word)
    df['_char_len'] = df[text_col].astype(str).str.len()
    df['_word_cnt'] = df[text_col].astype(str).str.split().str.len()
    print("\nCharacter length stats:")
    print(df['_char_len'].describe())
    print("\nWord count stats:")
    print(df['_word_cnt'].describe())

    return df, text_col, label_col


# Load
raw_df, TEXT_COL, LABEL_COL = load_and_explore_data(DATA_PATH, TEXT_COL, LABEL_COL)

# %% [markdown]
# ## Visualise raw class balance & message lengths

# %%
def plot_raw_eda(df, text_col, label_col):
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    # class balance
    sns.countplot(x=label_col, data=df, ax=axes[0])
    axes[0].set_title('Label Counts')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    # message length
    df['_char_len'] = df[text_col].astype(str).str.len()
    sns.histplot(df['_char_len'], bins=50, ax=axes[1])
    axes[1].set_title('Message Length (chars)')
    axes[1].set_xlabel('Chars')
    plt.tight_layout()
    plt.show()

if SHOW_PLOTS:
    plot_raw_eda(raw_df, TEXT_COL, LABEL_COL)

# %% [markdown]
# ## Preprocess text
# - Lowercase
# - Remove punctuation & digits (keep words)
# - Collapse whitespace
# - Drop NA / empty rows

# %%
TEXT_CLEAN_COL = 'message_processed'

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ''
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)   # drop punctuation
    s = re.sub(r"\d+", " ", s)        # drop digits (optional)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_data(df, text_col, label_col):
    df = df.copy()
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    df[TEXT_CLEAN_COL] = df[text_col].apply(clean_text)
    df = df[df[TEXT_CLEAN_COL] != '']

    # Feature add‑ons (optional, exploratory)
    df['message_length'] = df[text_col].str.len()
    df['word_count'] = df[text_col].str.split().str.len()
    df['exclamation_count'] = df[text_col].str.count('!')
    df['question_count'] = df[text_col].str.count('\?')

    print(f"Preprocessing done: {df.shape[0]} rows remain.")
    return df

proc_df = preprocess_data(raw_df, TEXT_COL, LABEL_COL)

# %% [markdown]
# ## Visualise processed data (pie + boxplots + top words)

# %%
def visualize_data_exploration(df, label_col):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Exploration Visualisations', fontsize=16, y=1.02)

    # 1. Class pie
    counts = df[label_col].value_counts()
    axes[0,0].pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('Class Distribution')

    # 2. Char length by class
    df.boxplot(column='message_length', by=label_col, ax=axes[0,1])
    axes[0,1].set_title('Message Length by Class')
    axes[0,1].set_xlabel('')

    # 3. Word count by class
    df.boxplot(column='word_count', by=label_col, ax=axes[1,0])
    axes[1,0].set_title('Word Count by Class')
    axes[1,0].set_xlabel('')

    # 4. Top words table
    sentiments = df[label_col].unique()
    word_freq = {s: Counter() for s in sentiments}
    for s in sentiments:
        words = ' '.join(df.loc[df[label_col]==s, TEXT_CLEAN_COL]).split()
        word_freq[s] = Counter(words).most_common(5)
    axes[1,1].axis('tight'); axes[1,1].axis('off')
    table_data = []
    for s in sentiments:
        top_words = ', '.join([f"{w}({c})" for w,c in word_freq[s]])
        table_data.append([s, top_words])
    table = axes[1,1].table(cellText=table_data, colLabels=['Class','Top 5 Words'], loc='center', cellLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1,2)

    plt.tight_layout(); plt.show()

if SHOW_PLOTS:
    visualize_data_exploration(proc_df, LABEL_COL)

# %% [markdown]
# ## Train/test split

# %%
X = proc_df[TEXT_CLEAN_COL].values
y = proc_df[LABEL_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Classes: {np.unique(y)}")

# %% [markdown]
# ## Build & train model (TF‑IDF + MultinomialNB)

# %%
def build_model(max_features=5000, ngram_range=(1,2), alpha=0.1):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            use_idf=True,
            sublinear_tf=True,
        )),
        ('clf', MultinomialNB(alpha=alpha)),
    ])

model = build_model(MAX_FEATURES, NGRAM_RANGE, ALPHA)
print("Training model…")
model.fit(X_train, y_train)
print("Done.")

# %% [markdown]
# ## Predictions & probabilities

# %%
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
classes = model.classes_

# %% [markdown]
# ## Metrics computation helper

# %%
def compute_metrics(y_true, y_pred, classes):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    m['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    m['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    m['precision'] = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    m['f1_score'] = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    m['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    m['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    m['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    m['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    m['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return m

metrics = compute_metrics(y_test, y_pred, classes)

print("\n=== METRICS SUMMARY ===")
for k,v in metrics.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: {np.round(v,4)}")
    else:
        print(f"{k}: {v:.4f}")

# %% [markdown]
# ## Classification report (text)

# %%
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, zero_division=0))

# %% [markdown]
# ## Visualise model performance

# %%
def visualize_model_performance(model, X_test, y_test, y_pred, y_proba, classes):
    n_classes = len(classes)
    fig = plt.figure(figsize=(20, 16))

    # Confusion Matrix
    ax1 = plt.subplot(3,3,1)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar_kws={'label':'Count'}, ax=ax1)
    ax1.set_title('Confusion Matrix'); ax1.set_ylabel('True'); ax1.set_xlabel('Pred')

    # Normalised CM
    ax2 = plt.subplot(3,3,2)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=classes, yticklabels=classes, cbar_kws={'label':'Proportion'}, ax=ax2)
    ax2.set_title('Normalised Confusion Matrix'); ax2.set_ylabel('True'); ax2.set_xlabel('Pred')

    # Classification report heatmap
    ax3 = plt.subplot(3,3,3)
    rpt = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    rpt_df = pd.DataFrame(rpt).transpose()
    rpt_df = rpt_df.iloc[:-3, :-1]  # drop support & averages rows
    sns.heatmap(rpt_df, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label':'Score'}, ax=ax3)
    ax3.set_title('Classification Report')

    # ROC OvR
    ax4 = plt.subplot(3,3,4)
    y_bin = label_binarize(y_test, classes=classes)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_proba[:,i])
        roc_auc = auc(fpr, tpr)
        ax4.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
    ax4.plot([0,1],[0,1], 'k--', lw=0.75)
    ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR'); ax4.set_title('ROC (OvR)'); ax4.legend()

    # Precision-Recall curves
    ax5 = plt.subplot(3,3,5)
    for i, cls in enumerate(classes):
        pr, rc, _ = precision_recall_curve(y_bin[:,i], y_proba[:,i])
        ap = average_precision_score(y_bin[:,i], y_proba[:,i])
        ax5.plot(rc, pr, label=f'{cls} (AP={ap:.3f})')
    ax5.set_xlabel('Recall'); ax5.set_ylabel('Precision'); ax5.set_title('PR Curves'); ax5.legend()

    # Top features for each class (first class shown; see helper below for full table)
    ax6 = plt.subplot(3,3,6)
    vec = model.named_steps['tfidf']; clf = model.named_steps['clf']
    feats = vec.get_feature_names_out()
    class_to_plot = classes[0]
    ci = list(classes).index(class_to_plot)
    top_idx = np.argsort(clf.feature_log_prob_[ci])[-10:][::-1]
    words = feats[top_idx]
    scores = clf.feature_log_prob_[ci][top_idx]
    ax6.barh(range(len(words)), scores)
    ax6.set_yticks(range(len(words))); ax6.set_yticklabels(words)
    ax6.invert_yaxis(); ax6.set_title(f'Top 10 Features: {class_to_plot}')

    # Prediction confidence dist
    ax7 = plt.subplot(3,3,7)
    max_proba = np.max(y_proba, axis=1)
    ax7.hist(max_proba, bins=30, alpha=0.7, edgecolor='black')
    ax7.axvline(max_proba.mean(), color='red', linestyle='--', label=f'Mean={max_proba.mean():.3f}')
    ax7.set_title('Prediction Confidence Dist'); ax7.set_xlabel('Max prob'); ax7.set_ylabel('Count'); ax7.legend()

    # Confidence: correct vs errors
    ax8 = plt.subplot(3,3,8)
    errors = (y_test != y_pred)
    err_conf = max_proba[errors]
    cor_conf = max_proba[~errors]
    ax8.hist(cor_conf, bins=20, alpha=0.5, label='Correct')
    ax8.hist(err_conf, bins=20, alpha=0.5, label='Errors')
    ax8.set_title('Confidence: Correct vs Errors'); ax8.set_xlabel('Max prob'); ax8.set_ylabel('Count'); ax8.legend()

    # Class-wise bar
    ax9 = plt.subplot(3,3,9)
    metrics_df = pd.DataFrame({
        'Precision': rpt_df['precision'],
        'Recall': rpt_df['recall'],
        'F1': rpt_df['f1-score'],
    })
    metrics_df.plot(kind='bar', ax=ax9)
    ax9.set_title('Per-Class Metrics'); ax9.set_ylabel('Score'); ax9.set_xlabel('Class'); ax9.set_ylim(0,1)
    plt.suptitle('Model Performance Summary', fontsize=16, y=0.995)
    plt.tight_layout(); plt.show()

if SHOW_PLOTS:
    visualize_model_performance(model, X_test, y_test, y_pred, y_proba, classes)

# %% [markdown]
# ## Detailed top features per class table

# %%
def get_top_features_per_class(model, top_n=20):
    vec = model.named_steps['tfidf']; clf = model.named_steps['clf']
    feats = np.array(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        top_idx = np.argsort(clf.feature_log_prob_[i])[-top_n:][::-1]
        out[cls] = list(zip(feats[top_idx], clf.feature_log_prob_[i][top_idx]))
    return out

feat_dict = get_top_features_per_class(model, top_n=20)
for cls, lst in feat_dict.items():
    print(f"\nTop features for class '{cls}':")
    for word,score in lst:
        print(f"  {word:20s} {score:.3f}")

# %% [markdown]
# ## Error analysis table (misclassified rows)

# %%
err_mask = (y_test != y_pred)
err_df = pd.DataFrame({
    'text': X_test[err_mask],
    'true': y_test[err_mask],
    'pred': y_pred[err_mask],
    'max_conf': np.max(y_proba[err_mask], axis=1),
})
print(f"Misclassified: {len(err_df)} / {len(X_test)}")
err_df.head()

# %% [markdown]
# ## Cross‑validation (StratifiedKFold)

# %%
def perform_cross_validation(model, X, y, cv_folds=5):
    print(f"\nPerforming {cv_folds}-fold CV…")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = ['accuracy','precision_weighted','recall_weighted','f1_weighted']
    results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
        results[metric] = scores
        print(f"{metric:18s}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8,5))
    pos = np.arange(len(scoring_metrics))
    for i, metric in enumerate(scoring_metrics):
        ax.boxplot(results[metric], positions=[i], widths=0.5)
        ax.scatter(np.repeat(i, len(results[metric])), results[metric], s=80, alpha=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels([m.replace('_',' ').title() for m in scoring_metrics])
    ax.set_ylabel('Score'); ax.set_title(f'{cv_folds}-Fold CV Results'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    return results

if SHOW_PLOTS:
    cv_results = perform_cross_validation(build_model(MAX_FEATURES, NGRAM_RANGE, ALPHA), X, y, CV_FOLDS)
else:
    cv_results = perform_cross_validation(build_model(MAX_FEATURES, NGRAM_RANGE, ALPHA), X, y, CV_FOLDS)

# %% [markdown]
# ## Example predictions visualised

# %%
def visualize_predictions(model, messages, classes):
    preds = model.predict(messages)
    proba = model.predict_proba(messages)
    # Build DF
    rows = []
    for i,msg in enumerate(messages):
        row = {'message': msg, 'prediction': preds[i], 'confidence': np.max(proba[i])}
        for j,cls in enumerate(classes):
            row[f'prob_{cls}'] = proba[i,j]
        rows.append(row)
    dfp = pd.DataFrame(rows)

    print("\nExample predictions:")
    display(dfp)

    # Stacked bar
    msgs_short = [m[:40] + ('…' if len(m)>40 else '') for m in messages]
    x = np.arange(len(messages))
    bottom = np.zeros(len(messages))
    plt.figure(figsize=(12,6))
    for j,cls in enumerate(classes):
        plt.bar(x, proba[:,j], bottom=bottom, label=cls)
        bottom += proba[:,j]
    plt.xticks(x, msgs_short, rotation=45, ha='right')
    plt.ylabel('Probability'); plt.title('Prediction Probabilities'); plt.legend(); plt.tight_layout(); plt.show()

    # Confidence bar
    plt.figure(figsize=(12,4))
    cols = ['green' if p==classes[0] else 'red' if p==classes[-1] else 'gray' for p in preds]  # simple heuristic
    plt.bar(x, dfp['confidence'], color=cols)
    for i,v in enumerate(dfp['confidence']):
        plt.text(i, v+0.01, preds[i], ha='center', va='bottom', fontsize=8)
    plt.xticks(x, msgs_short, rotation=45, ha='right')
    plt.ylim(0,1)
    plt.ylabel('Confidence'); plt.title('Prediction Confidence'); plt.tight_layout(); plt.show()

if SHOW_EXAMPLE_PREDICTIONS:
    demo_messages = [
        "Why isn't the class starting?",
        "Thanks! That was really helpful.",
        "This is absolutely terrible",
        "I love this course!",
        "I'm confused about this topic",
        "The service was okay, nothing special",
        "I'm extremely disappointed with this product",
        "Best experience ever! Highly recommend!",
        "Not sure if this is worth the price",
        "Worst customer service I've ever experienced",
    ]
    visualize_predictions(model, demo_messages, classes)

# %% [markdown]
# ## Save model (optional)

# %%
def save_model(model, metrics, path):
    if path is None: return
    bundle = {
        'model': model,
        'metrics': metrics,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    joblib.dump(bundle, path)
    print(f"Model saved -> {path}")

save_model(model, metrics, SAVE_MODEL_PATH)

# %% [markdown]
# ## Load model (demo)

# %%
def load_model(path):
    if not path or not os.path.exists(path):
        print(f"No model at {path}."); return None, None
    bundle = joblib.load(path)
    print(f"Loaded model trained at {bundle.get('timestamp','?')}")
    return bundle['model'], bundle.get('metrics',{})

_, _ = load_model(SAVE_MODEL_PATH)

# %% [markdown]
# ## Wrap everything in a single callable (optional)
# Use this if you want to re‑run programmatically.

# %%
def run_pipeline(csv_path,
                 text_col=None,
                 label_col=None,
                 test_size=0.2,
                 random_state=42,
                 max_features=5000,
                 ngram_range=(1,2),
                 alpha=0.1,
                 show_plots=True,
                 show_examples=False,
                 save_model_path=None):
    raw_df, text_col, label_col = load_and_explore_data(csv_path, text_col, label_col)
    if show_plots: plot_raw_eda(raw_df, text_col, label_col)
    proc_df = preprocess_data(raw_df, text_col, label_col)
    if show_plots: visualize_data_exploration(proc_df, label_col)

    X = proc_df[TEXT_CLEAN_COL].values
    y = proc_df[label_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    model = build_model(max_features, ngram_range, alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_
    metrics = compute_metrics(y_test, y_pred, classes)
    print(classification_report(y_test, y_pred, zero_division=0))
    if show_plots: visualize_model_performance(model, X_test, y_test, y_pred, y_proba, classes)
    cv_results = perform_cross_validation(build_model(max_features, ngram_range, alpha), X, y, CV_FOLDS)
    if show_examples:
        demo_messages = ["Thanks that was great","Horrible experience","It was okay"]
        visualize_predictions(model, demo_messages, classes)
    save_model(model, metrics, save_model_path)
    return {
        'raw_df': raw_df,
        'proc_df': proc_df,
        'model': model,
        'metrics': metrics,
        'cv_results': cv_results,
        'classes': classes,
    }
