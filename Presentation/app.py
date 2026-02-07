import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# MLflow imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from PIL import Image
    import os
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not installed. Install with: pip install mlflow pillow")

# Page configuration
st.set_page_config(
    page_title="Clickbait Detection | Gemma + UMAP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

# MLflow Settings - CHANGE THIS TO YOUR PATH!
MLFLOW_URI = "http://127.0.0.1:5000/"
EXPERIMENT_NAME = "Final_Evaluation_Rescaled"
USE_MLFLOW = True  # Set to False to disable MLflow features

# ============================================================================
# MLFLOW HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_mlflow_runs(_mlflow_uri, _experiment_name):
    """
    Load all runs from MLflow experiment
    
    Args:
        _mlflow_uri: MLflow tracking URI
        _experiment_name: Experiment name
        
    Returns:
        DataFrame with runs or None
    """
    if not MLFLOW_AVAILABLE or not USE_MLFLOW:
        return None
    
    try:
        mlflow.set_tracking_uri(_mlflow_uri)
        experiment = mlflow.get_experiment_by_name(_experiment_name)
        
        if experiment is None:
            st.error(f"❌ Experiment '{_experiment_name}' not found")
            st.info(f"💡 Create experiment by running a notebook with MLflow logging")
            return None
        
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_f1 DESC"]
        )
        
        return runs_df
        
    except Exception as e:
        st.error(f"❌ Error loading MLflow: {e}")
        return None


@st.cache_resource
def get_mlflow_client(_mlflow_uri):
    """Get MLflow client (cached)"""
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        mlflow.set_tracking_uri(_mlflow_uri)
        return MlflowClient()
    except Exception as e:
        st.error(f"Error creating MLflow client: {e}")
        return None

def load_artifact_image(run_id, artifact_name, mlflow_uri):
    """
    Load image artifact from MLflow run
    
    Args:
        run_id: MLflow run ID
        artifact_name: Artifact filename (e.g., 'confusion_matrix.png')
        mlflow_uri: MLflow tracking URI
        
    Returns:
        PIL Image or None
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    try:
        client = get_mlflow_client(mlflow_uri)
        if client is None:
            return None
        
        # Download artifact
        local_path = client.download_artifacts(run_id, artifact_name)
        
        # Load image
        img = Image.open(local_path)
        return img
        
    except FileNotFoundError:
        st.warning(f"⚠️ Artifact '{artifact_name}' not found in run {run_id[:8]}...")
        return None
    except Exception as e:
        st.error(f"❌ Error loading artifact: {e}")
        return None

def get_run_metrics(run_id, mlflow_uri):
    """Get all metrics for a specific run"""
    if not MLFLOW_AVAILABLE:
        return {}
    
    try:
        client = get_mlflow_client(mlflow_uri)
        if client is None:
            return {}
        
        run = client.get_run(run_id)
        return run.data.metrics
        
    except Exception as e:
        st.error(f"Error getting metrics: {e}")
        return {}

def get_run_params(run_id, mlflow_uri):
    """Get all parameters for a specific run"""
    if not MLFLOW_AVAILABLE:
        return {}
    
    try:
        client = get_mlflow_client(mlflow_uri)
        if client is None:
            return {}
        
        run = client.get_run(run_id)
        return run.data.params
        
    except Exception as e:
        st.error(f"Error getting parameters: {e}")
        return {}

# Custom CSS for better styling - Black, Grey, Orange theme
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B35 0%, #FF8C42 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2b2b2b;
        color: #e0e0e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 1rem 0;
    }
    .metric-card h3, .metric-card h4 {
        color: #FF8C42;
    }
    .insight-box {
        background-color: #1a1a1a;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    .stMetric {
        background-color: #2b2b2b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3a3a3a;
    }
    .stMetric label {
        color: #999 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FF6B35 !important;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation with custom styling
    st.sidebar.markdown("""
        <style>
        /* Custom Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a1a 0%, #2b2b2b 100%);
        }
        
        /* Custom Radio Button Styling */
        .stRadio > label {
            background-color: transparent !important;
        }
        
        .stRadio > div {
            gap: 0.5rem;
        }
        
        .stRadio > div > label {
            background: linear-gradient(135deg, #2b2b2b 0%, #1a1a1a 100%);
            border-left: 3px solid #FF6B35;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 4px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            display: flex;
            align-items: center;
        }
        
        .stRadio > div > label:hover {
            background: linear-gradient(135deg, #3a3a3a 0%, #2b2b2b 100%);
            border-left: 3px solid #FF8C42;
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(255, 107, 53, 0.3);
        }
        
        .stRadio > div > label[data-baseweb="radio"] > div:first-child {
            background-color: #FF6B35 !important;
            border-color: #FF6B35 !important;
        }
        
        .stRadio > div > label > div:last-child {
            color: #e0e0e0 !important;
            font-weight: 500;
            font-size: 15px;
        }
        
        /* Sidebar Title */
        .sidebar-title {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(90deg, #FF6B35 0%, #FF8C42 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        /* Info boxes in sidebar */
        .sidebar-info {
            background: #1a1a1a;
            border: 1px solid #FF6B35;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
            color: #e0e0e0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="sidebar-title"> Navigation</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "",  # Empty label since we have custom title
        [
            "🏠 Home",
            "📊 Dataset Overview",
            "🔬 Methodology",
            "🤖 ML Algorithms",
            "📈 Results & Analysis",
            "🔍 The Scaling Paradox",
            "🏆 Model Comparison",
            "📚 Conclusions"
        ],
        label_visibility="collapsed"
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div class="sidebar-info">
            <h3 style="color: #FF6B35; margin-top: 0;">🎯 Το Κεντρικό Ερώτημα</h3>
            <p style="font-size: 13px; line-height: 1.5;">
            "Μπορούμε να εντοπίσουμε τον εντυπωσιασμό 
            (sensationalism) όχι κοιτώντας τις λέξεις, 
            αλλά τη <strong>γεωμετρική θέση</strong> του κειμένου 
            στον σημασιολογικό χώρο;"
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div class="sidebar-info">
            <h3 style="color: #FF8C42; margin-top: 0;">🛠️ Tech Stack</h3>
            <ul style="font-size: 13px; line-height: 1.8; padding-left: 20px;">
                <li>Google Gemma (7B)</li>
                <li>UMAP (Manifold Learning)</li>
                <li>Gradient Boosting</li>
                <li>Optuna + MLflow</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Dataset Overview":
        show_dataset()
    elif page == "🔬 Methodology":
        show_methodology()
    elif page == "🤖 ML Algorithms":
        show_ml_algorithms()
    elif page == "🎯 Live Demo":
        show_demo()
    elif page == "📈 Results & Analysis":
        show_results()
    elif page == "🔍 The Scaling Paradox":
        show_scaling_paradox()
    elif page == "🏆 Model Comparison":
        show_model_comparison()
    elif page == "📚 Conclusions":
        show_conclusions()

def show_home():
    st.markdown('<div class="main-header">Ανίχνευση Clickbait</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">με χρήση Large Language Models & Τοπολογικής Ανάλυσης</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📖 ο Σκοπός της εργασίας")
        
        st.markdown("""
        ### 🎓 Υπότιτλος
        Μια προσέγγιση **Manifold Learning** με Google Gemma & UMAP
        
        ### ❓ Το Κεντρικό Ερώτημα
        *"Μπορούμε να εντοπίσουμε πως ένας τίτλος προσεγγίζει clicks, όχι κοιτώντας μεμονωμένα τις λέξεις,
        αλλά τη 'γεωμετρική θέση' του κειμένου στον σημασιολογικό χώρο;"*
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ### 🏛️ Οι Τρεις Πυλώνες της Εργασίας
        """)
        
        st.markdown("""
        <div class="metric-card">
            <h4>1. Beyond Keywords 🔤</h4>
            <p>Απομάκρυνση από προσεγγίσεις Bag-of-Words (π.χ. καταμέτρηση λέξεων όπως "ΣΟΚ").</p>
        </div>
        
        <div class="metric-card">
            <h4>2. Semantics (Σημασιολογία) 🧠</h4>
            <p>Χρήση του Gemma LLM για την κατανόηση του <strong>ύφους</strong>, του <strong>σαρκασμού</strong> και της <strong>δομής</strong>.</p>
        </div>
        
        <div class="metric-card">
            <h4>3. Geometry (Γεωμετρία) 📐</h4>
            <p>Χρήση του UMAP όχι απλά για μείωση διαστάσεων, αλλά για την αποκάλυψη της <strong>τοπολογικής δομής</strong> των δεδομένων.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("## 📊 Quick Stats")
        
        st.metric(
            label="Best Model Accuracy",
            value="91.02%",
            delta="Gradient Boosting"
        )
        
        st.metric(
            label="F1 Score",
            value="0.88",
            delta="Champion Model"
        )
        
        st.metric(
            label="Training Samples",
            value="~85,000",
            delta="Μετά καθαρισμό"
        )
        
        st.metric(
            label="UMAP Dimensions",
            value="500",
            delta="από 768"
        )
    
    st.markdown("---")
    
    # The Problem
    st.markdown("## ⚠️ Το Πρόβλημα")
    
    st.markdown("""
    ### 🎯 Ορισμός: Τι είναι το "Clickbait"?
    
    Το **Clickbait** είναι η τέχνη της εκμετάλλευσης του **"Curiosity Gap"** (Κενό Περιέργειας). 
    Ο τίτλος υπόσχεται μια πληροφορία που λείπει, δημιουργώντας ψυχολογική δυσφορία στον χρήστη μέχρι να κάνει κλικ.
    Στην παρούσα εργασία, δε θα μελετήσουμε την ποιότητα του άρθρου σε σχέση με το εάν ένας τίτλος είναι clickbait, παρά μόνο εάν κατάφερε να προσεγγίσει τον χρήστη.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🗣️ Πρόκληση 1</h4>
            <h5>Γλωσσική Ασάφεια</h5>
            <p>Τίτλοι όπως "Δεν θα πιστεύετε τι συνέβη!" δεν περιέχουν καμία ουσιαστική πληροφορία για το περιεχόμενο.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>⚖️ Πρόκληση 2</h4>
            <h5>Ανισορροπία Κλάσεων</h5>
            <p>News: ~61% vs Clickbait: ~39%. Ένα αφελές μοντέλο θα μπορούσε να έχει υψηλό Accuracy προβλέποντας συνέχεια "News" (γι' αυτό εστιάζουμε στο F1-Score).</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🔄 Πρόκληση 3</h4>
            <h5>Ετερογένεια</h5>
            <p>Τα clickbait αλλάζουν μορφή συνεχώς και απαιτούν μοντέλα που γενικεύουν (generalization).</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Pipeline visualization
    st.markdown("## 🔄 Η Λύση: Pipeline Δύο Βημάτων")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>1️⃣ Data Engineering</h3>
            <ul>
                <li>Text cleaning</li>
                <li>Merging sources</li>
                <li>De-duplication</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>2️⃣ Semantic Embedding</h3>
            <ul>
                <li><strong>Gemma LLM (7B)</strong></li>
                <li>Hidden State (768D)</li>
                <li>Contextual meaning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3️⃣ Topological Unrolling</h3>
            <ul>
                <li><strong>UMAP</strong></li>
                <li>500 components</li>
                <li>Density preserved</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>4️⃣ Classification</h3>
            <ul>
                <li>4 algorithm families</li>
                <li>Optuna tuning</li>
                <li>MLflow tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("## 🚀 Εξερευνήστε την Έρευνα")
    st.info("👈 Χρησιμοποιήστε το sidebar για να πλοηγηθείτε στις διαφορετικές ενότητες της ανάλυσης.")

def show_dataset():
    st.markdown("## 📊 Τα Δεδομένα (The Data)")
    
    st.markdown("""
    Λεπτομερής παρουσίαση του dataset που χρησιμοποιήθηκε για την εκπαίδευση και αξιολόγηση.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Πηγές & Όγκος")
        
        st.markdown("""
        <div class="metric-card">
            <h4>🔗 Πηγές Δεδομένων</h4>
            <ul>
                <li><strong>Kaggle</strong> Datasets</li>
                <li><strong>Webis-Clickbait-22</strong></li>
                <li><strong>GitHub</strong> Repositories</li>
            </ul>
            <p><em>Συνένωση (merging) διαφορετικών πηγών για μεγαλύτερη ποικιλομορφία</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        data_stats = {
            "Συνολικά Samples": "~85,000",
            "Non-Clickbait (News)": "~61%",
            "Clickbait": "~39%",
            "Train Set": "70%",
            "Validation Set": "15%",
            "Test Set": "15%"
        }
        
        st.markdown("### 📊 Κατανομή")
        for key, value in data_stats.items():
            st.metric(label=key, value=value)
    
    with col2:
        st.markdown("### 🔍 Data Engineering Steps")
        
        st.markdown("""
        <div class="metric-card">
            <h4>1. Cleaning (Καθαρισμός)</h4>
            <ul>
                <li>Αφαίρεση URLs</li>
                <li>Αφαίρεση ειδικών χαρακτήρων</li>
                <li>Αφαίρεση emoji</li>
                <li>Normalization κειμένου</li>
            </ul>
        </div>
        
        <div class="metric-card">
            <h4>2. Merging (Ενοποίηση)</h4>
            <p>Συγχώνευση διαφορετικών πηγών σε ένα ενιαίο σώμα κειμένων (corpus)</p>
        </div>
        
        <div class="metric-card">
            <h4>3. De-duplication</h4>
            <p>Αφαίρεση διπλότυπων εγγραφών για αποφυγή data leakage</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚖️ Η Πρόκληση της Ανισορροπίας")
    
    st.markdown("""
    <div class="insight-box">
        <p><strong>Γιατί εστιάζουμε στο F1-Score;</strong></p>
        <p>Οι ειδήσεις (News) είναι πολύ περισσότερες (~61%) από τα Clickbait (~39%). 
        Ένα αφελές μοντέλο θα μπορούσε να έχει υψηλό <strong>Accuracy</strong> προβλέποντας συνέχεια "News", 
        αλλά να είναι <strong>άχρηστο</strong> στην πράξη.</p>
        <p>Το <strong>F1-Score</strong> εξισορροπεί Precision & Recall, δίνοντας μια πιο ρεαλιστική εικόνα της απόδοσης.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📝 Παραδείγματα Τίτλων")
    
    examples = [
        {"headline": "Δεν θα πιστεύετε τι συνέβη μετά!", "label": "Clickbait", "color": "#FF6B35"},
        {"headline": "Έρευνα δείχνει ότι ο καφές μειώνει τον κίνδυνο καρδιοπάθειας", "label": "Non-Clickbait", "color": "#4CAF50"},
        {"headline": "Αυτό το ΕΝΑΣ κόλπο θα αλλάξει τη ζωή σας ΠΑΝΤΑ", "label": "Clickbait", "color": "#FF6B35"},
        {"headline": "Η κυβέρνηση ανακοίνωσε νέα κλιματική πολιτική", "label": "Non-Clickbait", "color": "#4CAF50"},
        {"headline": "You Won't BELIEVE What Happened Next!", "label": "Clickbait", "color": "#FF6B35"},
        {"headline": "Stock Market Closes Lower Amid Fed Concerns", "label": "Non-Clickbait", "color": "#4CAF50"}
    ]
    
    for ex in examples:
        st.markdown(f"""
        <div style="background-color: {ex['color']}22; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid {ex['color']};">
            <strong>{ex['headline']}</strong><br/>
            <small style="color: {ex['color']};">■ {ex['label']}</small>
        </div>
        """, unsafe_allow_html=True)

def show_methodology():
    st.markdown("## 🔬 Μεθοδολογία (Pipeline)")
    
    st.markdown("""
    Λεπτομερής παρουσίαση της τεχνικής προσέγγισης που χρησιμοποιήθηκε στην έρευνα.
    """)
    
    # Feature Engineering
    st.markdown("### 1️⃣ Feature Extraction με Gemma")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Google Gemma LLM (7B parameters)</h4>
            <ul>
                <li><strong>Input:</strong> Τίτλος (Text)</li>
                <li><strong>Output:</strong> Hidden State της τελευταίας στρώσης (768 dimensions)</li>
                <li><strong>Στόχος:</strong> Contextual Embeddings (όχι απλά keywords)</li>
            </ul>
            <p><em>"Το σύστημα καταλαβαίνει ότι το 'Σοκαριστικό!' και το 'Απίστευτο!' 
            βρίσκονται κοντά στον σημασιολογικό χώρο."</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 🎯 Γιατί Gemma;
        - **State-of-the-art** κατανόηση φυσικής γλώσσας
        - Κατανόηση **ύφους**, **σαρκασμού** και **δομής**
        - **Υψηλής ποιότητας** σημασιολογικές αναπαραστάσεις
        - **Αποδοτικό** για large-scale processing
        """)
    
    with col2:
        st.code("""
# Gemma Embedding Process
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "google/gemma-7b"
)

# Get embeddings
with torch.no_grad():
    outputs = model(
        input_ids,
        attention_mask=mask
    )
    
# Extract last hidden state
embeddings = outputs.last_hidden_state
# Shape: [batch, seq_len, 768]
        """, language="python")
    
    st.markdown("---")
    
    # Manifold Learning
    st.markdown("### 2️⃣ Manifold Learning: UMAP vs PCA")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>❌ Γιατί ΟΧΙ PCA;</h4>
            <p><strong>Το PCA είναι γραμμικός αλγόριθμος.</strong></p>
            <p>Όταν προβάλλει πολύπλοκα γλωσσικά δεδομένα, τείνει να δημιουργεί μια 
            <strong>"μουντζούρα"</strong>, χάνοντας τις λεπτές σημασιολογικές σχέσεις.</p>
            <h5>Προβλήματα του PCA:</h5>
            <ul>
                <li>Γραμμική προβολή</li>
                <li>Χάνει μη-γραμμικές σχέσεις</li>
                <li>Δεν διατηρεί τοπικές δομές</li>
                <li>Αδιάφορο στην πυκνότητα</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.image("pca.png", caption="PCA Projection: The 'Hairball' Problem", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>✅ Γιατί UMAP;</h4>
            <p><strong>Uniform Manifold Approximation and Projection</strong></p>
            <p>Είναι <strong>τοπολογικός αλγόριθμος</strong>. Αντιλαμβάνεται ότι τα δεδομένα 
            βρίσκονται πάνω σε "καμπυλωμένες" επιφάνειες (manifolds).</p>
            <h5>Τι διατηρεί το UMAP:</h5>
            <ul>
                <li><strong>Γειτονιές</strong> (Local Structure)</li>
                <li><strong>Πυκνότητα</strong> (Density) των σημείων</li>
                <li><strong>Τοπολογία</strong> του manifold</li>
                <li><strong>Αποστάσεις</strong> μεταξύ clusters</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.image("umap.png", caption="UMAP Projection: Clear Semantic Clusters", use_container_width=True)

    st.markdown("---")
    
    st.markdown("""
    <div class="insight-box">
        <h3>🎓 Topological Unrolling (Τοπολογικό Ξεδίπλωμα)</h3>
        <p>Το UMAP λειτουργεί ως <strong>"Manifold Unroller"</strong>, μετατρέποντας τις πολύπλοκες 
        σημασίες του LLM σε <strong>γεωμετρικά διαχωρίσιμες περιοχές</strong>.</p>
        <p>Μειώνουμε από <strong>768 διαστάσεις → 500 διαστάσεις</strong> διατηρώντας:</p>
        <ul>
            <li>Τις σχέσεις γειτνίασης (π.χ. παρόμοιοι τίτλοι μένουν κοντά)</li>
            <li>Την πληροφορία πυκνότητας (clusters με clickbait vs news)</li>
            <li>Τη γεωμετρική δομή που επιτρέπει linear separability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # UMAP Parameters
    st.markdown("### ⚙️ UMAP Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
from umap import UMAP

reducer = UMAP(
    n_components=500,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)

reduced_embeddings = reducer.fit_transform(
    gemma_embeddings
)
        """, language="python")
    
    with col2:
        st.markdown("""
        #### Παράμετροι Επεξήγηση:
        
        - **n_components**: 500 διαστάσεις (balance μεταξύ πληροφορίας και αποδοτικότητας)
        - **n_neighbors**: 15 (μέγεθος local neighborhood)
        - **min_dist**: 0.1 (ελάχιστη απόσταση σημείων στην προβολή)
        - **metric**: cosine (βέλτιστο για text embeddings)
        """)
    
    st.markdown("---")
    
    # Hyperparameter Tuning
    st.markdown("### 3️⃣ Hyperparameter Optimization")
    
    st.markdown("""
    <div class="metric-card">
        <h4>🔧 Optuna Framework</h4>
        <ul>
            <li><strong>Algorithm:</strong> Tree-structured Parzen Estimator (TPE)</li>
            <li><strong>Trials:</strong> 20 ανά μοντέλο</li>
            <li><strong>Tracking:</strong> MLflow για reproducibility</li>
            <li><strong>Validation:</strong> Stratified Train/Val split</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train_umap, y_train)
    
    preds = model.predict(X_val_umap)
    return f1_score(y_val, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
    """, language="python")

def show_ml_algorithms():
    st.markdown("## 🤖 Οι Αλγόριθμοι Μηχανικής Μάθησης")
    
    st.markdown("""
    Σε αυτή την ενότητα παρουσιάζονται **οπτικοποιήσεις** που δείχνουν πώς λειτουργεί κάθε αλγόριθμος
    που χρησιμοποιήθηκε στην έρευνα.
    """)
    
    # Algorithm selector
    st.markdown("---")
    
    algorithm = st.selectbox(
        "Επιλέξτε αλγόριθμο για visualization:",
        [
            "Logistic Regression",
            "Support Vector Machine (SVM)",
            "Gradient Boosting",
            "Stochastic Gradient Descent (SGD)"
        ]
    )
    
    st.markdown("---")
    
    if algorithm == "Logistic Regression":
        show_logistic_regression()
    elif algorithm == "Support Vector Machine (SVM)":
        show_svm()
    elif algorithm == "Gradient Boosting":
        show_gradient_boosting()
    elif algorithm == "Stochastic Gradient Descent (SGD)":
        show_sgd()

def show_logistic_regression():
    st.markdown("### 📊 Logistic Regression")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Τι είναι;</h4>
            <p>Η <strong>Logistic Regression</strong> είναι ένας γραμμικός ταξινομητής που μοντελοποιεί 
            την πιθανότητα μιας κλάσης χρησιμοποιώντας τη σιγμοειδή συνάρτηση.</p>
            <h4>Πώς Λειτουργεί;</h4>
            <ol>
                <li>Υπολογίζει ένα γραμμικό συνδυασμό: <code>z = w₁x₁ + w₂x₂ + b</code></li>
                <li>Εφαρμόζει τη sigmoid function: <code>σ(z) = 1/(1 + e⁻ᶻ)</code></li>
                <li>Αν σ(z) > 0.5 → Clickbait, αλλιώς → News</li>
            </ol>
            <h4>Γιατί το Χρησιμοποιήσαμε;</h4>
            <ul>
                <li>✅ <strong>Baseline Model:</strong> Απλό και γρήγορο</li>
                <li>✅ <strong>Interpretable:</strong> Μπορούμε να δούμε τα βάρη</li>
                <li>✅ <strong>Linear Separability Test:</strong> Ελέγχει αν το UMAP δημιούργησε linearly separable features</li>
            </ul>
            <h4>Αποτελέσματα:</h4>
            <ul>
                <li>F1 Score: <strong>0.86</strong></li>
                <li>Accuracy: <strong>0.87</strong></li>
                <li>Training Time: <strong>~2 min</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create animated logistic regression visualization
        import plotly.graph_objects as go
        import numpy as np
        
        # Generate sample data
        np.random.seed(42)
        n_points = 100
        
        # Class 0 (News) - bottom left cluster
        X0 = np.random.randn(n_points, 2) * 0.5 + np.array([-1.5, -1.5])
        
        # Class 1 (Clickbait) - top right cluster
        X1 = np.random.randn(n_points, 2) * 0.5 + np.array([1.5, 1.5])
        
        # Create mesh for decision boundary
        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        # Create frames for animation
        frames = []
        n_frames = 20
        
        for i in range(n_frames):
            # Gradually rotate the decision boundary
            angle = -np.pi/4 + (i / n_frames) * 0.2
            w1, w2 = np.cos(angle), np.sin(angle)
            
            # Decision boundary: w1*x + w2*y = 0
            Z = 1 / (1 + np.exp(-(w1 * xx + w2 * yy)))
            
            frame = go.Frame(
                data=[
                    go.Contour(
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        z=Z,
                        colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                        opacity=0.3,
                        showscale=False,
                        contours=dict(start=0, end=1, size=0.1),
                        hoverinfo='skip'
                    ),
                    go.Scatter(
                        x=X0[:, 0], y=X0[:, 1],
                        mode='markers',
                        marker=dict(size=8, color='#4CAF50', line=dict(width=1, color='white')),
                        name='News',
                        hovertemplate='News<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                    ),
                    go.Scatter(
                        x=X1[:, 0], y=X1[:, 1],
                        mode='markers',
                        marker=dict(size=8, color='#FF6B35', line=dict(width=1, color='white')),
                        name='Clickbait',
                        hovertemplate='Clickbait<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                    ),
                    go.Scatter(
                        x=[-4, 4],
                        y=[-w1/w2*(-4), -w1/w2*4] if w2 != 0 else [0, 0],
                        mode='lines',
                        line=dict(color='white', width=3, dash='dash'),
                        name='Decision Boundary',
                        hoverinfo='skip'
                    )
                ],
                name=f'frame{i}'
            )
            frames.append(frame)
        
        # Initial frame
        angle = -np.pi/4
        w1, w2 = np.cos(angle), np.sin(angle)
        Z = 1 / (1 + np.exp(-(w1 * xx + w2 * yy)))
        
        fig = go.Figure(
            data=[
                go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=Z,
                    colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                    opacity=0.3,
                    showscale=False,
                    contours=dict(start=0, end=1, size=0.1),
                    hoverinfo='skip'
                ),
                go.Scatter(
                    x=X0[:, 0], y=X0[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='#4CAF50', line=dict(width=1, color='white')),
                    name='News'
                ),
                go.Scatter(
                    x=X1[:, 0], y=X1[:, 1],
                    mode='markers',
                    marker=dict(size=8, color='#FF6B35', line=dict(width=1, color='white')),
                    name='Clickbait'
                ),
                go.Scatter(
                    x=[-4, 4],
                    y=[-w1/w2*(-4), -w1/w2*4],
                    mode='lines',
                    line=dict(color='white', width=3, dash='dash'),
                    name='Decision Boundary'
                )
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="Logistic Regression: Finding the Decision Boundary",
            xaxis_title="Feature 1 (UMAP Dimension)",
            yaxis_title="Feature 2 (UMAP Dimension)",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2b2b2b',
            font=dict(color='#e0e0e0'),
            hovermode='closest',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 200, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate'
                        }]
                    },
                    {
                        'label': '⏸ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'
                        }]
                    }
                ]
            }],
            legend=dict(
                bgcolor='#2b2b2b',
                bordercolor='#FF6B35',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🎬 **Πατήστε Play** για να δείτε πώς το μοντέλο βρίσκει το βέλτιστο decision boundary!")

def show_svm():
    st.markdown("### 🎯 Support Vector Machine (SVM)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Τι είναι;</h4>
            <p>Το <strong>SVM</strong> είναι ένας αλγόριθμος που βρίσκει το βέλτιστο υπερεπίπεδο 
            (hyperplane) που διαχωρίζει τις κλάσεις με τη <strong>μέγιστη απόσταση (margin)</strong>.</p>
            <h4>Πώς Λειτουργεί;</h4>
            <ol>
                <li><strong>Βρίσκει τα Support Vectors:</strong> Τα πιο "κρίσιμα" σημεία κοντά στο decision boundary</li>
                <li><strong>Μεγιστοποιεί το Margin:</strong> Η απόσταση μεταξύ των πιο κοντινών σημείων των δύο κλάσεων</li>
                <li><strong>RBF Kernel:</strong> Προβάλλει σε υψηλότερες διαστάσεις για μη-γραμμικό διαχωρισμό</li>
            </ol>
            <h4>Γιατί το Χρησιμοποιήσαμε;</h4>
            <ul>
                <li>✅ <strong>Maximum Margin:</strong> Θεωρητικά πιο robust στο overfitting</li>
                <li>✅ <strong>Kernel Trick:</strong> Μπορεί να χειριστεί μη-γραμμικά patterns</li>
                <li>⚠️ <strong>Distance-based:</strong> Εδώ εντοπίσαμε το Scaling Paradox!</li>
            </ul>
            <h4>Αποτελέσματα:</h4>
            <ul>
                <li>F1 Score (No Scaling): <strong>0.83</strong> ✅</li>
                <li>F1 Score (With Scaling): <strong>0.68</strong> ❌</li>
                <li>Training Time: <strong>~45 min</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        import plotly.graph_objects as go
        import numpy as np
        
        np.random.seed(42)
        n_points = 50
        
        # Create non-linearly separable data
        X0 = np.random.randn(n_points, 2) * 0.5 + np.array([-1, 0])
        X1_a = np.random.randn(n_points//2, 2) * 0.3 + np.array([1, 1])
        X1_b = np.random.randn(n_points//2, 2) * 0.3 + np.array([1, -1])
        X1 = np.vstack([X1_a, X1_b])
        
        # Create mesh
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        frames = []
        n_frames = 30
        
        for i in range(n_frames):
            # Simulate RBF kernel decision boundary
            gamma = 0.5 + (i / n_frames) * 1.5
            
            Z = np.zeros_like(xx)
            for x0, y0 in X0[:5]:  # Use first 5 as support vectors
                Z += np.exp(-gamma * ((xx - x0)**2 + (yy - y0)**2))
            for x1, y1 in X1[:5]:
                Z -= np.exp(-gamma * ((xx - x1)**2 + (yy - y1)**2))
            
            Z = 1 / (1 + np.exp(-Z))
            
            # Find support vectors (closest points to boundary)
            support_indices_0 = [0, 1, 2]
            support_indices_1 = [0, 1, 2]
            
            frame = go.Frame(
                data=[
                    go.Contour(
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        z=Z,
                        colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                        opacity=0.3,
                        showscale=False,
                        contours=dict(start=0, end=1, size=0.1),
                        hoverinfo='skip'
                    ),
                    go.Scatter(
                        x=X0[:, 0], y=X0[:, 1],
                        mode='markers',
                        marker=dict(size=8, color='#4CAF50', line=dict(width=1, color='white')),
                        name='News',
                        hovertemplate='News<extra></extra>'
                    ),
                    go.Scatter(
                        x=X1[:, 0], y=X1[:, 1],
                        mode='markers',
                        marker=dict(size=8, color='#FF6B35', line=dict(width=1, color='white')),
                        name='Clickbait',
                        hovertemplate='Clickbait<extra></extra>'
                    ),
                    go.Scatter(
                        x=X0[support_indices_0, 0],
                        y=X0[support_indices_0, 1],
                        mode='markers',
                        marker=dict(size=14, color='#4CAF50', 
                                  line=dict(width=3, color='yellow'),
                                  symbol='circle'),
                        name='Support Vectors',
                        showlegend=True if i == 0 else False,
                        hovertemplate='Support Vector<extra></extra>'
                    ),
                    go.Scatter(
                        x=X1[support_indices_1, 0],
                        y=X1[support_indices_1, 1],
                        mode='markers',
                        marker=dict(size=14, color='#FF6B35',
                                  line=dict(width=3, color='yellow'),
                                  symbol='circle'),
                        showlegend=False,
                        hovertemplate='Support Vector<extra></extra>'
                    )
                ],
                name=f'frame{i}'
            )
            frames.append(frame)
        
        # Initial frame
        gamma = 0.5
        Z = np.zeros_like(xx)
        for x0, y0 in X0[:5]:
            Z += np.exp(-gamma * ((xx - x0)**2 + (yy - y0)**2))
        for x1, y1 in X1[:5]:
            Z -= np.exp(-gamma * ((xx - x1)**2 + (yy - y1)**2))
        Z = 1 / (1 + np.exp(-Z))
        
        fig = go.Figure(
            data=[
                go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=Z,
                    colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                    opacity=0.3,
                    showscale=False,
                    contours=dict(start=0, end=1, size=0.1)
                ),
                go.Scatter(x=X0[:, 0], y=X0[:, 1], mode='markers',
                          marker=dict(size=8, color='#4CAF50', line=dict(width=1, color='white')),
                          name='News'),
                go.Scatter(x=X1[:, 0], y=X1[:, 1], mode='markers',
                          marker=dict(size=8, color='#FF6B35', line=dict(width=1, color='white')),
                          name='Clickbait'),
                go.Scatter(x=X0[[0,1,2], 0], y=X0[[0,1,2], 1], mode='markers',
                          marker=dict(size=14, color='#4CAF50', line=dict(width=3, color='yellow')),
                          name='Support Vectors')
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="SVM: Maximizing the Margin with RBF Kernel",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2b2b2b',
            font=dict(color='#e0e0e0'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 150}, 'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            legend=dict(bgcolor='#2b2b2b', bordercolor='#FF6B35', borderwidth=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("🎬 Τα **κίτρινα σημεία** είναι τα Support Vectors - τα πιο κρίσιμα σημεία για το decision boundary!")

def show_gradient_boosting():
    st.markdown("### 🌲 Gradient Boosting (The Champion)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Τι είναι;</h4>
            <p>Το <strong>Gradient Boosting</strong> είναι ένα ensemble μοντέλο που κατασκευάζει 
            ένα ισχυρό ταξινομητή προσθέτοντας διαδοχικά "ασθενή" δέντρα απόφασης.</p>
            <h4>Πώς Λειτουργεί;</h4>
            <ol>
                <li><strong>Βήμα 1:</strong> Εκπαιδεύει ένα απλό δέντρο</li>
                <li><strong>Βήμα 2:</strong> Βρίσκει τα λάθη (residuals) του πρώτου δέντρου</li>
                <li><strong>Βήμα 3:</strong> Εκπαιδεύει νέο δέντρο για να διορθώσει αυτά τα λάθη</li>
                <li><strong>Επανάληψη:</strong> Προσθέτει περισσότερα δέντρα μέχρι να φτάσει στην καλύτερη απόδοση</li>
            </ol>
            <h4>Μαθηματικά:</h4>
            <p><code>F_m(x) = F_(m-1)(x) + γ_m × h_m(x)</code></p>
            <p>Όπου κάθε <code>h_m</code> είναι ένα νέο δέντρο που μαθαίνει από τα residuals.</p>
            <h4>Γιατί Κέρδισε;</h4>
            <ul>
                <li>🏆 <strong>Scaling Invariant:</strong> Δεν επηρεάζεται από την κλίμακα</li>
                <li>🏆 <strong>Non-linear Patterns:</strong> Βρίσκει πολύπλοκες σχέσεις</li>
                <li>🏆 <strong>Robust:</strong> Αντέχει σε overfitting</li>
                <li>🏆 <strong>Interpretable:</strong> Feature importance analysis</li>
            </ul>
            <h4>Αποτελέσματα:</h4>
            <ul>
                <li>F1 Score: <strong>0.88</strong> 🥇</li>
                <li>Accuracy: <strong>0.91</strong></li>
                <li>Training Time: <strong>~12 min</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        import plotly.graph_objects as go
        import numpy as np
        
        np.random.seed(42)
        n_points = 80
        
        # Create data with complex boundary
        theta = np.linspace(0, 2*np.pi, n_points)
        r0 = 1 + 0.3 * np.random.randn(n_points)
        r1 = 2 + 0.3 * np.random.randn(n_points)
        
        X0 = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta)])
        X1 = np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)])
        
        # Create mesh
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        
        frames = []
        n_estimators_list = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
        
        for n_est in n_estimators_list:
            # Simulate boosting: better approximation with more trees
            Z = np.sqrt(xx**2 + yy**2)
            
            # Add complexity with more estimators
            for i in range(min(n_est // 10, 5)):
                angle = i * 2 * np.pi / 5
                Z += 0.1 * np.sin(n_est * (np.arctan2(yy, xx) - angle))
            
            # Decision boundary at r ≈ 1.5
            Z = 1 / (1 + np.exp(-3 * (Z - 1.5)))
            
            frame = go.Frame(
                data=[
                    go.Contour(
                        x=np.linspace(x_min, x_max, 100),
                        y=np.linspace(y_min, y_max, 100),
                        z=Z,
                        colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                        opacity=0.4,
                        showscale=False,
                        contours=dict(start=0, end=1, size=0.1),
                        hoverinfo='skip'
                    ),
                    go.Scatter(
                        x=X0[:, 0], y=X0[:, 1],
                        mode='markers',
                        marker=dict(size=6, color='#4CAF50', line=dict(width=0.5, color='white')),
                        name='News',
                        hovertemplate='News<extra></extra>'
                    ),
                    go.Scatter(
                        x=X1[:, 0], y=X1[:, 1],
                        mode='markers',
                        marker=dict(size=6, color='#FF6B35', line=dict(width=0.5, color='white')),
                        name='Clickbait',
                        hovertemplate='Clickbait<extra></extra>'
                    )
                ],
                layout=go.Layout(
                    title_text=f"Gradient Boosting: {n_est} Trees",
                    annotations=[
                        dict(
                            text=f"<b>Estimators: {n_est}</b>",
                            xref="paper", yref="paper",
                            x=0.5, y=1.05, showarrow=False,
                            font=dict(size=14, color='#FF6B35')
                        )
                    ]
                ),
                name=f'frame{n_est}'
            )
            frames.append(frame)
        
        # Initial frame
        Z = np.sqrt(xx**2 + yy**2)
        Z = 1 / (1 + np.exp(-3 * (Z - 1.5)))
        
        fig = go.Figure(
            data=[
                go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=Z,
                    colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                    opacity=0.4,
                    showscale=False,
                    contours=dict(start=0, end=1, size=0.1)
                ),
                go.Scatter(x=X0[:, 0], y=X0[:, 1], mode='markers',
                          marker=dict(size=6, color='#4CAF50', line=dict(width=0.5, color='white')),
                          name='News'),
                go.Scatter(x=X1[:, 0], y=X1[:, 1], mode='markers',
                          marker=dict(size=6, color='#FF6B35', line=dict(width=0.5, color='white')),
                          name='Clickbait')
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="Gradient Boosting: Adding Trees to Improve Fit",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2b2b2b',
            font=dict(color='#e0e0e0'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 400}, 'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            legend=dict(bgcolor='#2b2b2b', bordercolor='#FF6B35', borderwidth=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("🎬 Παρακολουθήστε πώς το decision boundary γίνεται πιο ακριβές καθώς προσθέτουμε περισσότερα δέντρα!")

def show_sgd():
    st.markdown("### ⚡ Stochastic Gradient Descent (SGD)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Τι είναι;</h4>
            <p>Το <strong>SGD</strong> είναι ένας αλγόριθμος βελτιστοποίησης που ενημερώνει τα βάρη 
            του μοντέλου επαναληπτικά χρησιμοποιώντας <strong>ένα δείγμα τη φορά</strong>.</p>
            <h4>Πώς Λειτουργεί;</h4>
            <ol>
                <li><strong>Αρχικοποίηση:</strong> Ξεκινάει με τυχαία βάρη w</li>
                <li><strong>Για κάθε δείγμα:</strong>
                    <ul>
                        <li>Υπολογίζει την πρόβλεψη</li>
                        <li>Υπολογίζει το σφάλμα (loss)</li>
                        <li>Ενημερώνει τα βάρη: <code>w = w - η × ∇L</code></li>
                    </ul>
                </li>
                <li><strong>Επανάληψη:</strong> Μέχρι τα βάρη να συγκλίνουν</li>
            </ol>
            <h4>Μαθηματικά:</h4>
            <p><code>w_(t+1) = w_t - η × ∇Q_i(w_t)</code></p>
            <p>Όπου <code>η</code> είναι το learning rate και <code>∇Q_i</code> η κλίση για το i-οστό δείγμα.</p>
            <h4>Γιατί το Χρησιμοποιήσαμε;</h4>
            <ul>
                <li>✅ <strong>Scalability:</strong> Πολύ γρήγορο για μεγάλα datasets</li>
                <li>✅ <strong>Online Learning:</strong> Μπορεί να μάθει από streaming data</li>
                <li>✅ <strong>Memory Efficient:</strong> Δεν χρειάζεται ολόκληρο dataset στη μνήμη</li>
            </ul>
            <h4>Αποτελέσματα:</h4>
            <ul>
                <li>F1 Score: <strong>0.81</strong></li>
                <li>Accuracy: <strong>0.82</strong></li>
                <li>Training Time: <strong>~3 min</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        import plotly.graph_objects as go
        import numpy as np
        
        np.random.seed(42)
        n_points = 60
        
        X0 = np.random.randn(n_points, 2) * 0.6 + np.array([-1, -1])
        X1 = np.random.randn(n_points, 2) * 0.6 + np.array([1, 1])
        
        # Create frames showing SGD updates
        frames = []
        n_iterations = 30
        
        # Start with random weights
        w = np.array([0.0, 1.0])
        learning_rate = 0.1
        
        weight_history = [w.copy()]
        
        for iteration in range(n_iterations):
            # Simulate SGD update
            # Pick random sample
            if iteration % 2 == 0:
                sample = X0[iteration % len(X0)]
                label = 0
            else:
                sample = X1[iteration % len(X1)]
                label = 1
            
            # Compute gradient (simplified)
            prediction = 1 / (1 + np.exp(-np.dot(w, sample)))
            gradient = (prediction - label) * sample
            
            # Update weights
            w = w - learning_rate * gradient
            weight_history.append(w.copy())
            
            # Create decision boundary
            x_range = np.array([-3, 3])
            if w[1] != 0:
                y_range = -w[0] / w[1] * x_range
            else:
                y_range = np.array([0, 0])
            
            # Create mesh for background
            x_min, x_max = -3, 3
            y_min, y_max = -3, 3
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                 np.linspace(y_min, y_max, 50))
            Z = 1 / (1 + np.exp(-(w[0] * xx + w[1] * yy)))
            
            # Highlight current sample
            current_sample = sample
            
            frame = go.Frame(
                data=[
                    go.Contour(
                        x=np.linspace(x_min, x_max, 50),
                        y=np.linspace(y_min, y_max, 50),
                        z=Z,
                        colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                        opacity=0.2,
                        showscale=False,
                        hoverinfo='skip'
                    ),
                    go.Scatter(
                        x=X0[:, 0], y=X0[:, 1],
                        mode='markers',
                        marker=dict(size=7, color='#4CAF50', line=dict(width=1, color='white')),
                        name='News',
                        hovertemplate='News<extra></extra>'
                    ),
                    go.Scatter(
                        x=X1[:, 0], y=X1[:, 1],
                        mode='markers',
                        marker=dict(size=7, color='#FF6B35', line=dict(width=1, color='white')),
                        name='Clickbait',
                        hovertemplate='Clickbait<extra></extra>'
                    ),
                    go.Scatter(
                        x=[current_sample[0]],
                        y=[current_sample[1]],
                        mode='markers',
                        marker=dict(size=20, color='yellow', 
                                  line=dict(width=3, color='white'),
                                  symbol='star'),
                        name='Current Sample',
                        showlegend=True if iteration == 0 else False,
                        hovertemplate='Learning from this!<extra></extra>'
                    ),
                    go.Scatter(
                        x=x_range,
                        y=y_range,
                        mode='lines',
                        line=dict(color='white', width=2, dash='dash'),
                        name='Decision Boundary',
                        showlegend=True if iteration == 0 else False
                    )
                ],
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"<b>Iteration: {iteration + 1}</b>",
                            xref="paper", yref="paper",
                            x=0.5, y=1.05, showarrow=False,
                            font=dict(size=14, color='#FF6B35')
                        )
                    ]
                ),
                name=f'frame{iteration}'
            )
            frames.append(frame)
        
        # Initial frame
        w_init = np.array([0.0, 1.0])
        x_range = np.array([-3, 3])
        y_range = -w_init[0] / w_init[1] * x_range if w_init[1] != 0 else np.array([0, 0])
        
        xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        Z = 1 / (1 + np.exp(-(w_init[0] * xx + w_init[1] * yy)))
        
        fig = go.Figure(
            data=[
                go.Contour(
                    x=np.linspace(-3, 3, 50),
                    y=np.linspace(-3, 3, 50),
                    z=Z,
                    colorscale=[[0, '#1a1a1a'], [0.5, '#FF6B35'], [1, '#FF8C42']],
                    opacity=0.2,
                    showscale=False
                ),
                go.Scatter(x=X0[:, 0], y=X0[:, 1], mode='markers',
                          marker=dict(size=7, color='#4CAF50', line=dict(width=1, color='white')),
                          name='News'),
                go.Scatter(x=X1[:, 0], y=X1[:, 1], mode='markers',
                          marker=dict(size=7, color='#FF6B35', line=dict(width=1, color='white')),
                          name='Clickbait'),
                go.Scatter(x=x_range, y=y_range, mode='lines',
                          line=dict(color='white', width=2, dash='dash'),
                          name='Decision Boundary')
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="SGD: Learning One Sample at a Time",
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#2b2b2b',
            font=dict(color='#e0e0e0'),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': 300}, 'fromcurrent': True}]},
                    {'label': '⏸ Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }],
            legend=dict(bgcolor='#2b2b2b', bordercolor='#FF6B35', borderwidth=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("🎬 Το **κίτρινο αστέρι** δείχνει το δείγμα από το οποίο μαθαίνει σε κάθε iteration!")

def show_results():
    st.markdown("## 📈 Results & Analysis")
    
    st.markdown("""
    Comprehensive evaluation των μοντέλων στο test set με **ακριβείς μετρικές** από τα πειράματα.
    """)

    # Exact metrics from Results notebook
    data = {
        "Model": [
            "Gradient Boosting",
            "Logistic Regression",
            "SGD Classifier",
            "SVM (RBF)",
            "SVM (RBF)",
            "Logistic Regression"
        ],
        "Configuration": [
            "No Scaling",
            "No Scaling",
            "No Scaling",
            "No Scaling",
            "With Scaling",
            "With Scaling"
        ],
        "Accuracy": [0.9087, 0.8934, 0.8903, 0.8868, 0.7135, 0.7020],
        "F1 Score": [0.8784, 0.8570, 0.8500, 0.8475, 0.7022, 0.6960],
        "Category": ["Tree-based", "Linear", "Linear", "Kernel-based", "Kernel-based", "Linear"]
    }
    
    df_results = pd.DataFrame(data)
    
    # Overall best model metrics
    st.markdown("### 🏆 Gradient Boosting - Best Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Accuracy",
        "90.87%",
        delta="+1.53% vs 2nd best",
        help="Gradient Boosting (No Scaling)"
    )
    col2.metric(
        "F1 Score",
        "0.8784",
        delta="+0.0214 vs 2nd best",
        help="Harmonic mean of Precision & Recall"
    )
    col3.metric(
        "Precision",
        "~0.89",
        help="Estimated από F1 και Recall"
    )
    col4.metric(
        "Recall",
        "~0.87",
        help="Estimated από F1 και Precision"
    )
    
    st.markdown("---")

    # Δεδομένα αντλημένα από το Book 2 (Evaluation Run)
    # Rank, Model, Config, F1, Accuracy, Precision, Recall, Inference Time
    data = [
        ["🥇", "Gradient Boosting", "No Scaling", 0.878, 0.909, 0.910, 0.849, "403 ms"],
        ["🥈", "Logistic Regression", "No Scaling", 0.857, 0.893, 0.894, 0.823, "84 ms"],
        ["🥉", "SGD Classifier", "No Scaling", 0.850, 0.890, 0.907, 0.800, "223 ms"],
        ["4", "SVM (RBF)", "No Scaling", 0.847, 0.887, 0.888, 0.810, "1.9 min"],
        ["5", "SVM (RBF)", "Scaled", 0.702, 0.714, 0.589, 0.869, "58.1 s"],
        ["6", "Logistic Regression", "Scaled", 0.696, 0.702, 0.576, 0.878, "95 ms"]
    ]

    # Δημιουργία DataFrame
    df_results = pd.DataFrame(data, columns=[
        "Rank", "Model", "Configuration", "F1 Score", "Accuracy", "Precision", "Recall", "Inference Time"
    ])

    # Συνάρτηση Styling
    def highlight_rows(row):
        # Πράσινο για τον Νικητή (Best F1)
        if row['Rank'] == "🥇":
            return ['background-color: #2d5016; color: white; font-weight: bold'] * len(row)
        # Κόκκινο/Σκούρο για τα Scaled (που απέτυχαν)
        elif row['Configuration'] == "Scaled":
            return ['background-color: #4a1a1a; color: #cccccc'] * len(row)
        # Γκρι για τη 2η και 3η θέση
        elif row['Rank'] in ["🥈", "🥉"]:
            return ['background-color: #3a3a1a; color: #e0e0e0'] * len(row)
        # Standard για τα υπόλοιπα
        else:
            return ['background-color: #2b2b2b; color: #e0e0e0'] * len(row)

    # Μορφοποίηση αριθμών
    styled_df = df_results.style.apply(highlight_rows, axis=1).format({
        'F1 Score': '{:.3f}',
        'Accuracy': '{:.3f}',
        'Precision': '{:.3f}',
        'Recall': '{:.3f}'
    })

    # Εμφάνιση πίνακα
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ========================================================================
    # MLFLOW INTEGRATION SECTION
    # ========================================================================
    
    st.markdown("### 🔥 MLflow Integration - Live Data Loading")
    
    if not MLFLOW_AVAILABLE:
        st.error("❌ MLflow not installed")
        st.code("pip install mlflow pillow", language="bash")
        st.stop()
    
    if not USE_MLFLOW:
        st.info("ℹ️ MLflow integration disabled in configuration")
        st.stop()
    
    # Configuration UI
    with st.expander("🔧 MLflow Configuration", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            mlflow_uri_input = st.text_input(
                "MLflow Tracking URI",
                value=MLFLOW_URI,
                help="Path to your mlruns directory or remote server URL"
            )
            
            experiment_input = st.text_input(
                "Experiment Name",
                value=EXPERIMENT_NAME,
                help="Name of your MLflow experiment"
            )
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📝 Quick Start</h4>
                <p style="font-size: 12px;">
                1. Ensure MLflow UI is running<br/>
                2. Check URI path is correct<br/>
                3. Click "Load Data"
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        load_data_btn = st.button("🔄 Load Data from MLflow", type="primary", use_container_width=True)
    
    # Load data when button clicked
    if load_data_btn:
        with st.spinner("📥 Loading runs from MLflow..."):
            runs_df = load_mlflow_runs(mlflow_uri_input, experiment_input)
        
        if runs_df is not None and len(runs_df) > 0:
            st.success(f"✅ Successfully loaded {len(runs_df)} runs!")
            
            # Store in session state
            st.session_state['mlflow_runs'] = runs_df
            st.session_state['mlflow_uri'] = mlflow_uri_input
            st.session_state['experiment_name'] = experiment_input
        else:
            st.error("❌ No runs found or error loading data")
            st.info("""
            **Troubleshooting:**
            - Check that MLflow UI is running
            - Verify the tracking URI path is correct
            - Ensure experiment name exists
            - Run notebooks with MLflow logging first
            """)

    # Display loaded runs
    if 'mlflow_runs' in st.session_state and st.session_state['mlflow_runs'] is not None:
        st.markdown("---")
        st.markdown("### 📊 MLflow Runs Data")

        runs_df = st.session_state['mlflow_runs']
        mlflow_uri = st.session_state.get('mlflow_uri', MLFLOW_URI)

        # Create display dataframe
        display_data = {
            'Model Name': runs_df.get('params.model_name', pd.Series(['Unknown'] * len(runs_df))),
            'Scaling': runs_df.get('params.data_scaling', pd.Series(['Unknown'] * len(runs_df))),
            'F1 Score': runs_df.get('metrics.gold_f1', pd.Series([0] * len(runs_df))).round(4),
            'Accuracy': runs_df.get('metrics.gold_accuracy', pd.Series([0] * len(runs_df))).round(4),
            'Precision': runs_df.get('metrics.gold_precision', pd.Series([0] * len(runs_df))).round(4),
            'Recall': runs_df.get('metrics.gold_recall', pd.Series([0] * len(runs_df))).round(4),
        }

        display_df = pd.DataFrame(display_data)

        # Display table
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # ====================================================================
        # PLOT VIEWER
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 📊 View Plots from MLflow")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Run selector
            run_names = runs_df.get('tags.mlflow.runName', pd.Series(['Unknown'] * len(runs_df))).tolist()
            run_ids = runs_df['run_id'].tolist()
            
            options = [f"{name} ({rid[:8]})" for name, rid in zip(run_names, run_ids)]
            
            selected_option = st.selectbox(
                "Select Run:",
                options,
                index=0,
                help="Select which model run to view plots from"
            )
            
            selected_idx = options.index(selected_option)
            selected_run_id = run_ids[selected_idx]
        
        with col2:
            # Plot type selector
            plot_type = st.selectbox(
                "Select Plot:",
                [
                    "Confusion Matrix",
                    "ROC Curve",
                    "Precision-Recall Curve",
                    "Feature Importance"
                ],
                help="Select which visualization to display"
            )
        
        # Artifact name mapping
        artifact_map = {
            "Confusion Matrix": "confusion_matrix_gold.png",
            "ROC Curve": "roc_curve_gold.png",
            "Precision-Recall Curve": "pr_curve_gold.png",
            "Feature Importance": "feature_importance_gold.png"
        }
        
        artifact_name = artifact_map[plot_type]
        
        # Load plot button
        if st.button("🔄 Load Plot", type="primary"):
            with st.spinner(f"📥 Loading {plot_type}..."):
                img = load_artifact_image(selected_run_id, artifact_name, mlflow_uri)
            
            if img is not None:
                st.image(
                    img,
                    caption=f"{plot_type} - {run_names[selected_idx]} ({selected_run_id[:8]})",
                    use_container_width=True
                )
                
                # Show run details in expander
                with st.expander("📊 Run Details", expanded=False):
                    metrics = get_run_metrics(selected_run_id, mlflow_uri)
                    params = get_run_params(selected_run_id, mlflow_uri)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Metrics:**")
                        if metrics:
                            for key, value in sorted(metrics.items()):
                                st.write(f"- **{key}:** {value:.4f}")
                        else:
                            st.write("No metrics found")
                    
                    with col2:
                        st.markdown("**Parameters:**")
                        if params:
                            for key, value in sorted(params.items()):
                                st.write(f"- **{key}:** {value}")
                        else:
                            st.write("No parameters found")
            else:
                st.error(f"❌ Could not load '{plot_type}'")
                st.info(f"""
                **Possible reasons:**
                - Plot '{artifact_name}' was not logged during training
                - Artifact name mismatch in your notebook
                - Run does not have this specific plot
                
                **Solution:** Check your training notebook and ensure this plot is logged with:
                ```python
                plt.savefig('{artifact_name}')
                mlflow.log_artifact('{artifact_name}')
                ```
                """)
                    
                st.success("✅ Βλέπε το παραπάνω code snippet για integration!")

        else:
            st.warning("⚠️ Παρακαλώ εισάγετε MLflow URI")
    
    st.markdown("---")
    
    # Training insights
    st.markdown("### ⚙️ Training Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>⏱️ Training Time Comparison</h4>
            <table style="width: 100%; color: #e0e0e0; font-size: 14px;">
                <tr>
                    <th style="text-align: left; padding: 8px; border-bottom: 2px solid #FF6B35;">Model</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 2px solid #FF6B35;">Time</th>
                </tr>
                <tr>
                    <td style="padding: 8px;">Gradient Boosting</td>
                    <td style="text-align: right; padding: 8px; color: #FF6B35;"><strong>~12 min</strong></td>
                </tr>
                <tr>
                    <td style="padding: 8px;">SVM (No Scaling)</td>
                    <td style="text-align: right; padding: 8px;">~45 min</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Logistic Regression</td>
                    <td style="text-align: right; padding: 8px; color: #4CAF50;"><strong>~2 min</strong></td>
                </tr>
                <tr>
                    <td style="padding: 8px;">SGD Classifier</td>
                    <td style="text-align: right; padding: 8px;">~3 min</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Best Parameters (Gradient Boosting)</h4>
            <ul style="font-size: 13px; line-height: 1.8;">
                <li><strong>n_estimators:</strong> 300</li>
                <li><strong>max_depth:</strong> 8</li>
                <li><strong>learning_rate:</strong> 0.15</li>
                <li><strong>min_samples_split:</strong> 10</li>
                <li><strong>subsample:</strong> 0.8</li>
                <li><strong>Optimization:</strong> Optuna (50 trials)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 💡 Κεντρικά Ευρήματα")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🏆 1. Gradient Boosting: Ο Αδιαμφισβήτητος Νικητής</h4>
        <p>Με <strong>F1: 0.8784</strong> και <strong>Accuracy: 90.87%</strong>, το Gradient Boosting υπερτερεί 
        χάρη στην ικανότητά του να διαχειρίζεται μη-γραμμικά όρια και στην <strong>ανοσία του στην κλιμάκωση</strong>.</p>
    </div>
    
    <div class="insight-box">
        <h4>📐 2. UMAP ως "Γραμμικοποιητής"</h4>
        <p>Η εξαιρετική επίδοση της Logistic Regression χωρίς scaling (<strong>F1: 0.8570</strong>) αποδεικνύει 
        ότι το UMAP λειτούργησε επιτυχώς ως μηχανισμός <strong>"ξεδιπλώματος" (manifold unrolling)</strong>, 
        δημιουργώντας σχεδόν γραμμικά διαχωρίσιμες κλάσεις.</p>
    </div>
    
    <div class="insight-box">
        <h4>⚠️ 3. Το Scaling Paradox</h4>
        <p>Τα μοντέλα με scaling (SVM: 0.7022, LogReg: 0.6960) υστερούν κατά <strong>~15-18%</strong> 
        λόγω καταστροφής της τοπολογικής πληροφορίας του UMAP.</p>
    </div>
    """, unsafe_allow_html=True)

def show_scaling_paradox():
    st.markdown("## 🔍 Το Παράδοξο του Scaling")
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 Κεντρική Ανακάλυψη της Έρευνας</h3>
        <p>Αυτή η έρευνα αποκάλυψε ένα <strong>κρίσιμο εύρημα</strong>: Το StandardScaler, 
        που θεωρείται απαραίτητο για distance-based models, <strong>καταστρέφει τη γεωμετρική πληροφορία</strong> 
        που κωδικοποιείται στα UMAP embeddings.</p>
        <p><strong>Αποτέλεσμα:</strong> Πτώση απόδοσης έως <strong>-18.8%</strong> σε F1 Score!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show exact metrics from Results notebook
    st.markdown("### 📊 Η Απόδειξη με Αριθμούς")
    
    st.markdown("#### 📈 Logistic Regression: Before & After Scaling")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #4CAF50;">
            <h3 style="color: #4CAF50; text-align: center;">✅ Χωρίς Scaling</h3>
            <h2 style="text-align: center; color: #FF6B35;">F1: 0.8570</h2>
            <h3 style="text-align: center; color: #FF8C42;">Acc: 89.34%</h3>
            <p style="text-align: center; margin-top: 10px;">
                <strong>Raw UMAP Features</strong><br/>
                Διατηρεί την τοπολογική δομή
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding-top: 60px;">
            <h1 style="color: #e0e0e0;">VS</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #f44336;">
            <h3 style="color: #f44336; text-align: center;">❌ Με Scaling</h3>
            <h2 style="text-align: center; color: #999;">F1: 0.6960</h2>
            <h3 style="text-align: center; color: #999;">Acc: 70.20%</h3>
            <p style="text-align: center; margin-top: 10px;">
                <strong>StandardScaler Applied</strong><br/>
                <span style="color: #f44336;">-18.8% F1 Drop!</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("#### 🎯 SVM (RBF Kernel): Before & After Scaling")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #4CAF50;">
            <h3 style="color: #4CAF50; text-align: center;">✅ Χωρίς Scaling</h3>
            <h2 style="text-align: center; color: #FF6B35;">F1: 0.8475</h2>
            <h3 style="text-align: center; color: #FF8C42;">Acc: 88.68%</h3>
            <p style="text-align: center; margin-top: 10px;">
                <strong>Raw UMAP Features</strong><br/>
                Διατηρεί density information
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding-top: 60px;">
            <h1 style="color: #e0e0e0;">VS</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #f44336;">
            <h3 style="color: #f44336; text-align: center;">❌ Με Scaling</h3>
            <h2 style="text-align: center; color: #999;">F1: 0.7022</h2>
            <h3 style="text-align: center; color: #999;">Acc: 71.35%</h3>
            <p style="text-align: center; margin-top: 10px;">
                <strong>StandardScaler Applied</strong><br/>
                <span style="color: #f44336;">-17.1% F1 Drop!</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.success("""
    **✅ Συστάσεις για UMAP Pipelines:**
    - ✅ Χρησιμοποιήστε raw features για Linear & Kernel models
    - ✅ Tree-based models λειτουργούν και με τα δύο
    - ❌ ΜΗΝ εφαρμόζετε StandardScaler/MinMaxScaler
    """)

def show_model_comparison():
    st.markdown("## 🏆 Model Comparison - Πλήρης Ανάλυση")

    st.markdown("""
    Συγκριτική αξιολόγηση **ΟΛΩΝ** των μοντέλων που δοκιμάστηκαν, 
    με και χωρίς scaling.
    """)

    # --- 1. Leaderboard Table (Παραμένει ως έχει) ---
    st.markdown("### 📊 Complete Leaderboard")

    leaderboard_data = {
        "Rank": ["🥇", "🥈", "🥉", "4", "5", "6"],
        "Model": [
            "Gradient Boosting",
            "Logistic Regression",
            "SGD Classifier",
            "SVM (RBF)",
            "SVM (RBF)",
            "Logistic Regression"
        ],
        "Configuration": [
            "No Scaling",
            "No Scaling",
            "No Scaling",
            "No Scaling",
            "With Scaling",
            "With Scaling"
        ],
        "F1 Score": [0.8784, 0.8570, 0.8500, 0.8475, 0.7022, 0.6960],
        "Accuracy": [0.9087, 0.8934, 0.8903, 0.8868, 0.7135, 0.7020],
        "Training Time": ["~12 min", "~2 min", "~3 min", "~45 min", "~50 min", "~2 min"],
        "Category": ["Tree-based", "Linear", "Linear", "Kernel", "Kernel", "Linear"]
    }

    df_leaderboard = pd.DataFrame(leaderboard_data)

    def highlight_row(row):
        if row['Rank'] == '🥇':
            return ['background-color: #2d5016; color: white; font-weight: bold'] * len(row)
        elif row['Configuration'] == 'With Scaling':
            return ['background-color: #4a1a1a; color: #999'] * len(row)
        elif row['Rank'] in ['🥈', '🥉']:
            return ['background-color: #3a3a1a; color: #e0e0e0'] * len(row)
        else:
            return ['background-color: #2b2b2b; color: #e0e0e0'] * len(row)

    styled_df = df_leaderboard.style.apply(highlight_row, axis=1).format({
        'F1 Score': '{:.4f}',
        'Accuracy': '{:.2%}'
    })

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- 2. F1 vs Accuracy Scatter Plot (Παραμένει ως έχει) ---
    st.markdown("### 📊 F1 Score vs Accuracy")

    fig = go.Figure()

    # No Scaling models
    df_no_scaling = df_leaderboard[df_leaderboard['Configuration'] == 'No Scaling']
    fig.add_trace(go.Scatter(
        x=df_no_scaling['Accuracy'],
        y=df_no_scaling['F1 Score'],
        mode='markers+text',
        marker=dict(size=20, color='#FF6B35', line=dict(color='white', width=2), symbol='circle'),
        text=df_no_scaling['Model'],
        textposition='top center',
        name='No Scaling',
        hovertemplate='<b>%{text}</b><br>Accuracy: %{x:.2%}<br>F1: %{y:.4f}<extra></extra>'
    ))

    # With Scaling models
    df_with_scaling = df_leaderboard[df_leaderboard['Configuration'] == 'With Scaling']
    fig.add_trace(go.Scatter(
        x=df_with_scaling['Accuracy'],
        y=df_with_scaling['F1 Score'],
        mode='markers+text',
        marker=dict(size=20, color='#7d7d7d', line=dict(color='white', width=2), symbol='x'),
        text=df_with_scaling['Model'],
        textposition='bottom center',
        name='With Scaling',
        hovertemplate='<b>%{text}</b><br>Accuracy: %{x:.2%}<br>F1: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2b2b2b',
        font=dict(color='#e0e0e0'),
        xaxis=dict(title='Accuracy', range=[0.65, 0.95]),
        yaxis=dict(title='F1 Score', range=[0.65, 0.95]),
        legend=dict(bgcolor='#2b2b2b', bordercolor='#FF6B35', borderwidth=1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 3. LOCAL PLOTS VIEWER (Τροποποιημένο) ---
    st.markdown("### 🖼️ Model Performance Visualizations")

    # Mapping Μοντέλων σε προθέματα αρχείων
    # ΠΡΟΣΟΧΗ: Τα ονόματα εδώ πρέπει να ταιριάζουν με τα ονόματα των αρχείων σου στο φάκελο plots
    models_map = {
        "Comparison": "Comparison"
    }

    # Mapping Τύπων Γραφημάτων σε επιθέματα αρχείων
    plots_map = {
        "Accuracy": "gold_accuracy.svg",
        "Training vs Inference Time": "output.png",
        "Average Precision": "gold_average_precision.svg",
        "F1": "gold_f1.svg",
        "Inference Time": "gold_inference_time_sec.svg",
        "Log Loss": "gold_log_loss.svg",
        "Precision": "gold_precision.svg",
        "Recall" : "gold_recall.svg",
        "Roc AUC": "gold_roc_auc.svg"
    }

    col1, col2 = st.columns(2)

    with col1:
        selected_model_name = st.selectbox(
            "Select Model:",
            list(models_map.keys()),
            key="local_model_selector"
        )

    with col2:
        selected_plot_name = st.selectbox(
            "Select Visualization:",
            list(plots_map.keys()),
            key="local_plot_selector"
        )

    plot_code = plots_map[selected_plot_name]

    # Αν το Feature Importance δεν υπάρχει για όλα (π.χ. SVM), μπορούμε να βάλουμε έλεγχο
    image_filename = f"{plot_code}"
    image_path = Path("plots") / image_filename

    st.markdown("---")

    # Έλεγχος και εμφάνιση εικόνας
    if image_path.exists():
        st.image(
            str(image_path),
            caption=f"{selected_plot_name} - {selected_model_name}",
            use_container_width=True
        )

        # Προαιρετικά: Hardcoded Metrics για να φαίνεται "γεμάτο"
        with st.expander("📊 Model Metrics Snapshot", expanded=False):
            # Παίρνουμε τα metrics από το leaderboard data για ευκολία
            row = df_leaderboard[(df_leaderboard['Model'] == selected_model_name) &
                                 (df_leaderboard['Configuration'] == 'No Scaling')]
            if not row.empty:
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("F1 Score", f"{row.iloc[0]['F1 Score']:.4f}")
                col_m2.metric("Accuracy", f"{row.iloc[0]['Accuracy']:.2%}")
            else:
                st.info("Metrics not available for this configuration in summary table.")

    else:
        st.warning(f"⚠️ Η εικόνα δεν βρέθηκε: `{image_filename}`")
        st.info(f"""
        **Οδηγίες:**
        Βεβαιώσου ότι υπάρχει φάκελος `plots` δίπλα στο `app.py` και περιέχει το αρχείο:
        `{image_filename}`
        """)

    st.markdown("---")

    # --- 4. Insights Section (Παραμένει ως έχει) ---
    st.markdown("### 💡 Κεντρικά Συμπεράσματα")

    st.markdown("""
    <div class="insight-box">
        <h4>1️⃣ Tree-based > Linear > Kernel (on UMAP)</h4>
        <p>Τα δενδρικά μοντέλα υπερτερούν γιατί δεν επηρεάζονται από scaling και 
        μπορούν να μοντελοποιήσουν εναπομείναντα μη-γραμμικά patterns.</p>
    </div>

    <div class="insight-box">
        <h4>2️⃣ UMAP Made Linear Models Competitive</h4>
        <p>Η Logistic Regression έφτασε <strong>97.6%</strong> της απόδοσης του champion 
        (0.8570 vs 0.8784), αποδεικνύοντας ότι το UMAP δημιούργησε σχεδόν γραμμικά 
        διαχωρίσιμες κλάσεις.</p>
    </div>

    <div class="insight-box">
        <h4>3️⃣ Scaling = -15% to -19% Performance</h4>
        <p>Τα models με scaling υστερούν δραματικά, επιβεβαιώνοντας το Scaling Paradox.</p>
    </div>
    """, unsafe_allow_html=True)

def show_conclusions():
    st.markdown("## 📚 Συμπεράσματα & Μελλοντική Εργασία")
    
    # Key Findings
    st.markdown("### 🎯 Κεντρικά Ευρήματα")
    
    st.markdown("""
    <div class="metric-card">
        <h3>✅ 1. Gemma + UMAP: Ένας Εξαιρετικά Δυνατός Συνδυασμός</h3>
        <p>Το UMAP λειτούργησε ως <strong>"Manifold Unroller"</strong>, μετατρέποντας τις πολύπλοκες 
        σημασίες του LLM σε γεωμετρικά διαχωρίσιμες περιοχές.</p>
        <ul>
            <li>Το Gemma (7B) κατάλαβε το <strong>ύφος</strong>, τον <strong>σαρκασμό</strong> και τη <strong>δομή</strong></li>
            <li>Το UMAP διατήρησε την <strong>τοπολογία</strong> και την <strong>πυκνότητα</strong></li>
            <li>Αποτέλεσμα: Σχεδόν τέλεια linear separability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>✅ 2. Manifold Unrolling: Η Κλειδί-Ανακάλυψη</h3>
        <p>Τα δεδομένα έγιναν τόσο καθαρά που ακόμα και απλά γραμμικά μοντέλα 
        (Logistic Regression) πλησίασαν την απόδοση πολύπλοκων μοντέλων.</p>
        <ul>
            <li><strong>Logistic Regression:</strong> 0.86 F1 (χωρίς scaling)</li>
            <li><strong>Gradient Boosting:</strong> 0.88 F1 (champion)</li>
            <li>Διαφορά μόνο <strong>2%</strong> - απόδειξη της ποιότητας των features</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>⚠️ 3. Το Παράδοξο του Scaling</h3>
        <p>Σε pipelines που περιλαμβάνουν τοπολογική μείωση διαστάσεων (όπως το UMAP), 
        η "βίαιη" κανονικοποίηση (Standard Scaling) μπορεί να είναι <strong>καταστροφική</strong>, 
        καθώς αλλοιώνει την πληροφορία της πυκνότητας.</p>
        <ul>
            <li><strong>Logistic Regression με scaling:</strong> F1 = 0.70 (-15% πτώση!)</li>
            <li><strong>SVM με scaling:</strong> F1 = 0.68 (-15% πτώση!)</li>
            <li><strong>Γιατί;</strong> Το UMAP κωδικοποιεί πληροφορία στην κλίμακα των διαστάσεων</li>
            <li>Το StandardScaler "ισοπεδώνει" όλες τις διαστάσεις, χάνοντας τη δομή</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🏆 4. Gradient Boosting: Ο Αδιαμφισβήτητος Champion</h3>
        <p>Το Gradient Boosting αναδείχθηκε ως το καλύτερο μοντέλο για τρεις λόγους:</p>
        <ul>
            <li><strong>Scaling Invariant:</strong> Δεν επηρεάζεται από την κλίμακα</li>
            <li><strong>Non-linear Modeling:</strong> Βρίσκει τις τελευταίες μη-γραμμικές σχέσεις</li>
            <li><strong>Robust:</strong> Αντέχει σε overfitting μέσω regularization</li>
            <li><strong>Αποτέλεσμα:</strong> 91% Accuracy, 0.88 F1</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Final Recommendations
    st.markdown("### 💡 Τελικές Προτάσεις")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Για Μέγιστη Ακρίβεια:</h4>
            <h3 style="color: #FF6B35;">Gradient Boosting</h3>
            <p><strong>Χωρίς Scaling</strong></p>
            <ul>
                <li>F1: 0.88</li>
                <li>Accuracy: 0.91</li>
                <li>Training: ~12 min</li>
                <li>✅ Best για production</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <h4>⚡ Για Ταχύτητα/Πόρους:</h4>
            <h3 style="color: #FF8C42;">Logistic Regression</h3>
            <p><strong>Χωρίς Scaling</strong></p>
            <ul>
                <li>F1: 0.86</li>
                <li>Accuracy: 0.87</li>
                <li>Training: ~2 min</li>
                <li>✅ Best για real-time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Limitations
    st.markdown("### ⚠️ Περιορισμοί της Έρευνας")
    
    st.markdown("""
    <div class="metric-card">
        <ul>
            <li><strong>Domain-Specific:</strong> Τα αποτελέσματα μπορεί να διαφέρουν για μη-ειδησεογραφικό κείμενο</li>
            <li><strong>Computational Cost:</strong> Τα Gemma embeddings απαιτούν GPU resources</li>
            <li><strong>Static Model:</strong> Δεν προσαρμόζεται σε evolving clickbait patterns χωρίς retraining</li>
            <li><strong>Language:</strong> Κυρίως δοκιμασμένο σε Αγγλικούς τίτλους</li>
            <li><strong>Interpretability:</strong> Τα UMAP features δεν είναι εύκολα ερμηνεύσιμα από ανθρώπους</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

if __name__ == "__main__":
    main()
