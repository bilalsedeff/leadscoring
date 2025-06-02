import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Temel dizinler
OUTPUT_DIR = Path("outputs")
FEATURE_IMP_DIR = OUTPUT_DIR / "feature_importance"
MODELS_DIR = OUTPUT_DIR / "models"
STATS_DIR = OUTPUT_DIR / "stats"

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Lead Conversion Analizi",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Lead Conversion Analiz Paneli")
page = st.sidebar.radio(
    "Sayfa Seçin:",
    ["📊 Genel Bakış", 
     "🔍 Feature Importance", 
     "📈 Model Performansı",
     "👥 Cohort Analizi",
     "🧪 MLflow Experiments"]
)

# Başlık
st.title("Lead Conversion Analiz Paneli")

# MLflow bağlantısı
@st.cache_resource
def get_mlflow_client():
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        return mlflow.tracking.MlflowClient()
    except Exception as e:
        st.error(f"MLflow bağlantısı kurulamadı: {e}")
        return None

# Veri yükleme fonksiyonları
@st.cache_data
def load_feature_importance():
    try:
        if (FEATURE_IMP_DIR / "importance.csv").exists():
            return pd.read_csv(FEATURE_IMP_DIR / "importance.csv")
        else:
            return None
    except Exception as e:
        st.error(f"Feature importance yüklenemedi: {e}")
        return None

@st.cache_data
def load_feature_types():
    try:
        if (FEATURE_IMP_DIR / "feature_types.csv").exists():
            return pd.read_csv(FEATURE_IMP_DIR / "feature_types.csv")
        else:
            return None
    except Exception as e:
        st.error(f"Feature tipleri yüklenemedi: {e}")
        return None

@st.cache_data
def load_selected_features():
    try:
        if (FEATURE_IMP_DIR / "selected_features.txt").exists():
            with open(FEATURE_IMP_DIR / "selected_features.txt", "r") as f:
                return [line.strip() for line in f.readlines()]
        else:
            return None
    except Exception as e:
        st.error(f"Seçilen özellikler yüklenemedi: {e}")
        return None

@st.cache_data
def load_model_metrics():
    try:
        metrics_files = list(MODELS_DIR.glob("*_metrics.csv"))
        if metrics_files:
            return pd.read_csv(metrics_files[0])
        else:
            return None
    except Exception as e:
        st.error(f"Model metrikleri yüklenemedi: {e}")
        return None

@st.cache_data
def load_cohort_analysis():
    try:
        cohort_files = list(STATS_DIR.glob("*_cohort_analysis.csv"))
        if cohort_files:
            return pd.read_csv(cohort_files[0])
        else:
            return None
    except Exception as e:
        st.error(f"Cohort analizi yüklenemedi: {e}")
        return None

# MLflow deneylerini yükle
@st.cache_data
def load_mlflow_experiments():
    client = get_mlflow_client()
    if client:
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            exp_runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in exp_runs:
                run_info = {
                    "experiment_name": exp.name,
                    "run_id": run.info.run_id,
                    "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
                    "status": run.info.status,
                }
                # Metrikleri ekle
                for key, value in run.data.metrics.items():
                    run_info[key] = value
                
                # Parametreleri ekle
                for key, value in run.data.params.items():
                    run_info[f"param_{key}"] = value
                
                runs.append(run_info)
        
        return pd.DataFrame(runs) if runs else None
    return None

# Genel Bakış sayfası
def overview_page():
    st.header("Genel Bakış")
    
    # İstatistikler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Toplam Feature Sayısı", 
                 value=len(load_feature_types()) if load_feature_types() is not None else "N/A")
    
    with col2:
        selected = load_selected_features()
        st.metric(label="Seçilen Feature Sayısı", 
                 value=len(selected) if selected is not None else "N/A")
    
    with col3:
        experiments = load_mlflow_experiments()
        st.metric(label="Toplam Experiment Sayısı", 
                 value=len(experiments['experiment_name'].unique()) if experiments is not None else "N/A")
    
    # Dizin yapısını göster
    st.subheader("Proje Yapısı")
    
    if OUTPUT_DIR.exists():
        output_structure = {}
        
        for component_dir in OUTPUT_DIR.iterdir():
            if component_dir.is_dir():
                files = []
                for file in component_dir.iterdir():
                    if file.is_file():
                        files.append({
                            "name": file.name,
                            "size": f"{file.stat().st_size / 1024:.2f} KB",
                            "modified": pd.to_datetime(file.stat().st_mtime, unit='s')
                        })
                
                output_structure[component_dir.name] = files
        
        # Tab gösterimi
        if output_structure:
            tabs = st.tabs(list(output_structure.keys()))
            
            for i, (component, files) in enumerate(output_structure.items()):
                if files:
                    with tabs[i]:
                        st.dataframe(pd.DataFrame(files), use_container_width=True)
                else:
                    with tabs[i]:
                        st.info(f"Bu bileşen için dosya bulunamadı.")
        else:
            st.info("Henüz hiçbir çıktı dosyası oluşturulmamış.")
    else:
        st.warning("outputs/ dizini bulunamadı. İlk önce analizleri çalıştırın.")
    
    # Son deneyler
    st.subheader("Son Deneyler")
    experiments = load_mlflow_experiments()
    
    if experiments is not None and not experiments.empty:
        # En son deneyleri göster
        recent_experiments = experiments.sort_values('start_time', ascending=False).head(5)
        st.dataframe(recent_experiments[['experiment_name', 'start_time', 'status']], use_container_width=True)
        
        # Metrics varsa göster
        metrics_cols = [col for col in experiments.columns if col not in ['experiment_name', 'run_id', 'start_time', 'status'] and not col.startswith('param_')]
        
        if metrics_cols:
            st.subheader("Model Performans Karşılaştırması")
            
            # Metrik seçimi
            selected_metric = st.selectbox("Karşılaştırma Metriği:", metrics_cols)
            
            # Metrik karşılaştırma grafiği
            fig = px.bar(
                experiments.sort_values(selected_metric, ascending=False).head(10),
                x='experiment_name',
                y=selected_metric,
                color='experiment_name',
                title=f"Experiment Karşılaştırması ({selected_metric})"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Henüz MLflow'da experiment bulunamadı. Eğitim çalıştırın.")

# Feature Importance sayfası
def feature_importance_page():
    st.header("Feature Importance Analizi")
    
    # Feature Importance yükle
    importance_df = load_feature_importance()
    feature_types = load_feature_types()
    selected_features = load_selected_features()
    
    if importance_df is not None:
        # Top 20 features
        st.subheader("En Önemli 20 Özellik")
        
        fig = px.bar(
            importance_df.head(20),
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Feature Importance",
            color='importance',
            color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature type dağılımı
        if feature_types is not None:
            st.subheader("Özellik Tipi Dağılımı")
            
            # Tiplerine göre feature'ları grupla
            type_counts = feature_types['type'].value_counts().reset_index()
            type_counts.columns = ['Tip', 'Sayı']
            
            fig = px.pie(
                type_counts,
                values='Sayı',
                names='Tip',
                title="Özellik Tiplerinin Dağılımı"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Seçilen özellikleri göster
        if selected_features is not None:
            st.subheader("Seçilen Özellikler")
            
            # Seçilen feature'ların importance değerlerini göster
            selected_importance = importance_df[importance_df['feature'].isin(selected_features)]
            st.dataframe(selected_importance, use_container_width=True)
            
            # Seçilen/seçilmeyen özellik dağılımı
            st.subheader("Seçilen vs. Seçilmeyen Özellikler")
            
            selection_status = pd.DataFrame({
                'Status': ['Seçilen', 'Seçilmeyen'],
                'Count': [len(selected_features), len(importance_df) - len(selected_features)]
            })
            
            fig = px.pie(
                selection_status,
                values='Count',
                names='Status',
                title="Özellik Seçim Durumu",
                color='Status',
                color_discrete_map={'Seçilen': 'green', 'Seçilmeyen': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP özet görselleri varsa göster
        shap_path = FEATURE_IMP_DIR / "shap_importance.png"
        if shap_path.exists():
            st.subheader("SHAP Özet Grafiği")
            st.image(str(shap_path))
        
        # Feature stability görseli
        stability_path = FEATURE_IMP_DIR / "feature_stability.png"
        if stability_path.exists():
            st.subheader("Feature Stability Analizi")
            st.image(str(stability_path))
    else:
        st.warning("Feature importance bilgileri bulunamadı. Lütfen önce feature importance analizi çalıştırın.")

# Model Performans sayfası
def model_performance_page():
    st.header("Model Performans Analizi")
    
    # Metrik dosyalarını arama
    metrics_files = list(MODELS_DIR.glob("*_metrics.csv"))
    performance_plots = list(MODELS_DIR.glob("*_performance.png"))
    confusion_matrices = list(MODELS_DIR.glob("*_confusion.png"))
    
    if metrics_files:
        metrics_df = pd.read_csv(metrics_files[0])
        
        # Metrik tablosu
        st.subheader("Model Performans Metrikleri")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Metrik görselleştirme
        if not metrics_df.empty:
            # İlk sütunu hariç tut (muhtemelen model adı)
            numeric_cols = metrics_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if numeric_cols:
                st.subheader("Metrik Karşılaştırması")
                
                # Radar chart için metrik seçimi
                selected_metrics = st.multiselect(
                    "Karşılaştırılacak metrikleri seçin:",
                    options=numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if selected_metrics:
                    # Radar chart için veri hazırlama
                    model_names = metrics_df.iloc[:, 0].tolist() if metrics_df.shape[1] > 1 else [f"Model {i+1}" for i in range(len(metrics_df))]
                    
                    fig = go.Figure()
                    
                    for i, model in enumerate(model_names):
                        values = metrics_df.iloc[i][selected_metrics].tolist()
                        # Listeyi kapatmak için ilk değeri sona ekle
                        values_closed = values + [values[0]]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values_closed,
                            theta=selected_metrics + [selected_metrics[0]],
                            fill='toself',
                            name=model
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Performans görselleri
    if performance_plots:
        st.subheader("Model Performans Grafikleri")
        for plot in performance_plots:
            st.image(str(plot), caption=plot.stem)
    
    # Confusion matrisleri
    if confusion_matrices:
        st.subheader("Confusion Matrisleri")
        for cm in confusion_matrices:
            st.image(str(cm), caption=cm.stem)
    
    # MLflow'dan deney metrikleri
    st.subheader("MLflow Deney Performansları")
    experiments = load_mlflow_experiments()
    
    if experiments is not None and not experiments.empty:
        # Metrik sütunları
        metric_cols = [col for col in experiments.columns if col not in ['experiment_name', 'run_id', 'start_time', 'status'] and not col.startswith('param_')]
        
        if metric_cols:
            # Experiment seçimi
            exp_names = sorted(experiments['experiment_name'].unique())
            selected_exp = st.selectbox("Experiment:", exp_names)
            
            # Seçilen deneyin metrikleri
            exp_df = experiments[experiments['experiment_name'] == selected_exp]
            
            if not exp_df.empty:
                st.dataframe(exp_df[['start_time', 'status'] + metric_cols], use_container_width=True)
                
                # Metrik bazlı karşılaştırma
                st.subheader("Metrik Detayları")
                selected_metric = st.selectbox("Metrik:", metric_cols)
                
                fig = px.bar(
                    exp_df.sort_values(selected_metric, ascending=False),
                    x='start_time',
                    y=selected_metric,
                    color=selected_metric,
                    title=f"{selected_exp} - {selected_metric} Değerleri"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("MLflow deneylerinde metrik bilgisi bulunamadı.")
    else:
        st.warning("MLflow deney bilgileri yüklenemedi.")

# Cohort Analizi sayfası
def cohort_analysis_page():
    st.header("Cohort Analizi")
    
    # Cohort analizi dosyasını arama
    cohort_files = list(STATS_DIR.glob("*_cohort_analysis.csv"))
    segment_files = list(STATS_DIR.glob("*_segment_performance.csv"))
    source_files = list(STATS_DIR.glob("*_source_performance.csv"))
    
    if cohort_files:
        # Cohort analizi
        cohort_df = pd.read_csv(cohort_files[0])
        
        st.subheader("Cohort Bazlı Conversion Oranları")
        st.dataframe(cohort_df, use_container_width=True)
        
        # Cohort görselleştirme
        if not cohort_df.empty:
            # Cohort sütunları
            cohort_cols = cohort_df.columns.tolist()
            
            # Cohort ve metrik seçimi
            cohort_col = st.selectbox("Cohort Sütunu:", [col for col in cohort_cols if col not in ['count', 'conversion_rate']])
            
            if 'conversion_rate' in cohort_cols:
                # Conversion rate heatmap
                pivot_df = cohort_df.pivot_table(
                    values='conversion_rate',
                    index=cohort_col,
                    columns='segment' if 'segment' in cohort_cols else None
                )
                
                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Segment" if 'segment' in cohort_cols else "", y=cohort_col, color="Conversion Rate"),
                    x=pivot_df.columns,
                    y=pivot_df.index,
                    color_continuous_scale='Viridis',
                    title=f"Conversion Oranları - {cohort_col} bazlı"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Segment performans analizi
    if segment_files:
        segment_df = pd.read_csv(segment_files[0])
        
        st.subheader("Segment Performans Analizi")
        st.dataframe(segment_df, use_container_width=True)
        
        # Segment görselleştirme
        if not segment_df.empty:
            segment_col = 'segment' if 'segment' in segment_df.columns else segment_df.columns[0]
            metric_cols = segment_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if metric_cols:
                selected_metric = st.selectbox("Segment Metriği:", metric_cols)
                
                fig = px.bar(
                    segment_df,
                    x=segment_col,
                    y=selected_metric,
                    color=segment_col,
                    title=f"Segment Bazlı {selected_metric}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Kaynak bazlı performans analizi
    if source_files:
        source_df = pd.read_csv(source_files[0])
        
        st.subheader("Kaynak Bazlı Performans Analizi")
        st.dataframe(source_df, use_container_width=True)
        
        # Kaynak görselleştirme
        if not source_df.empty:
            source_col = 'source' if 'source' in source_df.columns else source_df.columns[0]
            metric_cols = source_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if metric_cols:
                selected_metric = st.selectbox("Kaynak Metriği:", metric_cols, key="source_metric")
                
                # Top 10 kaynak göster (çok fazlaysa)
                plot_df = source_df.sort_values(selected_metric, ascending=False).head(10)
                
                fig = px.bar(
                    plot_df,
                    x=source_col,
                    y=selected_metric,
                    color=source_col,
                    title=f"Kaynak Bazlı {selected_metric} (Top 10)"
                )
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
    
    if not any([cohort_files, segment_files, source_files]):
        st.warning("Cohort analiz sonuçları bulunamadı. Lütfen önce istatistiksel analiz çalıştırın.")

# MLflow Experiments sayfası
def mlflow_experiments_page():
    st.header("MLflow Experiment Analizi")
    
    # MLflow client
    client = get_mlflow_client()
    
    if client:
        # Experiment ve run'ları getir
        experiments = load_mlflow_experiments()
        
        if experiments is not None and not experiments.empty:
            # Experiment seçimi
            exp_names = sorted(experiments['experiment_name'].unique())
            selected_exp = st.selectbox("Experiment:", exp_names)
            
            # Seçilen deneyin run'ları
            exp_runs = experiments[experiments['experiment_name'] == selected_exp]
            
            # Run seçimi
            run_times = exp_runs['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            run_ids = exp_runs['run_id'].tolist()
            
            selected_index = st.selectbox(
                "Run:",
                range(len(run_times)),
                format_func=lambda i: f"{run_times[i]} ({run_ids[i][:8]}...)"
            )
            
            selected_run = exp_runs.iloc[selected_index]
            
            # Run detayları
            st.subheader("Run Detayları")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Run ID:** `{selected_run['run_id']}`")
                st.markdown(f"**Başlangıç Zamanı:** {selected_run['start_time']}")
                st.markdown(f"**Durum:** {selected_run['status']}")
            
            # Parametreler
            param_cols = [col for col in selected_run.index if col.startswith('param_')]
            
            if param_cols:
                with col2:
                    st.subheader("Parametreler")
                    for param in param_cols:
                        param_name = param.replace('param_', '')
                        st.markdown(f"**{param_name}:** {selected_run[param]}")
            
            # Metrikler
            metric_cols = [col for col in selected_run.index if col not in ['experiment_name', 'run_id', 'start_time', 'status'] and not col.startswith('param_')]
            
            if metric_cols:
                st.subheader("Metrikler")
                
                metrics_df = pd.DataFrame({
                    'Metrik': metric_cols,
                    'Değer': [selected_run[col] for col in metric_cols]
                })
                
                st.dataframe(metrics_df, use_container_width=True)
                
                # Metrik görselleştirme
                fig = px.bar(
                    metrics_df,
                    x='Metrik',
                    y='Değer',
                    color='Metrik',
                    title="Metrik Değerleri"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Artifacts
            st.subheader("Artifacts")
            
            try:
                artifacts = client.list_artifacts(selected_run['run_id'])
                
                if artifacts:
                    artifacts_df = pd.DataFrame([
                        {'İsim': a.path, 'Boyut': f"{a.file_size/1024:.2f} KB", 'Tip': 'Klasör' if a.is_dir else 'Dosya'}
                        for a in artifacts
                    ])
                    
                    st.dataframe(artifacts_df, use_container_width=True)
                else:
                    st.info("Bu run için artifact bulunamadı.")
            except Exception as e:
                st.error(f"Artifact listesi alınamadı: {e}")
            
            # MLflow UI linki
            st.markdown(f"[MLflow UI'da görüntüle](http://localhost:5000/#/experiments/{client.get_experiment_by_name(selected_exp).experiment_id}/runs/{selected_run['run_id']})")
        else:
            st.warning("MLflow'da herhangi bir experiment bulunamadı.")
    else:
        st.error("MLflow bağlantısı kurulamadı. MLflow servisinin çalıştığından emin olun.")

# Sayfalara yönlendirme
if page == "📊 Genel Bakış":
    overview_page()
elif page == "🔍 Feature Importance":
    feature_importance_page()
elif page == "📈 Model Performansı":
    model_performance_page()
elif page == "👥 Cohort Analizi":
    cohort_analysis_page()
elif page == "🧪 MLflow Experiments":
    mlflow_experiments_page()
