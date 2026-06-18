"""Frontend Streamlit pour tester l'API de prédiction des prix de voitures."""
from __future__ import annotations

import os
import datetime
import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import mlflow
from mlflow.tracking import MlflowClient

API_INTERNAL_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
AIRFLOW_INTERNAL_URL = os.environ.get("AIRFLOW_URL", "http://airflow:8080")
MLFLOW_INTERNAL_URL = os.environ.get("MLFLOW_URL", "http://mlflow:5000")

EXTERNAL_IP = "88.96.57.190"
AIRFLOW_EXTERNAL_URL = f"http://{EXTERNAL_IP}:8080"
MLFLOW_EXTERNAL_URL = f"http://{EXTERNAL_IP}:5000"
API_EXTERNAL_URL = f"http://{EXTERNAL_IP}:8000/docs"

st.set_page_config(page_title="AutoPrice Pro", layout="wide", initial_sidebar_state="collapsed")

# Initialisation de l'historique dans la session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

def check_service(internal_url: str, external_url: str, name: str) -> dict:
    try:
        # Pour mlflow et airflow on teste juste la racine ou /health
        # On met un timeout très court pour ne pas bloquer
        endpoint = f"{internal_url}/health" if name == "API" else internal_url
        response = httpx.get(endpoint, timeout=2.0)
        is_up = response.status_code < 500
        return {"name": name, "status": "🟢 En ligne" if is_up else "🔴 Erreur", "url": external_url}
    except Exception:
        return {"name": name, "status": "🔴 Hors ligne", "url": external_url}

@st.cache_data(ttl=60)
def get_model_metrics():
    try:
        mlflow.set_tracking_uri(MLFLOW_INTERNAL_URL)
        client = MlflowClient()
        versions = client.search_model_versions("name='price_predictor'")
        if not versions:
            return None, "Aucune version du modèle 'price_predictor' n'a été trouvée."
        
        latest = max(versions, key=lambda v: int(v.version))
        run_id = latest.run_id
        run = client.get_run(run_id)
        
        # Download scatter plot if available
        import os
        scatter_path = None
        try:
            scatter_path = client.download_artifacts(run_id, "scatter.png")
        except Exception:
            pass
            
        return {
            "version": latest.version,
            "metrics": run.data.metrics,
            "run_id": run_id,
            "scatter_path": scatter_path
        }, None
    except Exception as e:
        return None, str(e)

# --- CSS CUSTOM PREMIUM AUTO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Background global plus sombre */
    .stApp {
        background: radial-gradient(circle at top, #1e1e38 0%, #0b0f19 100%);
    }

    /* Masquer le header/footer par défaut de Streamlit */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.3); /* Accent violet/bleu */
    }

    /* En-tête de la page (Titre principal) */
    .main-title {
        background: linear-gradient(135deg, #00E5FF, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
        padding-bottom: 0;
        text-shadow: 0px 0px 20px rgba(139, 92, 246, 0.3);
    }
    
    /* Sous-titre */
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Style des onglets Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255,255,255,0.05);
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
        color: #94a3b8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(139,92,246,0.1) 0%, rgba(255,255,255,0.05) 100%);
        color: #fff;
        border-top: 2px solid #8b5cf6;
    }

    /* Style des inputs Streamlit (Formulaire) */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.3) !important;
    }

    /* Bouton principal Streamlit (Submit) */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 20px rgba(139, 92, 246, 0.4) !important;
    }

    /* Boutons HTML custom (Airflow/MLflow) */
    .custom-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 20px;
        margin: 5px;
        background: rgba(255, 255, 255, 0.05);
        color: #fff;
        text-decoration: none;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .custom-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #06b6d4;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.3);
        transform: translateY(-2px);
    }
    .custom-btn.airflow { border-left: 3px solid #00E5FF; }
    .custom-btn.mlflow { border-left: 3px solid #3b82f6; }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER CUSTOM ---
col_title, col_btns = st.columns([2, 1])
with col_title:
    st.markdown('<h1 class="main-title">🏎️ AutoPrice Pro</h1>', unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Plateforme d'estimation intelligente par Johan Ribeiro Lamy</p>", unsafe_allow_html=True)

with col_btns:
    st.markdown(f"""
        <div style="text-align: right; margin-top: 15px;">
            <a href="{AIRFLOW_EXTERNAL_URL}" target="_blank" class="custom-btn airflow">🌪️ Airflow</a>
            <a href="{MLFLOW_EXTERNAL_URL}" target="_blank" class="custom-btn mlflow">📊 MLflow</a>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- ONGLETS ---
tab_home, tab_eval, tab_predict, tab_history, tab_surprise = st.tabs([
    "🏠 Accueil & État", 
    "📊 Évaluation Modèle", 
    "🔮 Prédiction", 
    "📋 Historique", 
    "🎁 Surprise"
])

with tab_home:
    st.markdown("""
        <div class="glass-card">
            <h3>Bienvenue sur AutoPrice Pro</h3>
            <p>Ce système de machine learning estime la valeur de revente de véhicules sur le marché en analysant leurs caractéristiques et leur historique.</p>
            <ul>
                <li><strong>Modèle prédictif</strong> entraîné sur des milliers de transactions réelles.</li>
                <li><strong>Pipeline automatisé</strong> via Airflow pour le réentraînement.</li>
                <li><strong>Tracking des modèles</strong> assuré par MLflow.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚦 État des Services")
    
    col_api, col_ml, col_air = st.columns(3)
    
    # Bouton de rafraichissement invisible qui force le rerun
    if st.button("🔄 Rafraîchir l'état", key="refresh_btn"):
        pass

    with st.spinner("Vérification des services..."):
        status_api = check_service(API_INTERNAL_URL, API_EXTERNAL_URL, "API")
        status_mlflow = check_service(MLFLOW_INTERNAL_URL, MLFLOW_EXTERNAL_URL, "MLflow")
        status_airflow = check_service(AIRFLOW_INTERNAL_URL, AIRFLOW_EXTERNAL_URL, "Airflow")

    with col_api:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <a href="{status_api['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                <h4>🚀 {status_api['name']}</h4>
            </a>
            <p style="font-size: 1.5rem; margin: 10px 0;">{status_api['status']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ml:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <a href="{status_mlflow['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                <h4>📊 {status_mlflow['name']}</h4>
            </a>
            <p style="font-size: 1.5rem; margin: 10px 0;">{status_mlflow['status']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_air:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <a href="{status_airflow['url']}" target="_blank" style="text-decoration: none; color: inherit;">
                <h4>🌪️ {status_airflow['name']}</h4>
            </a>
            <p style="font-size: 1.5rem; margin: 10px 0;">{status_airflow['status']}</p>
        </div>
        """, unsafe_allow_html=True)

with tab_eval:
    st.markdown("### 📊 Évaluation du Modèle Actif")
    st.markdown("Cette section affiche les performances du modèle actuellement déployé en production.")
    
    with st.spinner("Récupération des métriques depuis MLflow..."):
        data, error = get_model_metrics()
        
    if error:
        st.error(f"Impossible de récupérer les métriques : {error}")
    elif data:
        st.markdown(f"**Modèle :** `price_predictor` (Version {data['version']})")
        metrics = data['metrics']
        
        # Les metriques varient selon si l'entrainement etait simple ou hyperopt,
        # ou si c'est l'evaluation par defaut.
        r2 = metrics.get('r2', metrics.get('r2_score', 0))
        rmse = metrics.get('rmse', metrics.get('root_mean_squared_error', 0))
        mae = metrics.get('mae', metrics.get('mean_absolute_error', 0))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-bottom: 3px solid #00E5FF; padding: 15px;">
                <h4 style="color: #94a3b8; font-weight: normal; margin-bottom: 5px;">Score R²</h4>
                <p class="metric-value" style="font-size: 2.2rem; margin: 0;">{r2:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-bottom: 3px solid #8b5cf6; padding: 15px;">
                <h4 style="color: #94a3b8; font-weight: normal; margin-bottom: 5px;">RMSE</h4>
                <p class="metric-value" style="font-size: 2.2rem; margin: 0;">{rmse:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-bottom: 3px solid #f43f5e; padding: 15px;">
                <h4 style="color: #94a3b8; font-weight: normal; margin-bottom: 5px;">MAE</h4>
                <p class="metric-value" style="font-size: 2.2rem; margin: 0;">{mae:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        import os
        if data.get('scatter_path') and os.path.exists(data['scatter_path']):
            st.markdown("### 📈 Prédictions vs Valeurs Réelles")
            st.markdown("Ce graphique montre la corrélation entre les prix réels (axe X) et les prix prédits par le modèle (axe Y). Plus les points sont proches de la ligne rouge, meilleure est la prédiction.")
            
            # On met une marge pour eviter que l'image prenne tout l'ecran
            col_img1, col_img2, col_img3 = st.columns([1, 4, 1])
            with col_img2:
                st.image(data['scatter_path'], use_container_width=True)

with tab_predict:
    st.markdown("### ⚙️ Caractéristiques du véhicule")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Identité")
            brand = st.text_input("Marque", value="Toyota", help="Ex: Toyota, Honda, BMW")
            model = st.text_input("Modèle", value="Corolla", help="Ex: Corolla, Civic, X5")
            year = st.number_input("Année de fabrication", min_value=1990, max_value=2025, value=2018, step=1)
            registration_age = st.number_input("Âge d'immatriculation (années)", min_value=0, max_value=40, value=6, step=1)
            color = st.text_input("Couleur", value="White")
            city = st.text_input("Ville", value="Mumbai")
            
        with col2:
            st.markdown("#### Mécanique")
            engine_cc = st.number_input("Cylindrée (CC)", min_value=500.0, max_value=8000.0, value=1500.0)
            horsepower = st.number_input("Puissance (HP)", min_value=30.0, max_value=1000.0, value=110.0)
            mileage_kmpl = st.number_input("Consommation (km/L)", min_value=1.0, max_value=50.0, value=18.5)
            kms_driven = st.number_input("Kilométrage", min_value=0.0, max_value=1000000.0, value=45000.0)
            fuel_type = st.selectbox("Carburant", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            
        with col3:
            st.markdown("#### Historique & Administratif")
            seats = st.number_input("Nombre de sièges", min_value=2, max_value=10, value=5, step=1)
            number_of_doors = st.number_input("Nombre de portes", min_value=2, max_value=5, value=5, step=1)
            owner_type = st.selectbox("Propriétaire", ["First", "Second", "Third", "Fourth & Above"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            insurance_valid = st.checkbox("✅ Assurance valide", value=True)
            service_history = st.checkbox("🛠️ Carnet d'entretien complet", value=True)
            tax_paid = st.checkbox("💰 Taxes à jour", value=True)
            accidents = st.number_input("💥 Nombre d'accidents", min_value=0, max_value=10, value=0, step=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Lancer l'estimation", use_container_width=True)

    if submitted:
        payload = {
            "Year": year,
            "Mileage_kmpl": mileage_kmpl,
            "Engine_CC": engine_cc,
            "Horsepower": horsepower,
            "Kms_Driven": kms_driven,
            "Insurance_Valid": 1 if insurance_valid else 0,
            "Service_History": 1 if service_history else 0,
            "Accidents": accidents,
            "Tax_Paid": 1 if tax_paid else 0,
            "Number_of_Doors": number_of_doors,
            "Seats": seats,
            "Registration_Age": registration_age,
            "Brand": brand,
            "Model": model,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Color": color,
            "City": city
        }
        
        with st.spinner("Analyse des données par le modèle..."):
            try:
                response = httpx.post(f"{API_INTERNAL_URL}/predict", json=payload, timeout=10.0)
                response.raise_for_status()
                result = response.json()
                
                price_inr = result['price']
                price_eur = price_inr * 0.0112 # Conversion approximative
                
                # Ajout à l'historique
                st.session_state.prediction_history.insert(0, {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "vehicule": f"{brand} {model} ({year})",
                    "prix_inr": f"{price_inr:,.0f} ₹",
                    "prix_eur": f"{price_eur:,.0f} €",
                    "details": f"{kms_driven}km • {transmission} • {fuel_type}"
                })
                # Garder seulement les 10 derniers
                st.session_state.prediction_history = st.session_state.prediction_history[:10]

                st.markdown(f"""
                <div class="glass-card" style="border: 1px solid #00E5FF; text-align: center; background: linear-gradient(135deg, rgba(0,229,255,0.1), rgba(139,92,246,0.1));">
                    <h2 style="color: #fff; margin-bottom: 20px;">Estimation réussie 🎉</h2>
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                        <div>
                            <p style="color: #94a3b8; font-size: 1.2rem; margin: 0;">Prix Estimé (INR)</p>
                            <p class="metric-value">{price_inr:,.0f} ₹</p>
                        </div>
                        <div>
                            <p style="color: #94a3b8; font-size: 1.2rem; margin: 0;">Équivalent (EUR)</p>
                            <p class="metric-value">{price_eur:,.0f} €</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except httpx.HTTPError as exc:
                st.error(f"Appel à l'API impossible : {exc}")
            except KeyError:
                st.error(f"Réponse inattendue de l'API : {result}")

with tab_history:
    st.markdown("### 🕒 Historique des estimations (session locale)")
    
    if not st.session_state.prediction_history:
        st.info("Aucune estimation n'a été réalisée durant cette session.")
    else:
        df_history = pd.DataFrame(st.session_state.prediction_history)
        df_history.columns = ["Date / Heure", "Véhicule", "Prix (INR)", "Prix (EUR)", "Détails"]
        st.dataframe(df_history, use_container_width=True, hide_index=True)

with tab_surprise:
    st.markdown("### 🏎️ En attendant la fin d'un train...")
    st.markdown("Fais un petit tour de piste ! Utilise les **flèches directionnelles** de ton clavier pour piloter.")
    
    game_html = """
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; background: #0b0f19; padding: 20px; border-radius: 16px;">
      <p style="color: #94a3b8; margin-bottom: 10px;"><i>Clique sur le circuit pour activer les contrôles clavier.</i></p>
      <canvas id="gameCanvas" width="800" height="400" style="border: 2px solid #8b5cf6; border-radius: 8px; box-shadow: 0 0 20px rgba(139,92,246,0.3); outline: none;" tabindex="0"></canvas>
      <script>
        const canvas = document.getElementById("gameCanvas");
        const ctx = canvas.getContext("2d");
        
        // Permet au canvas de capter les fleches sans scroller la page
        canvas.focus();
        canvas.addEventListener('click', () => canvas.focus());

        const keys = {};
        window.addEventListener("keydown", e => {
          if(["ArrowUp","ArrowDown","ArrowLeft","ArrowRight"].includes(e.code)) {
              e.preventDefault(); // Bloque le scroll
          }
          keys[e.code] = true;
        }, { passive: false });
        window.addEventListener("keyup", e => keys[e.code] = false);
        window.addEventListener("blur", () => {
          for(let k in keys) keys[k] = false;
        });


        const car = {
          x: 280, y: 80,
          width: 24, height: 12,
          angle: 0, speed: 0,
          maxSpeed: 4.5, acceleration: 0.15,
          friction: 0.05, grassFriction: 0.15,
          rotationSpeed: 0.07
        };

        const track = {
          cx1: 280, cy1: 200, r: 120,
          cx2: 520, cy2: 200, r: 120,
          thickness: 80
        };

        function onTrack(x, y) {
          const d1 = Math.hypot(x - track.cx1, y - track.cy1);
          const d2 = Math.hypot(x - track.cx2, y - track.cy2);
          
          const onCircle1 = Math.abs(d1 - track.r) < track.thickness / 2;
          const onCircle2 = Math.abs(d2 - track.r) < track.thickness / 2;
          
          return onCircle1 || onCircle2;
        }

        function drawTrack() {
          // Herbe
          ctx.fillStyle = "#166534";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          
          ctx.lineWidth = track.thickness;
          ctx.lineCap = "round";
          
          // Asphalte
          ctx.strokeStyle = "#334155";
          
          ctx.beginPath();
          ctx.arc(track.cx1, track.cy1, track.r, 0, Math.PI * 2);
          ctx.stroke();
          
          ctx.beginPath();
          ctx.arc(track.cx2, track.cy2, track.r, 0, Math.PI * 2);
          ctx.stroke();
          
          // Ligne de depart
          ctx.strokeStyle = "white";
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(track.cx1, track.cy1 - track.r - track.thickness/2);
          ctx.lineTo(track.cx1, track.cy1 - track.r + track.thickness/2);
          ctx.stroke();
        }

        function drawCar() {
          ctx.save();
          ctx.translate(car.x, car.y);
          ctx.rotate(car.angle);
          
          // Chassis
          ctx.fillStyle = "#ef4444";
          ctx.fillRect(-car.width/2, -car.height/2, car.width, car.height);
          
          // Pare-brise
          ctx.fillStyle = "#000";
          ctx.fillRect(-car.width/4, -car.height/2 + 2, car.width/4, car.height - 4);
          
          // Phares
          ctx.fillStyle = "#fbbf24";
          ctx.fillRect(car.width/2 - 2, -car.height/2 + 1, 2, 3);
          ctx.fillRect(car.width/2 - 2, car.height/2 - 4, 2, 3);

          ctx.restore();
        }

        function update() {
          let currentFriction = onTrack(car.x, car.y) ? car.friction : car.grassFriction;

          if (keys["ArrowUp"]) car.speed += car.acceleration;
          if (keys["ArrowDown"]) car.speed -= car.acceleration;
          
          if (Math.abs(car.speed) > 0.2) {
            const dir = car.speed > 0 ? 1 : -1;
            if (keys["ArrowLeft"]) car.angle -= car.rotationSpeed * dir;
            if (keys["ArrowRight"]) car.angle += car.rotationSpeed * dir;
          }

          if (car.speed > 0) {
            car.speed -= currentFriction;
            if (car.speed < 0) car.speed = 0;
          } else if (car.speed < 0) {
            car.speed += currentFriction;
            if (car.speed > 0) car.speed = 0;
          }

          if (car.speed > car.maxSpeed) car.speed = car.maxSpeed;
          if (car.speed < -car.maxSpeed/2) car.speed = -car.maxSpeed/2;

          car.x += Math.cos(car.angle) * car.speed;
          car.y += Math.sin(car.angle) * car.speed;
          
          if (car.x < 0) car.x = 0;
          if (car.x > canvas.width) car.x = canvas.width;
          if (car.y < 0) car.y = 0;
          if (car.y > canvas.height) car.y = canvas.height;
        }

        function loop() {
          update();
          drawTrack();
          drawCar();
          requestAnimationFrame(loop);
        }

        // Init
        loop();
      </script>
    </div>
    """
    
    components.html(game_html, height=450, scrolling=False)
