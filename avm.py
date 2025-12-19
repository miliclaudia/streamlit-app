import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURARE PAGINĂ ---
st.set_page_config(
    page_title="Tema EDA cu Streamlit",
    page_icon="📊",
    layout="wide"
)

# --- 2. CSS PENTRU UI (ADAPTABIL LIGHT/DARK) ---
st.markdown("""
<style>
    /* Stiluri Generale */
    .main-header {
        font-size: 2.5rem;
        color: #4da6ff; 
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 2px solid #4da6ff;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ffa600; 
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #ffa600;
        padding-left: 10px;
    }
    
    /* Stiluri Pagina Home */
    .home-title {
        font-size: 2.8rem;
        font-weight: bold;
        /* Folosim o culoare care se vede bine pe ambele teme sau lasam default */
        text-align: center;
        padding-bottom: 20px;
        margin-bottom: 30px;
        border-bottom: 1px solid var(--text-color);
    }
    
    .section-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #ff9f43; 
        margin-top: 25px;
        margin-bottom: 10px;
    }
    
    .description-text {
        font-size: 1.1rem;
        line-height: 1.7;
        text-align: justify;
        margin-bottom: 15px;
        /* Nu fortam culoarea textului, lasam Streamlit sa decida (negru pe light, alb pe dark) */
    }
    
    /* Container subtil pentru cerinte - Adaptabil */
    .req-box {
        padding: 15px 20px;
        border-radius: 8px;
        /* Fundal usor vizibil pe ambele teme (gri foarte deschis transparent) */
        background-color: rgba(128, 128, 128, 0.1); 
        margin-bottom: 20px;
        border-left: 4px solid #4da6ff;
    }

    /* CARD STUDENT - DESIGN NOU ADAPTABIL */
    .student-card {
        /* Folosim variabile CSS Streamlit pentru a se adapta la tema */
        background-color: var(--secondary-background-color);
        border: 1px solid #4da6ff;
        border-radius: 15px;
        padding: 40px 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .student-name {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 15px;
        /* Culoare adaptabila */
        color: var(--text-color);
    }
    
    .student-group {
        font-size: 1.4rem;
        color: #ffa600;
        font-weight: bold;
        margin-bottom: 0px;
    }
    
    .icon-big {
        font-size: 5rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNCȚII UTILITARE ---
@st.cache_data
def load_data(file):
    """Încarcă datele din CSV sau Excel."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        return None

# --- 4. MENIU LATERAL ---
def sidebar_menu():
    st.sidebar.title("Tema EDA")
    st.sidebar.write("Navigare:")
    
    options = [
        "Descriere Aplicație",
        "Cerinta 1: Incarcare si Filtrare",
        "Cerinta 2: Structura si Valori Lipsa",
        "Cerinta 3: Analiza Univariata (Num)",
        "Cerinta 4: Analiza Categorica",
        "Cerinta 5: Corelatie si Outlieri"
    ]
    
    return st.sidebar.radio("Mergi la:", options)

# --- 5. IMPLEMENTARE PAGINI ---

def page_home():
    # Titlu Principal
    st.markdown('<div class="home-title">Descrierea Aplicației EDA cu Streamlit</div>', unsafe_allow_html=True)
    
    # Text introductiv
    st.markdown("""
    <div class="description-text">
    Aplicația a fost dezvoltată utilizând framework-ul Streamlit și are ca obiectiv realizarea unei analize exploratorii a 
    datelor (Exploratory Data Analysis – EDA) pe seturi de date de mari dimensiuni, furnizate de utilizator sub forma 
    fișierelor CSV sau Excel. Soluția propusă pune accent pe interactivitate și accesibilitate, permițând explorarea și 
    înțelegerea datelor fără a necesita cunoștințe avansate de programare.
    <br><br>
    Interfața grafică este intuitivă și organizată modular, facilitând investigarea structurii datasetului, evaluarea 
    calității datelor, analiza distribuțiilor, examinarea variabilelor categorice, identificarea relațiilor dintre 
    variabile numerice și detectarea valorilor atipice (outlieri). Funcționalitățile aplicației sunt grupate în cinci 
    secțiuni distincte, accesibile prin intermediul meniului lateral, fiecare corespunzând unei cerințe specifice ale temei.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Secțiunea 1
    with st.container():
        st.markdown('<div class="section-title">📂 Secțiunea 1 – Încărcarea și filtrarea datelor</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="req-box">
        <div class="description-text">
        Aplicația permite importarea fișierelor CSV sau Excel, efectuând automat verificări pentru a confirma încărcarea 
        corectă a acestora. După procesul de import, utilizatorului i se afișează un mesaj informativ care include 
        dimensiunile setului de date, precum și un eșantion reprezentativ format din primele 10 înregistrări. În plus, 
        sunt puse la dispoziție mecanisme interactive de filtrare: slidere pentru variabilele numerice și liste multiselect 
        pentru cele categorice. Numărul de observații inițiale este comparat cu cel rezultat după filtrare, permițând 
        delimitarea rapidă a subseturilor relevante.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Secțiunea 2
    with st.container():
        st.markdown('<div class="section-title">📋 Secțiunea 2 – Structura datasetului și analiza valorilor lipsă</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="req-box">
        <div class="description-text">
        Această etapă oferă o imagine de ansamblu asupra structurii datelor, incluzând dimensiunea datasetului și tipurile 
        de date asociate fiecărei coloane. Aplicația identifică automat variabilele care conțin valori lipsă și calculează 
        proporția acestora, prezentând rezultatele sub forma unui grafic de bare pentru o evaluare vizuală rapidă a calității 
        datelor. Totodată, sunt calculate statisticile descriptive fundamentale pentru variabilele numerice, precum media, 
        mediana, deviația standard, valorile extreme și cuartilele.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Secțiunea 3
    with st.container():
        st.markdown('<div class="section-title">📈 Secțiunea 3 – Analiza univariată a variabilelor numerice</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="req-box">
        <div class="description-text">
        Utilizatorul poate selecta orice variabilă numerică pentru a-i analiza distribuția. Aplicația generează o histogramă 
        interactivă, cu posibilitatea ajustării numărului de intervale, oferind flexibilitate în interpretarea distribuției. 
        Complementar, este afișat un box plot care evidențiază dispersia datelor și posibilele valori extreme. Principalii 
        indicatori statistici sunt calculați și afișați pentru a susține analiza vizuală.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Secțiunea 4
    with st.container():
        st.markdown('<div class="section-title">📊 Secțiunea 4 – Analiza variabilelor categorice</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="req-box">
        <div class="description-text">
        Această secțiune se concentrează pe variabilele de tip categoric sau text, identificate automat de aplicație. 
        După selectarea unei coloane, este generat un grafic de tip bar chart care reflectă frecvența fiecărei categorii. 
        În paralel, este afișat un tabel detaliat ce conține atât frecvențele absolute, cât și procentele corespunzătoare 
        fiecărei clase, oferind o perspectivă clară asupra distribuției categoriilor.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # Secțiunea 5
    with st.container():
        st.markdown('<div class="section-title">🔗 Secțiunea 5 – Corelații și identificarea outlierilor</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="req-box">
        <div class="description-text">
        Ultima componentă a aplicației este dedicată explorării relațiilor dintre variabilele numerice și identificării 
        valorilor anormale. Se calculează matricea de corelație, care este reprezentată grafic printr-un heatmap interactiv. 
        Utilizatorul poate analiza relația dintre două variabile selectate folosind un scatter plot, alături de coeficientul 
        de corelație Pearson. În plus, aplicația implementează metoda IQR (Interquartile Range) pentru detectarea automată 
        a outlierilor, afișând atât numărul și procentul acestora, cât și reprezentarea lor grafică.
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    # Footer simplu
    st.markdown('<div style="text-align: center; color: #888;">Autor Proiect: <b>Militaru Maria Claudia</b> | Grupa 1027 BDSA</div>', unsafe_allow_html=True)

def page_cerinta_1():
    st.markdown('<h1 class="main-header">CERINȚA 1: Încărcare și Filtrare Date</h1>', unsafe_allow_html=True)

    # 1. Încărcare fișier
    uploaded_file = st.file_uploader("📂 Încarcă fișier CSV sau Excel", type=['csv', 'xlsx'])

    # Logica de încărcare și salvare în sesiune
    if uploaded_file is not None:
        df_new = load_data(uploaded_file)
        if df_new is not None:
            st.session_state['df_raw'] = df_new
            # Resetăm filtrarea doar dacă se încarcă un fișier nou
            if 'df' not in st.session_state or len(st.session_state['df']) != len(df_new):
                st.session_state['df'] = df_new
            st.success(f"✅ Fișierul **{uploaded_file.name}** a fost citit corect!")
    
    
    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw']
        
        # 3. Afișare primele 10 rânduri
        st.markdown('<div class="sub-header">Previzualizare Dataset (Primele 10 rânduri)</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="sub-header">Filtrare Avansată</div>', unsafe_allow_html=True)
        
        # Lucrăm pe o copie pentru filtrare
        df_filtered = df.copy()
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns(2)

        # --- FILTRARE NUMERICĂ ---
        with col1:
            st.info("🔢 Filtrare Coloane Numerice")
            cols_to_filter_num = st.multiselect("Alege coloanele numerice de filtrat:", numeric_cols)
            
            for col in cols_to_filter_num:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                
                # Verificăm dacă există un interval valid
                if min_val < max_val:
                    # Folosim key unic pentru a păstra starea slider-ului
                    values = st.slider(
                        f"Interval pentru **{col}**:", 
                        min_val, max_val, (min_val, max_val), 
                        key=f"slider_{col}"
                    )
                    df_filtered = df_filtered[
                        (df_filtered[col] >= values[0]) & 
                        (df_filtered[col] <= values[1])
                    ]
                else:
                    st.warning(f"Coloana {col} are o singură valoare ({min_val}), nu se poate filtra.")

        # --- FILTRARE CATEGORICĂ ---
        with col2:
            st.info("🔠 Filtrare Coloane Categorice")
            cols_to_filter_cat = st.multiselect("Alege coloanele categorice de filtrat:", cat_cols)
            
            for col in cols_to_filter_cat:
                unique_vals = df[col].dropna().unique()
                # Folosim key unic
                selected_vals = st.multiselect(
                    f"Valori pentru **{col}**:", 
                    unique_vals, default=unique_vals, 
                    key=f"multi_{col}"
                )
                
                if selected_vals:
                    df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]

        st.markdown("---")
        
        # 6. Rezultate Filtrare
        st.markdown("### Rezultate Filtrare")
        m1, m2 = st.columns(2)
        m1.metric("Total Rânduri Inițiale", len(df))
        delta_rows = len(df_filtered) - len(df)
        m2.metric("Total Rânduri Filtrate", len(df_filtered), delta=f"{delta_rows} rânduri eliminate" if delta_rows != 0 else "0")

        # 7. Afișare dataframe filtrat
        st.markdown('<div class="sub-header">Dataframe Filtrat</div>', unsafe_allow_html=True)
        st.dataframe(df_filtered, use_container_width=True)
        
        # Actualizăm 'df' curent în sesiune pentru celelalte pagini
        st.session_state['df'] = df_filtered

    elif uploaded_file is None:
        st.info("👈 Te rog să încarci un fișier CSV sau Excel pentru a începe.")

def page_cerinta_2():
    st.markdown('<h1 class="main-header">CERINȚA 2: Analiză Exploratorie</h1>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠ Mergi la CERINTA 1 și încarcă datele!")
        return

    df = st.session_state['df']

    # --- METRICI GLOBALE ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rânduri", len(df))
    m2.metric("Total Coloane", len(df.columns))
    
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    m3.metric("Memorie", f"{mem_mb:.2f} MB")
    
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    if total_cells > 0:
        pct_missing = (total_missing / total_cells) * 100
    else:
        pct_missing = 0
    m4.metric("Valori Lipsă", f"{pct_missing:.1f}%")

    st.markdown("---")

    # --- STRUCTURA PE TAB-URI ---
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Info", "Statistici", "Vizualizare"])

    with tab1:
        st.subheader("Primele Rânduri")
        n_rows = st.slider("Număr rânduri de afișat:", 5, len(df), 10)
        st.dataframe(df.head(n_rows), use_container_width=True)
        with st.expander("Ultimele Rânduri"):
            st.dataframe(df.tail(n_rows), use_container_width=True)

    with tab2:
        st.subheader("Informații Dataset")
        col_info1, col_info2 = st.columns([1.5, 1])
        
        with col_info1:
            st.write("**Tipuri de Date:**")
            info_df = pd.DataFrame({
                'Coloană': df.columns,
                'Tip': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col_info2:
            st.write("**Distribuția Tipurilor:**")
            type_counts = df.dtypes.astype(str).value_counts()
            fig_pie = px.pie(values=type_counts.values, names=type_counts.index, title="Tipuri de Date")
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.subheader("Statistici Descriptive")
        st.write("**Coloane Numerice:**")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("Nu există coloane numerice.")
        
        st.write("**Coloane Categorice:**")
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            cat_summary = pd.DataFrame({
                'Valori Unice': [df[col].nunique() for col in cat_cols],
                'Cel Mai Comun': [df[col].mode()[0] if not df[col].mode().empty else "N/A" for col in cat_cols],
                'Frecvență': [df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0 for col in cat_cols]
            }, index=cat_cols)
            cat_summary['Procent'] = (cat_summary['Frecvență'] / len(df) * 100).round(2).astype(str) + '%'
            st.dataframe(cat_summary, use_container_width=True)
        else:
            st.info("Nu există coloane categorice.")

    with tab4:
        st.subheader("Vizualizare Valori Lipsă")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            missing_df = pd.DataFrame({
                'Coloană': missing_data.index,
                'Număr Lipsă': missing_data.values,
                'Procent': (missing_data.values / len(df) * 100).round(4)
            })

            fig_missing = px.bar(missing_df, x='Coloană', y='Procent', 
                                 title="Procentul Valorilor Lipsă pe Coloană",
                                 text='Număr Lipsă', color_discrete_sequence=['#87CEEB'])
            fig_missing.update_traces(textposition='outside')
            st.plotly_chart(fig_missing, use_container_width=True)
            
            st.dataframe(missing_df, use_container_width=True)
            
            st.write("**Heatmap Valori Lipsă (primele 50 rânduri)**")
            st.write('<div style="text-align:center">Galben = Lipsă, Albastru = Prezent</div>', unsafe_allow_html=True)
            
            fig_heat, ax = plt.subplots(figsize=(12, 5))
            # Ajustare pentru tema dark/light la plot-ul matplotlib
            fig_heat.patch.set_alpha(0) # Transparent background
            ax.patch.set_alpha(0)
            
            # Culori: Albastru (prezent), Galben (lipsa)
            custom_colors = ['#000099', '#ffff00'] 
            sns.heatmap(df.head(50).isnull(), yticklabels=False, cbar=False, cmap=sns.color_palette(custom_colors), ax=ax)
            
            # Ajustare culoare text axe pentru vizibilitate
            ax.tick_params(colors='gray', which='both')
            
            st.pyplot(fig_heat)
        else:
            st.success("✅ Nu există valori lipsă în dataset!")

def page_cerinta_3():
    st.markdown('<h1 class="main-header">CERINȚA 3: Analiză Numerică</h1>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠ Mergi la CERINTA 1 și încarcă datele!")
        return

    df = st.session_state['df']
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("Lipsă coloane numerice în setul de date.")
        return

    st.markdown('<div class="sub-header">1. Selectare Variabilă</div>', unsafe_allow_html=True)
    col_sel = st.selectbox("Selectează coloana numerică pentru analiză:", numeric_cols)

    st.markdown('<div class="sub-header">2. Indicatori Statistici</div>', unsafe_allow_html=True)
    val_mean = df[col_sel].mean()
    val_median = df[col_sel].median()
    val_std = df[col_sel].std()
    c1, c2, c3 = st.columns(3)
    c1.metric("Media (Mean)", f"{val_mean:.2f}")
    c2.metric("Mediana (Median)", f"{val_median:.2f}")
    c3.metric("Deviația Standard (Std)", f"{val_std:.2f}")

    st.markdown('<div class="sub-header">3. Histogramă Interactivă</div>', unsafe_allow_html=True)
    bins = st.slider("Numărul de bins (intervale):", min_value=10, max_value=100, value=30)
    fig_hist = px.histogram(df, x=col_sel, nbins=bins, title=f"Histograma: {col_sel} ({bins} bins)", color_discrete_sequence=['#636EFA'], opacity=0.8)
    fig_hist.update_layout(bargap=0.1) 
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="sub-header">4. Box Plot</div>', unsafe_allow_html=True)
    fig_box = px.box(df, y=col_sel, title=f"Box Plot: {col_sel}", points="all", color_discrete_sequence=['#EF553B'])
    st.plotly_chart(fig_box, use_container_width=True)

def page_cerinta_4():
    st.markdown('<h1 class="main-header">CERINȚA 4: Analiză Categorică</h1>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠ Mergi la CERINTA 1 și încarcă datele!")
        return

    df = st.session_state['df']
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_cols:
        st.error("Nu există coloane categorice în acest set de date.")
        return

    st.markdown('<div class="sub-header">1. Selectare Variabilă Categorică</div>', unsafe_allow_html=True)
    col_sel = st.selectbox("Alege coloana categorică pentru analiză:", cat_cols)
    
    val_counts = df[col_sel].value_counts()
    val_percent = (df[col_sel].value_counts(normalize=True) * 100).round(2)
    freq_df = pd.DataFrame({'Frecvență Absolută': val_counts.values, 'Procent (%)': val_percent.values}, index=val_counts.index)
    freq_df.index.name = 'Categorie'
    freq_df.reset_index(inplace=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="sub-header">2. Grafic de Frecvențe (Count Plot)</div>', unsafe_allow_html=True)
        fig = px.bar(freq_df, x='Categorie', y='Frecvență Absolută', text='Frecvență Absolută', color='Frecvență Absolută', title=f"Distribuția pentru variabila: {col_sel}", color_continuous_scale='Viridis')
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="sub-header">3. Tabel de Frecvențe</div>', unsafe_allow_html=True)
        st.dataframe(freq_df, use_container_width=True)

def page_cerinta_5():
    st.markdown('<h1 class="main-header">CERINȚA 5: Corelații & Outlieri</h1>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠ Mergi la CERINTA 1 și încarcă datele!")
        return

    df = st.session_state['df']
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Nu sunt suficiente coloane numerice pentru analiza de corelație.")
        return

    st.markdown('<div class="sub-header">1. Matricea de Corelație (Heatmap)</div>', unsafe_allow_html=True)
    corr_matrix = df[numeric_cols].corr()
    fig_heat = px.imshow(corr_matrix, text_auto='.2f', aspect='auto', color_continuous_scale='RdBu_r', color_continuous_midpoint=0, title='Heatmap Corelație (Coeficient Pearson)')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="sub-header">2. Analiză Bivariată (Scatter Plot)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    x_val = c1.selectbox("Alege variabila X:", numeric_cols, index=0)
    y_val = c2.selectbox("Alege variabila Y:", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
    pearson = df[x_val].corr(df[y_val])
    c3.metric("Coeficient Pearson", f"{pearson:.4f}")
    fig_scatter = px.scatter(df, x=x_val, y=y_val, title=f"Relația dintre {x_val} și {y_val}", opacity=0.7)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="sub-header">3. Analiza Outlierilor (Metoda IQR)</div>', unsafe_allow_html=True)
    st.info("Metoda IQR (Interquartile Range) definește outlierii ca fiind valorile din afara intervalului [Q1 - 1.5*IQR, Q3 + 1.5*IQR].")
    
    outlier_stats = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        pct_outliers = (num_outliers / len(df)) * 100
        outlier_stats.append({"Coloană": col, "Nr. Outlieri": num_outliers, "Procent Outlieri (%)": round(pct_outliers, 2), "Limită Inferioară": round(lower_bound, 2), "Limită Superioară": round(upper_bound, 2)})
    
    outlier_df = pd.DataFrame(outlier_stats)
    st.dataframe(outlier_df, use_container_width=True)
    
    st.markdown('<div class="sub-header">4. Vizualizare Outlieri pe Grafic</div>', unsafe_allow_html=True)
    viz_col = st.selectbox("Selectează coloana pentru a vizualiza outlierii:", numeric_cols)
    fig_out = px.box(df, y=viz_col, points='all', title=f"Distribuția și Outlierii pentru: {viz_col}", color_discrete_sequence=['#FF4B4B'])
    st.plotly_chart(fig_out, use_container_width=True)

# --- 6. MAIN APP ---
if __name__ == "__main__":
    selected_page = sidebar_menu()
    
    if "Descriere Aplicație" in selected_page:
        page_home()
    elif "Cerinta 1" in selected_page:
        page_cerinta_1()
    elif "Cerinta 2" in selected_page:
        page_cerinta_2()
    elif "Cerinta 3" in selected_page:
        page_cerinta_3()
    elif "Cerinta 4" in selected_page:
        page_cerinta_4()
    elif "Cerinta 5" in selected_page:
        page_cerinta_5()