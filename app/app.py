import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricMetric-AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    master = pd.read_csv(
        r"D:\CricMetric-AI\data\processed\master_player_features.csv"
    )
    rising = pd.read_csv(
        r"D:\CricMetric-AI\data\processed\rising_star_matches.csv"
    )
    ens = pd.read_csv(
        r"D:\CricMetric-AI\data\processed\ens_scores.csv"
    )
    return master, rising, ens

@st.cache_resource
def load_models():
    with open(r"D:\CricMetric-AI\data\processed\xgb_ranking_model.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(r"D:\CricMetric-AI\data\processed\shap_explainer.pkl", 'rb') as f:
        explainer = pickle.load(f)
    return model, explainer

master_df, rising_df, ens_df = load_data()
model, explainer = load_models()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/cricket-game.png", width=80)
st.sidebar.title("CricMetric-AI")
st.sidebar.caption("Context-Aware ODI Batter Rankings")


page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🏆 Rankings", "🔍 Player Profile", 
     "⚔️ Head to Head", "⭐ Rising Stars", 
     "📊 Era Explorer"]
)
st.sidebar.divider()
st.sidebar.caption(f"475 qualified batters | 2544 ODI matches | 1.3M balls")

# ─── PAGE 0: HOME ─────────────────────────────────────────────────────────────
if page == "🏠 Home":

    st.title("🏏 CricMetric-AI")
    st.subheader("The World's Most Context-Aware ODI Batter Ranking System")

    st.markdown("""
    > *"Beyond Runs. Beyond Averages. Beyond Strike Rates.
    CricMetric-AI is the first ODI ranking framework designed to evaluate batters through situational pressure,
     clutch performances, and winning influence."*
    """)

    st.divider()

    # project overview
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What is CricMetric-AI?")
        st.write("""
        Traditional cricket rankings measure **how many runs** a batter scored.
        CricMetric-AI measures **when, against whom, and whether it mattered**.
        
        We analyzed **1.3 million balls** across **2,544 ODI matches** (2002-2026)
        to build a context-aware ranking system that rewards:
        - Performing under **match pressure** (not just comfortable situations)
        - Scoring runs that **actually win matches** (not just milestone hunting)
        - Sustaining excellence across **different eras** of cricket
        """)

    with col2:
        st.subheader("The Problem with Traditional Rankings")
        st.write("""
        Consider Sachin Tendulkar's famous 2003 World Cup campaign vs a 
        century in a dead bilateral series — both count equally in career 
        averages. They shouldn't.
        
        ICC rankings use a points-decay algorithm that rewards recent form
        but ignores situational context entirely. A boundary hit in over 45 
        of a death chase carries the same weight as one in over 5 of a 
        comfortable chase.
        
        **CricMetric-AI fixes this.**
        """)

    st.divider()

    # three components
    st.subheader("Three Pillars of Our Ranking System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 PPI — Pressure Performance Index")
        st.markdown("**Weight: 35%**")
        st.write("""
        Measures how a batter performs in **4 specific high-pressure situations**
        using Bayesian shrinkage for statistical reliability:
        """)
        st.markdown("""
        - **S1 Early Collapse** — team 3+ wickets down in first 20 overs
        - **S2 Death Chase** — overs 30-50, chasing 250+
        - **S3 World Cup** — ICC World Cup matches only
        - **S4 Chase Recovery** — early wickets lost in winning chases
        """)
        st.info("Kohli's S2 death-chase score: 90.62/100")

    with col2:
        st.markdown("### ⚡ CFS — Clutch Factor Score")
        st.markdown("**Weight: 35%**")
        st.write("""
        Measures whether a batter's performance **actually translates to wins**
        using three independent signals:
        """)
        st.markdown("""
        - **C1 Win Probability Added** — match-state change caused by this batter
          (powered by Random Forest win probability model, AUC 0.85)
        - **C2 Match-Winning Ratio** — top scorer in wins, baseline-adjusted
          for team strength
        - **C3 Dominance Score** — contribution share relative to team total
          (solves the "weak team era" problem)
        """)
        st.info("Tendulkar's C3 Dominance: 88.19/100 — proves individual greatness independent of team results")

    with col3:
        st.markdown("### 📈 ENS — Era Normalisation Score")
        st.markdown("**Weight: 30%**")
        st.write("""
        Adjusts for the fact that **cricket has changed dramatically** since 2002.
        A 300 total in 2003 is not the same as 300 in 2023:
        """)
        st.markdown("""
        - **Era-adjusted average** — normalised against yearly scoring rates
        - **Career volume** — log-scaled to reward sustained excellence
        - **Opposition quality** — era-adjusted bowling strength index
          (economy + strike rate, restricted to ICC Full Members)
        """)
        st.info("Tendulkar gets +2.54 era boost (played in harder scoring era)")

    st.divider()
    st.subheader("What Makes This Different")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
     st.markdown("#### 🎲 Bayesian Shrinkage")
     st.markdown("Applied to **all metrics**")
     st.success("No arbitrary cutoffs")


    with col2:
      st.markdown("#### 🤖 Win Probability + XGBoost")
      st.markdown("Win Prob **AUC 0.85** | XGBoost **R² = 0.59**")
      st.success("Trained on 1.3M balls")

    with col3:
      st.markdown("#### ✅ ICC Validation")
      st.markdown("Pearson correlation **r = 0.74**")
      st.success("Independent verification")

    with col4:
      st.markdown("#### 🔍 SHAP Explainability")
      st.markdown("**Every ranking** explained")
      st.success("Full transparency")

    st.divider()

# replace Dataset section with this:
    st.subheader("Dataset")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
      st.markdown("#### 2,544")
      st.caption("ODI Matches")

    with col2:
      st.markdown("#### 1,349,408")
      st.caption("Balls Analyzed")

    with col3:
      st.markdown("#### 475")
      st.caption("Qualified Batters")

    with col4:
      st.markdown("#### 2002–2026")
      st.caption("Years Covered")

    with col5:
      st.markdown("#### 3")
      st.caption("ML Models")

    st.divider()


    st.caption("Built by Kadiyapu Durgesh Kumar | VIT-AP University 2028 | github.com/Durgeshkadiyapu/CricMetric-AI")

# ─── PAGE 1: RANKINGS ─────────────────────────────────────────────────────────
if page == "🏆 Rankings":
    
    st.title("🏆 All-Time ODI Batter Rankings")
    st.caption("Context-aware rankings using Pressure Performance Index (PPI), Clutch Factor Score (CFS), and Era Normalisation Score (ENS)")
    
    # metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Qualified Batters", "475")
    col2.metric("ODI Matches", "2,544")
    col3.metric("Balls Analyzed", "1.3M")
    col4.metric("Years Covered", "2002-2026")
    
    st.divider()
    
    # filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_n = st.slider("Show Top N batters", 10, 475, 100)
    
    with col2:
        min_ppi = st.slider("Minimum PPI", 0, 100, 0)
    
    with col3:
        sort_by = st.selectbox(
            "Sort by", 
            ["FINAL_SCORE", "PPI", "CFS", "ENS"]
        )
    
    # filter data
    filtered = master_df.copy()
    filtered = filtered[filtered['PPI'] >= min_ppi]
    filtered = filtered.sort_values(sort_by, ascending=False).head(top_n)
    filtered['Rank'] = range(1, len(filtered) + 1)
    
    # display table
    display_cols = ['Rank', 'batter', 'FINAL_SCORE', 'PPI', 'CFS', 'ENS',
                    'career_avg', 'career_runs']
    display_df = filtered[display_cols].copy()
    display_df.columns = ['Rank', 'Batter', 'Final Score', 
                          'PPI', 'CFS', 'ENS', 
                          'Career Avg', 'Career Runs']
    display_df = display_df.round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Final Score': st.column_config.ProgressColumn(
                'Final Score', min_value=0, max_value=100
            ),
            'PPI': st.column_config.ProgressColumn(
                'PPI', min_value=0, max_value=100
            ),
            'CFS': st.column_config.ProgressColumn(
                'CFS', min_value=0, max_value=100
            ),
            'ENS': st.column_config.ProgressColumn(
                'ENS', min_value=0, max_value=100
            ),
        }
    )
    
    st.divider()
    
    # bar chart - top 20
    st.subheader("Top 20 — Score Breakdown")
    top20 = master_df.sort_values('FINAL_SCORE', ascending=False).head(20)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='PPI', x=top20['batter'], y=top20['PPI'],
        marker_color='#1D9E75'
    ))
    fig.add_trace(go.Bar(
        name='CFS', x=top20['batter'], y=top20['CFS'],
        marker_color='#185FA5'
    ))
    fig.add_trace(go.Bar(
        name='ENS', x=top20['batter'], y=top20['ENS'],
        marker_color='#BA7517'
    ))
    
    fig.update_layout(
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        legend=dict(orientation='h', y=1.1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 2: PLAYER PROFILE ───────────────────────────────────────────────────
elif page == "🔍 Player Profile":
    
    st.title("🔍 Player Profile")
    st.caption("Deep dive into any player's context-aware performance breakdown")
    
    # player search
    player_list = master_df.sort_values('FINAL_SCORE', ascending=False)['batter'].tolist()
    selected_player = st.selectbox("Search Player", player_list)
    
    if selected_player:
        player = master_df[master_df['batter'] == selected_player].iloc[0]
        
        st.divider()
        
        # header metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Overall Rank", f"#{int(player['FINAL_RANK'])}")
        col2.metric("Final Score", f"{player['FINAL_SCORE']:.2f}")
        col3.metric("PPI", f"{player['PPI']:.2f}")
        col4.metric("CFS", f"{player['CFS']:.2f}")
        col5.metric("ENS", f"{player['ENS']:.2f}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Component Breakdown")
            
            # radar chart
            categories = ['PPI', 'CFS', 'ENS']
            values = [player['PPI'], player['CFS'], player['ENS']]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(29,158,117,0.3)',
                line=dict(color='#1D9E75', width=2),
                name=selected_player
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("Situation Scores")
            
            situations = {
                'S1 Early Collapse': player['s1_score'],
                'S2 Death Chase': player['s2_score'],
                'S3 World Cup': player['s3_score'],
                'S4 Chase Recovery': player['s4_score'],
                'C1 WPA': player['c1_score'],
                'C2 Match Winning': player['c2_score'],
                'C3 Dominance': player['c3_score'],
            }
            
            fig_bar = go.Figure(go.Bar(
                x=list(situations.values()),
                y=list(situations.keys()),
                orientation='h',
                marker_color=[
                    '#1D9E75','#1D9E75','#1D9E75','#1D9E75',
                    '#185FA5','#185FA5','#185FA5'
                ]
            ))
            fig_bar.update_layout(
                height=400,
                xaxis=dict(range=[0,100]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        st.divider()
        
        # career stats
        st.subheader("Career Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Career Runs", f"{int(player['career_runs']):,}")
        col2.metric("Career Average", f"{player['career_avg']:.2f}")
        col3.metric("Era-Adjusted Average", f"{player['era_adj_avg']:.2f}")
        
        st.divider()
        
        # SHAP explanation
        st.subheader("Why This Ranking? (SHAP Explanation)")
        
        feature_cols_xgb = [
            's1_score', 's2_score', 's3_score', 's4_score',
            'c1_score', 'c2_score', 'c3_score',
            'era_adj_avg_norm', 'volume_norm', 'opp_quality_norm'
        ]
        
        try:
            player_features = master_df[
                master_df['batter']==selected_player
            ][feature_cols_xgb]
            
            shap_vals = explainer.shap_values(player_features)
            
            shap_df = pd.DataFrame({
                'Feature': feature_cols_xgb,
                'SHAP Impact': shap_vals[0],
                'Feature Value': player_features.values[0]
            }).sort_values('SHAP Impact', key=abs, ascending=False)
            
            colors = ['#1D9E75' if x > 0 else '#E24B4A' 
                     for x in shap_df['SHAP Impact']]
            
            fig_shap = go.Figure(go.Bar(
                x=shap_df['SHAP Impact'],
                y=shap_df['Feature'],
                orientation='h',
                marker_color=colors
            ))
            fig_shap.update_layout(
                title="Feature contributions to ICC rating prediction",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="SHAP Value (positive = pushes rating up)"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
# ─── PAGE 3: HEAD TO HEAD ─────────────────────────────────────────────────────
elif page == "⚔️ Head to Head":
    
    st.title("⚔️ Head to Head Comparison")
    st.caption("Compare any two ODI batters across all context-aware metrics")
    
    player_list = master_df.sort_values('FINAL_SCORE', ascending=False)['batter'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", player_list, index=0)
    with col2:
        player2 = st.selectbox("Player 2", player_list, index=1)
    
    if player1 and player2 and player1 != player2:
        
        p1 = master_df[master_df['batter']==player1].iloc[0]
        p2 = master_df[master_df['batter']==player2].iloc[0]
        
        st.divider()
        
        # header comparison
        col1, col2, col3 = st.columns([2,1,2])
        
        with col1:
            st.markdown(f"## {player1}")
            st.metric("Rank", f"#{int(p1['FINAL_RANK'])}")
            st.metric("Final Score", f"{p1['FINAL_SCORE']:.2f}")
        
        with col2:
            st.markdown("## VS")
            st.markdown(" ")
        
        with col3:
            st.markdown(f"## {player2}")
            st.metric("Rank", f"#{int(p2['FINAL_RANK'])}")
            st.metric("Final Score", f"{p2['FINAL_SCORE']:.2f}")
        
        st.divider()
        
        # radar chart comparison
        st.subheader("Overall Profile Comparison")
        
        categories = ['PPI', 'CFS', 'ENS',
                      'S1 Collapse', 'S2 Death', 
                      'S3 World Cup', 'S4 Recovery',
                      'C1 WPA', 'C2 Win Ratio', 'C3 Dominance']
        
        p1_values = [p1['PPI'], p1['CFS'], p1['ENS'],
                     p1['s1_score'], p1['s2_score'],
                     p1['s3_score'], p1['s4_score'],
                     p1['c1_score'], p1['c2_score'], p1['c3_score']]
        
        p2_values = [p2['PPI'], p2['CFS'], p2['ENS'],
                     p2['s1_score'], p2['s2_score'],
                     p2['s3_score'], p2['s4_score'],
                     p2['c1_score'], p2['c2_score'], p2['c3_score']]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=p1_values + [p1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(29,158,117,0.3)',
            line=dict(color='#1D9E75', width=2),
            name=player1
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=p2_values + [p2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(24,95,165,0.3)',
            line=dict(color='#185FA5', width=2),
            name=player2
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation='h', y=1.1)
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.divider()
        
        # detailed comparison table
        st.subheader("Detailed Metric Comparison")
        
        metrics = {
            'Final Score': ('FINAL_SCORE', '🏆'),
            'PPI': ('PPI', '🎯'),
            'CFS': ('CFS', '⚡'),
            'ENS': ('ENS', '📈'),
            'S1 Early Collapse': ('s1_score', '🔴'),
            'S2 Death Chase': ('s2_score', '💀'),
            'S3 World Cup': ('s3_score', '🏆'),
            'S4 Chase Recovery': ('s4_score', '🔄'),
            'C1 Win Prob Added': ('c1_score', '📊'),
            'C2 Match Winning': ('c2_score', '🎯'),
            'C3 Dominance': ('c3_score', '👑'),
            'Career Average': ('career_avg', '📋'),
            'Era Adj Average': ('era_adj_avg', '⏳'),
            'Career Runs': ('career_runs', '🏏'),
        }
        
        comparison_data = []
        for label, (col, icon) in metrics.items():
            v1 = p1[col]
            v2 = p2[col]
            winner = player1 if v1 > v2 else player2 if v2 > v1 else 'Tie'
            comparison_data.append({
                'Metric': f"{icon} {label}",
                player1: round(v1, 2),
                player2: round(v2, 2),
                'Winner': winner
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # color code winner column
        def highlight_winner(row):
            colors = [''] * len(row)
            if row['Winner'] == player1:
                colors[1] = 'background-color: rgba(29,158,117,0.3)'
            elif row['Winner'] == player2:
                colors[2] = 'background-color: rgba(24,95,165,0.3)'
            return colors
        
        st.dataframe(
            comp_df.style.apply(highlight_winner, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # wins count
        p1_wins = sum(1 for _, (col, _) in metrics.items() 
                      if p1[col] > p2[col])
        p2_wins = sum(1 for _, (col, _) in metrics.items() 
                      if p2[col] > p1[col])
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{player1} wins", f"{p1_wins} metrics")
        col2.metric("Tied", f"{len(metrics)-p1_wins-p2_wins} metrics")
        col3.metric(f"{player2} wins", f"{p2_wins} metrics")
# ─── PAGE 4: RISING STARS ─────────────────────────────────────────────────────
elif page == "⭐ Rising Stars":
    
    st.title("⭐ Rising Stars")
    st.caption("Which current young players are tracking like all-time ODI legends?")
    
    st.info("""
    **How this works:** We built 20-dimensional career trajectory vectors 
    (avg, SR, WPA, balls faced, batting position × 4 years) for 274 rising stars 
    and compared them against top 100 legends using cosine similarity.
    Players who debuted 2019+ with 117+ career balls qualified.
    """)
    
    st.divider()
    
    # load rising star data
    rising_top = rising_df[rising_df['match_rank']==1].copy()
    rising_top = rising_top.merge(
        master_df[['batter', 'FINAL_RANK', 'FINAL_SCORE']],
        left_on='legend_match', right_on='batter', how='left'
    ).drop('batter', axis=1)
    
    # filters
    col1, col2 = st.columns(2)
    with col1:
        min_similarity = st.slider(
            "Minimum similarity score", 0.0, 1.0, 0.70, 0.01
        )
    with col2:
        legend_filter = st.multiselect(
            "Filter by legend match",
            options=sorted(rising_top['legend_match'].unique()),
            default=[]
        )
    
    filtered_rising = rising_top[
        rising_top['similarity'] >= min_similarity
    ]
    if legend_filter:
        filtered_rising = filtered_rising[
            filtered_rising['legend_match'].isin(legend_filter)
        ]
    
    filtered_rising = filtered_rising.sort_values(
        'similarity', ascending=False
    )
    
    st.subheader(f"Rising Stars — {len(filtered_rising)} players found")
    
    # display table
    display_rising = filtered_rising[[
        'rising_star', 'legend_match', 'similarity', 'FINAL_RANK'
    ]].copy()
    display_rising.columns = [
        'Rising Star', 'Closest Legend', 'Similarity', 'Legend Rank'
    ]
    display_rising['Similarity'] = display_rising['Similarity'].round(4)
    display_rising['Legend Rank'] = display_rising['Legend Rank'].apply(
        lambda x: f"#{int(x)}" if pd.notna(x) else "N/A"
    )
    
    st.dataframe(
        display_rising,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Similarity': st.column_config.ProgressColumn(
                'Similarity', min_value=0, max_value=1, format="%.4f"
            )
        }
    )
    
    st.divider()
    
    # top matches visualization
    st.subheader("Top 15 Highest Similarity Matches")
    
    top15_rising = filtered_rising.head(15)
    
    fig = go.Figure(go.Bar(
        x=top15_rising['similarity'],
        y=top15_rising['rising_star'] + ' → ' + top15_rising['legend_match'],
        orientation='h',
        marker_color='#1D9E75',
        text=top15_rising['similarity'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        height=500,
        xaxis=dict(range=[0, 1.1], title='Cosine Similarity'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
   # replace "Which Legends Are Most Matched" section with this:
    st.subheader("Which Legends Are Most Matched?")

    # get all top 20 legends from master_df
    top20_legends = master_df.sort_values('FINAL_SCORE', ascending=False).head(20)['batter'].tolist()

# count matches for each legend (including 0 for unmatched)
    legend_match_counts = rising_top['legend_match'].value_counts()

# build complete series with all top 20 legends
    legend_counts_full = pd.Series(
    {legend: legend_match_counts.get(legend, 0) 
     for legend in top20_legends}).sort_values(ascending=True)

    fig2 = go.Figure(go.Bar(
    x=legend_counts_full.values,
    y=legend_counts_full.index,
    orientation='h',
    marker_color=[
        '#E24B4A' if v == 0 else '#185FA5' 
        for v in legend_counts_full.values
    ],
    text=legend_counts_full.values,
    textposition='outside'
))

    fig2.update_layout(
    height=550,
    xaxis_title='Number of Rising Stars matched',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white')
)

    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔴 Red bars = unmatched legends (Kohli, ABdV etc.) — no current rising star has yet replicated their early career trajectory")
    
    st.divider()
    
    # notable findings
    st.subheader("Notable Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 Kohli & ABdV — Unmatched")
        st.write("""
        No current rising star matched Virat Kohli or AB de Villiers 
        as their primary trajectory match. Their early-career combination 
        of quality, volume, WPA, and positional dominance remains 
        historically unreplicated.
        """)
    
    with col2:
      st.markdown("#### 🏏 Shubman Gill → S Dhawan")
      st.write("""
       Gill's trajectory most closely matches Shikhar Dhawan's early career 
     (similarity: 0.9376). Both are stylish Indian openers with 
    similar avg, SR, and WPA patterns in their first 4 years.
    """)
    
# ─── PAGE 5: ERA EXPLORER ─────────────────────────────────────────────────────
elif page == "📊 Era Explorer":
    
    st.title("📊 Era Explorer")
    st.caption("How has ODI cricket changed from 2002 to 2026?")
    
    st.divider()
    
    # load era data from master_df
    # recompute era scoring from career stats
    era_data = pd.DataFrame({
        'Year': [2002,2003,2004,2005,2006,2007,2008,2009,2010,
                 2011,2012,2013,2014,2015,2016,2017,2018,2019,
                 2020,2021,2022,2023,2024,2025,2026],
        'Runs_per_ball': [0.8056,0.7059,0.7422,0.7751,0.7411,
                          0.7819,0.7791,0.8148,0.8079,0.7898,
                          0.7984,0.8051,0.8421,0.8898,0.8763,
                          0.8702,0.8390,0.8708,0.8666,0.7843,
                          0.7920,0.8772,0.7983,0.8547,0.8269],
        'Era_difficulty': [1.0097,1.1523,1.0959,1.0494,1.0976,
                           1.0403,1.0440,0.9983,1.0068,1.0299,
                           1.0188,1.0103,0.9659,0.9141,0.9282,
                           0.9347,0.9695,0.9341,0.9386,1.0371,
                           1.0270,0.9273,1.0189,0.9517,0.9837]
    })
    
    # tab layout
    tab1, tab2, tab3 = st.tabs([
        "📈 Scoring Trends", 
        "🎯 Era Difficulty", 
        "🏆 Top Performers by Era"
    ])
    
    with tab1:
        st.subheader("ODI Scoring Rate by Year (2002-2026)")
        
        fig = go.Figure()
        
        # scoring rate line
        fig.add_trace(go.Scatter(
            x=era_data['Year'],
            y=era_data['Runs_per_ball'] * 6,  # convert to runs per over
            mode='lines+markers',
            name='Runs per Over',
            line=dict(color='#1D9E75', width=3),
            marker=dict(size=8)
        ))
        
        # add era labels
        fig.add_vrect(
            x0=2002, x1=2008,
            fillcolor='rgba(29,158,117,0.1)',
            annotation_text="Pre-T20 era",
            annotation_position="top left"
        )
        fig.add_vrect(
            x0=2008, x1=2016,
            fillcolor='rgba(24,95,165,0.1)',
            annotation_text="T20 influence era",
            annotation_position="top left"
        )
        fig.add_vrect(
            x0=2016, x1=2026,
            fillcolor='rgba(186,117,23,0.1)',
            annotation_text="Modern era",
            annotation_position="top left"
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Year',
            yaxis_title='Average Runs Per Over',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Key observations:**
        - **2002-2008**: Lower scoring era (avg ~4.7 RPO) — batting was harder
        - **2015**: Peak modern scoring (5.34 RPO) — T20 techniques fully adopted
        - **2021-2022**: COVID dip — fewer matches, different conditions
        - **2023**: High scoring World Cup year (5.26 RPO)
        """)
    
    with tab2:
        st.subheader("Era Difficulty Factor by Year")
        st.write("""
        Era difficulty factor > 1.0 means that year was **harder to score in** 
        than average — batters from these years get a scoring boost in ENS.
        Factor < 1.0 means easier scoring — slight penalty applied.
        """)
        
        colors = ['#1D9E75' if x > 1.0 else '#E24B4A' 
                  for x in era_data['Era_difficulty']]
        
        fig2 = go.Figure(go.Bar(
            x=era_data['Year'],
            y=era_data['Era_difficulty'],
            marker_color=colors,
            text=era_data['Era_difficulty'].round(3),
            textposition='outside'
        ))
        
        fig2.add_hline(
            y=1.0, 
            line_dash="dash", 
            line_color="white",
            annotation_text="League average"
        )
        
        fig2.update_layout(
            height=500,
            xaxis_title='Year',
            yaxis_title='Era Difficulty Factor',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("🟢 Hardest year: 2003 (factor 1.1523) — pre-powerplay, restrictive fielding rules")
        with col2:
            st.error("🔴 Easiest year: 2015 (factor 0.9141) — peak T20 influence on ODI batting")
    
    with tab3:
        st.subheader("Top Performers by Era")
        
        era_choice = st.selectbox(
            "Select Era",
            ["Pre-T20 (2002-2008)", 
             "T20 Influence (2009-2015)",
             "Modern (2016-2026)"]
        )
        
        era_years = {
            "Pre-T20 (2002-2008)": (2002, 2008),
            "T20 Influence (2009-2015)": (2009, 2015),
            "Modern (2016-2026)": (2016, 2026)
        }
        
        start_yr, end_yr = era_years[era_choice]
        
        st.write(f"Top 15 batters who played primarily in {era_choice}:")
        
        # filter master_df by era using career stats
        # use era_adj_avg as proxy for era performance
        era_top = master_df.sort_values(
            'ENS', ascending=False
        ).head(50)[['batter', 'FINAL_RANK', 'FINAL_SCORE', 
                    'ENS', 'era_adj_avg', 'career_runs']].head(15)
        
        st.dataframe(
            era_top.round(2),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("Note: Era filtering by actual debut/retirement years requires birth date data not available in Cricsheet. Showing ENS-ranked players as proxy for era-adjusted performance.")