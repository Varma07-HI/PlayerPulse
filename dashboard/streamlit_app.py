import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'dashboard_data'

st.set_page_config(page_title='PlayerPulse', page_icon='💓', layout='wide')

# ---------------------------------------------------------------- styling
ACCENT = '#6C5CE7'
ACCENT_2 = '#00B894'
WARN = '#E17055'
COLORWAY = [ACCENT, ACCENT_2, WARN, '#0984E3', '#FDCB6E', '#E84393', '#636E72']

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
      h1 {{ font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0; }}
      .pp-sub {{ opacity: .65; margin: .15rem 0 1.2rem 0; font-size: 1.02rem; }}
      div[data-testid="stMetric"] {{
          background: rgba(127,127,127,.08);
          border: 1px solid rgba(127,127,127,.20);
          border-left: 5px solid {ACCENT};
          padding: 16px 20px; border-radius: 14px;
      }}
      div[data-testid="stMetricLabel"] p {{ font-size: .85rem; opacity: .75; text-transform: uppercase; letter-spacing: .04em; }}
      div[data-testid="stMetricValue"] {{ font-weight: 700; }}
      button[data-baseweb="tab"] {{ font-weight: 600; padding: 10px 18px; }}
      .pp-badge {{
          display: inline-block; padding: 4px 12px; border-radius: 999px; margin: 2px 6px 2px 0;
          background: rgba(108,92,231,.14); color: {ACCENT}; font-weight: 600; font-size: .85rem;
      }}
      .pp-sim {{
          background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.20);
          border-radius: 14px; padding: 6px 16px 14px 16px; margin-top: .4rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, height=420):
    fig.update_layout(
        template='plotly_white',
        colorway=COLORWAY,
        height=height,
        margin={'l': 10, 'r': 10, 't': 55, 'b': 10},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'size': 13},
        title_font={'size': 17},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1},
        hoverlabel={'font_size': 13},
    )
    fig.update_xaxes(gridcolor='rgba(127,127,127,.18)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(127,127,127,.18)', zeroline=False)
    return fig


def show_chart(fig):
    try:
        st.plotly_chart(fig, width='stretch')
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_df(df, **kwargs):
    try:
        st.dataframe(df, width='stretch', hide_index=True, **kwargs)
    except TypeError:  # older Streamlit
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    return {
        'scores': pd.read_parquet(D / 'scores.parquet'),
        'test': pd.read_parquet(D / 'test_predictions.parquet'),
        'ret': pd.read_parquet(D / 'retention.parquet'),
        'fair': pd.read_csv(D / 'fairness.csv'),
        'meta': json.loads((ROOT / 'models' / 'meta.json').read_text()),
        'drift': json.loads((D / 'drift_summary.json').read_text()),
    }


try:
    data = load()
except FileNotFoundError as e:
    st.error(f'Missing data file: `{e.filename}`. Run the pipeline to generate the dashboard data first.')
    st.stop()

scores, test, meta = data['scores'], data['test'], data['meta']

# ---------------------------------------------------------------- header
st.title('💓 PlayerPulse: churn early-warning')
st.markdown('<div class="pp-sub">Predict which players are about to leave, and decide who to reach out to.</div>',
            unsafe_allow_html=True)

tabs = st.tabs(['Overview', 'At-risk players', 'Retention', 'Model performance',
                'Monitoring', 'Fairness', 'Ask the data'])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric('Active players', f'{len(scores):,}')
    c2.metric('Predicted 30-day churn rate', f'{scores.churn_prob.mean():.1%}')
    c3.metric('Players at risk', f'{int(scores.at_risk.sum()):,}')
    st.markdown(
        f"<span class='pp-badge'>Model: {meta['chosen_model']}</span>"
        f"<span class='pp-badge'>Outreach threshold: {meta['threshold']:.2f}</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- at-risk
with tabs[1]:
    st.write('Highest-risk players. (+) means the feature pushes risk up, (-) down.')
    top = scores.sort_values('churn_prob', ascending=False).head(200)
    cols = ['player_id', 'churn_prob', 'reasons', 'matches_30d', 'days_since_last']
    show_df(
        top[cols],
        column_config={
            'player_id': st.column_config.TextColumn('Player'),
            'churn_prob': st.column_config.ProgressColumn(
                'Churn probability', min_value=0.0, max_value=1.0, format='%.2f'),
            'reasons': st.column_config.TextColumn('Top reasons', width='large'),
            'matches_30d': st.column_config.NumberColumn('Matches (30d)', format='%d'),
            'days_since_last': st.column_config.NumberColumn('Days since last match', format='%d'),
        },
    )

# ---------------------------------------------------------------- retention
with tabs[2]:
    ret = data['ret'].copy()
    ret['cohort'] = ret['cohort'].astype(str)
    fig = px.line(ret, x='month_idx', y='retention', color='cohort',
                  labels={'month_idx': 'Months since first match',
                          'retention': 'Share still playing', 'cohort': 'Cohort'})
    fig.update_traces(line={'width': 2.5})
    fig.update_yaxes(tickformat='.0%')
    show_chart(style(fig, 480))

# ---------------------------------------------------------------- model performance
with tabs[3]:
    left, right = st.columns(2)

    prec, rec, _ = precision_recall_curve(test.churn, test.p)
    pr = px.area(x=rec, y=prec, title='Precision-recall curve',
                 labels={'x': 'Recall', 'y': 'Precision'})
    pr.update_traces(line={'color': ACCENT, 'width': 2.5}, fillcolor='rgba(108,92,231,.15)')
    pr.update_xaxes(range=[0, 1])
    pr.update_yaxes(range=[0, 1.02])
    with left:
        show_chart(style(pr))

    frac, mean_p = calibration_curve(test.churn, test.p, n_bins=10)
    cal = px.line(x=mean_p, y=frac, markers=True, title='Calibration',
                  labels={'x': 'Predicted probability', 'y': 'Actual churn rate'})
    cal.update_traces(line={'color': ACCENT_2, 'width': 2.5}, marker={'size': 8})
    cal.add_shape(type='line', x0=0, y0=0, x1=1, y1=1,
                  line={'dash': 'dash', 'color': 'rgba(127,127,127,.8)'})
    cal.update_xaxes(range=[0, 1])
    cal.update_yaxes(range=[0, 1])
    with right:
        show_chart(style(cal))

    st.subheader('Intervention simulator')
    st.markdown('<div class="pp-sim">', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    default_thr = min(max(float(meta['threshold']), 0.05), 0.95)
    thr = a.slider('Threshold', 0.05, 0.95, default_thr, 0.01)
    cost = b.number_input('Cost per outreach', value=1.0, min_value=0.0)
    value = c.number_input('Value of a retained player', value=10.0, min_value=0.0)
    save = d.slider('Outreach save rate', 0.0, 1.0, 0.3, 0.05)

    flagged = test[test.p >= thr]
    caught = int(flagged.churn.sum())
    outreach_cost = len(flagged) * cost
    retained_value = caught * save * value
    net = retained_value - outreach_cost

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Players reached', f'{len(flagged):,}')
    k2.metric('Churners caught', f'{caught:,}')
    k3.metric('Outreach cost', f'{outreach_cost:,.0f}')
    k4.metric('Retained value', f'{retained_value:,.0f}')
    k5.metric('Net', f'{net:,.0f}', delta=f'{net:,.0f}', delta_color='normal')
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- monitoring
with tabs[4]:
    dr = data['drift']
    m1, m2, m3 = st.columns(3)
    m1.metric('Dataset drift', 'YES' if dr['dataset_drift'] else 'no')
    m2.metric('Drifted features', dr['n_drifted'])
    m3.metric('Max null rate', f"{dr['max_null_rate']:.1%}")

    test2 = test.assign(date=pd.to_datetime(test.cutoff, unit='s').dt.date.astype(str))
    test2 = test2.sort_values('date')
    box = px.box(test2, x='date', y='p', title='Prediction distribution over time',
                 labels={'date': 'Date', 'p': 'Predicted probability'})
    box.update_traces(marker_color=ACCENT, line={'width': 1.5})
    show_chart(style(box, 440))

    report = D / 'drift_report.html'
    if report.exists():
        with st.expander('Full drift report', expanded=True):
            components.html(report.read_text(encoding='utf-8'), height=700, scrolling=True)

# ---------------------------------------------------------------- fairness
with tabs[5]:
    fair = data['fair']
    show_df(fair)
    bar = px.bar(fair, x='group', y='false_negative_rate', facet_col='slice', color='group',
                 title='Missed churners (false negative rate) by group',
                 labels={'group': 'Group', 'false_negative_rate': 'False negative rate'})
    bar.update_xaxes(matches=None, showticklabels=True)
    bar.update_yaxes(tickformat='.0%')
    bar.update_layout(showlegend=False)
    bar.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
    show_chart(style(bar, 460))

# ---------------------------------------------------------------- ask the data
with tabs[6]:
    st.info('Coming in v2.1: ask questions in plain English, answered by a SQL agent.')