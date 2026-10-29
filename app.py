import streamlit as st
import pandas as pd
import plotly.express as px
from ortools.linear_solver import pywraplp

st.set_page_config(page_title='PEATRA | Restoration Optimizer', page_icon='🌿', layout='wide')
COST='Cost (Rp million)'; SITE='Site'; ACTION='Action'; FEAS='Technically feasible'
INDICATORS=['Fire-risk reduction','Hydrological benefit','Ecological benefit']
DEFAULT=[
('A - Sungai Utama','Repair existing block',20,35,45,30,True),('A - Sungai Utama','Build new block',35,75,78,70,True),
('B - Kanal Barat','Repair existing block',15,20,25,20,True),('B - Kanal Barat','Build new block',30,65,70,60,True),
('C - Kanal Selatan','Repair existing block',25,40,45,35,True),('C - Kanal Selatan','Build new block',40,90,88,92,True),
('D - Area Timur','Repair existing block',18,30,38,32,True),('D - Area Timur','Build new block',32,70,73,75,True),
('E - Area Gambut Dalam','Repair existing block',22,42,50,48,False),('E - Area Gambut Dalam','Build new block',38,82,85,88,True),
('F - Kanal Utara','Repair existing block',16,25,30,28,True),('F - Kanal Utara','Build new block',29,62,66,64,True)]

def default_df():
    return pd.DataFrame(DEFAULT, columns=[SITE,ACTION,COST,*INDICATORS,FEAS])

def score_data(df, weights):
    out=df.copy()
    out['Benefit score']=sum(pd.to_numeric(out[k])*weights[k] for k in INDICATORS)
    return out

def validate(df):
    required=[SITE,ACTION,COST,*INDICATORS,FEAS]
    if any(c not in df.columns for c in required): return 'A required column is missing.'
    if df[required].isna().any().any(): return 'Complete all fields; blank values are not allowed.'
    if (pd.to_numeric(df[COST],errors='coerce')<0).any(): return 'Costs cannot be negative.'
    for col in INDICATORS:
        vals=pd.to_numeric(df[col],errors='coerce')
        if vals.isna().any() or ((vals<0)|(vals>100)).any(): return f'{col} must be numeric and between 0 and 100.'
    if pd.to_numeric(df[COST],errors='coerce').isna().any(): return 'Costs must be numeric.'
    if df.duplicated([SITE,ACTION]).any(): return 'Each site-action combination must be unique.'
    return None

def solve_milp(df,budget):
    solver=pywraplp.Solver.CreateSolver('CBC')
    if solver is None: raise RuntimeError('OR-Tools CBC solver could not be initialized.')
    x={i:solver.BoolVar(f'x_{i}') for i in df.index}
    for i,row in df.iterrows():
        if not bool(row[FEAS]): solver.Add(x[i]==0)
    solver.Add(solver.Sum(float(df.loc[i,COST])*x[i] for i in df.index)<=float(budget))
    for site in df[SITE].unique():
        ids=[i for i in df.index if df.loc[i,SITE]==site]
        solver.Add(solver.Sum(x[i] for i in ids)<=1)
    solver.Maximize(solver.Sum(float(df.loc[i,'Benefit score'])*x[i] for i in df.index))
    status=solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL,pywraplp.Solver.FEASIBLE): return None,'No feasible solution'
    chosen=[i for i in df.index if x[i].solution_value()>.5]
    return df.loc[chosen].copy(), ('OPTIMAL' if status==pywraplp.Solver.OPTIMAL else 'FEASIBLE')

def greedy_benchmark(df,budget):
    cand=df[df[FEAS] & (df[COST]>0)].copy()
    cand['Score/cost']=cand['Benefit score']/cand[COST]
    cand=cand.sort_values(['Score/cost','Benefit score'],ascending=False)
    selected=[]; sites=set(); spend=0
    for i,r in cand.iterrows():
        if r[SITE] in sites: continue
        if spend+float(r[COST])<=budget+1e-8:
            selected.append(i); sites.add(r[SITE]); spend+=float(r[COST])
    return df.loc[selected].copy()

def total(res):
    return (float(res[COST].sum()),float(res['Benefit score'].sum())) if len(res) else (0.,0.)

st.title('🌿 PEATRA')
st.subheader('Peatland Restoration Adaptive Optimization')
st.caption('Interactive decision-support prototype · Streamlit + Python + OR-Tools MILP')
st.warning('ILLUSTRATIVE DATA ONLY. Site names, costs, and indicator scores are synthetic examples, not verified field data from Pelalawan. Do not use this demo for operational decisions.')

with st.sidebar:
    st.header('Optimization settings')
    budget=st.slider('Available budget (Rp million)',20,200,100,5)
    st.divider(); st.subheader('Benefit weights (%)')
    wf=st.slider('Fire-risk reduction',0,100,40,5)
    wh=st.slider('Hydrological benefit',0,100,35,5)
    we=st.slider('Ecological benefit',0,100,25,5)
    wsum=wf+wh+we
    st.metric('Weight total',f'{wsum}%')
    if wsum!=100: st.error('Weights must sum to 100% before running the solver.')
    st.caption('Objective: maximize weighted benefit. Constraints: budget, at most one action per site, technical feasibility.')

st.markdown('### 1 · Intervention dataset')
st.write('Edit the synthetic alternatives. Indicator scores are illustrative values on a 0–100 scale.')
raw=st.data_editor(default_df(),num_rows='dynamic',hide_index=True,use_container_width=True,
    column_config={SITE:st.column_config.TextColumn(required=True),ACTION:st.column_config.SelectboxColumn(options=['Repair existing block','Build new block'],required=True),COST:st.column_config.NumberColumn(min_value=0,step=1),**{c:st.column_config.NumberColumn(min_value=0,max_value=100,step=1) for c in INDICATORS},FEAS:st.column_config.CheckboxColumn()})
err=validate(raw)
weights={INDICATORS[0]:wf/100,INDICATORS[1]:wh/100,INDICATORS[2]:we/100}
if err: st.error(err); st.stop()
scored=score_data(raw,weights)
st.markdown('### 2 · Calculated weighted benefit')
view=scored.copy(); view['Benefit score']=view['Benefit score'].round(2)
st.dataframe(view,use_container_width=True,hide_index=True)

if st.button('⚙️ Run optimization',type='primary',use_container_width=True,disabled=(wsum!=100)):
    try:
        opt,status=solve_milp(scored,budget)
        if opt is None: st.error(status)
        else:
            st.session_state['result']={'opt':opt,'greedy':greedy_benchmark(scored,budget),'status':status,'budget':budget,'weights':weights.copy(),'data':scored.copy()}
    except Exception as e: st.error(f'Solver error: {e}')

if 'result' in st.session_state:
    r=st.session_state['result']
    stale=(r['budget']!=budget or r['weights']!=weights or not r['data'].reset_index(drop=True).astype(str).equals(scored.reset_index(drop=True).astype(str)))
    if stale: st.info('Inputs changed after the previous run. Click **Run optimization** to refresh the solution.')
    opt,greedy=r['opt'],r['greedy']; oc,oscore=total(opt); gc,gscore=total(greedy)
    st.markdown('### 3 · Solver results')
    a,b,c,d=st.columns(4); a.metric('Solver status',r['status']); b.metric('Optimized cost',f'Rp {oc:,.1f}m'); c.metric('Benefit score',f'{oscore:,.2f}'); d.metric('Budget remaining',f'Rp {r["budget"]-oc:,.1f}m')
    st.markdown('**Recommended interventions**')
    if opt.empty: st.info('No interventions fit the current budget and feasibility constraints.')
    else:
        show=opt[[SITE,ACTION,COST,'Benefit score']].copy(); show['Benefit score']=show['Benefit score'].round(2)
        st.dataframe(show.sort_values(SITE),use_container_width=True,hide_index=True)
    st.markdown('**Comparison with a simple greedy benchmark**')
    comp=pd.DataFrame({'Method':['MILP optimum','Greedy benefit/cost'],'Total cost (Rp million)':[oc,gc],'Benefit score':[oscore,gscore]})
    st.dataframe(comp.round(2),hide_index=True,use_container_width=True)
    st.plotly_chart(px.bar(comp,x='Method',y='Benefit score',text_auto='.2f',title='Weighted benefit score by method'),use_container_width=True)
    st.caption('The greedy method is a heuristic benchmark, not guaranteed optimal. MILP status OPTIMAL means the solver proved optimality for the current inputs and model.')
    with st.expander('Mathematical model'):
        st.latex(r'U_{ij}=w_1S_{ij1}+w_2S_{ij2}+w_3S_{ij3}')
        st.latex(r'\max Z=\sum_i\sum_j U_{ij}x_{ij}')
        st.latex(r'\sum_i\sum_j C_{ij}x_{ij}\leq B')
        st.latex(r'\sum_jx_{ij}\leq1\quad\forall i')
        st.latex(r'x_{ij}\leq a_{ij},\qquad x_{ij}\in\{0,1\}')
        st.write('U = weighted benefit score; S = indicator score; w = indicator weight; C = intervention cost; B = budget; a = technical feasibility (0/1); x = binary selection decision.')

st.divider()
st.markdown('### How to use this prototype')
x,y,z=st.columns(3); x.info('**Input**\n\nEdit costs, indicators, and technical feasibility.'); y.info('**Optimize**\n\nSet budget and weights totaling 100%, then run the solver.'); z.info('**Validate**\n\nReplace synthetic values with sourced and expert-validated field data before real-world use.')
st.caption('PEATRA MVP · Research and competition demonstration only.')
