"""NIMBO Lab — real trace dashboard. Run: streamlit run app/dashboard.py"""
import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title='NIMBO Lab',page_icon='🤖',layout='wide')
st.markdown('''<style>
.stApp{background:radial-gradient(circle at 15% 3%,#17384c 0%,#0b1829 35%,#07111d 100%);color:#edf9ff}
[data-testid="stMetric"]{background:#13283a;border:1px solid #28495c;border-radius:16px;padding:14px}
.robot{margin:10px auto;width:155px;text-align:center;animation:bob 2.5s ease-in-out infinite}
.robot .head{font-size:85px;filter:drop-shadow(0 0 18px #39c7cf)}
.robot .caption{font-size:12px;letter-spacing:3px;color:#64dbd5}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
.trace{background:#132b3c;border-left:4px solid #43e0d3;padding:12px;border-radius:10px;margin:5px 0}
</style>''',unsafe_allow_html=True)
st.title('NIMBO  /  Adaptive Compute Lab')
st.caption('Smarter AI. Less Compute. Same Ambition.  ·  HUMAIN Challenge 2')
mode=st.segmented_control('Workspace',['Live Text Agent','Benchmark Evidence','Image Research'],default='Live Text Agent')

@st.cache_resource(show_spinner='Loading real SST-2 classifiers…')
def get_models():
    from nimbo.text_backend import load_classifier
    from experiments.run_sst2 import SMALL,LARGE
    return load_classifier(SMALL,sst2_labels=True),load_classifier(LARGE,sst2_labels=True)

if mode=='Live Text Agent':
    left,mid,right=st.columns([1,2,1])
    with left:
        st.markdown('<div class="robot"><div class="head">🤖</div><div class="caption">NIMBO ONLINE</div></div>',unsafe_allow_html=True)
        threshold=st.slider('Routing confidence threshold',.5,1.0,.9,.01)
        st.caption('Low-confidence predictions escalate to the BERT baseline. Confidence is not yet calibrated.')
    with mid:
        st.subheader('Real Inference')
        sample=st.text_area('Customer message / movie review','The movie was surprisingly engaging and beautifully acted.',height=130)
        if st.button('Run NIMBO',type='primary',use_container_width=True):
            from nimbo.core import NimboAgent,Modality,Request
            with st.spinner('Running real model inference…'):
                small,large=get_models()
                agent=NimboAgent(small,large,lambda x:None,lambda x:None,threshold)
                trace=agent.run(Request(Modality.TEXT,sample))
            st.session_state['trace']=trace
        if 'trace' in st.session_state:
            trace=st.session_state['trace']
            st.success('Result: '+str(trace.result['label']))
            st.metric('Measured execution time',f'{trace.total_latency_ms:.2f} ms')
            st.metric('Actual route',trace.decision['route'])
            st.caption(trace.decision['reason'])
    with right:
        st.subheader('Decision Trace')
        if 'trace' in st.session_state:
            for e in st.session_state['trace'].events:
                st.markdown(f'<div class="trace"><b>{e["state"].upper()}</b><br><small>{e["elapsed_ms"]:.2f} ms</small></div>',unsafe_allow_html=True)
        else: st.info('Submit a request to inspect the live agent trace.')
elif mode=='Benchmark Evidence':
    st.subheader('Measured comparisons')
    candidates=list(Path('results').glob('benchmark_v2*.json'))+list(Path('results').glob('sst2*.json')) if Path('results').exists() else []
    if not candidates: st.warning('No benchmark report found. Run python -m experiments.benchmark_v2 first.')
    else:
        chosen=st.selectbox('Measured report',candidates,format_func=lambda f:f.name)
        result=json.loads(chosen.read_text())
        if 'strategies' in result:
            strategies=result['strategies']
            cols=st.columns(3)
            for col,key in zip(cols,['small','adaptive','large']):
                with col:
                    st.markdown('**'+key.upper()+'**')
                    st.metric('Accuracy',f'{strategies[key]["accuracy"]:.1%}')
                    st.metric('Mean latency',f'{strategies[key]["mean_latency_ms"]:.2f} ms')
            st.bar_chart({key:v['mean_latency_ms'] for key,v in strategies.items()},horizontal=True)
            st.caption('Results were measured by local benchmark; no unmeasured GPU energy or cost estimates.')
        else: st.json(result)
else:
    st.subheader('Diffusion Optimization')
    st.info('Image pipeline is under evaluation on NVIDIA T4. This panel deliberately does not invent image benchmarks or live generation.')
    st.markdown('**Planned real paths:** Standard SD 1.5 versus SD 1.5 + LCM-LoRA with measured latency, VRAM and side-by-side outputs.')
