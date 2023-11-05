import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime

if 'user_defined_facet_number' not in st.session_state:
    st.session_state['logger'] = []
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    st.session_state['search_type'] = ['none']
    st.session_state['search_query'] = ['none']
    st.session_state['number_search'] = 0
    st.session_state['number_slider_change'] = 0
    st.session_state['number_new_slider_change'] = 0
    st.session_state['number_similiarity_slider_change'] = 0
    st.session_state['start_time'] = datetime.datetime.now().timestamp()
    st.session_state['end_time'] = 0
    st.session_state['claim_candidate'] = []
    st.session_state['time_series'] = [{'start': datetime.datetime.now().timestamp()}]
    st.session_state.selected_claims = []
    st.session_state.value_watcher = [0,0,0,0]
    st.session_state.query_similarity = 0
    st.session_state.similarity_weight_boolean = True

# get random string of letters and digits
selected_claims = []
if st.session_state['logger']:
    claim_id = (id for id in range(1, 100_00))
    for item in st.session_state['logger']:
        with st.container():
            if item[2]['user_query']:
                st.markdown(f"""
                    ##### {item[2]['user_query'][-1]['type']}: {item[2]['user_query'][-1]['query']}  
                    """)
            else:
                st.markdown(f"""
                    ##### No query
                    """)
            for i in item[1]['selected_claims']:
                claim = "claim_checkbox_" + str(next(claim_id))
                if st.checkbox(i['tweet_text'], key=claim, value=False):
                    selected_claims.append([i['tweet_text'], i['tweet_id']])
            
            st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
else:
    st.write("You haven't selected any claims~")

# st.write(st.session_state['logger'])

json_string = json.dumps([st.session_state['logger'], selected_claims, st.session_state['time_series']])

st.download_button(
    label = f"Download task data",
    data = json_string,
    file_name = f'interface_A.json',
    mime='application/json'
)

# st.write(st.session_state['time_series'])
# st.write(st.session_state['number_slider_change'])
# st.write(st.session_state['number_search'])