import streamlit as st
import pandas as pd
import numpy as np
import json
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as componentsvalue_watcher
import plotly.figure_factory as ff
import requests
import datetime
import base64
from PIL import Image
import time
import streamlit_antd_components as sac
# import wikipedia
# from streamlit_searchbox import st_searchbox
# from typing import Any, List

######## page config ########

st.set_page_config(layout="wide")
## modify interface lookup
st.markdown("""
    <style>
    # div.mantine-Group-root.mantine-1qj7haw{
    #     gap: 0rem;
    # }
    button.css-nqowgj.e1ewe7hr3{
        display: none;
    }
    ::-webkit-scrollbar {
        width: 5px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    # .css-1oe5cao {
    #     padding-top: 3rem;
    # }
    [data-testid=stVerticalBlock]{
        gap: 0.5rem;
    }
    .css-1629p8f h4 {
        font-weight: 300;
    }
    # label.st-bl {
    #     padding-top: 9px;
    #     padding-right: 0px;
    # }
    # label.st-d5 {
    #     padding-top: 9px;
    #     padding-right: 0px;
    # }
    # label.st-e2 {
    #     padding-top: 9px;
    #     padding-right: 0px;
    # }
    button[title="View fullscreen"]{
        visibility: hidden;
    }
    div.css-1dx1gwv {
        visibility: hidden;
        padding-top: 0px;
    }
    </style>
    """,unsafe_allow_html=True)

## data and model cache
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    data['similarity_numeric'] = 0
    return data

@st.cache_data(show_spinner=False)
def filter_data(data, facet):
    data = data.loc[(data[facet] == 1)]
    data = data.sort_values(by='weighted_score', ascending=False)
    return data

@st.cache_resource()
def load_sentenceBert():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource()
def load_embedding():
    # path = Path(__file__).parents
    embedding_model = load_sentenceBert()
    corpus_embedding = embedding_model.encode(list(init_data['tweet_text']))
    return corpus_embedding

## session state
if 'user_defined_facet_number' not in st.session_state:
    st.session_state['logger'] = []
    st.session_state['user_defined_facet'] = []
    st.session_state['user_defined_prompts'] = []
    st.session_state['user_defined_facet_number'] = 0
    st.session_state['GPT_filtered_data'] = pd.DataFrame([])
    # old log variables
    st.session_state['search_type'] = ['none']
    st.session_state['search_query'] = [{'type':'none', 'query':'none'}]
    # new log variables
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

## detect feature changes
def increment_predefined_counter():
    st.session_state['number_slider_change'] += 1
    st.session_state['time_series'].append({'slider': datetime.datetime.now().timestamp()})

def increment_similarity_counter():
    st.session_state['number_similiarity_slider_change'] += 1
    st.session_state['time_series'].append({'similarity_slider': datetime.datetime.now().timestamp()})

def increment_customized_counter():
    st.session_state['number_new_slider_change'] += 1
    st.session_state['time_series'].append({'customized_slider': datetime.datetime.now().timestamp()})

def increment_search_counter():
    st.session_state['number_search'] += 1 
    # st.session_state['search_content'].append({'type': query_search ,'query': query})
    st.session_state['time_series'].append({'search': datetime.datetime.now().timestamp()})

## initiate feature state
st.session_state.verifiable = True
st.session_state.false_info = True
st.session_state.interest_to_public = True
st.session_state.general_harm = True
# initiate customized facet
if st.session_state['user_defined_facet']:
    for item in st.session_state['user_defined_facet']:
        new_facet = item['facet_name']
        st.session_state[new_facet] = True

## functions
def similarity_search(query, data):
    # st.dataframe(data)
    search_model = load_sentenceBert()
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    corpus_embedding = load_embedding()
    top_k = util.semantic_search(query_embedding, corpus_embedding, top_k=len(init_data))
    top_id = [i['corpus_id'] for i in top_k[0]]
    sim_score = {}
    for idx, item in zip(top_id, [i['score'] for i in top_k[0]]):
        sim_score[idx] = item
    new_id = []
    for i in top_id:
        if i in list(data.index): 
            new_id.append(i)
    data = init_data.iloc[new_id]
    data['similarity_numeric'] = [sim_score[i] for i in new_id]
    data['weighted_score'] = data['weighted_score'] + data['similarity_numeric']*0.5
    # st.write(data)
    data = data.sort_values(by='weighted_score', ascending=False)
    # st.write(data)
    return data

def boolean_search(query, data):
    data = data[data['tweet_text'].str.contains(query) == True]
    return data

def draw_graph(data, name, prob):
    df_fig = data[data[name] == 1]
    fig = ff.create_distplot([df_fig[prob]*10], group_labels=['x'], bin_size=.1, show_rug=False, show_curve=False, colors=['rgba(255, 75, 75, 0.65)'])
    fig.update_layout(showlegend=False, height=50, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_layout(yaxis={'visible': False, 'showticklabels': False}, xaxis={'visible': False, 'showticklabels': False})
    if name + '_slider' in st.session_state:
        fig.add_vrect(x0=st.session_state[name + '_slider'][0]*10, x1=st.session_state[name + '_slider'][1]*10,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
    else:
        fig.add_vrect(x0=0.0, x1=10.0,fillcolor="rgba(255, 75, 75, 0.35)", opacity=0.5,layer="below", line_width=1.5, line_color="rgba(255, 75, 75, 0.7)")
    graph = st.plotly_chart(fig, theme='streamlit', config={'staticPlot': True}, use_container_width=True)
    return graph

def re_rank(data):
    data['weighted_score'] = (data['verifiable']*data['verifiable_numeric']*verifiable_weight_slider
                                        + data['false_info']*data['false_info_numeric']*false_info_weight_slider
                                        + data['interest_to_public']*data['interest_to_public_numeric']*interest_to_public_weight_slider
                                        + data['general_harm']*data['general_harm_numeric']*general_harm_weight_slider)
    if st.session_state['user_defined_facet']:
        for item in st.session_state['user_defined_facet']:
            new_facet = item['facet_name']
            # new_facet_weight = new_facet + '_weight_slider'
            data['weighted_score'] = data['weighted_score'] + data[new_facet]*data[new_facet + "_prob"]*st.session_state[new_facet + '_weight_slider']
    if data['similarity_numeric'].empty != True:
        data['weighted_score'] = data['weighted_score'] + data['similarity_numeric']*similarity_weight_slider
    if sum(data['weighted_score']) != 0:
        data = data.sort_values(by='weighted_score', ascending=False)
    else:
        data = data.sort_index()
    return data
        
def split_frame(data, rows):
    data = [data.loc[i : i + rows - 1, :] for i in range(0, len(data), rows)]
    return data

# def search_wikipedia(searchterm: str) -> List[any]:
#     st.session_state['number_search'] += 1 
#     # st.session_state['search_content'].append({'type': query_search ,'query': query})
#     st.session_state['time_series'].append({'search': datetime.datetime.now().timestamp()})
#     return wikipedia.search(searchterm) if searchterm else []

## load data
TEST_URL = './user_test_data_a.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df_filter_data = init_data
df_filter_data['search'] = ''
df_filter_data['topics'] = ''
df_filter_data['preview'] = 'tweet'

######## interface layout ########

## search input
query_search = 'similarity'
# query_search = st.radio("xx", ('Similarity Search', 'Keyword Search'), horizontal=True, label_visibility='collapsed', on_change=increment_search_counter)
# query = st_searchbox(
#     search_wikipedia,
#     key="wiki_searchbox"
# )
query = st.text_input("search:", label_visibility="collapsed", placeholder="Search claims", on_change=increment_search_counter)

## sidebar
with st.sidebar:

    # if query_search == 'Similarity Search' and query:
    if query:
        # st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
        st.markdown('## Query similarity')
        similarity_weight_slider = st.slider('Query similarity weight', key='query_similarity_weight', min_value=0.0, value=0.5, max_value=1.0, format="%f", label_visibility='hidden', on_change=increment_similarity_counter)
        df_filter_data = similarity_search(query, df_filter_data)
        df_filter_data['search'] = query
        st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    else:
        similarity_weight_slider = 0

    st.markdown('## Preset')

    col1, col2 = st.columns([6, 1])
    with col1:
        verifiable = st.checkbox('Verifiable', help="The system helps you rank tweets that are likely to be verifiable at the top.", value=True, on_change=increment_predefined_counter)
    with col2:
        if verifiable:
            verifiable_select = st.toggle('', key='verifiable_select', label_visibility='hidden')
    if verifiable:
        verifiable_weight_slider = st.slider('verifiable', key='verifiable_weight', min_value=0.0, value=0.1, max_value=1.0, format="%f", label_visibility='collapsed', on_change=increment_predefined_counter)
        if verifiable_select:
            st.session_state.verifiable = False
            draw_graph(df_filter_data, 'verifiable', 'verifiable_numeric')
            if verifiable_weight_slider == 0.00:
                st.session_state.verifiable = True
            verifiable_slider = st.slider('Select a range of values',0.00, 1.00, (0.00, 1.00), format="%f",
                                        key='verifiable_slider', disabled=st.session_state.verifiable, label_visibility='collapsed', on_change=increment_predefined_counter)
    else:
        verifiable_weight_slider = 0
        
    # st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    with col1:
        false_info = st.checkbox('Likely to be false', help="The system helps you rank tweets that are likely to contain false information at the top.", value=True, on_change=increment_predefined_counter)
    with col2:
        if false_info:
            false_info_select = st.toggle('', key='false_info_select', label_visibility='hidden')
    if false_info:
        false_info_weight_slider = st.slider('false_info', key='false_info_weight', min_value=0.0, value=0.1, max_value=1.0, format="%f", label_visibility='collapsed', on_change=increment_predefined_counter)
        if false_info_select:
            st.session_state.false_info = False
            draw_graph(df_filter_data, 'false_info', 'false_info_numeric')
            if false_info_weight_slider == 0.00:
                st.session_state.false_info = True
            false_info_slider = st.slider('Select a range of values',0.0, 1.0, (0.0, 1.0), format="%f",
                                        key='false_info_slider', disabled=st.session_state.false_info, label_visibility='collapsed', on_change=increment_predefined_counter)
    else:
        false_info_weight_slider = 0
    
    # st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    with col1:
        general_harm = st.checkbox('Likely to cause harm', help="The system helps you rank claims that are likely to cause harm to the society at the top.", value=True, on_change=increment_predefined_counter)
    with col2:
        if general_harm:
            general_harm_select = st.toggle('', key='general_harm_select', label_visibility='hidden')
    if general_harm:
        general_harm_weight_slider = st.slider('general_harm', key='general_harm_weight', min_value=0.0, value=0.1, max_value=1.0, format="%f", label_visibility='collapsed', on_change=increment_predefined_counter)
        if general_harm_select:
            st.session_state.general_harm = False
            if general_harm_weight_slider == 0.00:
                st.session_state.general_harm = True
            draw_graph(df_filter_data, 'general_harm', 'general_harm_numeric')
            general_harm_slider = st.slider('Select a range of values',0.0, 1.0, (0.0, 1.0), format="%f",
                                        key='general_harm_slider', disabled=st.session_state.general_harm, label_visibility='collapsed', on_change=increment_predefined_counter)
    else:
        general_harm_weight_slider = 0
        
    # st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)

    col1, col2 = st.columns([6, 1])
    with col1:
        public_interest = st.checkbox('Interest to the public', help="The system helps you rank claims that the public might be more interested in at the top.", value=True, on_change=increment_predefined_counter)
    with col2:
        if public_interest:
            interest_to_public_select = st.toggle('', key='interest_to_public_select', label_visibility='hidden')
    if public_interest:
        interest_to_public_weight_slider = st.slider('interest_to_public', key='interest_to_public_weight', min_value=0.0, value=0.1, max_value=1.0, format="%f", label_visibility='collapsed', on_change=increment_predefined_counter)
        if interest_to_public_select:
            st.session_state.interest_to_public = False
            if interest_to_public_weight_slider == 0.00:
                st.session_state.interest_to_public = True
            draw_graph(df_filter_data, 'interest_to_public', 'interest_to_public_numeric')
            interest_to_public_slider = st.slider('Select a range of values',0.0, 1.0, (0.0, 1.0), format="%f",
                                        key='interest_to_public_slider', disabled=st.session_state.interest_to_public, label_visibility='collapsed', on_change=increment_predefined_counter)
    else:
        interest_to_public_weight_slider = 0 
        
    # st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    
    weight_slider_list = [verifiable_weight_slider, false_info_weight_slider, general_harm_weight_slider, interest_to_public_weight_slider]
    criteria_list = ['verifiable', 'false_info', 'general_harm', 'interest_to_public']

    st.markdown("""<hr style="margin:1em 0px" /> """, unsafe_allow_html=True)
    st.markdown('## Customized')

    if st.session_state['user_defined_facet']:
        st.session_state.reset = False
        for item in st.session_state['user_defined_facet']:
            new_facet = item['facet_name']
            new_facet_slider = new_facet + '_slider'
            new_facet_check = new_facet + '_check'
            col1, col2 = st.columns([6, 1])
            with col1:
                new_facet_check = st.checkbox("""{new_facet}""".format(new_facet=item['facet_name'].capitalize()), key=new_facet + '_check', value=True, on_change=increment_customized_counter)
            with col2:
                if new_facet_check:
                    new_facet_select = st.toggle('', key=new_facet + '_select', label_visibility='hidden')
            # st.write(st.session_state[new_facet + '_check'])
            if st.session_state[new_facet + '_check']:
                new_facet_weight_slider = st.slider('xx', key=new_facet + '_weight_slider', min_value=0.0, value=0.1, max_value=1.0, format="%f", label_visibility='collapsed', on_change=increment_customized_counter)
                # st.write(new_facet_weight_slider)
                weight_slider_list.append(st.session_state[new_facet + '_weight_slider'])
                criteria_list.append(new_facet)
                if new_facet_select:
                    st.session_state[new_facet] = False
                    if new_facet_weight_slider == 0.00:
                        st.session_state[new_facet] = True
                    draw_graph(df_filter_data, new_facet, new_facet + "_prob")
                    new_facet_slider = st.slider('Select a range of values',0.0, 1.0, (0.0, 1.0), format="%f",
                                            key=new_facet_slider, disabled=st.session_state[new_facet], label_visibility='collapsed', on_change=increment_customized_counter)
            else:
                # new_facet_weight_slider = new_facet + '_weight_slider'
                st.session_state[new_facet + '_weight_slider'] = 0
    else:
        st.session_state.reset = True

    st.markdown("""<br> """, unsafe_allow_html=True)

    reset = st.button('reset customized facet', type="secondary", disabled=st.session_state.reset)

## search results

# filter data
df_filter_data = re_rank(df_filter_data)
if verifiable:
    if verifiable_select:
        df_filter_data = filter_data(df_filter_data, 'verifiable')
        df_filter_data = df_filter_data[df_filter_data['verifiable_numeric'].between(st.session_state['verifiable_slider'][0], st.session_state['verifiable_slider'][1])]

if false_info:    
    if false_info_select:
        df_filter_data = filter_data(df_filter_data, 'false_info')
        df_filter_data = df_filter_data[df_filter_data['false_info_numeric'].between(st.session_state['false_info_slider'][0], st.session_state['false_info_slider'][1])]

if public_interest:
    if interest_to_public_select:
        df_filter_data = filter_data(df_filter_data, 'interest_to_public')
        df_filter_data = df_filter_data[df_filter_data['interest_to_public_numeric'].between(st.session_state['interest_to_public_slider'][0], st.session_state['interest_to_public_slider'][1])]

if general_harm:
    if general_harm_select:
        df_filter_data = filter_data(df_filter_data, 'general_harm')
        df_filter_data = df_filter_data[df_filter_data['general_harm_numeric'].between(st.session_state['general_harm_slider'][0], st.session_state['general_harm_slider'][1])]

if st.session_state['user_defined_facet']:
    for item in st.session_state['user_defined_facet']:
        new_facet = item['facet_name']
        new_slider = new_facet + '_slider'
        if st.session_state[new_facet + '_check']:
            if st.session_state[new_facet + '_select']:
                df_filter_data = filter_data(df_filter_data, new_facet)
                df_filter_data = df_filter_data[df_filter_data[new_facet + '_prob'].between(st.session_state[new_slider][0], st.session_state[new_slider][1])]

## re-rank data based on user interactions

for ele1, ele2 in zip(weight_slider_list, st.session_state.value_watcher):
    if ele1 != ele2:
        df_filter_data = re_rank(df_filter_data)

## topic selection
st.markdown("""<span style="margin:1em 0px 2em 0px" /> """, unsafe_allow_html=True)
# topics = sac.checkbox(
#     items=['Covid', 'Vaccine', 'India', 'China', 'Johnson', 'United States', 'Moderna', 'Pandemic',
#     'Pfizer', 'Astrazeneca', 'Trump', 'Biden', 'Russia','Deaths', 'Doses' ,'Effectiveness'], 
#     label='Frequent claim topics:', align='end', check_all=False
#     )

# if topics:
#     df_filter_data = df_filter_data.fillna('0')
#     df_filter_data['topics'] = ','.join(topics)
#     mask = df_filter_data['tweet_text'].apply(lambda x: any(item for item in topics if item.lower() in x.lower()))
#     df_filter_data = df_filter_data[mask]

## pagination
pagination = st.container()
st.markdown("""<span style="margin:1em 0px 2em 0px" /> """, unsafe_allow_html=True)
# st.markdown("""<br/> <br/>""", unsafe_allow_html=True)
current_page = sac.pagination(total=len(df_filter_data), page_size=20, align='start', simple=False, show_total=True)
batch_size = 20

## render search results
df_filter_data = df_filter_data.reset_index()
pages = split_frame(df_filter_data, batch_size)
# st.dataframe(pages[0])
# st.markdown('**Select Claims:**')

## ouput search results into AgGrid table
if pages:
    df_render = pages[current_page - 1][['tweet_text', 'tweet_id', 'search', 'topics', 'preview']]
    df_render['tweet_id'] = df_render['tweet_id'].apply(str)
else:
    df_render = pd.DataFrame(columns=['tweet_text'])

tooltip_renderer = JsCode("""
        class CustomTooltip {
            init(params) {
                const eGui = (this.eGui = document.createElement('div'));
                const color = 'white';
                const data = params.data.tweet_id;
                const text = params.data.tweet_text;
                const url = "https://raw.githubusercontent.com/JialingJia/houjiang-website/master/research_data/checkworthy_tweet/img/" + params.data.tweet_id + ".png" ;
                console.log(url);
                          
                eGui.classList.add('custom-tooltip');
                eGui.innerHTML = `<img src="${url}" width="400px" height=auto>`;
            }
            getGui() {
                return this.eGui;
            }
        }
        """)

myRenderer = JsCode("""
        class BoldCellRenderer {

            init(params) {
                this.eGui = document.createElement('span');

                var substr_keywords = params.data.search.split(' ');
                var substr_topics = params.data.topics.split(',');          
                var substr = substr_keywords.concat("", substr_topics);

                var strRegExp = new RegExp('(' + substr.filter(n => n).join('|') + ')', 'ig');

                this.eGui.innerHTML = params.value.replace(strRegExp, '<span style="color: rgb(255, 75, 75); font-weight: 600">$&</span>');

            }
            getGui(){
                return this.eGui;
            }
        }
    """)

with st.form('my_form'):       
    edited_df = GridOptionsBuilder.from_dataframe(df_render)
    # edited_df.configure_default_column(tooltipField="preview")
    edited_df.configure_column('tweet_id', hide=True)
    edited_df.configure_column('search', hide=True)
    edited_df.configure_column('topics', hide=True)
    edited_df.configure_column('preview', hide=True)
    # edited_df.configure_column('preview', tooltipComponent=tooltip_renderer)
    # edited_df.configure_column('preview', **{'width':100, 'cellStyle': {"color":"grey"}})
    edited_df.configure_column('tweet_text', wrapText=True, autoHeight=True, cellRenderer=myRenderer)
    edited_df.configure_column('tweet_text', header_name='Select tweets', **{'width':1000})
    edited_df.configure_selection(selection_mode="multiple", use_checkbox=True)
    edited_df.configure_grid_options(tooltipShowDelay=1000, tooltipHideDelay=100000)
    gridOptions = edited_df.build()
    grid_table = AgGrid(df_render, 
                                reload_data = False,
                                gridOptions = gridOptions,
                                fit_columns_on_grid_load = True,
                                height = 1300,
                                width = '100%',
                                custom_css = {
                                    ".ag-cell-value": {'font-size': '16px', 'line-height': '22px','padding': '10px 20px 10px 10px'}, 
                                    "#gridToolBar": {'display':'none'},
                                    ".custom-tooltip": {'width': '400px', 'height': 'auto', 'overflow': 'hidden', 'box-shadow': '0 0 0.25rem grey', 'border-radius':'6px'},
                                    ".custom-tooltip p": {'margin': '5px', 'white-space': 'nowrap'},
                                    ".custom-tooltip p:first-of-type": {'font-weight': 'bold'}
                                              },
                                allow_unsafe_jscode= True
                                )
    # grid_table['selected_rows']
    submitted = st.form_submit_button(f"Save selection")
    if submitted:
        st.session_state['time_series'].append({'selection': datetime.datetime.now().timestamp()})
        selected_claims = grid_table['selected_rows']
        st.session_state['end_time'] = datetime.datetime.now().timestamp()
        st.session_state['time_series'].append({'end': datetime.datetime.now().timestamp()})
        st.toast('Claims have been successfully selected!', icon="🎉")
        # time.sleep(.5)
        # st.toast('Selected claims are saved in the selection page.')
        # time.sleep(.5)
        # st.toast('Continue finding more claims!')
    else:
        selected_claims = []

## logger
propability_range = []
for item in criteria_list:
    try:
        propability_range.append({item:st.session_state[item + "_slider"]})
    except:
        propability_range.append({item:[0,0]})

if query and {'type':query_search, 'query':query} != st.session_state['search_query'][-1]:
    st.session_state['search_query'].append({'type':query_search, 'query':query})
#     # st.session_state['number_search']  = st.session_state['number_search'] + 1
#     st.session_state['time_series'].append({'search': datetime.datetime.now().timestamp()})

# if query and st.session_state.query_similarity != similarity_weight_slider:
    # st.session_state['number_similiarity_slider_change'] = st.session_state['number_similiarity_slider_change'] + 1
#     st.session_state['time_series'].append({'similarity_slider': datetime.datetime.now().timestamp()})

logger = [
    {'user_id': st.experimental_user.email}, 
    {'selected_claims': selected_claims}, 
    {'user_query': st.session_state['search_query']},
    {'number_query': st.session_state['number_search']}, 
    {'number_slider_change': st.session_state['number_slider_change']}, 
    {'number_similiarity_slider_change': st.session_state['number_similiarity_slider_change']},
        {'page': {
            'current_page': current_page, 
            'batch_size': batch_size, 
            # 'total_pages': total_pages
            }}, 
        {'time': {
            'start_time': st.session_state['start_time'],
            'end_time': st.session_state['end_time']
            }},
    {'criteria_list': criteria_list}, 
    {'criteria_weight': weight_slider_list}, 
    {'criteria_probability_range': propability_range},
    {'similarity_weight': similarity_weight_slider}, 
    {'user_prompts': st.session_state['user_defined_prompts']},
    {'number_customized_slider_change': st.session_state['number_new_slider_change']}
        ]

## update start time

st.session_state.value_watcher = weight_slider_list
st.session_state.query_similarity = similarity_weight_slider

if selected_claims:
    st.session_state.claim_selected = False
    st.session_state['start_time'] = st.session_state['end_time']
    st.session_state['logger'].append(logger)
    ## clear previous log
    st.session_state['search_query'] = [{'type':'none', 'query':'none'}]
    st.session_state['number_search'] = 0
    st.session_state['number_slider_change'] = 0
    st.session_state['number_new_slider_change'] = 0
    st.session_state['number_similiarity_slider_change'] = 0
else:
    st.session_state.claim_selected = True

with st.sidebar:
    if reset:
        st.error("Reset facet might delete the GPT-processed data. Do you really want to do this?")
        if st.button("Yes I'm ready to rumble"):
            del st.session_state['user_defined_facet']
            del st.session_state['user_defined_prompts']
            del st.session_state['user_defined_facet_number']
            del st.session_state['GPT_filtered_data']
            st.experimental_rerun()


# st.write("number_slider_change", st.session_state['number_slider_change'])
# st.write("number_similiarity_slider_change", st.session_state['number_similiarity_slider_change'])
# st.write("number_new_slider_change", st.session_state['number_new_slider_change'])
# st.write("number_search", st.session_state['number_search'])
# st.write("search_query", st.session_state['search_query'])

# st.write(logger)
# st.write(st.session_state['time_series'])