import streamlit as st
import pandas as pd
import numpy as np
import string
import time
import random
import joblib
from sentence_transformers import SentenceTransformer, util
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode, DataReturnMode
from prompt_template import Template, GPT
import datetime


# page config
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    button.css-nqowgj.e1ewe7hr3{
        display: none;
    }
    ::-webkit-scrollbar {
        width: 10px;
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
    # .css-lrlib {
    #     padding-top: 3rem;
    # }
    </style>
""", unsafe_allow_html=True)

# data and model cache
@st.cache_data()
def load_data(url):
    data = pd.read_csv(url)
    data['similarity_numeric'] = 0
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

# functions
def similarity_search(query, data):
    search_model = load_sentenceBert()
    query_embedding = search_model.encode(query, convert_to_tensor=True)
    corpus_embedding = load_embedding()
    top_k = util.semantic_search(query_embedding, corpus_embedding, top_k=len(data))
    top_id = [i['corpus_id'] for i in top_k[0]]
    new_id = []
    for i in top_id:
        if i in list(data.index): 
            new_id.append(i)
    # data = data.sort_values(by='index', key=lambda column: column.map(lambda e: new_id.index(e)))
    # st.write(new_id)
    data = init_data.iloc[new_id]
    return data

def boolean_search(query, data):
    data = data[data['tweet_text'].str.contains(query) == True]
    return data

def generate_random(data):
    random_list = []
    for i in range(0,len(data)):
        random_list.append(round(random.uniform(0.0, 1.0), 3))
    return random_list

def generate_random_boolean(data):
    random_list = []
    for i in range(0,len(data)):
        # random_list.append(random.choice([0, 1]))
        random_list.append(1)
    return random_list

# initiate session state
if 'select_search' not in st.session_state:
    st.session_state.before_example = []
    st.session_state.middle_example = []
    st.session_state.after_example = []

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
    
# load data
TEST_URL = './user_test_data_a.csv'
original_data = load_data(TEST_URL)
if st.session_state['GPT_filtered_data'].empty:
    init_data = original_data
else:
    init_data = st.session_state['GPT_filtered_data'].sort_index()

df = init_data

# UI
st.subheader('Create and add new criterion')

st.info(f'You are going to use ChatGPT to create new criteria to filter claims. Please provide **:red[detailed descriptions of the new criteria]** to ChatGPT so that it helps to preprocess claims that are more likely to match the new criteria.')

def start_GPT():
    st.session_state['time_series'].append({'GPT_edit_prompt': datetime.datetime.now().timestamp()})

col1, col2, col3 = st.columns([5,0.25,5])

with col1:
    facet_name = st.text_input(f'**Criterion name**: what is your new criterion?',
                                    placeholder="propaganda", 
                                    on_change=start_GPT)
    prompts_1 = st.text_area(f'**Descriptions**: how would you describe the new criterion?',
                                    placeholder='Claims are sensationalized or contain hyperbolic language grabbing public attention, thus easily calling people to action.',
                                    height= 160,
                                    on_change=start_GPT)

    criterion_name = facet_name
    descriptions = prompts_1
            
with col3:
    prompt_template = st.selectbox(
        f'**View prompt examples:**',
        ('Propaganda', 'Difficult', 'Urgent')
    )

    if prompt_template == 'Propaganda':
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"I want to identify claims that are propaganda.")
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"Claims are sensationalized or contain hyperbolic language grabbing public attention, thus easily calling people to action.")

    if prompt_template == 'Difficult':
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"I want to identify claims that are difficult to check.")
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"To fact-check this claim is very difficult because there is no evidence, it requires a long time, or it is too subjective.")

    if prompt_template == 'Urgent':
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"I want to identify claims that are urgent.")
        with st.chat_message("user", avatar = "🧑‍💻"):
            st.write(f"Fact-checker should immediately fact-check this claim otherwise it might cause distrastous harm to the public.")
            
    #  st.markdown('''
    #              Criterion: Propaganda
                 
    #              Description: Claims are sensationalized or contain hyperbolic language 
    #              grabbing public attention, thus easily calling people to action.
                 
    #              Criterion: Urgent
                 
    #              Description: Fact-checker should immediately fact-check this claim 
    #              otherwise it might cause distrastous harm to the public.
    #              ''')
    #  st.markdown(f'**Example prompt templates**: what is your new criterion?')


# # Examples
# query = st.text_input(f"**Examples (optional)**: which claims match the new criterion?", placeholder="search claims using keywords")
# query_search = st.radio("xx", ('Similarity Search', 'Boolean Search'), horizontal=True, label_visibility='collapsed', key = 'select_search')
# if query_search == 'Similarity Search' and query:
#     df = similarity_search(query, df)
# if query_search == 'Boolean Search' and query:
#     df = boolean_search(query, df)
# # df = search(query, df)

# st.session_state.before_example = list(df['tweet_text'])

# delect_js = JsCode("""
#     function(e) {
#         let api = e.api;        
#         let sel = api.getSelectedRows();
#         setTimeout(function(){
#             api.applyTransaction({remove: sel})
#         }, 500);
#     };
#     """)

# # AgGrid version
# custom_css = {
#                 ".ag-cell-value": {'line-height': '20px','padding': '10px'}, 
#                 "#gridToolBar": {'display':'none'}
#                 }


# col1, col2 = st.columns(2)
# df_before = pd.DataFrame(st.session_state.before_example, columns=['tweet_text'])
# with col1:
#     edited_df = GridOptionsBuilder.from_dataframe(df_before)
#     edited_df.configure_column('tweet_text', wrapText=True, autoHeight=True)
#     edited_df.configure_column('tweet_text', header_name='tweets (click to add)', **{'width':1000})
#     edited_df.configure_selection(selection_mode="single", use_checkbox=True)
#     # edited_df.configure_grid_options(onRowSelected=delect_js)
#     gridOptions = edited_df.build()
#     grid_table = AgGrid(df_before, 
#                             update_mode=GridUpdateMode.SELECTION_CHANGED,
#                             gridOptions = gridOptions,
#                             fit_columns_on_grid_load=True,
#                             height = 600,
#                             width = '100%',
#                             custom_css = custom_css,
#                             allow_unsafe_jscode = True,
#                                 # reload_data = False
#                             )
#     # st.session_state.before_example = list(grid_table.data['tweet_text'])
#     for i in grid_table.selected_rows:
#         if i['tweet_text'] not in st.session_state.middle_example:
#             st.session_state.middle_example.append(i['tweet_text'])

# df_middle = pd.DataFrame(st.session_state.middle_example, columns=['tweet_text'])

# with col2:
#     df_middle['label'] = 1
#     selected_df = GridOptionsBuilder.from_dataframe(df_middle)
#     selected_df.configure_column('label', editable=True)
#     selected_df.configure_column('tweet_text', wrapText=True, autoHeight=True, editable = True)
#     selected_df.configure_column('tweet_text', header_name='selected tweets (click to remove)', **{'width':1000})
#     selected_df.configure_selection(selection_mode="single", use_checkbox=True)
#     selected_df.configure_grid_options(onRowSelected=delect_js)
#     gridOptions = selected_df.build()
#     selected_table = AgGrid(df_middle, 
#                             gridOptions = gridOptions,
#                             editable = True,
#                             # reload_data = False,
#                             fit_columns_on_grid_load=True,
#                             height = 600,
#                             width = '100%',
#                             custom_css = custom_css,
#                             allow_unsafe_jscode = True,
#                             data_return_mode = DataReturnMode.AS_INPUT,
#                             update_mode = GridUpdateMode.MODEL_CHANGED
#                             )
#     if selected_table.data.empty == False:
#         st.session_state.after_example = list(selected_table.data['tweet_text'])
#     # else:
#     #     st.session_state.middle_example = []
#         # st.experimental_rerun()
        
#     # for i in selected_table.selected_rows:
#     #     st.session_state.before_example.append(i['tweet_text'])
#     # df_before = pd.DataFrame(st.session_state.before_example, columns=['tweet_text'])
#     st.session_state.middle_example = st.session_state.after_example

# # st.write(selected_table.data)

prompts_2 = []
# for idx, row in selected_table.data.iterrows():
#     prompts_2.append([row['tweet_text'], row['label']])
# # st.write(prompts_2)
#     # st.session_state.after_example = [] 
#     # for i in selected_table.data:
#     #     st.session_state.after_example.append(i['tweet_text'])
#     # for i in selected_table.selected_rows:
#     #     st.session_state.before_example.append(i['tweet_text'])
    
# prompts_2 = st.session_state.temp_example
final_submission = st.button('Confirm and add new criterion', type='primary')
if final_submission:
    if facet_name and (prompts_1 or prompts_2):
        st.session_state['time_series'].append({'GPT_start': datetime.datetime.now().timestamp()})
        if facet_name in [items['facet_name'] for items in st.session_state['user_defined_facet']]:
            facet_name = facet_name + "_new"
        message = st.warning("The GPT is processing your texts. Don't leave this page otherwise the data will be lost.")
        # record user input in session state
        st.session_state['user_defined_facet_number'] += 1
        t = st.session_state['user_defined_facet_number']
        t_facet = {'facet_name':facet_name}
        st.session_state['user_defined_facet'].append(t_facet)
        if prompts_1:
            st.session_state['user_defined_prompts'].append({'prompt':prompts_1})
        if prompts_2:
            st.session_state['user_defined_prompts'].append({'prompt':prompts_2})
        # request GPT
        # k = 50
        GPT_response_list = []
        # progress_text = "The GPT is processing your texts"
        progress_bar = st.progress(0)
        percent_complete = 0
        ## generate prompts
        for text in df['tweet_text']:
            prompt = Template(facet_name, text, prompts_1, prompts_2)
            # st.write(prompt.prompt())
            GPT_response = GPT(prompt.prompt())
            GPT_response_list.append(GPT_response.generate_1())
            progress_bar.progress(percent_complete)
            percent_complete = percent_complete + 1/len(df)
        # for text in df['tweet_text'][k::]:
        #     progress_bar.progress(percent_complete)
        #     percent_complete = percent_complete + 1/len(df)
        time.sleep(0.1)    
        progress_bar.progress(100)
        # new_facet_answer = [item[0] for item in GPT_response_list] + generate_random_boolean(df)[k::]
        # new_facet_prob = [item[1] for item in GPT_response_list] + generate_random(df)[k::]
        new_facet_answer = [item[0] for item in GPT_response_list]
        new_facet_prob = [item[1] for item in GPT_response_list]
        df[[facet_name, facet_name + '_prob']] = list(zip(new_facet_answer, new_facet_prob))
        # generate random number for testing
        # df[facet_name] = generate_random_boolean(df)
        # df[facet_name + '_prob'] = generate_random(df)
        # st.dataframe(df)
        st.session_state['GPT_filtered_data'] = df
        # record end time
        st.session_state['time_series'].append({'GPT_end': datetime.datetime.now().timestamp()})
        # message.empty
        st.toast('You have successfully added the new facet to the facet browsing!', icon="🎉")
        time.sleep(1)
        st.toast('Go back to the select page and play with it!')
        # else:
        #     st.warning('You have already created this facet. Please give it a different name for the new facet.', icon="👻")
    else:
        st.error('Please complete both of the name and prompt for your new facet. The prompt could be either the description or the example.', icon="👻")

# st.write(st.session_state['user_defined_facet'])
# st.write(st.session_state['user_defined_prompts'])
# st.dataframe(st.session_state['GPT_filtered_data'])