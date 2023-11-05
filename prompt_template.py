import numpy as np
import streamlit as st
import os
import openai
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ['OPENAI_API_KEY']

class Template:
    def __init__ (self, facet, input, description, example):
        self.facet = facet
        self.input = input
        self.description = description
        self.example = example

    def des_context(self):
        context = """\n###\nContext:""" + self.description
        return context

    def ex_context(self):
        # context = """\nHere are some examples that match the criterion""" + self.facet
        # context = """\n###\n"""
        for idx, example in enumerate(self.example):
            if example[1] == 1:
                context_example = """\n###\nInput: """ + example[0] + """\n###\nOutput: yes"""
            if example[1] == 0:
                context_example = """\n###\nInput: """ + example[0] + """\n###\nOutput: no"""
            context = context_example
        return context

    def prompt(self):
        intro = """\n###\nInput:""" + self.input
        user_defined_instruct = """\n###\nIdentify whether the input claim is """ + f"""{self.facet}""" + """ and output yes or no."""
        output = """\n###\nOutput:"""
        if self.description and self.example:
            context = self.des_context() + self.ex_context()
        elif self.description:
            context = self.des_context()
        elif self.example:
            context = self.ex_context()

        prompt = user_defined_instruct + context + intro + output
        # st.markdown(prompt)
        return prompt 
    
class GPT:
    def __init__(self, prompt):
        self.prompt = prompt

    def generate_1(self):
        GPT_response = openai.Completion.create(
            model= 'gpt-3.5-turbo-instruct',
            prompt= self.prompt,
            max_tokens=128,
            temperature=0,
            logprobs=5
        )

        # print(GPT_response['choices'][0])
        tokens = GPT_response['choices'][0]['logprobs']['tokens']
        tokens_prob = GPT_response['choices'][0]['logprobs']['top_logprobs']

        # yes_prob = 0.5
        # no_prob = 0.5
        # st.write(tokens)
        # st.write(tokens_prob)
        for token, prob in zip(tokens, tokens_prob):
            # print('token that is currently examined:', token)
            # st.write('token that is currently examined:', token)
            if token.strip().lower() in ['yes','no']:
                # st.write('exist')
                yes_prob = 0
                no_prob = 0
                for name, value in prob.items():
                    if name.strip().lower() == 'no':
                        # st.write('no_pro:', np.exp(value))
                        no_prob = no_prob + np.exp(value)
                    if name.strip().lower() == 'yes':
                        # st.write('yes_pro:', np.exp(value))
                        yes_prob = yes_prob + np.exp(value)
                # st.write('prob of answer no:', no_prob)
                # st.write('prob of answer yes:', yes_prob)
                break

        yes_prob_normalized = round(yes_prob/(no_prob+yes_prob), 3)
        no_prob_normalized = round(no_prob/(no_prob+yes_prob), 3)

        return [1, yes_prob_normalized]
        # if no_prob_normalized < yes_prob_normalized:
        #     return [1, yes_prob_normalized]
        # else:
        #     return [0, no_prob_normalized]
        
    def generate_2(self):
        GPT_response = openai.Completion.create(
            model= 'gpt-3.5-turbo-instruct',
            prompt= self.prompt,
            max_tokens=128,
            temperature=0,
            logprobs=5
        )

        # print(GPT_response['choices'][0])
        tokens = GPT_response['choices'][0]['logprobs']['tokens']
        tokens_prob = GPT_response['choices'][0]['logprobs']['top_logprobs']

        yes_prob = 0
        no_prob = 0
        # st.write(tokens)
        # st.write(tokens_prob)
        for name, value in tokens_prob[0].items():
            if name.strip().lower() == 'no':
                # st.write('no_pro:', np.exp(value))
                no_prob = no_prob + np.exp(value)
            if name.strip().lower() == 'yes':
                # st.write('yes_pro:', np.exp(value))
                yes_prob = yes_prob + np.exp(value)
                # st.write('prob of answer no:', no_prob)
                # st.write('prob of answer yes:', yes_prob)

        yes_prob_normalized = round(yes_prob/(no_prob+yes_prob), 3)
        no_prob_normalized = round(no_prob/(no_prob+yes_prob), 3)

        # if no_prob_normalized < yes_prob_normalized:
        #     return [1, yes_prob_normalized]
        # else:
        #     return [0, no_prob_normalized]
        
        return [1, yes_prob_normalized]
        
