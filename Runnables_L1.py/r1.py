import random
class DummyLlm:
    def __init__(self):
        print("llm created")

    def predict(self,prompt):
        response_list=["r1","r2","r3"]
        return {"response" :random.choice(response_list)}

class DummyPromptTemplate:
    def __init__(self,template,input_variable):
        self.template=template
        self.input_variable=input_variable

    def format(self,input_dict):
        return self.template.format(**input_dict)
llm=DummyLlm()
llm.predict("any random response")