from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel         
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,                                        
    temperature=0.3
)
model1 = ChatHuggingFace(llm=llm1)                             

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    max_new_tokens=1024,                                       
    temperature=0.3
)
model2 = ChatHuggingFace(llm=llm2)                             

prompt1 = ChatPromptTemplate.from_messages([
    ("human", "Generate short and simple notes from the following text:\n{text}")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("human", "Generate 5 short question answers from the following text:\n{text}")
])

prompt3 = ChatPromptTemplate.from_messages([
    ("human", "Merge the provided notes and quiz into a single document\nnotes -> {notes} and quiz -> {quiz}")
])

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz":  prompt2 | model2 | parser
})

merging_chain = prompt3 | model1 | parser

chain = parallel_chain | merging_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({'text': text})

print(result)