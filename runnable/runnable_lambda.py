from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a short essay about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.4,
    api_key=api_key
    
)

parser = StrOutputParser()

essay_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'essay': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x:len(x.split()))
})

final_chain = RunnableSequence(essay_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'rainy season in india'})

final_result = """{} \n word count - {}""".format(result['essay'], result['word_count'])

print(final_result)