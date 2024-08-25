import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.assistant.llm import LLM
from src.assistant.pgvector import PGVectorUtils
from src.core.consts import PGVECTOR_DSN


logger = logging.getLogger(__name__)


class Chain:

    def __init__(
        self,
        prompt: str = None,
        prompt_path: str = "./prompt.txt"
    ) -> None:

        self.prompt = prompt
        self.prompt_path = prompt_path
        self.llm_class = LLM()
        self.vectorstore_utils = PGVectorUtils(PGVECTOR_DSN)

    def fill_prompt(self):
        if not (self.prompt or self.prompt_path):
            logger.warn("Prompt and prompt file not found.")
            self.prompt = "You are a ai assistant. {input}"

        if self.prompt_path:
            with open(self.prompt_path, 'r', encoding='utf-8') as file:
                self.prompt = file.read()

    def get_prompt_template(self):
        self.fill_prompt()
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            ("human", "{input}"),
        ])
        return prompt

    def get_rag_chain(self):
        model = self.llm_class.get_model()
        retriever = self.vectorstore_utils.get_retriever()
        prompt_template = self.get_prompt_template()

        question_answer_chain = create_stuff_documents_chain(
            model, prompt_template
        )
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain, retriever

    def chat(self, message: str):
        rag_chain, retriever = self.get_rag_chain()
        response = rag_chain.invoke({
            "input": message,
            "question": RunnablePassthrough(),
            "context": retriever
        })
        return response
