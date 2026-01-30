import os

from fastapi import HTTPException
from google import genai
from google.genai.errors import ClientError


class RAGGenerator:
    """Gera respostas (Answer Generation) usando RAG:
    Query + Documento → Prompt → Gemini (API nova) → Resposta.
    """

    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada no ambiente (.env).")

        # Cliente da nova API
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

        print(f"[RAG] Modelo '{model_name}' carregado.")

    def build_prompt(self, query: str, context: str) -> str:
        """Constrói o prompt para o modelo baseado na query e no contexto."""
        return f"""
        CONTEXTO DO DOCUMENTO:
        \"\"\"
        {context}
        \"\"\"

        PERGUNTA:
        \"\"\"
        {query}
        \"\"\"

        Responde APENAS com base no contexto acima.
        Não inventes factos.
        """

    def answer(self, query: str, context: str) -> str:
        """Gera a resposta final usando a API nova (google-genai)."""
        prompt = self.build_prompt(query, context)

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text.strip()  # type: ignore

        except ClientError as e:
            # A nova Google API usa e.code para o status HTTP
            if getattr(e, "code", None) == 429:
                raise HTTPException(status_code=429, detail="Gemini API quota exceeded")

            if getattr(e, "code", None) == 503:
                raise HTTPException(
                    status_code=503, detail="Gemini API service overloaded try again later"
                )

            # Outros erros da API Gemini
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
