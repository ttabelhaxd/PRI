import os
from google import genai

class QueryExpander:
    """Expande a query antes de enviar para o motor de pesquisa."""

    def __init__(self, model_name="gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada no ambiente (.env).")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def expand(self, query: str) -> str:
        prompt = f"""
        Expande a seguinte consulta de pesquisa, adicionando termos relevantes,
        sinónimos e expressões alternativas. Mantém tudo numa única frase separada por vírgulas.
        Query original: "{query}"

        Responde APENAS com a query expandida, nada mais.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            expanded = response.text.strip()
            return expanded

        except Exception as e:
            print("[QueryExpander] Erro ao contactar LLM:", e)
            print("[QueryExpander] A usar fallback simples.")

            return f"{query}, {query} história, informação sobre {query}"

