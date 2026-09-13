# IBM RAG and Agentic AI

Repositorio de estudos e projetos praticos desenvolvidos ao longo do curso **IBM Generative AI: RAG and Agentic AI Systems**. Reune desde experimentos simples com Gradio ate um projeto final com um sistema multi-agente de recomendacao.

## Estrutura do repositorio

| Pasta / Arquivo | Descricao |
|---|---|
| `finalProject/` | Projeto final do curso: laboratorios de RAG, agentes especializados, sistemas multi-agente com LangGraph e uma interface de chatbot (MCP) para recomendacao de restaurantes e receitas na California. |
| `cal_coach_app/` | Aplicacao Flask que analisa fotos de refeicoes e retorna estimativa de calorias e nutrientes usando um modelo multimodal do watsonx.ai. |
| `genai_flask_app/` | Aplicacao Flask de chat que compara respostas entre diferentes LLMs (Llama, Granite e Mistral). |
| `icebreaker/` | Bot que gera quebra-gelos personalizados a partir de perfis do LinkedIn, usando LlamaIndex e watsonx.ai (possui README proprio). |
| `labs/` | Notebooks de apoio sobre prompt engineering, busca por similaridade e sumarizacao de documentos. |
| `ytbot.py` | Bot que extrai a transcricao de um video do YouTube e responde perguntas sobre o conteudo (RAG com FAISS). |
| `llm_chat.py`, `gradio_demo.py`, `app.py`, `common_input_types.py` | Pequenos exemplos e testes com Gradio e watsonx.ai usados durante o aprendizado. |

## Principais tecnologias

- Python
- IBM watsonx.ai (modelos de linguagem e embeddings)
- LangChain / LangChain IBM
- LlamaIndex
- LangGraph (workflows multi-agente)
- Gradio e Flask (interfaces web)
- FAISS (busca vetorial)

## Como executar

Cada projeto e independente e possui suas proprias dependencias. De forma geral:

1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   source venv/bin/activate  # Linux/Mac
   ```
2. Instale as dependencias do projeto desejado (verifique o `requirements.txt` da pasta, quando existir).
3. Configure as credenciais necessarias em um arquivo `.env` (chave de API do IBM watsonx.ai, project ID e, em alguns projetos, chave da OpenRouter). Esse arquivo nunca deve ser versionado.
4. Execute o arquivo principal do projeto, por exemplo:
   ```bash
   python cal_coach_app/app.py
   python genai_flask_app/app.py
   python ytbot.py
   ```

## Observacao

Este repositorio tem fins educacionais, feito durante o curso da IBM na plataforma Coursera/Skills Network. Alguns arquivos sao templates de laboratorio fornecidos pelo curso e completados como exercicio.
