import boto3
import json
import uuid
from datetime import datetime
import os
import pandas as pd
import PyPDF2
from PIL import Image

PROFILE_NAME = os.environ.get('AWS_PROFILE', '')

def get_boto3_client(service_name, region_name='us-east-1', profile_name=''):
    """
    Retorna um cliente do serviço AWS usando IAM Role da instância.
    """
    try:
        # Primeiro tenta usar o IAM Role (modo de produção)
        session = boto3.Session(region_name=region_name)
        client = session.client(service_name)
        
        print(f"DEBUG: Usando IAM Role para acessar '{service_name}' na região '{region_name}'")
        return client
        
    except Exception as e:
        print(f"ERRO: Não foi possível acessar a AWS: {str(e)}")
        print("ATENÇÃO: Verifique se o IAM Role está corretamente associado à instância EC2.")
        return None



def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna como string."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt(file_path):
    """Lê o conteúdo de um arquivo TXT e retorna como string."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv(file_path):
    """Lê o conteúdo de um arquivo CSV e retorna como string."""
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"
    
def format_context(context, source="Contexto Adicional"):
    """Formata o contexto para ser adicionado ao prompt."""
    return f"\n\n{source}:\n{context}\n\n"

#ALTERAR
def generate_chat_prompt(user_message, conversation_history=None, context="Robo que analisa imagens de materiais recicláveis"):
    """
    Gera um prompt de chat completo com histórico de conversa e contexto opcional.
    """
    system_prompt = """
    Você é um robô especialista em analisar imagens de materiais recicláveis.
    Ao receber uma imagem, observe atentamente todos os detalhes visuais (formas, cores, texturas e tamanhos)
    para identificar corretamente os materiais presentes.
    
    Regras:
      - Retorne uma lista de itens, onde cada item é um objeto JSON com as chaves "Tipo de objeto", "Tipo de material" e "Quantidade".
      - Utilize nomes simples e padronizados para os objetos, por exemplo, "garrafa plástica", "lata", etc.
      - Informe a quantidade EXATA detectada para cada objeto na imagem.
      - Retorne somente o material externo de cada obejto, não o conteúdo interno.
      
    Exemplo de saída:
    [
      {
        "Tipo de objeto": "garrafa plástica",
        "Tipo de material": "PET",
        "Quantidade": 3
      },
      {
        "Tipo de objeto": "lata",
        "Tipo de material": "Vidro",
        "Quantidade": 1
      }
    ]
    
    Por favor, processe a imagem conforme descrito e retorne apenas o JSON, sem nenhum texto adicional.
    """

    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "Histórico da conversa:\n"
        recent_messages = conversation_history[-8:]
        for message in recent_messages:
            role = "Usuário" if message.get('role') == 'user' else "Assistente"
            conversation_context += f"{role}: {message.get('content')}\n"
        conversation_context += "\n"

    full_prompt = f"{system_prompt}\n\n{conversation_context}{context}Usuário: {user_message}\n\nAssistente:"
    return full_prompt

#ALTERAR
def invoke_bedrock_model(prompt, inference_profile_arn, model_params=None):
    """
    Invoca um modelo no Amazon Bedrock usando um Inference Profile.
    """
    if model_params is None:
        model_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 200,
        "max_tokens": 800
        }

    bedrock_runtime = get_boto3_client('bedrock-runtime')

    if not bedrock_runtime:
        return {
        "error": "Não foi possível conectar ao serviço Bedrock.",
        "answer": "Erro de conexão com o modelo.",
        "sessionId": str(uuid.uuid4())
        }

    try:
        body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": model_params["max_tokens"],
        "temperature": model_params["temperature"],
        "top_p": model_params["top_p"],
        "top_k": model_params["top_k"],
        "messages": [
        {
        "role": "user",
        "content": [
        {
        "type": "text",
        "text": prompt
        }
    ]
    }
    ]
    })

        response = bedrock_runtime.invoke_model(
        modelId=inference_profile_arn,  # Usando o ARN do Inference Profile
        body=body,
        contentType="application/json",
        accept="application/json"
    )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
            
        return {
            "answer": answer,
            "sessionId": str(uuid.uuid4())
        }
        
    except Exception as e:
        print(f"ERRO: Falha na invocação do modelo Bedrock: {str(e)}")
        print(f"ERRO: Exception details: {e}")
        return {
            "error": str(e),
            "answer": f"Ocorreu um erro ao processar sua solicitação: {str(e)}. Por favor, tente novamente.",
            "sessionId": str(uuid.uuid4())
        }
def read_pdf_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo PDF carregado pelo Streamlit."""
    try:
        import io
        from PyPDF2 import PdfReader
        
        pdf_bytes = io.BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Erro ao ler PDF: {str(e)}"

def read_txt_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo TXT carregado pelo Streamlit."""
    try:
        return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        return f"Erro ao ler TXT: {str(e)}"

def read_csv_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de um arquivo CSV carregado pelo Streamlit."""
    try:
        import pandas as pd
        import io
        
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        return df.to_string()
    except Exception as e:
        return f"Erro ao ler CSV: {str(e)}"

def read_image_from_uploaded_file(uploaded_file):
    """Lê o conteúdo de uma imagem carregada pelo Streamlit."""
    try:
        image = Image.open(uploaded_file)
        return f"Imagem carregada com dimensões: {image.size}"
    except Exception as e:
        return f"Erro ao processar a imagem: {str(e)}"