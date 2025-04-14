# Desfoque de Faces com OpenCV

Uma ferramenta para desfocar faces em imagens usando OpenCV e Streamlit. Este projeto é uma adaptação do trabalho original de [Face-Blurring-OpenCV](https://github.com/arkalsekar/Face-Blurring-OpenCV) por [@arkalsekar](https://github.com/arkalsekar).

## Funcionalidades

- Interface gráfica amigável com Streamlit
- Dois métodos de desfoque: simples e pixelado
- Processamento de imagem única ou em lote
- Ajuste do nível de confiança da detecção facial
- Personalização do nível de pixelização
- Download das imagens processadas

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/luizzzvictor/blur-face

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Como Usar

1. Inicie a aplicação:

```bash
streamlit run app.py
```

2. Acesse a interface web através do seu navegador

3. Escolha o modo de processamento:

   - Imagem Única: para processar uma imagem por vez
   - Múltiplas Imagens: para processar várias imagens de uma vez

4. Configure as opções de desfoque no menu lateral:

   - Método de Desfoque (Simples ou Pixelado)
   - Nível de Confiança
   - Blocos de Pixelização (quando usar método pixelado)

5. Faça upload das imagens e processe!

## Créditos

Este projeto é uma adaptação do [Face-Blurring-OpenCV](https://github.com/arkalsekar/Face-Blurring-OpenCV) criado por [@arkalsekar](https://github.com/arkalsekar), que por sua vez foi inspirado nos tutoriais do PyImageSearch. Nossa versão adiciona uma interface gráfica com Streamlit e suporte a processamento em lote.

## Licença

MIT

## Tecnologias Utilizadas

- Python
- OpenCV
- Streamlit
- NumPy
