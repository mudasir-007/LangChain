# YouTube Chatbot — HuggingFace (Free, no paid API)

## What changed from the original
| Original (YouTuber)        | This version                              |
|----------------------------|-------------------------------------------|
| `OpenAIEmbeddings`         | `HuggingFaceEmbeddings` (all-MiniLM-L6-v2)|
| `ChatOpenAI` (gpt-4o-mini) | `HuggingFaceEndpoint` (Mistral-7B)        |
| OpenAI API key             | HuggingFace token (free)                  |
| Old `langchain_openai`     | `langchain-huggingface` (updated syntax)  |

---

## VS Code Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your HuggingFace token
Get a free token from https://huggingface.co/settings/tokens

Either export it:
```bash
# Windows
set HF_TOKEN=your_token_here

# Mac/Linux
export HF_TOKEN=your_token_here
```

Or paste it directly in `chatbot.py` line:
```python
HF_TOKEN = "your_token_here"
```

### 3. Run
```bash
python chatbot.py
```

---

## Google Colab Setup
1. Open `youtube_chatbot.ipynb` in Colab
2. Paste your HuggingFace token in the CONFIG cell
3. Run all cells top to bottom
