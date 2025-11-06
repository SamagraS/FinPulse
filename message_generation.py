import json
import os
from openai import OpenAI  # pip install openai>=1.2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("PERPLEXITY_API_KEY")
if not API_KEY:
    logger.error("API key 'PERPLEXITY_API_KEY' not found in environment variables.")
    exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

def build_prompt(payload: dict) -> list:
    """
    Build system and user messages for chat completion API with multilingual support.
    """
    lang = payload["customer"]["preferred_language"]
    scheme_name = payload.get("banking_scheme", {}).get("name", None)
    # Simple map for greetings or other per-language additions if needed
    language_map = {
        "English": "Dear",
        "Hindi": "नमस्ते",
        "Bengali": "নমস্কার",
        "Marathi": "नमस्कार",
        "Telugu": "నమస్కారం",
        "Tamil": "வணக்கம்",
        "Gujarati": "નમસ્તે",
        "Urdu": "السلام علیکم",
        "Kannada": "ನಮಸ್ಕಾರ",
        "Odia": "ନମସ୍କାର",
        "Malayalam": "നമസ്കാരം"
    }
    greeting = language_map.get(lang, "Hello")

    if scheme_name:
        scheme_intro = (
            f"• Include a detailed explanation of the banking scheme '{scheme_name}'. "
            f"Explain the main benefits, eligibility, and how it helps the customer. "
            f"Make this explanation informative yet concise to fit the SMS-length message.\n"
        )
    else:
        scheme_intro = ""


    
    system_msg = (
        f"You are an outreach copywriter for India Post Payments Bank. "
        f"Generate a SINGLE SMS-length message (≤700 {lang} chars) that:\n"
        f"• Greets the customer with '{greeting}' appropriately for the upcoming festival.\n"
        f"• Include a detailed explanation of the banking scheme {payload['banking_scheme']['name']}. "
        f"Explain the main benefits, eligibility, and how it helps the customer. "
        f"Make this explanation informative.\n"
        f"• Explains that a ₹{payload['product']['amount']} {payload['culture']['regional_terms']['loan']} "
        f"is available for {payload['culture']['local_crop']} Rabi sowing.\n"
        f"• Removes the {payload['constraints']['primary']} barrier and reassures about {payload['constraints']['secondary']} "
        f"using {payload['constraints']['mitigation']}.\n"
        f"• Tone: {payload['customer']['communication_style']}. Literacy: {payload['customer']['literacy']}. Language: {lang}.\n"
    )
    user_msg = json.dumps(payload, ensure_ascii=False, indent=2)
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

def generate_message(payload: dict, temperature: float = 0.7, max_tokens: int = 350) -> str:
    """
    Use AI model to generate personalized outreach message, supporting multiple languages.
    """
    messages = build_prompt(payload)
    try:
        resp = client.chat.completions.create(
            model="sonar",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        message_content = resp.choices[0].message.content.strip()
        logger.info("Message generation successful.")
        return message_content
    except Exception as e:
        logger.error(f"Error during message generation API call: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        logger.error("Please provide path to input JSON payload file as the first argument.")
        sys.exit(1)
    
    payload_file = sys.argv[1]
    
    try:
        with open(payload_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load payload JSON file: {e}")
        sys.exit(1)

    message = generate_message(data)
    print(message)
