"""
🌾 UZHAVAN AI - Ollama LLM Service

Connects UZHAVAN AI to a local Ollama instance.

Default:
    Ollama: http://127.0.0.1:11434
    Model : qwen2.5:7b

The service generates Tamil + English responses in ONE Ollama request.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Optional


class OllamaService:

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: Optional[str] = None
    ):
        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        self.host = os.getenv("OLLAMA_HOST", host).rstrip("/")

        self.model = os.getenv(
            "OLLAMA_MODEL",
            model or "qwen2.5:7b"
        )

        self.timeout = float(
            os.getenv("OLLAMA_TIMEOUT", "120")
        )

        # Keep Qwen loaded in RAM/VRAM.
        # This avoids repeatedly loading the 7B model.
        self.keep_alive = os.getenv(
            "OLLAMA_KEEP_ALIVE",
            "30m"
        )

        print("\n" + "=" * 70)
        print("🌾 UZHAVAN AI - Ollama Service")
        print("=" * 70)
        print(f"Host       : {self.host}")
        print(f"Model      : {self.model}")
        print(f"Timeout    : {self.timeout}s")
        print(f"Keep Alive : {self.keep_alive}")
        print("=" * 70)

        self._ensure_model()

    # =============================================================
    # CHECK OLLAMA
    # =============================================================

    def _ensure_model(self):
        """
        Check Ollama and select the requested model.
        """

        try:

            print("[OLLAMA] Checking Ollama server...")

            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET"
            )

            with urllib.request.urlopen(
                req,
                timeout=5
            ) as resp:

                if resp.status != 200:
                    print(
                        f"[OLLAMA] ❌ Server returned HTTP {resp.status}"
                    )
                    return

                raw = resp.read().decode("utf-8")
                data = json.loads(raw)

                models = [
                    m.get("name", "")
                    for m in data.get("models", [])
                ]

                print(
                    f"[OLLAMA] Installed models: {models}"
                )

                if not models:
                    print(
                        "[OLLAMA] ⚠️ No models installed."
                    )
                    return

                # Requested model exists
                if self.model in models:
                    print(
                        f"[OLLAMA] ✅ Using requested model: "
                        f"{self.model}"
                    )
                    return

                # Try known fallback models
                preferred_models = [
                    "qwen2.5:7b",
                    "gemma4:latest",
                    "llama3.2"
                ]

                for preferred in preferred_models:

                    if preferred in models:

                        self.model = preferred

                        print(
                            f"[OLLAMA] ⚠️ Requested model unavailable."
                        )

                        print(
                            f"[OLLAMA] Using fallback: {self.model}"
                        )

                        return

                # Last resort
                self.model = models[0]

                print(
                    f"[OLLAMA] ⚠️ Using first available model: "
                    f"{self.model}"
                )

        except Exception as e:

            print(
                f"[OLLAMA] ❌ Model detection failed: "
                f"{type(e).__name__}: {e}"
            )

    # =============================================================
    # AVAILABILITY
    # =============================================================

    def is_available(self) -> bool:
        """
        Return True if Ollama is reachable.
        """

        try:

            req = urllib.request.Request(
                f"{self.host}/api/tags",
                method="GET"
            )

            with urllib.request.urlopen(
                req,
                timeout=5
            ) as resp:

                available = resp.status == 200

                if available:
                    print(
                        "[OLLAMA] ✅ Server available"
                    )

                return available

        except Exception as e:

            print(
                f"[OLLAMA] ❌ Server unavailable: "
                f"{type(e).__name__}: {e}"
            )

            return False

    # =============================================================
    # GENERATE TAMIL + ENGLISH RESPONSE
    # =============================================================

    def generate_dual_response(
        self,
        question: str,
        context: str = ""
    ) -> Optional[Dict[str, str]]:

        """
        Send ONE request to Qwen and generate:

        Tamil:
        <Tamil answer>

        English:
        <English translation>

        Returns:

        {
            "ta": "...",
            "en": "..."
        }
        """

        print("\n")
        print("=" * 70)
        print("🤖 [OLLAMA] GENERATE REQUEST")
        print("=" * 70)

        print(f"Host     : {self.host}")
        print(f"Model    : {self.model}")
        print(f"Question : {question}")
        print(
            f"Context  : {len(context)} characters"
        )

        # ---------------------------------------------------------
        # Safety
        # ---------------------------------------------------------

        question = str(question or "").strip()
        context = str(context or "").strip()

        if not question:

            print(
                "[OLLAMA] ❌ Empty question"
            )

            return None

        # ---------------------------------------------------------
        # Agricultural system prompt
        # ---------------------------------------------------------

        system_prompt = """
You are UZHAVAN AI, an agricultural AI assistant designed for
farmers and rural communities in Tamil Nadu.

Your job is to answer agricultural questions accurately,
simply, practically, and safely.

IMPORTANT LANGUAGE RULES:

1. The user may write in:
   - Tamil
   - English
   - Regional Tamil dialect
   - Tamil-English code-mixed language

2. Understand common Tamil agricultural terminology correctly.

Examples:

நெல் = rice / paddy
பயிர் = crop
உரம் = fertilizer
விதை = seed
நீர் = water
பாசனம் = irrigation
பூச்சி = pest/insect
நோய் = disease
மண் = soil
விவசாயி = farmer

3. Do NOT translate agricultural words literally into unrelated
English meanings.

4. If the question is in Tamil, answer in simple Tamil.

5. Then provide an accurate English translation of the SAME answer.

6. Do NOT answer with unrelated greetings.

7. Do NOT invent information when reliable context is provided.

8. If the context does not contain enough information, say that
the information is insufficient and provide a safe general answer.

9. Never claim that an AI prediction is a guaranteed agricultural
diagnosis.

10. Keep answers concise and useful for farmers.

OUTPUT FORMAT:

TAMIL:
<simple Tamil answer>

ENGLISH:
<accurate English translation of the Tamil answer>

Return ONLY these two sections.
"""

        # ---------------------------------------------------------
        # Build user prompt
        # ---------------------------------------------------------

        if context:

            user_prompt = f"""
AGRICULTURAL KNOWLEDGE CONTEXT:
{context}

USER QUESTION:
{question}

Using the agricultural context above, answer the user's question.
"""

        else:

            user_prompt = f"""
No external agricultural context is available.

USER QUESTION:
{question}

Answer using your general agricultural knowledge, but do not
invent specific government benefits, fertilizer dosages, prices,
weather information, or medical/agricultural guarantees.
"""

        # ---------------------------------------------------------
        # Complete prompt
        # ---------------------------------------------------------

        full_prompt = (
            system_prompt.strip()
            + "\n\n"
            + user_prompt.strip()
        )

        # ---------------------------------------------------------
        # Ollama payload
        # ---------------------------------------------------------

        payload = {

            "model": self.model,

            "prompt": full_prompt,

            "stream": False,

            # Keep model loaded
            "keep_alive": self.keep_alive,

            "options": {

                "temperature": 0.2,

                "top_p": 0.9,

                "num_predict": 400,

                "repeat_penalty": 1.05
            }
        }

        print(
            f"[OLLAMA] Prompt size: "
            f"{len(full_prompt)} characters"
        )

        print(
            "[OLLAMA] 🚀 Sending POST "
            "/api/generate ..."
        )

        # ---------------------------------------------------------
        # Encode UTF-8
        # ---------------------------------------------------------

        try:

            req_data = json.dumps(
                payload,
                ensure_ascii=False
            ).encode("utf-8")

        except Exception as e:

            print(
                f"[OLLAMA] ❌ JSON encoding error: {e}"
            )

            return None

        # ---------------------------------------------------------
        # HTTP request
        # ---------------------------------------------------------

        try:

            req = urllib.request.Request(

                f"{self.host}/api/generate",

                data=req_data,

                headers={
                    "Content-Type":
                        "application/json; charset=utf-8"
                },

                method="POST"
            )

            # -----------------------------------------------------
            # Send request
            # -----------------------------------------------------

            with urllib.request.urlopen(
                req,
                timeout=self.timeout
            ) as resp:

                status_code = resp.status

                print(
                    f"[OLLAMA] HTTP status: "
                    f"{status_code}"
                )

                if status_code != 200:

                    print(
                        "[OLLAMA] ❌ "
                        f"Unexpected HTTP status: "
                        f"{status_code}"
                    )

                    return None

                # -------------------------------------------------
                # Read response
                # -------------------------------------------------

                response_bytes = resp.read()

                response_text = (
                    response_bytes
                    .decode("utf-8")
                )

                data = json.loads(
                    response_text
                )

                raw_text = (
                    data
                    .get("response", "")
                    .strip()
                )

                # -------------------------------------------------
                # Debug
                # -------------------------------------------------

                print(
                    "[OLLAMA] ✅ Response received"
                )

                print(
                    "[OLLAMA] Raw response:"
                )

                print(
                    raw_text[:1500]
                )

                print("=" * 70)

                if not raw_text:

                    print(
                        "[OLLAMA] ❌ Empty response"
                    )

                    return None

                # -------------------------------------------------
                # Parse Tamil + English
                # -------------------------------------------------

                result = self._parse_dual_text(
                    raw_text
                )

                print(
                    "[OLLAMA] Parsed Tamil:"
                )

                print(
                    result["ta"][:500]
                )

                print(
                    "[OLLAMA] Parsed English:"
                )

                print(
                    result["en"][:500]
                )

                print("=" * 70)

                return result

        # ---------------------------------------------------------
        # HTTP errors
        # ---------------------------------------------------------

        except urllib.error.HTTPError as e:

            try:
                error_body = (
                    e.read()
                    .decode("utf-8")
                )
            except Exception:
                error_body = ""

            print(
                "[OLLAMA] ❌ HTTP ERROR"
            )

            print(
                f"Status: {e.code}"
            )

            print(
                f"Reason: {e.reason}"
            )

            if error_body:
                print(
                    f"Response: {error_body}"
                )

            return None

        # ---------------------------------------------------------
        # Connection errors
        # ---------------------------------------------------------

        except urllib.error.URLError as e:

            print(
                "[OLLAMA] ❌ CONNECTION ERROR"
            )

            print(
                f"Could not connect to: "
                f"{self.host}"
            )

            print(
                f"Reason: {e.reason}"
            )

            return None

        # ---------------------------------------------------------
        # Timeout
        # ---------------------------------------------------------

        except TimeoutError:

            print(
                "[OLLAMA] ❌ REQUEST TIMEOUT"
            )

            print(
                f"Timeout: {self.timeout}s"
            )

            return None

        # ---------------------------------------------------------
        # JSON errors
        # ---------------------------------------------------------

        except json.JSONDecodeError as e:

            print(
                "[OLLAMA] ❌ Invalid JSON response"
            )

            print(e)

            return None

        # ---------------------------------------------------------
        # Anything else
        # ---------------------------------------------------------

        except Exception as e:

            print(
                "[OLLAMA] ❌ GENERATION ERROR"
            )

            print(
                f"Type: {type(e).__name__}"
            )

            print(
                f"Error: {e}"
            )

            return None

    # =============================================================
    # PARSE DUAL RESPONSE
    # =============================================================

    def _parse_dual_text(
        self,
        text: str
    ) -> Dict[str, str]:

        """
        Parse different formats Qwen may produce.

        Expected:

        TAMIL:
        ...

        ENGLISH:
        ...
        """

        text = text.strip()

        # ---------------------------------------------------------
        # Remove common markdown wrappers
        # ---------------------------------------------------------

        text = text.replace(
            "```text",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

        # ---------------------------------------------------------
        # Find English section
        # ---------------------------------------------------------

        english_markers = [
            "ENGLISH:",
            "English:",
            "🇬🇧 English:",
            "2. 🇬🇧 English:",
            "2. English:",
            "2. 🇬🇧 ENGLISH:",
            "2. ENGLISH:"
        ]

        english_position = -1
        english_marker_used = ""

        for marker in english_markers:

            position = text.find(marker)

            if position != -1:

                english_position = position
                english_marker_used = marker
                break

        # ---------------------------------------------------------
        # English section found
        # ---------------------------------------------------------

        if english_position >= 0:

            tamil_part = text[
                :english_position
            ].strip()

            english_part = text[
                english_position
                + len(english_marker_used):
            ].strip()

        else:

            tamil_part = text
            english_part = ""

        # ---------------------------------------------------------
        # Clean Tamil heading
        # ---------------------------------------------------------

        tamil_markers = [
            "TAMIL:",
            "Tamil:",
            "தமிழ்:",
            "1. TAMIL:",
            "1. Tamil:",
            "1. 🇮🇳 தமிழ்:",
            "1. 🇮🇳 தமிழ் (Tamil):",
            "1. 🇮🇳 TAMIL:"
        ]

        for marker in tamil_markers:

            tamil_part = tamil_part.replace(
                marker,
                ""
            )

        tamil_part = tamil_part.strip()

        # ---------------------------------------------------------
        # Clean English heading
        # ---------------------------------------------------------

        for marker in english_markers:

            english_part = english_part.replace(
                marker,
                ""
            )

        english_part = english_part.strip()

        # ---------------------------------------------------------
        # If Qwen didn't produce English,
        # don't pretend Tamil is English.
        # ---------------------------------------------------------

        if not english_part:

            english_part = (
                "English translation "
                "was not generated."
            )

        # ---------------------------------------------------------
        # Final result
        # ---------------------------------------------------------

        return {
            "ta": tamil_part,
            "en": english_part
        }


# =================================================================
# SINGLETON
# =================================================================

_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:

    global _ollama_service

    if _ollama_service is None:

        _ollama_service = OllamaService()

    return _ollama_service