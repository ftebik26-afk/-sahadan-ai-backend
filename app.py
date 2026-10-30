
import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "deepseek-ai/deepseek-v4.1-flash"

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}

    home = data.get("home", "")
    away = data.get("away", "")

    if not home or not away:
        return jsonify({"error": "İki takımın adını girin."}), 400

    if not NVIDIA_API_KEY:
        return jsonify({
            "error": "NVIDIA API anahtarı sunucuya eklenmemiş."
        }), 500

    prompt = f"""
Sen profesyonel futbol veri analistisin.

Maç: {home} - {away}

Verilen verileri değerlendir.
Eksik verileri uydurma.
Kesin sonuç iddiasında bulunma.

Şunları analiz et:
- Maç sonucu 1 X 2
- İlk yarı sonucu
- İY/MS
- Karşılıklı gol
- 1.5, 2.5, 3.5 gol alt üst
- Muhtemel skorlar
- Korner ve kart beklentisi
- Tahmin olasılıkları ve güven seviyesi

Veri olmadan gerçek kadro, sakatlık veya istatistik iddiası üretme.
"""

    try:
        response = requests.post(
            NVIDIA_BASE_URL + "/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1800
            },
            timeout=90
        )

        response.raise_for_status()
        result = response.json()

        return jsonify({
            "success": True,
            "analysis": result["choices"][0]["message"]["content"]
        })

    except requests.RequestException as e:
        return jsonify({
            "error": "NVIDIA bağlantı hatası: " + str(e)
        }), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
                                           
