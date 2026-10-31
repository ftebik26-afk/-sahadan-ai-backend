
import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "deepseek-ai/deepseek-v4.1-flash"


# CORS: HTML uygulamasının backend'e erişmesine izin ver
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# Ana sayfa
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# Sunucu sağlık kontrolü
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "server": "SAHADAN AI",
        "model": MODEL,
        "api_key_configured": bool(NVIDIA_API_KEY)
    })


# Yapay zeka maç analizi
@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():

    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "success": False,
            "error": "Geçerli JSON veri gönderilmedi."
        }), 400

    home_team = data.get("home", "").strip()
    away_team = data.get("away", "").strip()

    match_url = data.get("matchUrl", "")

    if not home_team or not away_team:
        return jsonify({
            "success": False,
            "error": "Ev sahibi ve deplasman takımı gerekli."
        }), 400

    if not NVIDIA_API_KEY:
        return jsonify({
            "success": False,
            "error": "NVIDIA_API_KEY Render ortam değişkenlerinde bulunamadı."
        }), 500

    prompt = f"""
Sen profesyonel futbol veri analisti ve maç simülasyon uzmanısın.

MAÇ: {home_team} - {away_team}
MAÇ LİNKİ: {match_url}

GÖREVİN:

Verilen gerçek istatistikleri ve oranları analiz et.
Erişemediğin verileri uydurma.
Veri eksikse bunu açıkça belirt.

Analiz başlıkları:
1. Maç sonucu 1-X-2 olasılıkları
2. İlk yarı sonucu
3. İY/MS tahmini
4. Karşılıklı gol olasılığı
5. 1.5, 2.5, 3.5 gol alt/üst
6. En olası 5 skor
7. Korner beklentisi
8. Sarı ve kırmızı kart beklentisi
9. Oyuncu ve kadro etkisi
10. Penaltı ihtimali
11. Sürpriz sonuç ihtimali
12. Tahmin güven seviyesi

Olasılıkları yüzde olarak belirt.
Tahminleri kesin sonuç gibi sunma.
Gerçek veriye ulaşamadıysan istatistik üretme.

Türkçe, ayrıntılı ve profesyonel rapor hazırla.
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
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 2500
            },
            timeout=120
        )

        if not response.ok:
            return jsonify({
                "success": False,
                "error": "NVIDIA API hatası",
                "details": response.text[:1500]
            }), 502

        result = response.json()

        analysis_text = result["choices"][0]["message"]["content"]

        return jsonify({
            "success": True,
            "home": home_team,
            "away": away_team,
            "analysis": analysis_text
        })

    except requests.Timeout:
        return jsonify({
            "success": False,
            "error": "Yapay zeka sunucusu zaman aşımına uğradı."
        }), 504

    except requests.RequestException as e:
        return jsonify({
            "success": False,
            "error": "NVIDIA bağlantı hatası: " + str(e)
        }), 502

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
        )
    
