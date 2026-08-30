from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import time
import hashlib
import psycopg2
from transformers import pipeline
import psutil
import json
from pathlib import Path

# Load model directly
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

load_dotenv()

app = Flask(__name__)

HISTORY_FILE = Path('translation_history.json')

# 2024 Helsinki OPUS-MT (opus-mt-tc-bible-big). One hub covers de↔en.
HUB_GMW_DEU_ENG_NLD = "Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_nld"
CURRENT_MODEL = "gmw-de-en"
TARGET_ISO3 = {"en": "eng", "de": "deu", "ru": "rus", "nl": "nld"}
LEGACY_MODEL_ALIASES = {
    "Helsinki-NLP/opus-mt-de-en": "gmw-de-en",
    "Helsinki-NLP/opus-mt-en-de": "gmw-en-de",
}

# Настройки базы данных
DATABASE_URL = "postgresql://koveh:Daniil77Daniil@65.109.88.77:5432/koveh"

MODEL_LIST = [
    {
        "name": HUB_GMW_DEU_ENG_NLD,
        "value": "gmw-de-en",
        "hub": HUB_GMW_DEU_ENG_NLD,
        "description": "German-English",
        "from": "de",
        "to": "en",
        "icon": "🇩🇪🇬🇧",
    },
    {
        "name": HUB_GMW_DEU_ENG_NLD,
        "value": "gmw-en-de",
        "hub": HUB_GMW_DEU_ENG_NLD,
        "description": "English-German",
        "from": "en",
        "to": "de",
        "icon": "🇬🇧🇩🇪",
    },
]


def find_model(model_value: str):
    if not model_value:
        return None
    aliased = LEGACY_MODEL_ALIASES.get(model_value, model_value)
    for model in MODEL_LIST:
        if model["value"] == aliased:
            return model
    for model in MODEL_LIST:
        if model.get("hub") == aliased:
            return model
    return None


def find_model_for_pair(source_lang: str, target_lang: str):
    return next(
        (model for model in MODEL_LIST if model["from"] == source_lang and model["to"] == target_lang),
        None,
    )


def with_target_tag(text: str, target_lang: str) -> str:
    tag = TARGET_ISO3.get(target_lang, target_lang)
    return f">>{tag}<< {text}"


HUB_PIPES = {}
for model in MODEL_LIST:
    hub = model["hub"]
    if hub not in HUB_PIPES:
        try:
            print(f"Loading Helsinki model: {hub}")
            HUB_PIPES[hub] = pipeline("translation", model=hub)
        except Exception as e:
            print(f"Ошибка загрузки модели {hub}: {str(e)}")
            HUB_PIPES[hub] = None
    model["pipe"] = HUB_PIPES.get(hub)


def load_history():
    try:
        if not HISTORY_FILE.exists():
            HISTORY_FILE.write_text('[]', encoding='utf-8')
        return json.loads(HISTORY_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"Ошибка загрузки истории: {str(e)}")
        return []

def save_history(history_item):
    history = load_history()
    history.append(history_item)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def get_db_connection():
    """Получить соединение с базой данных"""
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def save_translation_request(request_body, response_body, request_text, response_text, provider="Helsinki-NLP"):
    """Сохранить данные перевода"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Создаем хеш запроса для уникальности
        request_str = json.dumps(request_body, sort_keys=True)
        request_hash = hashlib.sha256(request_str.encode()).hexdigest()
        
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO api_request_data (request_hash, request_body, response_body, request_messages, response_message)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (request_hash) DO UPDATE SET
                    response_body = EXCLUDED.response_body,
                    response_message = EXCLUDED.response_message,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
            """, (
                request_hash,
                json.dumps(request_body),
                json.dumps(response_body),
                json.dumps([{"role": "user", "content": request_text}]),
                response_text
            ))
            
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None
            
    except Exception as e:
        print(f"Error saving translation request: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return None

def save_translation_transaction(api_key, provider, model, method, endpoint, request_text, response_text, duration_ms, status, request_data_id, user_ip=None, user_agent=None, error_message=None):
    """Сохранить транзакцию перевода"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        # Приблизительный подсчет токенов (символы / 4)
        request_tokens = len(request_text) // 4 if request_text else 0
        response_tokens = len(response_text) // 4 if response_text else 0
        total_tokens = request_tokens + response_tokens
        
        # Стоимость для Helsinki-NLP (бесплатно)
        usd_cost = 0.0  # Helsinki-NLP бесплатен
        
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO api_transactions (
                    user_id, api_key, provider, model, method, endpoint,
                    request_tokens, response_tokens, total_tokens, usd_cost,
                    request_time, response_time, duration_ms, status,
                    request_data_id, user_ip, user_agent, error_message
                ) VALUES (
                    NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, %s, %s,
                    %s, %s, %s, %s
                ) RETURNING id
            """, (
                api_key or 'anonymous', provider, model, method, endpoint,
                request_tokens, response_tokens, total_tokens, usd_cost,
                duration_ms, status, request_data_id, user_ip, user_agent, error_message
            ))
            
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None
            
    except Exception as e:
        print(f"Error saving translation transaction: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return None


@app.route("/screen_translator", methods=["POST"])
def screen_translator():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data.get('text')
        source_lang = data.get('source', 'de')  # default to German
        target_lang = data.get('target', 'en')  # default to English
        
        selected_model = find_model_for_pair(source_lang, target_lang)
        
        if not selected_model or not selected_model.get("pipe"):
            return jsonify({"error": f"Unsupported language pair: {source_lang}-{target_lang}"}), 400
            
        if not text.strip():
            return jsonify({"error": "Empty text"}), 400
            
        translated_text = selected_model["pipe"](with_target_tag(text, target_lang))
        actual_text = translated_text[0]['translation_text']
        
        # Return format matching Google Translate API
        return jsonify({
            "text": actual_text,
            "source": source_lang,
            "target": target_lang,
            "time": time.time()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/translate", methods=["POST"])
def translate():
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data.get('text')
        model_value = data.get('model', CURRENT_MODEL)  # Get selected model or use default
        
        selected_model = find_model(model_value)
        if not selected_model or not selected_model.get("pipe"):
            return jsonify({"error": "Invalid model"}), 400
            
        if not text.strip():
            return jsonify({"error": "Empty text"}), 400
            
        translated_text = selected_model["pipe"](with_target_tag(text, selected_model["to"]))
        actual_text = translated_text[0]['translation_text']
        
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)
        
        response_data = {
            "translated_text": actual_text, 
            "source_language": selected_model["from"], 
            "target_language": selected_model["to"],
            "model": selected_model.get("hub") or selected_model["value"],
            "time": end_time
        }
        
        # Сохранение данных запроса и ответа
        request_data_id = save_translation_request(
            request_body=data,
            response_body=response_data,
            request_text=text,
            response_text=actual_text,
            provider="Helsinki-NLP"
        )
        
        # Сохранение транзакции
        save_translation_transaction(
            api_key=request.headers.get('X-API-Key', 'anonymous'),
            provider="Helsinki-NLP",
            model=model_value,
            method="POST",
            endpoint="/translate",
            request_text=text,
            response_text=actual_text,
            duration_ms=duration_ms,
            status="completed",
            request_data_id=request_data_id,
            user_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        end_time = time.time()
        duration_ms = int((end_time - start_time) * 1000)
        
        # Сохранение ошибки
        error_response = {"error": str(e)}
        request_data_id = save_translation_request(
            request_body=data if 'data' in locals() else {},
            response_body=error_response,
            request_text=data.get('text', '') if 'data' in locals() else '',
            response_text=None,
            provider="Helsinki-NLP"
        )
        
        save_translation_transaction(
            api_key=request.headers.get('X-API-Key', 'anonymous'),
            provider="Helsinki-NLP",
            model=data.get('model', CURRENT_MODEL) if 'data' in locals() else CURRENT_MODEL,
            method="POST",
            endpoint="/translate",
            request_text=data.get('text', '') if 'data' in locals() else '',
            response_text=None,
            duration_ms=duration_ms,
            status="error",
            request_data_id=request_data_id,
            user_ip=request.remote_addr,
            user_agent=request.headers.get('User-Agent'),
            error_message=str(e)
        )
        
        return jsonify(error_response), 400

@app.route("/translator", methods=["POST"])
def translator():
    return translate()  # переиспользуем существующую функцию

# add cpu info
@app.route("/get_cpu_info")
def get_cpu_info():
    return jsonify({"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent})
    

@app.route("/")
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Translation App</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f6f7fb;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 24px;
            }
            h1 {
                color: #333;
                font-size: 24px;
                margin-bottom: 24px;
            }
            .translation-box {
                display: flex;
                gap: 24px;
                margin-bottom: 24px;
            }
            .translation-column {
                flex: 1;
            }
            label {
                display: block;
                color: #666;
                margin-bottom: 8px;
                font-size: 14px;
            }
            textarea {
                width: 100%;
                padding: 12px;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                resize: vertical;
                font-size: 16px;
                min-height: 150px;
                box-sizing: border-box;
            }
            textarea:focus {
                outline: none;
                border-color: #4b73ff;
            }
            button {
                background-color: #4b73ff;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.2s;
            }
            button:hover {
                background-color: #3d5ce0;
            }
            .language-label {
                display: inline-block;
                background-color: #f1f3f4;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 14px;
                margin-bottom: 12px;
            }
            .system-info {
                color: #666;
                font-size: 14px;
                margin-left: 10px;
            }
            .chart-container {
                margin-top: 20px;
                padding: 20px;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            canvas {
                width: 100% !important;
                height: 300px !important;
            }
            .cards-container {
                max-width: 1000px;
                margin: 0 auto;
                display: flex;
                flex-direction: column;
                gap: 24px;
            }
            
            .card {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 24px;
            }
            
            .chart-wrapper {
                position: relative;
                height: 500px;
                width: 100%;
                margin-bottom: 0;
            }
            
            .chart-card {
                padding: 24px 24px 12px 24px;
            }
            
            .chart-title {
                color: #333;
                font-size: 20px;
                margin-bottom: 20px;
            }
            
            .model-select {
                margin-bottom: 16px;
                padding: 8px;
                border-radius: 8px;
                border: 1px solid #e5e5e5;
                width: 100%;
                font-size: 14px;
            }
            
            .feedback-buttons {
                margin-top: 12px;
                display: flex;
                gap: 12px;
            }
            
            .feedback-button {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                border: none;
                background-color: #f1f3f4;
                cursor: pointer;
                transition: background-color 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .feedback-button:hover {
                background-color: #e5e5e5;
            }
            
            .feedback-button.like {
                color: #4CAF50;
            }
            
            .feedback-button.dislike {
                color: #f44336;
            }
            
            .translation-controls {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 16px;
            }
            
            .feedback-buttons {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .feedback-button {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border: none;
                background-color: #f1f3f4;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .feedback-button:hover {
                background-color: #e5e5e5;
                transform: scale(1.1);
            }
            
            .feedback-button.like {
                color: #4CAF50;
            }
            
            .feedback-button.dislike {
                color: #f44336;
            }
            
            .feedback-hint {
                color: #666;
                font-size: 12px;
            }

            .chart-wrapper {
                position: relative;
                height: 500px;
                width: 100%;
                margin-bottom: 0;
            }
            
            .chart-card {
                padding: 24px 24px 12px 24px;
            }

            .model-select {
                display: flex;
                align-items: center;
                padding: 8px 12px;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
                font-size: 14px;
                margin-bottom: 16px;
                width: 100%;
            }

            .model-option {
                display: flex;
                align-items: center;
                gap: 8px;
            }
        </style>
    </head>
    <body>
        <div class="cards-container">
            <div class="card">
                <h1>Translation App <span class="system-info" id="system-info"></span></h1>
                <select class="model-select" id="model-select" onchange="updateLanguageLabels()">
                    ''' + ''.join([f'<option value="{model["value"]}" data-from="{model["from"]}" data-to="{model["to"]}" data-from-full="{model["description"].split("-")[0]}" data-to-full="{model["description"].split("-")[1]}">{model["icon"]} {model["description"]} ({model["value"]})</option>' for model in MODEL_LIST]) + '''
                </select>
                <div class="translation-box">
                    <div class="translation-column">
                        <div class="language-label" id="from-label">German (Deutschland)</div>
                        <textarea id="de-text" placeholder="Enter text to translate"></textarea>
                    </div>
                    <div class="translation-column">
                        <div class="language-label" id="to-label">English (United Kingdom) <span id="translation-time"></span></div>
                        <textarea id="en-text" readonly placeholder="Translation"></textarea>
                    </div>
                </div>
                <div class="translation-controls">
                    <button onclick="translateText()">Translate</button>
                    <div class="feedback-buttons">
                        <button class="feedback-button like" onclick="saveFeedback(true)" title="Like (← Left Arrow)">
                            <i class="fas fa-thumbs-up"></i>
                        </button>
                        <button class="feedback-button dislike" onclick="saveFeedback(false)" title="Dislike (→ Right Arrow)">
                            <i class="fas fa-thumbs-down"></i>
                        </button>
                        <span class="feedback-hint">Use arrow up and down for feedback</span>
                    </div>
                </div>
            </div>

            <div class="card chart-card">
                <h2 class="chart-title">Translation Performance History</h2>
                <div class="chart-wrapper">
                    <canvas id="historyChart"></canvas>
                </div>
            </div>
        </div>

        <script>
            let startTime;
            let translationHistory = [];
            let historyChart;

            // Инициализация графика
            function initChart() {
                const ctx = document.getElementById('historyChart').getContext('2d');
                historyChart = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Translation Speed',
                            data: [],
                            backgroundColor: 'rgba(75, 115, 255, 0.6)',
                            borderColor: 'rgba(75, 115, 255, 1)',
                            pointRadius: 6,
                            pointHoverRadius: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,  // Важно для контроля высоты
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Text Length (symbols)'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Time per Symbol (seconds)'
                                }
                            }
                        },
                        plugins: {
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Length: ${context.parsed.x} symbols, Time/Symbol: ${context.parsed.y.toFixed(3)}s`;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // Обновление графика
            function updateChart(textLength, timePerSymbol) {
                translationHistory.push({
                    x: textLength,
                    y: timePerSymbol
                });

                historyChart.data.datasets[0].data = translationHistory;
                historyChart.update();
            }

            let lastTranslation = null;

            function translateText() {
                const deText = document.getElementById('de-text').value;
                if (!deText.trim()) return;
                
                const select = document.getElementById('model-select');
                const option = select.options[select.selectedIndex];
                const fromLang = option.getAttribute('data-from');
                const toLang = option.getAttribute('data-to');
                
                startTime = Date.now();
                document.getElementById('translation-time').innerText = '';
                
                fetch('/translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: deText,
                        source_language: fromLang,
                        target_language: toLang,
                        model: select.value
                    })
                })
                .then(response => response.json())
                .then(data => {
                    const translatedText = data.translated_text;
                    document.getElementById('en-text').value = translatedText;
                    
                    const endTime = Date.now();
                    const seconds = ((endTime - startTime) / 1000).toFixed(2);
                    const inputSymbols = deText.length;
                    const outputSymbols = translatedText.length;
                    const timePerSymbol = seconds / outputSymbols;
                    
                    const timeInfo = `(${seconds}s | ${timePerSymbol.toFixed(3)}s per symbol | ` +
                        `in: ${inputSymbols} symbols | out: ${outputSymbols} symbols)`;
                    document.getElementById('translation-time').innerText = timeInfo;
                    
                    // Сохраняем данные последнего перевода
                    lastTranslation = {
                        model: document.getElementById('model-select').value,
                        input_text: deText,
                        translated_text: translatedText,
                        seconds: parseFloat(seconds),
                        time_per_symbol: timePerSymbol,
                        input_symbols: inputSymbols,
                        output_symbols: outputSymbols,
                        timestamp: new Date().toISOString()
                    };
                    
                    updateChart(outputSymbols, timePerSymbol);
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred during translation');
                    document.getElementById('translation-time').innerText = '';
                });
            }

            function saveFeedback(isLike) {
                if (!lastTranslation) {
                    alert('Please translate something first');
                    return;
                }

                const feedbackData = {
                    ...lastTranslation,
                    feedback: isLike ? 'like' : 'dislike'
                };

                fetch('/save_feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(feedbackData)
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    alert(isLike ? 'Thanks for the like!' : 'Thanks for the feedback!');
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Failed to save feedback');
                });
            }

            // Добавляем обработчик события keydown для текстового поля
            document.getElementById('de-text').addEventListener('keydown', function(e) {
                // Проверяем, была ли нажата клавиша Enter без Shift
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault(); // Предотвращаем добавление новой строки
                    translateText(); // Вызываем функцию перевода
                }
            });

            // Функция обновления системной информации
            function updateSystemInfo() {
                fetch('/get_cpu_info')
                    .then(response => response.json())
                    .then(data => {
                        const systemInfo = document.getElementById('system-info');
                        systemInfo.textContent = `CPU: ${data.cpu_percent.toFixed(1)}% | RAM: ${data.memory_percent.toFixed(1)}%`;
                        document.title = `Translation App (CPU: ${data.cpu_percent.toFixed(1)}% | RAM: ${data.memory_percent.toFixed(1)}%)`;
                    })
                    .catch(error => console.error('Error:', error));
            }

            // Обновление каждые 200мс
            setInterval(updateSystemInfo, 200);
            
            // Начальное обновление
            updateSystemInfo();

            // Инициализирум график при загрузке страницы
            document.addEventListener('DOMContentLoaded', initChart);

            // Обработчик клавиш
            document.addEventListener('keydown', function(e) {
                if (lastTranslation) {
                    if (e.key === 'ArrowUp') {
                        e.preventDefault();
                        saveFeedback(true);
                    } else if (e.key === 'ArrowDown') {
                        e.preventDefault();
                        saveFeedback(false);
                    }
                }
            });

            const COUNTRY_NAMES = {
                'de': 'German (Deutschland)',
                'en': 'English (United Kingdom)',
                'fr': 'French (France)',
                'ru': 'Russian (Россия)'
            };

            function updateLanguageLabels() {
                const select = document.getElementById('model-select');
                const option = select.options[select.selectedIndex];
                const fromLang = option.getAttribute('data-from');
                const toLang = option.getAttribute('data-to');
                
                document.getElementById('from-label').textContent = COUNTRY_NAMES[fromLang];
                document.getElementById('to-label').innerHTML = 
                    `${COUNTRY_NAMES[toLang]} <span id="translation-time"></span>`;
                
                // Update placeholders
                document.getElementById('de-text').placeholder = 
                    `Enter text in ${COUNTRY_NAMES[fromLang]}`;
                document.getElementById('en-text').placeholder = 
                    `Translation in ${COUNTRY_NAMES[toLang]}`;
            }

            // Initialize labels on page load
            document.addEventListener('DOMContentLoaded', function() {
                updateLanguageLabels();
                initChart();
            });
        </script>
    </body>
    </html>
    '''

@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    data = request.get_json()
    try:
        save_history(data)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import torch
    print(torch.__version__)
    
    # Используем Gunicorn в продакшене, Flask для разработки
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(host="0.0.0.0", port=8003, debug=True)
    else:
        # В продакшене используем Gunicorn через командную строку:
        # gunicorn -c gunicorn_config.py app:app
        pass
# docker run -p 8002:8002 translate_flask
# curl -X POST "http://127.0.0.1:8002/translate" -H "Content-Type: application/json" -d '{"text": "Hello, world!", "source_language": "en", "target_language": "fr"}'
# curl -X POST "http://127.0.0.1:8002/translate" -H "Content-Type: application/json" -d '{"text": "Hello, world!", "source_language": "en", "target_language": "fr"}'