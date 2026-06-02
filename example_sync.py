import time
import requests

# Configuración del LLM local
URL_LLM = "http://localhost:1234/v1/chat/completions"

def consultar_llm_sync(prompt, id_tarea):
    """Envía un prompt al LLM local de forma síncrona (bloqueante)."""
    payload = {
        "model": "openai/gpt-oss-20b", 
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    print(f"🔄 [Tarea {id_tarea}] Enviando prompt al LLM...")
    
    try:
        # La ejecución SE CONGELA aquí hasta que el LLM responda por completo
        response = requests.post(URL_LLM, json=payload, timeout=60.0)
        response.raise_for_status()
        
        data = response.json()
        respuesta_texto = data['choices'][0]['message']['content']
        
        print(f"✅ [Tarea {id_tarea}] ¡Respuesta recibida!")
        return respuesta_texto

    except Exception as e:
        print(f"❌ [Tarea {id_tarea}] Error en la petición: {e}")
        return None

def main():
    prompts = [
          "Cuéntame un chiste corto de programadores.",
        "Explica qué es un agujero negro en una sola frase.",
        "Escribe un poema de veinte líneas sobre el café.",
        "¿Qué es la inteligencia artificial?",
        "¿Cuál es la capital de Francia?",
        "¿En qué año se independizó México?",
        "¿Cuál es la fórmula química del agua?",
        "¿Qué es el cambio climático?",
        "¿Cuál es la capital de España?",  
        "¿Qué es el internet?",
        "¿Qué es el hardware?",
        "¿Qué es el software?"
    ]
    
    tiempo_inicio = time.time()
    resultados = []
    
    # Ejecución secuencial estricta
    for i, prompt in enumerate(prompts, start=1):
        resultado = consultar_llm_sync(prompt, i)
        resultados.append(resultado)
    
    # Mostrar los resultados obtenidos
    print("\n" + "="*40)
    print("🧠 RESULTADOS DEL LLM (SÍNCRONO)")
    print("="*40)
    for i, res in enumerate(resultados, start=1):
        print(f"\n🤖 Respuesta a Tarea {i}:\n{res}")
        
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n⏱️ Tiempo total de ejecución síncrona: {tiempo_total:.2f} segundos")

if __name__ == "__main__":
    main()