import asyncio
import time
import httpx

# Configuración del LLM local
URL_LLM = "http://localhost:1234/v1/chat/completions"

async def consultar_llm(client, prompt, id_tarea):
    """Envía un prompt al LLM local de forma asíncrona."""
    payload = {
        # LM Studio suele ignorar el nombre del modelo o usa el que esté cargado
        "model": "openai/gpt-oss-20b", 
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    print(f"⏳ [Tarea {id_tarea}] Enviando prompt al LLM...")
    
    try:
        # Realizamos la petición POST asíncrona (con un timeout amplio por si el LLM tarda)
        response = await client.post(URL_LLM, json=payload, timeout=60.0)
        response.raise_for_status()
        
        # Extraemos la respuesta del JSON
        data = response.json()
        respuesta_texto = data['choices'][0]['message']['content']
        
        print(f"✅ [Tarea {id_tarea}] ¡Respuesta recibida!")
        return respuesta_texto

    except Exception as e:
        print(f"❌ [Tarea {id_tarea}] Error en la petición: {e}")
        return None

async def main():
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
    
    # Creamos un único cliente asíncrono para reutilizar las conexiones
    async with httpx.AsyncClient() as client:
        # Creamos una lista de tareas (corutinas)
        tareas = [
            consultar_llm(client, prompt, i) 
            for i, prompt in enumerate(prompts, start=1)
        ]
        
        # Ejecutamos todas las tareas en paralelo y esperamos a que terminen
        resultados = await asyncio.gather(*tareas)
    
    # Mostrar los resultados obtenidos
    print("\n" + "="*40)
    print("🧠 RESULTADOS DEL LLM")
    print("="*40)
    for i, res in enumerate(resultados, start=1):
        print(f"\n🤖 Respuesta a Tarea {i}:\n{res}")
        
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n⏱️ Tiempo total de ejecución asíncrona: {tiempo_total:.2f} segundos")

# Ejecutar el bucle de eventos de asyncio
if __name__ == "__main__":
    asyncio.run(main())