import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import re
from utils.nlp_utils import tokenizar_texto, lematizar_texto, pos_tagging, crear_embedding, similaridad_textos, extraer_keywords
from utils.analysis import analizar_empresa, generar_mensaje_resultado

# Descargar recursos de NLTK
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuración de la página
st.set_page_config(
    page_title="FinanzBot - Análisis Empresarial",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuraciones CSS personalizadas con colores mejorados
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #303030;
        margin-bottom: 1.5rem;
    }
    .chat-message-user {
        background-color: #3F51B5;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 5px solid #1A237E;
        color: #FFFFFF;
        font-weight: 500;
    }
    .chat-message-bot {
        background-color: #1565C0;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 5px solid #0D47A1;
        color: #FFFFFF;
        font-weight: 500;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #263238;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: #ECEFF1;
    }
    .highlight-metric {
        background-color: #37474F;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-weight: bold;
        color: #ECEFF1;
    }
    .estado-excelente {
        background-color: #2E7D32;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .estado-bueno {
        background-color: #0277BD;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .estado-regular {
        background-color: #EF6C00;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .estado-critico {
        background-color: #C62828;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .recomendacion {
        background-color: #455A64;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #FFC107;
        color: #ECEFF1;
        font-weight: 500;
    }
    .sidebar {
        background-color: #263238;
        color: #ECEFF1;
    }
    .stButton button {
        background-color: #1976D2;
        color: white;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #1565C0;
    }
    .stTextInput>div>div>input {
        background-color: #455A64;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #455A64;
        color: white;
    }
    .stNumberInput>div>div>input {
        background-color: #455A64;
        color: white;
    }
    .info-box {
        background-color: #1976D2;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #90CAF9;
    }
    .btn-navigation {
        background-color: #1976D2;
        color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        margin: 5px;
        font-weight: bold;
        cursor: pointer;
    }
    .custom-metric {
        background-color: #37474F;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-label {
        font-weight: bold;
        font-size: 1.2rem;
        color: #90CAF9;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
        margin: 10px 0;
    }
    .metric-status-good {
        color: #66BB6A;
        font-weight: bold;
    }
    .metric-status-bad {
        color: #EF5350;
        font-weight: bold;
    }
    .chat-input {
        background-color: #455A64;
        border: none;
        color: white;
        padding: 10px;
        border-radius: 5px;
        width: 100%;
        margin-top: 10px;
    }
    .strong-text {
        font-weight: bold;
        color: #4FC3F7;
    }
</style>
""", unsafe_allow_html=True)

# Función para mostrar animación de "pensando"
def mostrar_procesamiento():
    with st.spinner('Analizando datos financieros y aplicando técnicas de NLP...'):
        # Simulación de procesamiento
        progress_bar = st.progress(0)
        for i in range(100):
            # Actualizar barra de progreso
            progress_bar.progress(i + 1)
            # Pequeña pausa para simular procesamiento
            time.sleep(0.01)
        st.success('¡Análisis completado!')

# Función mejorada para respuestas del chatbot
def chatbot_response(mensaje, datos_empresa=None):
    """
    Genera respuestas del chatbot basadas en el mensaje del usuario y los datos de la empresa.
    
    Args:
        mensaje (str): Mensaje del usuario
        datos_empresa (dict, optional): Datos de la empresa
        
    Returns:
        str: Respuesta del chatbot
    """
    # Aplicar NLP al mensaje
    tokens = tokenizar_texto(mensaje)
    lemas = lematizar_texto(mensaje)
    pos_tags = pos_tagging(mensaje)
    keywords = extraer_keywords(mensaje, 3)
    
    # Mensajes predefinidos para diferentes situaciones
    mensajes_predefinidos = {
        'saludo': [
            "👋 ¡Hola! Soy FinanzBot, tu asistente especializado en análisis financiero empresarial. ¿En qué puedo ayudarte hoy?",
            "¡Saludos! Estoy aquí para ayudarte a entender mejor la situación financiera de tu empresa. ¿Qué te gustaría saber?",
            "Hola, soy tu asistente de análisis económico. Puedo ayudarte a interpretar tus indicadores financieros y darte recomendaciones personalizadas."
        ],
        'agradecimiento': [
            "¡Es un placer ayudarte! El análisis financiero es mi especialidad. ¿Hay algo más que quieras saber?",
            "No hay de qué. Recuerda que puedo explicarte cualquier indicador financiero de tu empresa con más detalle.",
            "¡De nada! Si tienes más preguntas sobre la salud financiera de tu empresa, no dudes en consultarme."
        ],
        'despedida': [
            "¡Hasta pronto! Recuerda revisar periódicamente tus indicadores financieros para mantener el control de tu empresa.",
            "Adiós. No olvides implementar las recomendaciones para mejorar la salud financiera de tu negocio. ¡Éxito!",
            "Que tengas un excelente día. Estaré aquí cuando necesites más análisis o interpretaciones de tus datos financieros."
        ],
        'endeudamiento': [
            "El ratio de endeudamiento muestra qué proporción de tus activos está financiada por deuda. Un valor menor generalmente indica una situación más sólida, aunque depende del sector.",
            "Para mejorar tu ratio de endeudamiento, podrías: 1) Aumentar el capital social, 2) Reinvertir beneficios, 3) Vender activos no productivos para reducir deuda, o 4) Renegociar plazos de pago.",
            "Es importante comparar tu ratio de endeudamiento con empresas similares del sector. Cada industria tiene sus particularidades y lo que es alto en un sector puede ser normal en otro."
        ],
        'rentabilidad': [
            "La rentabilidad sobre activos (ROA) indica cuánto beneficio generas por cada peso invertido en activos. Un ROA más alto significa que estás aprovechando mejor tus recursos.",
            "Para mejorar tu rentabilidad podrías: 1) Aumentar precios si el mercado lo permite, 2) Reducir costos operativos, 3) Optimizar la gestión de inventarios, o 4) Deshacerte de activos poco productivos.",
            "Tu ROA debe compararse con la media del sector. Si está por debajo, podría ser momento de replantearse la estrategia de negocio o buscar nuevas oportunidades de mercado."
        ],
        'productividad': [
            "La productividad por empleado muestra cuánto genera cada trabajador en términos de ingresos. Es un indicador clave de la eficiencia operativa.",
            "Para mejorar la productividad podrías: 1) Invertir en capacitación, 2) Mejorar procesos y tecnología, 3) Implementar sistemas de incentivos basados en resultados, o 4) Revisar la distribución de tareas.",
            "Una baja productividad puede indicar exceso de personal, falta de tecnología adecuada, o procesos ineficientes. Análisis más profundos te ayudarán a identificar los cuellos de botella."
        ],
        'cartera': [
            "La rotación de cartera indica cuántos días tardas en cobrar tus ventas a crédito. Una rotación más baja es generalmente mejor, ya que mejora tu liquidez.",
            "Para mejorar tu rotación de cartera, considera: 1) Revisar políticas de crédito, 2) Implementar descuentos por pronto pago, 3) Mejorar el seguimiento de cobros, o 4) Evaluar factoring para cuentas problemáticas.",
            "Una cartera que rota lentamente puede generar problemas de liquidez. Es importante balancear las políticas de crédito para no perder clientes pero tampoco arriesgar tu flujo de caja."
        ],
        'liquidez': [
            "La liquidez se refiere a la capacidad de tu empresa para cumplir con sus obligaciones a corto plazo. Con los datos proporcionados, puedo hacer una estimación básica.",
            "Un buen ratio de liquidez suele estar entre 1.5 y 2.0, indicando que puedes cubrir tus deudas a corto plazo sin problemas.",
            "Si tienes problemas de liquidez, podrías: 1) Mejorar la gestión de cobros, 2) Renegociar plazos con proveedores, 3) Establecer líneas de crédito, o 4) Revisar tu ciclo de conversión de efectivo."
        ],
        'general': [
            "Basándome en los datos proporcionados, puedo analizar varios aspectos financieros de tu empresa. ¿Hay algún indicador específico que te interese conocer más a fondo?",
            "¿Sabías que el análisis financiero debe ser periódico? Te recomiendo revisar estos indicadores al menos trimestralmente para detectar tendencias y actuar a tiempo.",
            "Recuerda que cada sector tiene sus propios estándares para los indicadores financieros. Lo importante es identificar tendencias y compararte con empresas similares."
        ],
        'ayuda': [
            "Puedo ayudarte a entender cualquier indicador de tu empresa. Puedes preguntarme sobre endeudamiento, rentabilidad, productividad o rotación de cartera.",
            "Estoy aquí para resolver tus dudas financieras. ¿Te gustaría saber más sobre algún aspecto específico de tu análisis?",
            "Puedes hacerme preguntas sobre la situación financiera de tu empresa o pedirme recomendaciones para mejorar cualquiera de tus indicadores."
        ],
        'recomendaciones': [
            "Basándome en tu análisis, te recomendaría enfocarte primero en las áreas con indicadores más débiles.",
            "Una estrategia efectiva es priorizar las mejoras en los indicadores críticos y luego pasar a optimizar los que ya están en buen estado.",
            "Recuerda que las mejoras financieras suelen requerir tiempo. Establece metas a corto y mediano plazo para ver cambios significativos."
        ],
        'comparacion': [
            "Comparar tus indicadores con los estándares del sector es clave para entender tu posición competitiva.",
            "Cada sector tiene particularidades que afectan sus indicadores financieros. Lo importante es estar por encima del promedio de tu industria.",
            "La comparación temporal de tus propios indicadores también es valiosa. Observa tendencias y cambios para evaluar el impacto de tus decisiones."
        ]
    }
    
    # Verificar el tipo de mensaje usando NLP mejorado
    # Usar más patrones y términos para mejorar la detección
    if any(palabra in mensaje.lower() for palabra in ['hola', 'buenos', 'saludos', 'que tal', 'hi', 'hello']):
        categoria = 'saludo'
    elif any(palabra in mensaje.lower() for palabra in ['gracias', 'agradecido', 'agradezco', 'thank', 'thanks', 'te agradezco']):
        categoria = 'agradecimiento'
    elif any(palabra in mensaje.lower() for palabra in ['adios', 'chao', 'hasta luego', 'nos vemos', 'bye', 'hasta pronto']):
        categoria = 'despedida'
    elif any(palabra in mensaje.lower() for palabra in ['ayuda', 'help', 'como', 'cómo', 'que puedes hacer', 'qué haces']):
        categoria = 'ayuda'
    elif any(palabra in mensaje.lower() for palabra in ['recomienda', 'recomendación', 'recomendaciones', 'consejos', 'sugerencias', 'mejorar']):
        categoria = 'recomendaciones'
    elif any(palabra in mensaje.lower() for palabra in ['comparar', 'comparación', 'sector', 'industria', 'promedio', 'media', 'estándar']):
        categoria = 'comparacion'
    elif any(palabra in mensaje.lower() for palabra in ['deuda', 'endeudamiento', 'pasivo', 'prestamo', 'financiacion', 'obligaciones']):
        categoria = 'endeudamiento'
    elif any(palabra in mensaje.lower() for palabra in ['rentabilidad', 'ganancia', 'beneficio', 'rendimiento', 'roa', 'margen', 'utilidad']):
        categoria = 'rentabilidad'
    elif any(palabra in mensaje.lower() for palabra in ['productividad', 'eficiencia', 'empleado', 'trabajador', 'personal', 'rendimiento']):
        categoria = 'productividad'
    elif any(palabra in mensaje.lower() for palabra in ['cartera', 'cobrar', 'credito', 'rotacion', 'cliente', 'factura', 'cobros']):
        categoria = 'cartera'
    elif any(palabra in mensaje.lower() for palabra in ['liquidez', 'efectivo', 'caja', 'corriente', 'solvencia', 'flujo']):
        categoria = 'liquidez'
    else:
        # Análisis más avanzado basado en similitud semántica
        temas = {
            'endeudamiento': "deudas financiación pasivos préstamos créditos obligaciones financieras endeudamiento apalancamiento préstamo",
            'rentabilidad': "beneficios ganancias rentabilidad margen utilidad rendimiento roa roi retorno inversión",
            'productividad': "empleados trabajadores personal productividad eficiencia rendimiento laboral desempeño equipo",
            'cartera': "cartera cobros créditos clientes facturas cuentas por cobrar rotación clientes pago",
            'liquidez': "liquidez efectivo caja flujo dinero solvencia corto plazo activos corrientes",
            'ayuda': "ayuda cómo funciona qué puedes hacer cómo te uso asistencia guía opciones",
            'recomendaciones': "recomendaciones consejos mejorar optimizar sugerencias acciones estrategias"
        }
        
        # Calcular similitud con cada tema
        mejores_similitudes = {}
        for tema, descripcion in temas.items():
            similitud = similaridad_textos(mensaje.lower(), descripcion)
            mejores_similitudes[tema] = similitud
        
        # Elegir el tema con mayor similitud si supera un umbral
        mejor_tema = max(mejores_similitudes.items(), key=lambda x: x[1])
        if mejor_tema[1] > 0.1:  # Umbral de similitud
            categoria = mejor_tema[0]
        else:
            categoria = 'general'
    
    # Si hay datos de empresa, personalizar respuesta
    if datos_empresa and 'resultados' in datos_empresa:
        resultados = datos_empresa['resultados']
        
        if categoria == 'endeudamiento':
            ratio = resultados['indicadores']['ratio_endeudamiento']
            evaluacion = resultados['evaluacion']['endeudamiento']
            
            # Respuesta detallada y personalizada
            respuesta = f"📊 **Análisis de Endeudamiento**\n\nTu ratio de endeudamiento es **{ratio:.2f}**, lo cual es considerado **{evaluacion}** para el sector {resultados['sector']}.\n\n"
            
            # Añadir interpretación según el valor
            if ratio < 0.4:
                respuesta += "Este valor indica un bajo nivel de endeudamiento, lo que es positivo para la estabilidad financiera, pero podría estar perdiendo oportunidades de apalancamiento para crecer más rápido.\n\n"
            elif ratio < 0.6:
                respuesta += "Este valor muestra un endeudamiento moderado y saludable, un buen balance entre capital propio y ajeno.\n\n"
            else:
                respuesta += "Este nivel de endeudamiento es elevado, lo que podría aumentar el riesgo financiero y dificultar el acceso a nuevo financiamiento.\n\n"
            
            # Recomendaciones personalizadas
            if ratio > 0.6:
                respuesta += "**Recomendaciones para reducir tu endeudamiento:**\n"
                respuesta += "1. Considera una ampliación de capital\n"
                respuesta += "2. Implementa un plan de reducción de deuda priorizando las de mayor costo\n"
                respuesta += "3. Evalúa la venta de activos no estratégicos\n"
                respuesta += "4. Negocia mejores condiciones con tus acreedores\n\n"
            
            return respuesta
            
        elif categoria == 'rentabilidad':
            rent = resultados['indicadores']['rentabilidad']
            evaluacion = resultados['evaluacion']['rentabilidad']
            
            # Respuesta detallada
            respuesta = f"💰 **Análisis de Rentabilidad**\n\nTu rentabilidad sobre activos (ROA) es **{rent:.2%}**, lo cual es considerada **{evaluacion}** para el sector {resultados['sector']}.\n\n"
            
            # Interpretación personalizada
            if rent < 0.05:
                respuesta += "Esta rentabilidad es baja. Cada $100 invertidos en activos están generando menos de $5 de beneficio, lo que sugiere revisar la eficiencia operativa y la estructura de costos.\n\n"
                respuesta += "**Estrategias para mejorar tu rentabilidad:**\n"
                respuesta += "1. Analiza tu estructura de costos para identificar ineficiencias\n"
                respuesta += "2. Revisa tu política de precios; considera aumentos selectivos\n"
                respuesta += "3. Optimiza el uso de activos o elimina los improductivos\n"
                respuesta += "4. Evalúa la posibilidad de subcontratar procesos no estratégicos\n\n"
            elif rent < 0.15:
                respuesta += "Esta rentabilidad es moderada. Tus activos están produciendo un retorno razonable, aunque siempre hay espacio para mejorar.\n\n"
                respuesta += "**Para optimizar aún más tu rentabilidad:**\n"
                respuesta += "1. Identifica tus líneas de negocio más rentables y poténcialas\n"
                respuesta += "2. Evalúa la implementación de tecnología para aumentar eficiencia\n"
                respuesta += "3. Considera estrategias de venta cruzada con tus clientes actuales\n\n"
            else:
                respuesta += "¡Excelente rentabilidad! Tus activos están siendo muy productivos generando un alto retorno sobre la inversión.\n\n"
                respuesta += "**Para mantener esta excelente rentabilidad:**\n"
                respuesta += "1. Analiza qué factores están contribuyendo más a este éxito\n"
                respuesta += "2. Considera reinvertir parte de los beneficios en innovación\n"
                respuesta += "3. Documenta tus procesos exitosos como mejores prácticas\n\n"
            
            return respuesta
            
        elif categoria == 'productividad':
            productividad = resultados['indicadores']['productividad']
            evaluacion = resultados['evaluacion']['productividad']
            
            # Respuesta detallada
            respuesta = f"👥 **Análisis de Productividad por Empleado**\n\nLa productividad por empleado es **${productividad:,.0f} COP**, lo cual es considerada **{evaluacion}** para el sector {resultados['sector']}.\n\n"
            
            # Interpretación personalizada según sector
            sector_limites = {
                'Tecnología': 100000000,
                'Comercio': 50000000,
                'Manufactura': 70000000,
                'Servicios': 60000000
            }
            
            limite = sector_limites.get(resultados['sector'], 60000000)
            
            if productividad < limite * 0.7:
                respuesta += "Esta productividad está por debajo del estándar del sector. Podría ser conveniente revisar procesos, capacitación y tecnología disponible para los empleados.\n\n"
                respuesta += "**Acciones concretas para mejorar la productividad:**\n"
                respuesta += "1. Implementa programas de capacitación enfocados en habilidades clave\n"
                respuesta += "2. Evalúa si el personal cuenta con las herramientas adecuadas\n"
                respuesta += "3. Revisa la distribución de responsabilidades y elimina duplicidades\n"
                respuesta += "4. Considera implementar un sistema de incentivos por resultados\n\n"
            elif productividad < limite * 1.2:
                respuesta += "Esta productividad está alineada con los estándares del sector, mostrando una operación eficiente.\n\n"
                respuesta += "**Para seguir mejorando:**\n"
                respuesta += "1. Investiga mejores prácticas de empresas líderes en tu sector\n"
                respuesta += "2. Fomenta la innovación y mejora continua entre tus empleados\n"
                respuesta += "3. Considera automatizar tareas repetitivas para liberar tiempo creativo\n\n"
            else:
                respuesta += "¡Excelente productividad! Tus empleados generan un valor significativamente superior al promedio del sector.\n\n"
                respuesta += "**Para mantener este alto rendimiento:**\n"
                respuesta += "1. Documenta y comparte las prácticas exitosas dentro de la organización\n"
                respuesta += "2. Implementa programas de retención para tu talento clave\n"
                respuesta += "3. Considera mentorías internas para transferir conocimientos\n\n"
            
            return respuesta
            
        elif categoria == 'cartera':
            rotacion = resultados['indicadores']['rotacion_cartera']
            evaluacion = resultados['evaluacion']['rotacion']
            
            # Respuesta detallada
            respuesta = f"📅 **Análisis de Rotación de Cartera**\n\nTu rotación de cartera es de **{rotacion:.1f} días**, lo cual es considerado **{evaluacion}** para el sector {resultados['sector']}.\n\n"
            
            # Interpretación personalizada
            if rotacion < 30:
                respuesta += "¡Excelente gestión de cobros! Tu empresa recupera el dinero rápidamente, lo que favorece la liquidez.\n\n"
                respuesta += "**Para mantener esta eficiencia:**\n"
                respuesta += "1. Documenta tus procesos actuales de cobro como mejores prácticas\n"
                respuesta += "2. Considera ofrecer más opciones de pago electrónico para facilitar aún más los cobros\n"
                respuesta += "3. Evalúa si tus políticas de crédito son demasiado restrictivas y podrían estar limitando ventas\n\n"
            elif rotacion < 60:
                respuesta += "Tu gestión de cobros es adecuada. Mantiene un buen equilibrio entre política comercial y necesidades de liquidez.\n\n"
                respuesta += "**Para optimizar aún más:**\n"
                respuesta += "1. Segmenta a tus clientes según su comportamiento de pago\n"
                respuesta += "2. Considera implementar descuentos por pronto pago en segmentos específicos\n"
                respuesta += "3. Automatiza recordatorios de cobro para aumentar eficiencia\n\n"
            else:
                respuesta += "Tu periodo de cobro es extenso, lo que podría estar afectando tu flujo de caja. Considera revisar tus políticas de crédito y cobranza.\n\n"
                respuesta += "**Plan de acción para mejorar la rotación:**\n"
                respuesta += "1. Revisa y ajusta tus condiciones de crédito (plazos, requisitos)\n"
                respuesta += "2. Implementa un sistema de seguimiento de facturas vencidas\n"
                respuesta += "3. Capacita a tu equipo de ventas en técnicas de negociación de pagos\n"
                respuesta += "4. Evalúa servicios de factoring para clientes grandes o recurrentes\n"
                respuesta += "5. Considera ofrecer descuentos significativos por pago anticipado\n\n"
            
            return respuesta
            
        elif categoria == 'liquidez':
            # Estimación aproximada de liquidez basada en cartera y deudas
            cartera = resultados['datos']['cartera']
            deudas = resultados['datos']['deudas']
            
            # Cálculo simplificado (solo con los datos disponibles)
            est_liquidez = cartera / deudas if deudas > 0 else "Alta"
            
            # Respuesta
            if isinstance(est_liquidez, str):
                respuesta = f"💧 **Estimación de Liquidez**\n\nBasado en los datos proporcionados, tu empresa parece tener una liquidez **{est_liquidez}**, ya que tus deudas son mínimas en comparación con tu cartera.\n\n"
                respuesta += "**Recomendaciones para gestionar el exceso de liquidez:**\n"
                respuesta += "1. Evalúa oportunidades de inversión para generar rendimientos adicionales\n"
                respuesta += "2. Considera expandir operaciones o líneas de negocio\n"
                respuesta += "3. Valora la posibilidad de reducir plazos de pago a proveedores para obtener descuentos\n\n"
            else:
                respuesta = f"💧 **Estimación de Liquidez**\n\nLa relación entre tu cartera y tus deudas es de **{est_liquidez:.2f}**.\n\n"
                
                if est_liquidez < 1:
                    respuesta += "Esto podría indicar problemas de liquidez a corto plazo, ya que tu cartera no cubriría todas tus deudas.\n\n"
                    respuesta += "**Acciones inmediatas para mejorar tu liquidez:**\n"
                    respuesta += "1. Implementa un sistema intensivo de cobro de cartera\n"
                    respuesta += "2. Negocia extensiones de plazo con proveedores\n"
                    respuesta += "3. Evalúa opciones de financiamiento a corto plazo\n"
                    respuesta += "4. Considera aplazar inversiones no esenciales\n\n"
                elif est_liquidez < 1.5:
                    respuesta += "Tu liquidez es ajustada pero manejable. Mantén un control cercano de tu flujo de caja.\n\n"
                    respuesta += "**Para mejorar tu posición de liquidez:**\n"
                    respuesta += "1. Implementa un sistema de pronóstico de flujo de caja semanal\n"
                    respuesta += "2. Revisa tus políticas de inventario para optimizar capital de trabajo\n"
                    respuesta += "3. Establece alertas tempranas para posibles problemas de liquidez\n\n"
                else:
                    respuesta += "Tu posición de liquidez parece sólida, con suficiente cartera para cubrir tus obligaciones.\n\n"
                    respuesta += "**Para optimizar tu gestión de liquidez:**\n"
                    respuesta += "1. Evalúa la posibilidad de inversiones temporales para excedentes\n"
                    respuesta += "2. Considera negociar descuentos con proveedores por pronto pago\n"
                    respuesta += "3. Revisa periódicamente la calidad de tu cartera (antigüedad, concentración)\n\n"
            
            return respuesta
            
        elif categoria == 'recomendaciones':
            # Priorizar recomendaciones basadas en los indicadores más débiles
            evaluaciones = resultados['evaluacion']
            indicadores_debiles = [k for k, v in evaluaciones.items() if v in ['alta', 'baja']]
            
            respuesta = f"🚀 **Recomendaciones Prioritarias para {resultados['nombre']}**\n\n"
            
            if not indicadores_debiles:
                respuesta += "¡Felicitaciones! Todos tus indicadores muestran un buen desempeño. Aquí hay algunas recomendaciones para mantener y mejorar tu excelente posición:\n\n"
                respuesta += "1. **Planificación estratégica**: Establece metas de crecimiento realistas para los próximos 3-5 años\n"
                respuesta += "2. **Innovación**: Destina recursos para investigación y desarrollo de nuevos productos o servicios\n"
                respuesta += "3. **Talento**: Implementa programas de retención y desarrollo para tu personal clave\n"
                respuesta += "4. **Digitalización**: Evalúa oportunidades para automatizar procesos y mejorar la experiencia del cliente\n"
                respuesta += "5. **Diversificación**: Considera expandirte a mercados relacionados para reducir riesgos\n\n"
            else:
                respuesta += f"Basado en el análisis, estas son las áreas prioritarias que requieren atención:\n\n"
                
                if 'endeudamiento' in indicadores_debiles:
                    respuesta += "**Para mejorar tu nivel de endeudamiento:**\n"
                    respuesta += "1. Elabora un plan de reducción de deuda priorizando las de mayor costo financiero\n"
                    respuesta += "2. Evalúa la posibilidad de convertir deuda a corto plazo en largo plazo\n"
                    respuesta += "3. Considera una ampliación de capital si es viable\n"
                    respuesta += "4. Optimiza la gestión de activos para reducir la necesidad de financiamiento\n\n"
                
                if 'rentabilidad' in indicadores_debiles:
                    respuesta += "**Para aumentar tu rentabilidad:**\n"
                    respuesta += "1. Analiza detalladamente tu estructura de costos para identificar ineficiencias\n"
                    respuesta += "2. Revisa tu estrategia de precios y considera aumentos selectivos\n"
                    respuesta += "3. Identifica y potencia las líneas de negocio más rentables\n"
                    respuesta += "4. Evalúa la posibilidad de discontinuar productos o servicios poco rentables\n\n"
                
                if 'productividad' in indicadores_debiles:
                    respuesta += "**Para mejorar la productividad de tus empleados:**\n"
                    respuesta += "1. Implementa programas de capacitación enfocados en habilidades clave\n"
                    respuesta += "2. Evalúa si tu personal cuenta con las herramientas tecnológicas adecuadas\n"
                    respuesta += "3. Considera implementar sistemas de incentivos basados en resultados\n"
                    respuesta += "4. Revisa los procesos actuales para eliminar actividades que no agregan valor\n\n"
                
                if 'rotacion' in indicadores_debiles:
                    respuesta += "**Para optimizar tu rotación de cartera:**\n"
                    respuesta += "1. Revisa y ajusta tus políticas de crédito y cobro\n"
                    respuesta += "2. Implementa un sistema de seguimiento y gestión de cuentas por cobrar\n"
                    respuesta += "3. Considera ofrecer descuentos por pronto pago\n"
                    respuesta += "4. Evalúa la posibilidad de utilizar factoring para cuentas de clientes grandes\n\n"
            
            respuesta += "¿Te gustaría que profundice en alguna de estas recomendaciones en particular?"
            return respuesta
        
        elif categoria == 'comparacion':
            # Datos para comparación sectorial
            sector = resultados['sector']
            
            # Valores estándar por sector (simplificados)
            estandares_sector = {
                'Tecnología': {
                    'endeudamiento': 0.6,
                    'rentabilidad': 0.15,
                    'productividad': 100000000,
                    'rotacion': 60
                },
                'Comercio': {
                    'endeudamiento': 0.5,
                    'rentabilidad': 0.08,
                    'productividad': 50000000,
                    'rotacion': 45
                },
                'Manufactura': {
                    'endeudamiento': 0.55,
                    'rentabilidad': 0.1,
                    'productividad': 70000000,
                    'rotacion': 50
                },
                'Servicios': {
                    'endeudamiento': 0.45,
                    'rentabilidad': 0.12,
                    'productividad': 60000000,
                    'rotacion': 30
                },
                'Otro': {
                    'endeudamiento': 0.5,
                    'rentabilidad': 0.1,
                    'productividad': 60000000,
                    'rotacion': 45
                }
            }
            
            # Usar el sector correcto o el valor por defecto
            estandar = estandares_sector.get(sector, estandares_sector['Otro'])
            
            # Preparar respuesta de comparación
            respuesta = f"📊 **Comparación con el Sector {sector}**\n\n"
            respuesta += "Aquí tienes cómo se comparan tus indicadores con los estándares del sector:\n\n"
            
            # Endeudamiento
            ratio = resultados['indicadores']['ratio_endeudamiento']
            respuesta += f"**Endeudamiento:**\n"
            respuesta += f"- Tu empresa: {ratio:.2f}\n"
            respuesta += f"- Promedio del sector: {estandar['endeudamiento']:.2f}\n"
            if ratio <= estandar['endeudamiento']:
                respuesta += f"- Situación: ✅ Mejor que el promedio (menor endeudamiento)\n\n"
            else:
                respuesta += f"- Situación: ⚠️ Mayor que el promedio del sector\n\n"
            
            # Rentabilidad
            rent = resultados['indicadores']['rentabilidad']
            respuesta += f"**Rentabilidad:**\n"
            respuesta += f"- Tu empresa: {rent:.2%}\n"
            respuesta += f"- Promedio del sector: {estandar['rentabilidad']:.2%}\n"
            if rent >= estandar['rentabilidad']:
                respuesta += f"- Situación: ✅ Mejor que el promedio\n\n"
            else:
                respuesta += f"- Situación: ⚠️ Por debajo del promedio del sector\n\n"
            
            # Productividad
            prod = resultados['indicadores']['productividad']
            respuesta += f"**Productividad por empleado:**\n"
            respuesta += f"- Tu empresa: ${prod:,.0f}\n"
            respuesta += f"- Promedio del sector: ${estandar['productividad']:,.0f}\n"
            if prod >= estandar['productividad']:
                respuesta += f"- Situación: ✅ Mejor que el promedio\n\n"
            else:
                respuesta += f"- Situación: ⚠️ Por debajo del promedio del sector\n\n"
            
            # Rotación de cartera
            rot = resultados['indicadores']['rotacion_cartera']
            respuesta += f"**Rotación de cartera:**\n"
            respuesta += f"- Tu empresa: {rot:.1f} días\n"
            respuesta += f"- Promedio del sector: {estandar['rotacion']} días\n"
            if rot <= estandar['rotacion']:
                respuesta += f"- Situación: ✅ Mejor que el promedio (menor tiempo de cobro)\n\n"
            else:
                respuesta += f"- Situación: ⚠️ Por encima del promedio del sector\n\n"
            
            respuesta += "Esta comparación te permite identificar fortalezas y oportunidades de mejora específicas respecto a tu sector. ¿Te gustaría que analicemos en profundidad alguno de estos indicadores?"
            
            return respuesta
    
    # Para otros tipos de mensajes, o cuando no hay datos de empresa
    # Usar análisis de texto basado en NLP para personalizar la respuesta
    
    # Verificar si el mensaje es una pregunta sobre definiciones financieras
    definiciones = {
        'endeudamiento': "El ratio de endeudamiento mide la proporción de activos financiados con deuda. Se calcula dividiendo el total de deudas entre el total de activos. Un ratio menor indica menor dependencia de financiamiento externo.",
        'rentabilidad': "La rentabilidad sobre activos (ROA) mide la eficiencia con la que una empresa utiliza sus activos para generar beneficios. Se calcula dividiendo las ganancias anuales entre el total de activos.",
        'productividad': "La productividad por empleado indica cuánto ingreso o beneficio genera cada trabajador en promedio. Se calcula dividiendo las ganancias anuales entre el número de empleados.",
        'rotacion': "La rotación de cartera mide el tiempo promedio que tarda una empresa en cobrar sus ventas a crédito. Se calcula multiplicando el valor en cartera por 365 días y dividiéndolo entre las ventas anuales.",
        'liquidez': "La liquidez se refiere a la capacidad de una empresa para cumplir con sus obligaciones financieras a corto plazo. Un buen nivel de liquidez garantiza que la empresa pueda operar sin contratiempos."
    }
    
    # Verificar si el mensaje pide una definición
    for termino, definicion in definiciones.items():
        if f"qué es {termino}" in mensaje.lower() or f"que es {termino}" in mensaje.lower() or f"definición de {termino}" in mensaje.lower() or f"definicion de {termino}" in mensaje.lower():
            return f"📘 **{termino.capitalize()}**\n\n{definicion}\n\n¿Te gustaría saber más sobre cómo interpretar este indicador o cómo mejorarlo?"
    
    # General: respuesta basada en análisis de texto
    # Usar POS tagging para identificar verbos y sustantivos clave
    verbos = [word for word, tag in pos_tags if tag == 'VERB']
    sustantivos = [word for word, tag in pos_tags if tag in ['NOUN', 'PROPN']]
    
    # Personalización básica basada en palabras extraídas
    if verbos and sustantivos:
        accion = verbos[0]
        tema = sustantivos[0] if sustantivos else "empresa"
        respuesta = f"Entiendo que quieres {accion} sobre {tema}. "
        respuesta += random.choice(mensajes_predefinidos[categoria])
        return respuesta
    
    # Si no hay suficiente contexto para personalizar
    return random.choice(mensajes_predefinidos[categoria])

# Variables para almacenar datos de la empresa y resultados
if 'datos_empresa' not in st.session_state:
    st.session_state.datos_empresa = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'page_view' not in st.session_state:
    st.session_state.page_view = "chat"  # Opciones: "chat", "form", "results"

# Sidebar con información y navegación
with st.sidebar:
    st.markdown('<div class="main-title">FinanzBot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Asistente Financiero</div>', unsafe_allow_html=True)
    
    # Opciones de navegación
    st.markdown("### Navegación")
    
    # Botones para navegar entre vistas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page_view = "chat"
    
    with col2:
        if st.button("📝 Formulario", use_container_width=True):
            st.session_state.page_view = "form"
    
    with col3:
        if st.button("📊 Resultados", use_container_width=True):
            st.session_state.page_view = "results"
    
    st.markdown("---")
    
    # Información sobre la aplicación
    st.markdown("### ℹ️ Acerca de")
    st.info("""
    Esta aplicación utiliza técnicas de NLP para analizar la salud económica de empresas:
    
    * **Tokenización**: Divide el texto en unidades individuales
    * **Lematización**: Reduce palabras a su forma base
    * **POS Tagging**: Etiquetado gramatical
    * **Embedding**: Representación vectorial del texto
    
    Los indicadores financieros evaluados son:
    * Ratio de endeudamiento
    * Rentabilidad sobre activos
    * Productividad por empleado
    * Rotación de cartera
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado con ❤️ usando Streamlit y Python")

# Vista Principal (cambia según la selección)
if st.session_state.page_view == "chat":
    st.markdown('<div class="main-title">💬 Chat con FinanzBot</div>', unsafe_allow_html=True)
    
    # Si no hay datos de empresa, mostrar mensaje
    if not st.session_state.datos_empresa:
        st.info("Para obtener respuestas personalizadas sobre tu empresa, completa primero el formulario de datos en la sección 'Formulario'.")
    
    # Contenedor para el historial de chat
    chat_container = st.container()
    
    # Mostrar historial de chat con estilos mejorados
    with chat_container:
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f'<div class="chat-message-user"><strong>Tú:</strong> {message}</div>', unsafe_allow_html=True)
            else:
                # Convertir marcadores markdown en el mensaje
                # Procesar negritas
                message = re.sub(r'\*\*(.*?)\*\*', r'<span class="strong-text">\1</span>', message)
                # Procesar itálicas
                message = re.sub(r'\*(.*?)\*', r'<em>\1</em>', message)
                
                st.markdown(f'<div class="chat-message-bot"><strong>FinanzBot:</strong> {message}</div>', unsafe_allow_html=True)
    
    # Input para chatbot
    mensaje_usuario = st.chat_input("Escribe tu pregunta aquí...")
    
    if mensaje_usuario:
        # Agregar mensaje del usuario al historial
        st.session_state.chat_history.append(("user", mensaje_usuario))
        
        # Generar respuesta del chatbot
        respuesta = chatbot_response(mensaje_usuario, st.session_state.datos_empresa)
        
        # Agregar respuesta al historial
        st.session_state.chat_history.append(("bot", respuesta))
        
        # Actualizar chat
        st.rerun()

elif st.session_state.page_view == "form":
    st.markdown('<div class="main-title">📝 Datos de la Empresa</div>', unsafe_allow_html=True)
    st.markdown("Complete el siguiente formulario para obtener un análisis financiero detallado de su empresa.")
    
    with st.form("formulario_empresa"):
        st.markdown("### Información General")
        nombre_empresa = st.text_input("Nombre de la Empresa")
        sector = st.selectbox(
            "Sector",
            ["Tecnología", "Comercio", "Manufactura", "Servicios", "Otro"]
        )
        
        st.markdown("### Información Financiera")
        # Crear dos columnas para los inputs numéricos
        col1_form, col2_form = st.columns(2)
        
        with col1_form:
            ganancias = st.number_input("Ganancias Anuales (COP)", min_value=0.0, format="%f")
            num_empleados = st.number_input("Número de Empleados", min_value=1, step=1)
            valor_cartera = st.number_input("Valor en Cartera (COP)", min_value=0.0, format="%f")
        
        with col2_form:
            valor_activos = st.number_input("Valor en Activos (COP)", min_value=0.0, format="%f")
            valor_deudas = st.number_input("Valor Deudas (COP)", min_value=0.0, format="%f")
        
        submitted = st.form_submit_button("Analizar Empresa")
        
        if submitted and nombre_empresa:
            # Preparar datos
            datos = {
                'nombre': nombre_empresa,
                'ganancias': ganancias,
                'sector': sector,
                'empleados': num_empleados,
                'activos': valor_activos,
                'cartera': valor_cartera,
                'deudas': valor_deudas
            }
            
            # Mostrar animación de procesamiento
            mostrar_procesamiento()
            
            # Realizar análisis
            resultados = analizar_empresa(datos)
            mensaje = generar_mensaje_resultado(resultados)
            
            # Guardar datos y resultados en sesión
            st.session_state.datos_empresa = {
                'datos': datos,
                'resultados': resultados,
                'mensaje': mensaje
            }
            
            # Agregar mensaje de bienvenida al chat si es el primer análisis
            if not st.session_state.chat_history:
                bienvenida = f"¡Hola! He analizado los datos de {nombre_empresa}. Puedes preguntarme sobre cualquier aspecto del análisis, como endeudamiento, rentabilidad, productividad o rotación de cartera."
                st.session_state.chat_history.append(("bot", bienvenida))
            
            # Cambiar a vista de resultados
            st.session_state.page_view = "results"
            st.rerun()

elif st.session_state.page_view == "results":
    if not st.session_state.datos_empresa:
        st.warning("No hay datos de empresa para mostrar. Por favor, completa el formulario primero.")
        st.session_state.page_view = "form"
        st.rerun()
    else:
        resultados = st.session_state.datos_empresa['resultados']
        
        st.markdown('<div class="main-title">📊 Análisis Financiero</div>', unsafe_allow_html=True)
        st.markdown(f"<div class='sub-title'>Empresa: {resultados['nombre']} | Sector: {resultados['sector']}</div>", unsafe_allow_html=True)
        
        # Estado general con estilo mejorado
        estado = resultados['estado_general']
        st.markdown(f"<div class='estado-{estado.lower()}'><h2>Estado: {estado}</h2></div>", unsafe_allow_html=True)
        
        # Contenedor para los indicadores principales
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Indicadores Clave")
        
        # Crear 4 columnas para mostrar los indicadores principales con estilo mejorado
        col1_ind, col2_ind, col3_ind, col4_ind = st.columns(4)
        
        with col1_ind:
            ratio = resultados['indicadores']['ratio_endeudamiento']
            eval_ind = resultados['evaluacion']['endeudamiento']
            status_class = "metric-status-good" if eval_ind == "bueno" else "metric-status-bad"
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-label">⚖️ Endeudamiento</div>
                <div class="metric-value">{ratio:.2f}</div>
                <div class="{status_class}">({eval_ind})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2_ind:
            rent = resultados['indicadores']['rentabilidad']
            eval_ind = resultados['evaluacion']['rentabilidad']
            status_class = "metric-status-good" if eval_ind == "buena" else "metric-status-bad"
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-label">💰 Rentabilidad</div>
                <div class="metric-value">{rent:.2%}</div>
                <div class="{status_class}">({eval_ind})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3_ind:
            prod = resultados['indicadores']['productividad']
            eval_ind = resultados['evaluacion']['productividad']
            status_class = "metric-status-good" if eval_ind == "buena" else "metric-status-bad"
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-label">👥 Productividad</div>
                <div class="metric-value">${prod:,.0f}</div>
                <div class="{status_class}">({eval_ind})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4_ind:
            rot = resultados['indicadores']['rotacion_cartera']
            eval_ind = resultados['evaluacion']['rotacion']
            status_class = "metric-status-good" if eval_ind == "buena" else "metric-status-bad"
            st.markdown(f"""
            <div class="custom-metric">
                <div class="metric-label">📅 Rotación</div>
                <div class="metric-value">{rot:.1f} días</div>
                <div class="{status_class}">({eval_ind})</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recomendaciones
        if resultados['recomendaciones']:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 💡 Recomendaciones")
            for rec in resultados['recomendaciones']:
                st.markdown(f"<div class='recomendacion'>{rec}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Gráfico de indicadores
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Visualización de Indicadores")
        
        # Crear datos para el gráfico de radar
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), facecolor='#263238')
        
        # Normalizar valores para el gráfico
        endeudamiento_norm = 1 - min(1, resultados['indicadores']['ratio_endeudamiento'] / 1.0)
        rentabilidad_norm = min(1, resultados['indicadores']['rentabilidad'] / 0.3)
        
        # Para productividad, normalizar según sector
        sector_limites = {
            'Tecnología': 100000000,
            'Comercio': 50000000,
            'Manufactura': 70000000,
            'Servicios': 60000000,
            'Otro': 60000000
        }
        
        limite_prod = sector_limites.get(resultados['sector'], 60000000)
        productividad_norm = min(1, resultados['indicadores']['productividad'] / limite_prod)
        
        # Para rotación, menor es mejor (normalizar de forma inversa)
        rotacion_norm = 1 - min(1, resultados['indicadores']['rotacion_cartera'] / 90)
        
        # Datos para el gráfico
        categorias = ['Endeudamiento', 'Rentabilidad', 'Productividad', 'Rot. Cartera']
        valores = [endeudamiento_norm, rentabilidad_norm, productividad_norm, rotacion_norm]
        
        # Completar el círculo repitiendo el primer valor
        valores += [valores[0]]
        categorias += [categorias[0]]
        
        # Ángulos para cada eje
        angulos = [n / float(len(categorias)-1) * 2 * 3.14159 for n in range(len(categorias))]
        
        # Configurar colores y estilos para el tema oscuro
        plt.rcParams.update({
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
        })
        
        # Dibujar los ejes y el gráfico
        ax.plot(angulos, valores, linewidth=2, linestyle='solid', color='#29B6F6')
        ax.fill(angulos, valores, alpha=0.4, color='#29B6F6')
        
        # Agregar etiquetas con mejor visibilidad
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias[:-1], color='white', fontweight='bold')
        
        # Agregar título con mejor visibilidad
        plt.title('Perfil Económico de la Empresa', size=15, y=1.1, color='white', fontweight='bold')
        
        # Configurar fondo y grid para mejor visibilidad
        ax.set_facecolor('#37474F')
        ax.grid(color='#90A4AE', linestyle='-', linewidth=0.5, alpha=0.5)
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Análisis NLP
        with st.expander("🧠 Detalles del Procesamiento NLP"):
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Tokenización", "Lematización", "POS Tagging", "Embedding"])
            
            with tab1:
                st.markdown("#### 🔍 Tokenización")
                st.markdown("Proceso de dividir el texto en unidades individuales (tokens):")
                st.code(str(resultados['nlp_ejemplo']['tokens']))
            
            with tab2:
                st.markdown("#### 📝 Lematización")
                st.markdown("Proceso de reducir palabras a su forma base (lema):")
                st.code(str(resultados['nlp_ejemplo']['lemas']))
            
            with tab3:
                st.markdown("#### 🏷️ POS Tagging")
                st.markdown("Etiquetado gramatical (Part-of-Speech):")
                st.code(str(resultados['nlp_ejemplo']['pos_tags']))
            
            with tab4:
                st.markdown("#### 🧮 Embedding")
                st.markdown("Representación vectorial del texto:")
                st.code(f"Dimensión del embedding: {resultados['nlp_ejemplo']['embedding_dim']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Botón para ir al chat
        if st.button("💬 Consultar al Chatbot", use_container_width=True):
            st.session_state.page_view = "chat"
            st.rerun()

# Código principal
if __name__ == "__main__":
    # Código para ejecutar la aplicación
    pass