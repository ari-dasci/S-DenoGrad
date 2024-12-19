# List of tabular datasets and how to import them

In order to use this datsets, the instalation of icimlrepo is necesary:

```
pip install ucimlrepo  
```

## RT-IoT2022

**English:**

The RT-IoT2022, a proprietary dataset derived from a real-time IoT infrastructure, is introduced as a comprehensive resource integrating a diverse range of IoT devices and sophisticated network attack methodologies. This dataset encompasses both normal and adversarial network behaviours, providing a general representation of real-world scenarios. Incorporating data from IoT devices such as ThingSpeak-LED, Wipro-Bulb, and MQTT-Temp, as well as simulated attack scenarios involving Brute-Force SSH attacks, DDoS attacks using Hping and Slowloris, and Nmap patterns, RT-IoT2022 offers a detailed perspective on the complex nature of network traffic. The bidirectional attributes of network traffic are meticulously captured using the Zeek network monitoring tool and the Flowmeter plugin. Researchers can leverage the RT-IoT2022 dataset to advance the capabilities of Intrusion Detection Systems (IDS), fostering the development of robust and adaptive security solutions for real-time IoT networks.

**Español:**

El RT-IoT2022, un conjunto de datos propietario derivado de una infraestructura IoT en tiempo real, se presenta como un recurso integral que integra una amplia variedad de dispositivos IoT y metodologías sofisticadas de ataques a redes. Este conjunto de datos abarca tanto comportamientos normales como adversarios en la red, proporcionando una representación general de escenarios del mundo real. Al incorporar datos de dispositivos IoT como ThingSpeak-LED, Wipro-Bulb y MQTT-Temp, así como escenarios simulados de ataques, incluyendo ataques de fuerza bruta a SSH, ataques DDoS utilizando Hping y Slowloris, y patrones de Nmap, el RT-IoT2022 ofrece una perspectiva detallada sobre la complejidad del tráfico de red. Los atributos bidireccionales del tráfico de red se capturan meticulosamente utilizando la herramienta de monitoreo de red Zeek y el complemento Flowmeter. Los investigadores pueden aprovechar el conjunto de datos RT-IoT2022 para avanzar en las capacidades de los Sistemas de Detección de Intrusos (IDS), fomentando el desarrollo de soluciones de seguridad robustas y adaptativas para redes IoT en tiempo real.

```python
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
rt_iot2022 = fetch_ucirepo(id=942) 
  
# data (as pandas dataframes) 
X = rt_iot2022.data.features 
y = rt_iot2022.data.targets 
  
# metadata 
print(rt_iot2022.metadata) 
  
# variable information 
print(rt_iot2022.variables) 
```

## SUPPORT2

**English:**

This dataset comprises 9105 individual critically ill patients across 5 United States medical centers, accessioned throughout 1989-1991 and 1992-1994. Each row concerns hospitalized patient records who met the inclusion and exclusion criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. The goal is to determine these patients' 2- and 6-month survival rates based on several physiologic, demographics, and disease severity information. It is an important problem because it addresses the growing national concern over patients' loss of control near the end of life. It enables earlier decisions and planning to reduce the frequency of a mechanical, painful, and prolonged dying process.

**Español:**

Este conjunto de datos incluye registros de 9,105 pacientes críticamente enfermos individuales, atendidos en cinco centros médicos de los Estados Unidos durante los periodos 1989-1991 y 1992-1994. Cada fila corresponde a los registros hospitalarios de pacientes que cumplieron con los criterios de inclusión y exclusión para nueve categorías de enfermedades: insuficiencia respiratoria aguda, enfermedad pulmonar obstructiva crónica, insuficiencia cardíaca congestiva, enfermedad hepática, coma, cáncer de colon, cáncer de pulmón, fallo multiorgánico con malignidad y fallo multiorgánico con sepsis.

El objetivo es determinar las tasas de supervivencia a 2 y 6 meses de estos pacientes, basándose en información fisiológica, demográfica y de gravedad de la enfermedad. Este es un problema relevante, ya que aborda la creciente preocupación nacional sobre la pérdida de control de los pacientes al final de la vida. Además, permite tomar decisiones anticipadas y planificar para reducir la frecuencia de un proceso de muerte mecánico, doloroso y prolongado.

```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
support2 = fetch_ucirepo(id=880) 
  
# data (as pandas dataframes) 
X = support2.data.features 
y = support2.data.targets 
  
# metadata 
print(support2.metadata) 
  
# variable information 
print(support2.variables) 
```

## Parkinsons Telemonitoring

**English:**

Oxford Parkinson's Disease Telemonitoring Dataset. This dataset is composed of a range of biomedical voice measurements from 42 people with early-stage Parkinson's disease recruited to a six-month trial of a telemonitoring device for remote symptom progression monitoring. The recordings were automatically captured in the patient's homes.


**Español:**

Conjunto de Datos de Telemonitoreo de la Enfermedad de Parkinson de Oxford. Este conjunto de datos está compuesto por una variedad de mediciones biomédicas de la voz provenientes de 42 personas con enfermedad de Parkinson en etapa temprana, reclutadas para un ensayo de seis meses de un dispositivo de telemonitoreo diseñado para el seguimiento remoto de la progresión de los síntomas. Las grabaciones se capturaron automáticamente en los hogares de los pacientes.

```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
parkinsons_telemonitoring = fetch_ucirepo(id=189) 
  
# data (as pandas dataframes) 
X = parkinsons_telemonitoring.data.features 
y = parkinsons_telemonitoring.data.targets 
  
# metadata 
print(parkinsons_telemonitoring.metadata) 
  
# variable information 
print(parkinsons_telemonitoring.variables)
```