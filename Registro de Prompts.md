# Registro de Prompts

**Modelo de Lenguaje Utilizado**: Google Gemini

## Prompt Utilizado 1

Actúa como un generador determinista de datos sintéticos educativos para el caso docente ficticio sobre riesgo de no continuidad estudiantil en educación superior.

Toma en cuenta las siguientes especificaciones obligatorias:

1. REQUISITO FUNDAMENTAL DE SALIDA:
   - Genera DIRECTAMENTE las filas del dataset sintético en texto plano con formato CSV (separado por comas), listo para ser copiado o guardado directamente en un archivo .csv.

2. REPRODUCIBILIDAD Y DETERMINISMO:
   - Asume una lógica de generación fija e invariable asociada a la SEMILLA = 42, asegurando que todos los registros generados sean consistentes y reproducibles.

3. ESTRUCTURA Y COLUMNAS OBLIGATORIAS:
   Genera exactamente 1000 filas de datos (muestra representativa de la cohorte de referencia) con las siguientes columnas exactas en la primera línea (header):
   id, mode, working_condition, attendance_percentage, academic_load, subjects_passed, previous_average, dropping_out_risk

   Especificaciones por variable:
   - `id`: Identificador sintético único ("EST-0001", "EST-0002", etc.).
   - `mode`: Categórica ("In Person" o "Hibrid").
   - `working_condition`: Categórica ("Works" o "Doesn't Work").
   - `attendance_percentage`: Numérico continuo entre 0.0 y 1.0 (con 4 decimales).
   - `academic_load`: Numérico entero entre 1 y 6 asignaturas.
   - `subjects_passed`: Numérico entero entre 0 y el valor de `academic_load`.
   - `previous_average`: Numérico continuo entre 1.0 y 7.0 (con 2 decimales).
   - `dropping_out_risk`: Variable binaria (0 = sin riesgo, 1 = en riesgo / requiere apoyo de orientación).

4. LÓGICA DE LA VARIABLE OBJETIVO (SIN FILTRACIÓN / TARGET LEAKAGE):
   - La asignación de `dropping_out_risk` debe ser probabilística y realista: mayor propensión a riesgo (1) cuando baja la asistencia, cuando aumentan las asignaturas reprobadas o si el estudiante trabaja.
   - La relación NO debe ser determinista ni depender al 100% de una sola variable (evitando filtración directa de datos).

5. FORMATO DE LA RESPUESTA ESPERADA:
   - Entregable 1: El dataset en un bloque de código en formato CSV listo para descargar/copiar.
   - Entregable 2: Una tabla en formato Markdown con el Diccionario de Datos estructurado de la siguiente forma:
     | Nombre de Variable | Descripción Operativa | Tipo de Dato | Rango / Valores | Rol en el Modelo |

## Respuesta

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

ids = [f'EST-{str(i).zfill(4)}' for i in range(1, n_samples + 1)]
modes = np.random.choice(['In Person', 'Hibrid'], n_samples, p=[0.65, 0.35])
working = np.random.choice(['Works', "Doesn't Work"], n_samples, p=[0.45, 0.55])

# Attendance depends on working condition
attendance = np.where(
    working == 'Works',
    np.clip(np.random.beta(5, 3, n_samples), 0, 1),
    np.clip(np.random.beta(7, 2, n_samples), 0, 1)
)

academic_load = np.random.randint(1, 7, n_samples)

previous_average = np.clip(np.random.normal(5.2, 0.9, n_samples), 1.0, 7.0)

# Pass probability depends on attendance and previous average
pass_prob = np.clip((attendance * 0.6) + ((previous_average - 1) / 6) * 0.4, 0, 1)
subjects_passed = np.random.binomial(academic_load, pass_prob)

# Dropping out risk logistic calculation
# Higher risk if: Works, low attendance, high fails, low average
fails = academic_load - subjects_passed
logit = -2.0 \
        + 1.2 * (working == 'Works') \
        - 3.5 * attendance \
        + 0.8 * fails \
        - 0.5 * (previous_average - 4.0)

prob = 1 / (1 + np.exp(-logit))
dropping_out_risk = np.random.binomial(1, prob)

df = pd.DataFrame({
    'id': ids,
    'mode': modes,
    'working_condition': working,
    'attendance_percentage': np.round(attendance, 4),
    'academic_load': academic_load,
    'subjects_passed': subjects_passed,
    'previous_average': np.round(previous_average, 2),
    'dropping_out_risk': dropping_out_risk
})

csv_string = df.to_csv(index=False)
print(csv_string)


```

```text
id,mode,working_condition,attendance_percentage,academic_load,subjects_passed,previous_average,dropping_out_risk
EST-0001,In Person,Works,0.6632,2,1,5.75,0
EST-0002,Hibrid,Doesn't Work,0.7922,1,0,5.81,0
EST-0003,Hibrid,Doesn't Work,0.7864,4,4,5.26,0
EST-0004,In Person,Doesn't Work,0.8916,3,2,5.94,0
EST-0005,In Person,Doesn't Work,0.8921,6,6,5.37,0
EST-0006,In Person,Doesn't Work,0.942,1,1,4.42,0
EST-0007,In Person,Doesn't Work,0.8626,5,4,5.02,0
EST-0008,Hibrid,Doesn't Work,0.8874,4,4,5.32,0
EST-0009,In Person,Works,0.6772,3,3,6.04,0
EST-0010,Hibrid,Doesn't Work,0.9649,4,4,5.4,0
EST-0011,In Person,Works,0.7055,2,1,4.69,0
EST-0012,Hibrid,Doesn't Work,0.8336,3,2,6.07,0
EST-0013,Hibrid,Doesn't Work,0.7079,5,5,5.54,0
EST-0014,In Person,Works,0.8177,3,2,6.59,0
EST-0015,In Person,Doesn't Work,0.6614,5,4,5.28,0
EST-0016,In Person,Doesn't Work,0.7955,2,1,4.27,0
EST-0017,In Person,Works,0.8045,5,4,6.26,0
EST-0018,In Person,Doesn't Work,0.9141,4,3,4.68,0
EST-0019,In Person,Doesn't Work,0.7205,1,0,5.47,0
EST-0020,In Person,Works,0.5302,4,1,5.37,0
EST-0021,In Person,Doesn't Work,0.4945,6,6,5.24,0
EST-0022,In Person,Works,0.7273,4,2,5.05,0
EST-0023,In Person,Doesn't Work,0.8521,3,3,3.71,0
EST-0024,In Person,Doesn't Work,0.7984,2,0,4.3,0
EST-0025,In Person,Doesn't Work,0.8078,6,6,6.21,0
EST-0026,Hibrid,Doesn't Work,0.89,6,5,5.59,0
EST-0027,In Person,Doesn't Work,0.3586,6,5,6.92,0
EST-0028,In Person,Doesn't Work,0.7655,1,1,4.27,0
EST-0029,In Person,Works,0.8102,5,3,5.17,0
EST-0030,In Person,Doesn't Work,0.5755,3,1,6.25,0
EST-0031,In Person,Works,0.5988,2,1,4.69,0
EST-0032,In Person,Doesn't Work,0.8742,1,1,4.76,0
EST-0033,In Person,Works,0.5185,4,2,4.94,0
EST-0034,Hibrid,Works,0.7413,2,2,4.8,0
EST-0035,Hibrid,Works,0.362,6,5,5.68,0
EST-0036,Hibrid,Doesn't Work,0.8255,2,2,3.6,0
EST-0037,In Person,Works,0.7981,2,2,6.68,0
EST-0038,In Person,Doesn't Work,0.8259,4,4,4.34,0
EST-0039,Hibrid,Works,0.4866,5,1,3.29,0
EST-0040,In Person,Works,0.6333,3,3,5.19,0
EST-0041,In Person,Doesn't Work,0.6105,6,5,4.86,0
EST-0042,In Person,Doesn't Work,0.7907,5,2,5.04,0
EST-0043,In Person,Works,0.6565,1,1,5.48,0
EST-0044,Hibrid,Doesn't Work,0.9297,6,5,5.78,0
EST-0045,In Person,Doesn't Work,0.8349,3,3,6.72,0
EST-0046,Hibrid,Doesn't Work,0.7339,1,1,4.58,0
EST-0047,In Person,Works,0.4824,4,4,5.9,0
EST-0048,In Person,Works,0.8127,5,5,5.84,0
EST-0049,In Person,Doesn't Work,0.8152,6,6,2.82,0
EST-0050,In Person,Works,0.7893,2,2,4.45,0
EST-0051,Hibrid,Doesn't Work,0.6532,3,3,4.04,0
EST-0052,Hibrid,Doesn't Work,0.5443,6,4,5.6,0
EST-0053,Hibrid,Doesn't Work,0.9781,2,2,5.5,0
EST-0054,Hibrid,Doesn't Work,0.7921,6,5,5.52,0
EST-0055,In Person,Doesn't Work,0.66,1,1,4.92,0
EST-0056,Hibrid,Works,0.5308,2,1,5.02,0
EST-0057,In Person,Doesn't Work,0.7827,2,1,5.17,0
EST-0058,In Person,Works,0.6365,6,5,4.84,0
EST-0059,In Person,Doesn't Work,0.5398,4,2,5.9,0
EST-0060,In Person,Doesn't Work,0.7274,4,2,4.24,0
EST-0061,In Person,Doesn't Work,0.7359,5,4,4.5,0
EST-0062,In Person,Doesn't Work,0.8889,1,1,5.31,0
EST-0063,Hibrid,Doesn't Work,0.4821,4,2,4.87,0
EST-0064,In Person,Works,0.6924,6,5,4.2,0
EST-0065,In Person,Works,0.4693,4,4,5.68,0
EST-0066,In Person,Doesn't Work,0.5787,5,4,4.68,0
EST-0067,In Person,Works,0.7136,3,3,4.44,0
EST-0068,Hibrid,Works,0.3694,5,2,5.04,1
EST-0069,In Person,Works,0.7607,2,1,4.4,0
EST-0070,Hibrid,Works,0.7325,2,2,5.17,0
EST-0071,Hibrid,Works,0.7894,4,2,4.88,1
EST-0072,In Person,Works,0.5949,4,1,4.74,0
EST-0073,In Person,Works,0.5594,5,1,4.74,0
EST-0074,Hibrid,Works,0.673,5,3,6.1,0
EST-0075,Hibrid,Doesn't Work,0.6646,6,5,5.53,0
EST-0076,Hibrid,Works,0.611,1,1,3.66,0
EST-0077,Hibrid,Works,0.5933,6,4,5.99,0
EST-0078,In Person,Doesn't Work,0.6811,2,1,5.66,0
EST-0079,In Person,Works,0.6901,6,3,5.61,0
EST-0080,In Person,Doesn't Work,0.8916,5,2,6.0,0
EST-0081,Hibrid,Doesn't Work,0.9209,6,5,5.21,0
EST-0082,In Person,Works,0.7923,3,3,5.76,0
EST-0083,In Person,Works,0.6741,1,1,4.78,0
EST-0084,In Person,Doesn't Work,0.8107,5,5,5.96,0
EST-0085,In Person,Doesn't Work,0.7867,4,3,5.4,0
EST-0086,In Person,Doesn't Work,0.9772,6,5,3.96,0
EST-0087,Hibrid,Doesn't Work,0.8379,5,4,6.13,0
EST-0088,In Person,Works,0.5669,4,2,5.08,0
EST-0089,Hibrid,Doesn't Work,0.7893,4,2,4.96,0
EST-0090,In Person,Works,0.589,2,1,4.8,0
EST-0091,In Person,Doesn't Work,0.8482,2,1,4.87,0
EST-0092,Hibrid,Doesn't Work,0.7856,5,5,5.41,0
EST-0093,Hibrid,Doesn't Work,0.8681,5,3,4.31,0
EST-0094,In Person,Doesn't Work,0.8087,3,2,3.79,0
EST-0095,Hibrid,Works,0.6801,1,1,4.24,0
EST-0096,In Person,Works,0.3989,1,1,5.54,0
EST-0097,In Person,Works,0.7461,5,2,5.05,1
EST-0098,In Person,Works,0.5615,2,2,6.2,0
EST-0099,In Person,Doesn't Work,0.7055,6,4,5.14,0
EST-0100,In Person,Works,0.634,3,3,5.96,0
EST-0101,In Person,Works,0.6405,1,0,4.58,0
EST-0102,In Person,Doesn't Work,0.8662,5,3,4.81,0
EST-0103,In Person,Doesn't Work,0.7827,2,2,5.21,0
EST-0104,In Person,Doesn't Work,0.6145,2,1,5.07,0
EST-0105,Hibrid,Doesn't Work,0.8311,2,0,4.98,1
EST-0106,In Person,Works,0.599,6,3,4.42,0
EST-0107,In Person,Works,0.7209,4,1,3.77,0
EST-0108,Hibrid,Doesn't Work,0.9496,6,6,5.77,0
EST-0109,In Person,Doesn't Work,0.5756,2,2,5.67,0
EST-0110,In Person,Works,0.2522,5,2,4.65,1
EST-0111,In Person,Doesn't Work,0.941,3,2,5.54,0
EST-0112,In Person,Works,0.7066,6,6,5.05,0
EST-0113,Hibrid,Works,0.6036,5,4,6.2,0
EST-0114,Hibrid,Doesn't Work,0.7123,2,2,4.02,0
EST-0115,In Person,Doesn't Work,0.9156,3,3,5.09,0
EST-0116,Hibrid,Doesn't Work,0.8256,1,0,5.01,0
EST-0117,Hibrid,Works,0.7601,1,1,5.35,0
EST-0118,In Person,Doesn't Work,0.7927,5,4,6.58,0
EST-0119,Hibrid,Works,0.5925,3,1,5.09,1
EST-0120,In Person,Works,0.5831,5,4,5.87,0
EST-0121,Hibrid,Doesn't Work,0.6892,3,2,5.07,0
EST-0122,Hibrid,Doesn't Work,0.8536,3,1,6.16,0
EST-0123,In Person,Doesn't Work,0.7491,5,2,5.88,0
EST-0124,In Person,Doesn't Work,0.6528,3,3,4.9,0
EST-0125,In Person,Doesn't Work,0.7608,6,6,4.84,0
EST-0126,In Person,Doesn't Work,0.6234,2,2,5.64,0
EST-0127,Hibrid,Works,0.4292,3,2,4.61,0
EST-0128,Hibrid,Works,0.5256,6,3,5.37,1
EST-0129,In Person,Works,0.3074,3,2,4.88,0
EST-0130,In Person,Works,0.4433,6,4,5.53,0
EST-0131,In Person,Doesn't Work,0.6863,5,4,5.06,0
EST-0132,In Person,Doesn't Work,0.6178,1,1,4.58,0
EST-0133,In Person,Works,0.7437,4,3,6.19,1
EST-0134,In Person,Works,0.618,3,2,5.42,0
EST-0135,Hibrid,Doesn't Work,0.8952,4,3,5.15,0
EST-0136,In Person,Works,0.577,3,2,5.58,0
EST-0137,In Person,Doesn't Work,0.7772,5,4,3.7,0
EST-0138,Hibrid,Doesn't Work,0.6383,5,5,7.0,0
EST-0139,In Person,Doesn't Work,0.8568,6,5,5.22,0
EST-0140,Hibrid,Doesn't Work,0.8985,3,3,3.43,0
EST-0141,Hibrid,Doesn't Work,0.9444,5,5,6.28,0
EST-0142,In Person,Works,0.5747,5,5,4.52,0
EST-0143,In Person,Doesn't Work,0.4354,3,3,6.14,0
EST-0144,In Person,Doesn't Work,0.3738,6,2,5.31,0
EST-0145,In Person,Works,0.5529,6,4,3.79,1
EST-0146,In Person,Doesn't Work,0.7808,6,4,5.44,0
EST-0147,In Person,Doesn't Work,0.9322,6,5,5.37,0
EST-0148,In Person,Doesn't Work,0.7277,1,1,6.24,0
EST-0149,In Person,Doesn't Work,0.9211,3,1,5.67,0
EST-0150,In Person,Doesn't Work,0.8663,1,1,4.08,0
EST-0151,Hibrid,Doesn't Work,0.7119,5,3,6.1,0
EST-0152,In Person,Doesn't Work,0.8718,1,0,4.83,0
EST-0153,In Person,Doesn't Work,0.8203,6,4,6.75,0
EST-0154,In Person,Doesn't Work,0.9132,5,5,4.88,0
EST-0155,Hibrid,Doesn't Work,0.8034,1,1,5.01,0
EST-0156,In Person,Doesn't Work,0.8539,1,1,4.46,0
EST-0157,Hibrid,Works,0.6138,3,2,4.75,0
EST-0158,Hibrid,Doesn't Work,0.9119,1,1,4.51,0
EST-0159,In Person,Doesn't Work,0.8145,5,5,6.26,0
EST-0160,Hibrid,Works,0.6466,6,3,4.0,0
EST-0161,In Person,Doesn't Work,0.6615,3,2,5.92,0
EST-0162,In Person,Works,0.5657,1,1,5.95,0
EST-0163,In Person,Works,0.8503,3,2,6.44,0
EST-0164,In Person,Doesn't Work,0.7467,4,2,5.24,0
EST-0165,In Person,Works,0.5779,3,3,4.28,0
EST-0166,Hibrid,Doesn't Work,0.9046,6,5,5.05,0
EST-0167,In Person,Doesn't Work,0.6413,4,4,5.74,0
EST-0168,In Person,Doesn't Work,0.7354,4,3,5.28,0
EST-0169,In Person,Works,0.7373,3,1,5.53,0
EST-0170,In Person,Works,0.7923,5,3,6.43,0
EST-0171,Hibrid,Doesn't Work,0.9149,6,6,5.72,0
EST-0172,In Person,Doesn't Work,0.8636,5,5,5.38,0
EST-0173,In Person,Doesn't Work,0.81,6,5,3.54,0
EST-0174,In Person,Works,0.6701,6,4,4.49,0
EST-0175,In Person,Doesn't Work,0.5313,5,4,4.84,0
EST-0176,In Person,Doesn't Work,0.7128,2,1,3.42,0
EST-0177,Hibrid,Doesn't Work,0.7259,4,4,4.2,0
EST-0178,In Person,Works,0.408,3,2,4.96,0
EST-0179,Hibrid,Doesn't Work,0.6216,5,4,5.1,0
EST-0180,In Person,Doesn't Work,0.4627,5,1,5.66,1
EST-0181,In Person,Doesn't Work,0.6327,3,1,5.02,0
EST-0182,In Person,Works,0.7558,4,3,5.42,0
EST-0183,Hibrid,Doesn't Work,0.8645,3,3,4.82,0
EST-0184,Hibrid,Doesn't Work,0.528,6,4,5.24,0
EST-0185,In Person,Works,0.7112,2,2,5.61,0
EST-0186,Hibrid,Doesn't Work,0.6327,6,4,6.53,0
EST-0187,Hibrid,Doesn't Work,0.5863,2,1,4.74,0
EST-0188,In Person,Works,0.7187,3,2,5.33,0
EST-0189,In Person,Works,0.5847,6,4,5.98,0
EST-0190,In Person,Doesn't Work,0.8917,4,3,3.9,0
EST-0191,In Person,Works,0.5839,6,3,7.0,0
EST-0192,Hibrid,Doesn't Work,0.9233,5,3,6.26,0
EST-0193,Hibrid,Works,0.8821,2,1,3.66,0
EST-0194,In Person,Doesn't Work,0.7324,2,2,5.53,0
EST-0195,In Person,Works,0.5594,4,2,4.69,0
EST-0196,In Person,Doesn't Work,0.8476,2,2,4.84,0
EST-0197,Hibrid,Doesn't Work,0.7256,2,1,5.52,0
EST-0198,Hibrid,Doesn't Work,0.938,4,4,6.15,0
EST-0199,Hibrid,Doesn't Work,0.6927,6,5,4.78,0
EST-0200,Hibrid,Works,0.6814,3,3,4.79,0
EST-0201,In Person,Doesn't Work,0.9053,6,5,5.53,0
EST-0202,In Person,Works,0.5193,5,2,5.71,1
EST-0203,In Person,Works,0.5165,5,2,4.29,0
EST-0204,Hibrid,Works,0.3786,2,1,6.75,0
EST-0205,In Person,Doesn't Work,0.6459,5,4,4.62,0
EST-0206,In Person,Doesn't Work,0.7209,3,2,6.07,0
EST-0207,In Person,Doesn't Work,0.8235,2,2,5.61,0
EST-0208,Hibrid,Works,0.5565,5,3,5.75,1
EST-0209,In Person,Works,0.7196,1,1,4.64,0
EST-0210,In Person,Doesn't Work,0.7186,1,1,3.65,0
EST-0211,In Person,Works,0.6469,1,1,5.18,0
EST-0212,Hibrid,Doesn't Work,0.8115,1,1,5.18,0
EST-0213,Hibrid,Works,0.5778,6,1,4.34,1
EST-0214,In Person,Doesn't Work,0.8758,1,1,6.03,0
EST-0215,Hibrid,Doesn't Work,0.8416,2,2,5.15,0
EST-0216,In Person,Works,0.4965,4,2,4.47,0
EST-0217,In Person,Works,0.6797,6,3,5.73,0
EST-0218,Hibrid,Doesn't Work,0.8818,4,4,5.75,0
EST-0219,In Person,Doesn't Work,0.8559,2,1,4.56,0
EST-0220,Hibrid,Works,0.7131,2,1,6.31,0
EST-0221,Hibrid,Doesn't Work,0.7685,3,3,7.0,0
EST-0222,In Person,Works,0.7146,6,4,4.22,0
EST-0223,In Person,Works,0.5737,4,3,5.03,0
EST-0224,In Person,Doesn't Work,0.3413,1,0,5.5,0
EST-0225,In Person,Works,0.3626,2,2,7.0,0
EST-0226,In Person,Doesn't Work,0.6709,1,1,4.54,0
EST-0227,Hibrid,Works,0.699,4,2,4.35,1
EST-0228,In Person,Works,0.5646,6,3,4.35,1
EST-0229,Hibrid,Doesn't Work,0.9166,5,5,6.3,0
EST-0230,In Person,Works,0.678,5,3,4.56,0
EST-0231,Hibrid,Doesn't Work,0.8689,2,2,4.43,0
EST-0232,In Person,Works,0.5461,3,2,5.41,0
EST-0233,In Person,Works,0.8283,1,1,5.48,0
EST-0234,In Person,Works,0.6484,2,1,6.16,0
EST-0235,In Person,Doesn't Work,0.6092,3,3,5.28,0
EST-0236,Hibrid,Works,0.3587,1,0,5.42,0
EST-0237,In Person,Doesn't Work,0.8574,4,3,5.11,0
EST-0238,In Person,Doesn't Work,0.9722,5,5,4.03,0
EST-0239,In Person,Doesn't Work,0.6865,3,2,5.02,0
EST-0240,In Person,Works,0.5907,1,1,5.26,1
EST-0241,Hibrid,Doesn't Work,0.5173,2,0,4.8,0
EST-0242,Hibrid,Doesn't Work,0.5834,1,0,3.79,0
EST-0243,Hibrid,Doesn't Work,0.7762,2,2,4.18,0
EST-0244,In Person,Doesn't Work,0.7341,5,4,5.16,0
EST-0245,In Person,Doesn't Work,0.8182,5,5,6.04,0
EST-0246,Hibrid,Doesn't Work,0.5884,1,1,6.71,0
EST-0247,In Person,Works,0.6522,1,1,5.03,0
EST-0248,Hibrid,Works,0.7315,2,2,4.55,0
EST-0249,Hibrid,Doesn't Work,0.6621,2,1,4.86,0
EST-0250,Hibrid,Doesn't Work,0.6837,3,3,4.26,1
EST-0251,In Person,Doesn't Work,0.4112,3,1,5.91,0
EST-0252,In Person,Doesn't Work,0.9123,3,3,7.0,0
EST-0253,Hibrid,Works,0.4596,1,1,4.58,0
EST-0254,In Person,Works,0.5707,6,2,4.13,0
EST-0255,In Person,Doesn't Work,0.742,6,5,6.11,0
EST-0256,In Person,Works,0.867,6,6,5.49,0
EST-0257,Hibrid,Works,0.4523,6,3,5.5,0
EST-0258,Hibrid,Doesn't Work,0.7764,3,3,6.63,0
EST-0259,In Person,Works,0.3887,1,0,3.97,1
EST-0260,In Person,Doesn't Work,0.9058,6,3,3.95,0
EST-0261,In Person,Works,0.669,5,1,5.05,1
EST-0262,Hibrid,Doesn't Work,0.677,4,2,5.26,0
EST-0263,In Person,Works,0.7322,5,3,4.88,0
EST-0264,In Person,Doesn't Work,0.7564,1,1,7.0,0
EST-0265,Hibrid,Works,0.5761,1,1,4.92,0
EST-0266,Hibrid,Works,0.7358,4,2,4.7,0
EST-0267,Hibrid,Works,0.7158,2,2,5.31,0
EST-0268,Hibrid,Doesn't Work,0.6583,2,2,6.07,0
EST-0269,In Person,Doesn't Work,0.826,2,2,4.55,0
EST-0270,In Person,Works,0.6333,3,2,5.07,0
EST-0271,Hibrid,Doesn't Work,0.8308,2,2,6.1,0
EST-0272,Hibrid,Doesn't Work,0.5113,3,2,6.15,0
EST-0273,Hibrid,Doesn't Work,0.8443,3,3,5.4,0
EST-0274,Hibrid,Works,0.5523,2,2,7.0,0
EST-0275,In Person,Doesn't Work,0.843,2,2,6.98,0
EST-0276,In Person,Works,0.7486,4,3,3.93,0
EST-0277,Hibrid,Works,0.7785,6,2,3.68,1
EST-0278,In Person,Doesn't Work,0.7803,4,4,5.92,0
EST-0279,Hibrid,Works,0.4701,3,3,6.43,0
EST-0280,Hibrid,Doesn't Work,0.7211,4,3,5.0,0
EST-0281,Hibrid,Doesn't Work,0.5523,3,2,4.75,0
EST-0282,In Person,Doesn't Work,0.7881,1,1,4.46,0
EST-0283,In Person,Doesn't Work,0.7501,3,3,4.15,0
EST-0284,In Person,Works,0.8828,3,1,6.22,0
EST-0285,In Person,Works,0.7309,4,2,4.86,0
EST-0286,In Person,Doesn't Work,0.8276,6,5,5.47,0
EST-0287,In Person,Works,0.5268,1,0,3.56,0
EST-0288,In Person,Works,0.4205,5,4,6.1,0
EST-0289,In Person,Works,0.8914,5,5,5.73,0
EST-0290,In Person,Works,0.776,6,4,3.99,0
EST-0291,In Person,Works,0.4921,5,2,5.43,0
EST-0292,In Person,Works,0.6734,1,1,5.2,0
EST-0293,Hibrid,Doesn't Work,0.5184,2,0,4.79,0
EST-0294,In Person,Doesn't Work,0.7301,1,1,2.59,0
EST-0295,In Person,Doesn't Work,0.6625,4,3,4.61,0
EST-0296,In Person,Works,0.5974,3,3,5.47,0
EST-0297,Hibrid,Works,0.666,3,3,4.71,0
EST-0298,In Person,Works,0.4683,4,1,5.0,0
EST-0299,In Person,Doesn't Work,0.6538,4,3,6.55,0
EST-0300,In Person,Doesn't Work,0.9034,3,3,5.13,0
EST-0301,In Person,Works,0.6171,6,1,3.31,1
EST-0302,In Person,Doesn't Work,0.5485,2,0,5.03,1
EST-0303,In Person,Works,0.8353,2,1,5.29,0
EST-0304,In Person,Doesn't Work,0.8473,2,0,4.46,0
EST-0305,Hibrid,Doesn't Work,0.799,4,4,5.86,0
EST-0306,Hibrid,Doesn't Work,0.8817,3,1,3.88,0
EST-0307,In Person,Doesn't Work,0.6838,6,4,5.08,0
EST-0308,In Person,Doesn't Work,0.866,3,1,3.87,0
EST-0309,Hibrid,Works,0.3253,3,2,4.93,0
EST-0310,In Person,Doesn't Work,0.6869,6,5,5.84,0
EST-0311,In Person,Works,0.5543,1,0,5.05,0
EST-0312,In Person,Doesn't Work,0.6833,6,5,5.0,0
EST-0313,In Person,Doesn't Work,0.9368,1,1,5.22,0
EST-0314,Hibrid,Doesn't Work,0.8628,5,5,5.27,0
EST-0315,Hibrid,Doesn't Work,0.953,1,0,5.82,0
EST-0316,Hibrid,Doesn't Work,0.8721,2,2,5.83,0
EST-0317,In Person,Doesn't Work,0.6194,2,2,6.48,0
EST-0318,In Person,Doesn't Work,0.98,6,6,5.35,0
EST-0319,In Person,Works,0.7143,2,0,3.71,0
EST-0320,In Person,Works,0.4044,2,1,5.97,0
EST-0321,In Person,Works,0.5836,5,3,6.41,0
EST-0322,Hibrid,Works,0.8425,2,1,7.0,0
EST-0323,Hibrid,Doesn't Work,0.857,4,4,6.51,0
EST-0324,In Person,Works,0.6242,4,3,4.76,0
EST-0325,Hibrid,Doesn't Work,0.8574,4,3,5.37,0
EST-0326,Hibrid,Works,0.5493,1,0,4.96,0
EST-0327,In Person,Works,0.6731,6,3,4.83,1
EST-0328,In Person,Doesn't Work,0.8108,6,5,5.1,0
EST-0329,In Person,Doesn't Work,0.8425,3,2,6.36,0
EST-0330,In Person,Doesn't Work,0.5547,5,3,3.77,0
EST-0331,In Person,Doesn't Work,0.963,4,3,4.3,0
EST-0332,Hibrid,Doesn't Work,0.9691,4,3,6.3,0
EST-0333,In Person,Works,0.7983,2,2,6.51,0
EST-0334,In Person,Doesn't Work,0.8626,4,2,4.61,0
EST-0335,In Person,Doesn't Work,0.6309,2,1,6.69,0
EST-0336,In Person,Doesn't Work,0.8098,2,1,6.08,0
EST-0337,Hibrid,Works,0.8343,3,3,4.96,0
EST-0338,Hibrid,Doesn't Work,0.5339,1,0,3.01,0
EST-0339,In Person,Works,0.5624,2,0,3.7,0
EST-0340,In Person,Works,0.6595,2,2,4.61,0
EST-0341,In Person,Works,0.7506,6,6,5.88,0
EST-0342,In Person,Doesn't Work,0.7793,3,1,5.89,0
EST-0343,In Person,Doesn't Work,0.9335,2,2,4.78,0
EST-0344,In Person,Doesn't Work,0.902,5,2,4.99,1
EST-0345,In Person,Doesn't Work,0.7056,1,1,5.17,0
EST-0346,In Person,Works,0.7467,5,4,5.0,0
EST-0347,In Person,Works,0.7154,4,2,5.35,0
EST-0348,In Person,Works,0.6996,1,1,5.17,1
EST-0349,In Person,Doesn't Work,0.8524,4,3,3.69,0
EST-0350,In Person,Works,0.723,6,4,5.01,0
EST-0351,In Person,Doesn't Work,0.6887,2,1,5.18,0
EST-0352,Hibrid,Doesn't Work,0.905,3,3,6.09,0
EST-0353,Hibrid,Works,0.3647,5,2,4.45,1
EST-0354,In Person,Doesn't Work,0.8458,4,4,5.38,0
EST-0355,In Person,Doesn't Work,0.7672,2,2,5.71,0
EST-0356,In Person,Doesn't Work,0.5419,5,4,5.07,0
EST-0357,In Person,Doesn't Work,0.8107,5,4,3.62,0
EST-0358,In Person,Works,0.7568,6,5,6.11,0
EST-0359,Hibrid,Works,0.689,6,6,5.87,0
EST-0360,In Person,Doesn't Work,0.5506,4,4,5.41,0
EST-0361,In Person,Doesn't Work,0.9188,2,2,5.24,0
EST-0362,In Person,Works,0.9397,3,2,3.6,0
EST-0363,In Person,Doesn't Work,0.808,4,2,5.9,0
EST-0364,In Person,Doesn't Work,0.8396,5,4,6.54,0
EST-0365,Hibrid,Doesn't Work,0.5337,5,5,4.2,0
EST-0366,In Person,Works,0.6592,1,1,6.41,0
EST-0367,Hibrid,Doesn't Work,0.9795,2,2,6.22,0
EST-0368,Hibrid,Doesn't Work,0.5383,1,0,4.17,0
EST-0369,In Person,Works,0.9745,6,3,4.85,0
EST-0370,In Person,Works,0.7851,3,2,3.94,0
EST-0371,In Person,Doesn't Work,0.7212,4,2,5.14,0
EST-0372,In Person,Works,0.7261,5,4,5.6,0
EST-0373,In Person,Doesn't Work,0.8787,2,2,4.01,0
EST-0374,Hibrid,Doesn't Work,0.815,5,3,5.55,0
EST-0375,In Person,Works,0.7778,6,6,5.78,0
EST-0376,In Person,Doesn't Work,0.7896,1,1,5.79,0
EST-0377,Hibrid,Works,0.6461,1,1,5.61,0
EST-0378,In Person,Doesn't Work,0.787,4,2,4.09,0
EST-0379,Hibrid,Works,0.7195,2,2,5.09,0
EST-0380,In Person,Doesn't Work,0.7831,6,3,5.52,0
EST-0381,In Person,Works,0.8276,2,2,6.51,0
EST-0382,Hibrid,Works,0.474,4,3,5.2,0
EST-0383,In Person,Doesn't Work,0.8909,6,5,5.58,0
EST-0384,Hibrid,Works,0.569,6,4,4.36,0
EST-0385,Hibrid,Doesn't Work,0.8044,1,1,4.16,0
EST-0386,Hibrid,Works,0.1237,1,0,4.05,0
EST-0387,In Person,Doesn't Work,0.9324,1,1,4.82,0
EST-0388,In Person,Works,0.6571,5,5,7.0,0
EST-0389,Hibrid,Doesn't Work,0.5919,3,1,4.84,0
EST-0390,Hibrid,Doesn't Work,0.7057,1,1,5.53,0
EST-0391,Hibrid,Works,0.8542,3,2,5.7,0
EST-0392,In Person,Works,0.504,3,2,4.22,1
EST-0393,In Person,Works,0.5726,2,1,4.81,0
EST-0394,Hibrid,Doesn't Work,0.966,4,4,5.76,0
EST-0395,In Person,Works,0.6828,5,2,5.59,0
EST-0396,Hibrid,Doesn't Work,0.8663,3,2,3.41,0
EST-0397,Hibrid,Works,0.8137,5,4,5.73,0
EST-0398,In Person,Works,0.598,2,2,4.7,0
EST-0399,Hibrid,Doesn't Work,0.8451,1,1,4.73,0
EST-0400,Hibrid,Doesn't Work,0.6889,6,4,5.56,0
EST-0401,In Person,Works,0.7394,3,1,6.57,0
EST-0402,Hibrid,Works,0.4499,4,3,5.02,1
EST-0403,In Person,Works,0.8174,6,6,6.26,0
EST-0404,Hibrid,Doesn't Work,0.8417,2,2,3.9,0
EST-0405,In Person,Doesn't Work,0.721,5,4,5.32,0
EST-0406,Hibrid,Doesn't Work,0.535,1,1,4.71,0
EST-0407,In Person,Doesn't Work,0.9671,5,4,5.45,0
EST-0408,In Person,Doesn't Work,0.6389,4,2,4.93,0
EST-0409,Hibrid,Doesn't Work,0.5922,3,1,3.99,0
EST-0410,In Person,Doesn't Work,0.8685,4,2,4.4,0
EST-0411,In Person,Works,0.6995,5,4,3.72,0
EST-0412,Hibrid,Works,0.5061,5,2,4.02,1
EST-0413,Hibrid,Works,0.7998,6,6,5.63,0
EST-0414,In Person,Works,0.3323,4,3,4.97,0
EST-0415,In Person,Doesn't Work,0.8537,4,4,5.41,0
EST-0416,In Person,Doesn't Work,0.5034,4,3,6.83,0
EST-0417,In Person,Works,0.8043,1,1,4.93,0
EST-0418,In Person,Works,0.6388,5,4,5.77,0
EST-0419,Hibrid,Doesn't Work,0.9951,2,2,5.23,0
EST-0420,Hibrid,Doesn't Work,0.8712,1,1,7.0,0
EST-0421,Hibrid,Works,0.5369,1,1,2.48,0
EST-0422,Hibrid,Doesn't Work,0.8071,1,0,4.55,0
EST-0423,In Person,Works,0.9024,6,6,5.79,0
EST-0424,In Person,Works,0.6646,2,1,4.52,0
EST-0425,In Person,Doesn't Work,0.8444,4,2,5.71,0
EST-0426,In Person,Doesn't Work,0.7831,6,6,5.15,0
EST-0427,In Person,Works,0.5152,5,2,5.98,0
EST-0428,Hibrid,Doesn't Work,0.8971,3,3,4.18,0
EST-0429,In Person,Doesn't Work,0.7214,6,4,5.36,0
EST-0430,In Person,Doesn't Work,0.7277,3,2,5.62,0
EST-0431,In Person,Doesn't Work,0.8734,5,4,5.32,0
EST-0432,Hibrid,Doesn't Work,0.9137,2,1,5.33,0
EST-0433,In Person,Doesn't Work,0.8946,3,3,6.28,0
EST-0434,In Person,Doesn't Work,0.8433,1,1,3.69,0
EST-0435,In Person,Works,0.6264,6,4,5.31,0
EST-0436,Hibrid,Works,0.618,3,1,4.11,0
EST-0437,In Person,Works,0.6211,5,4,5.88,0
EST-0438,Hibrid,Doesn't Work,0.4539,1,0,4.45,0
EST-0439,Hibrid,Works,0.5693,3,1,2.97,0
EST-0440,In Person,Doesn't Work,0.8082,5,5,5.9,0
EST-0441,In Person,Doesn't Work,0.7229,2,1,4.96,0
EST-0442,Hibrid,Works,0.7323,5,5,5.52,0
EST-0443,In Person,Doesn't Work,0.6491,2,2,6.48,0
EST-0444,In Person,Works,0.4192,1,1,5.53,0
EST-0445,Hibrid,Doesn't Work,0.3184,5,2,5.35,0
EST-0446,Hibrid,Works,0.83,4,3,5.16,0
EST-0447,Hibrid,Doesn't Work,0.9757,6,5,4.17,0
EST-0448,Hibrid,Doesn't Work,0.6171,4,3,5.01,0
EST-0449,In Person,Doesn't Work,0.9754,4,4,6.98,0
EST-0450,In Person,Doesn't Work,0.5712,4,3,4.86,1
EST-0451,Hibrid,Doesn't Work,0.6678,6,4,5.06,0
EST-0452,In Person,Doesn't Work,0.7113,4,2,6.33,0
EST-0453,In Person,Doesn't Work,0.902,2,1,3.81,0
EST-0454,Hibrid,Doesn't Work,0.7351,5,5,5.76,0
EST-0455,In Person,Works,0.4311,1,0,3.77,1
EST-0456,In Person,Doesn't Work,0.9418,4,2,5.56,0
EST-0457,In Person,Doesn't Work,0.8545,4,3,5.18,0
EST-0458,In Person,Doesn't Work,0.8379,1,1,5.75,0
EST-0459,In Person,Works,0.6731,1,0,4.29,0
EST-0460,In Person,Doesn't Work,0.9281,4,4,4.72,0
EST-0461,In Person,Works,0.7951,2,2,5.8,0
EST-0462,In Person,Doesn't Work,0.8166,1,0,4.74,0
EST-0463,Hibrid,Works,0.6844,5,3,5.88,0
EST-0464,In Person,Doesn't Work,0.7421,5,5,5.39,0
EST-0465,Hibrid,Doesn't Work,0.689,2,2,4.8,0
EST-0466,In Person,Works,0.6802,6,5,4.92,0
EST-0467,In Person,Doesn't Work,0.5255,3,1,5.2,0
EST-0468,Hibrid,Works,0.6172,2,2,5.0,0
EST-0469,In Person,Doesn't Work,0.9579,4,4,5.86,0
EST-0470,Hibrid,Doesn't Work,0.8275,3,1,4.87,1
EST-0471,In Person,Doesn't Work,0.5761,4,3,4.68,0
EST-0472,Hibrid,Doesn't Work,0.625,5,5,6.02,0
EST-0473,In Person,Doesn't Work,0.8656,1,1,4.82,0
EST-0474,Hibrid,Works,0.8172,4,4,4.71,0
EST-0475,In Person,Works,0.6193,5,3,2.96,1
EST-0476,Hibrid,Doesn't Work,0.8327,2,0,4.26,0
EST-0477,In Person,Doesn't Work,0.8691,4,2,4.45,0
EST-0478,In Person,Doesn't Work,0.94,5,4,4.71,0
EST-0479,Hibrid,Doesn't Work,0.789,1,1,4.55,0
EST-0480,In Person,Works,0.7107,4,3,6.08,0
EST-0481,In Person,Doesn't Work,0.8978,6,5,4.5,0
EST-0482,Hibrid,Works,0.7941,5,5,4.57,0
EST-0483,In Person,Doesn't Work,0.8464,4,2,5.34,1
EST-0484,In Person,Works,0.7826,1,1,4.21,0
EST-0485,In Person,Works,0.8902,4,4,6.61,0
EST-0486,Hibrid,Works,0.7696,4,3,5.92,0
EST-0487,In Person,Works,0.7792,6,4,4.15,0
EST-0488,In Person,Doesn't Work,0.742,5,5,6.11,0
EST-0489,Hibrid,Works,0.3855,3,3,6.3,0
EST-0490,Hibrid,Works,0.7079,4,3,4.17,0
EST-0491,In Person,Works,0.4856,3,1,4.15,0
EST-0492,In Person,Works,0.7354,2,2,4.65,0
EST-0493,In Person,Works,0.3476,4,1,3.93,0
EST-0494,In Person,Works,0.3969,3,2,4.5,0
EST-0495,In Person,Doesn't Work,0.8823,2,1,4.84,0
EST-0496,In Person,Doesn't Work,0.9557,6,6,5.65,0
EST-0497,In Person,Doesn't Work,0.7395,5,5,5.27,0
EST-0498,In Person,Doesn't Work,0.8744,4,3,5.07,0
EST-0499,Hibrid,Works,0.6362,5,2,4.77,1
EST-0500,Hibrid,Doesn't Work,0.7448,2,2,4.02,0
EST-0501,Hibrid,Doesn't Work,0.6864,5,4,5.73,0
EST-0502,In Person,Doesn't Work,0.7276,5,5,5.71,0
EST-0503,In Person,Works,0.7474,1,1,4.64,0
EST-0504,Hibrid,Works,0.6839,2,1,4.21,0
EST-0505,Hibrid,Works,0.8021,5,5,6.6,0
EST-0506,In Person,Works,0.7006,3,1,5.42,0
EST-0507,Hibrid,Doesn't Work,0.3756,2,1,5.56,0
EST-0508,Hibrid,Doesn't Work,0.7191,1,1,5.6,0
EST-0509,Hibrid,Doesn't Work,0.7976,4,3,4.12,0
EST-0510,Hibrid,Doesn't Work,0.7484,6,5,6.55,0
EST-0511,In Person,Doesn't Work,0.7628,5,5,4.84,0
EST-0512,In Person,Doesn't Work,0.5628,3,1,5.94,0
EST-0513,Hibrid,Doesn't Work,0.8391,1,0,6.51,0
EST-0514,Hibrid,Works,0.8037,2,2,5.27,0
EST-0515,In Person,Works,0.5131,6,5,4.14,0
EST-0516,In Person,Works,0.6448,6,4,2.96,0
EST-0517,In Person,Doesn't Work,0.7627,5,3,5.17,0
EST-0518,Hibrid,Doesn't Work,0.9018,1,1,5.43,0
EST-0519,Hibrid,Works,0.8654,1,0,4.68,0
EST-0520,In Person,Doesn't Work,0.9673,3,2,5.32,0
EST-0521,In Person,Works,0.7835,4,3,5.59,0
EST-0522,In Person,Doesn't Work,0.7077,4,3,6.09,0
EST-0523,Hibrid,Works,0.9339,4,3,4.99,0
EST-0524,Hibrid,Works,0.798,6,5,5.16,0
EST-0525,Hibrid,Doesn't Work,0.8066,6,5,5.04,0
EST-0526,In Person,Works,0.7729,2,1,5.23,0
EST-0527,In Person,Works,0.5396,6,3,4.74,0
EST-0528,In Person,Works,0.6046,6,3,4.73,1
EST-0529,In Person,Doesn't Work,0.6207,1,1,4.79,0
EST-0530,Hibrid,Doesn't Work,0.6178,2,1,5.34,0
EST-0531,Hibrid,Works,0.7975,1,1,4.44,0
EST-0532,Hibrid,Doesn't Work,0.8807,3,2,4.79,0
EST-0533,Hibrid,Doesn't Work,0.7345,3,2,6.83,0
EST-0534,In Person,Works,0.5402,2,1,6.01,0
EST-0535,Hibrid,Works,0.8112,2,2,4.34,0
EST-0536,Hibrid,Works,0.5896,6,6,5.98,0
EST-0537,Hibrid,Works,0.5967,3,2,5.76,0
EST-0538,In Person,Doesn't Work,0.7787,5,4,5.11,0
EST-0539,In Person,Works,0.4672,4,0,4.56,0
EST-0540,In Person,Doesn't Work,0.7692,6,5,4.46,0
EST-0541,Hibrid,Doesn't Work,0.937,4,4,5.4,0
EST-0542,In Person,Works,0.4406,3,1,4.44,0
EST-0543,In Person,Doesn't Work,0.6345,4,4,6.19,0
EST-0544,Hibrid,Works,0.7965,5,4,6.44,0
EST-0545,In Person,Doesn't Work,0.7204,5,4,5.16,0
EST-0546,In Person,Doesn't Work,0.8181,1,1,6.22,0
EST-0547,In Person,Doesn't Work,0.7018,2,1,4.83,0
EST-0548,Hibrid,Doesn't Work,0.5685,4,2,4.56,0
EST-0549,In Person,Works,0.8135,5,3,6.19,0
EST-0550,Hibrid,Doesn't Work,0.7607,2,2,6.62,0
EST-0551,In Person,Doesn't Work,0.9247,2,2,6.97,0
EST-0552,Hibrid,Doesn't Work,0.635,1,0,5.5,0
EST-0553,In Person,Doesn't Work,0.9225,2,1,5.98,0
EST-0554,In Person,Doesn't Work,0.8227,2,2,5.48,0
EST-0555,Hibrid,Works,0.6098,6,6,6.46,0
EST-0556,In Person,Works,0.7699,3,3,5.59,0
EST-0557,In Person,Doesn't Work,0.708,1,1,5.92,0
EST-0558,In Person,Works,0.7748,1,0,4.18,0
EST-0559,Hibrid,Doesn't Work,0.8578,4,3,4.99,0
EST-0560,In Person,Doesn't Work,0.7136,2,2,5.93,0
EST-0561,Hibrid,Works,0.3132,4,1,5.85,1
EST-0562,In Person,Works,0.4449,6,4,6.01,0
EST-0563,In Person,Doesn't Work,0.7754,5,4,5.31,0
EST-0564,In Person,Works,0.6099,2,0,4.61,0
EST-0565,In Person,Doesn't Work,0.9333,5,3,5.15,0
EST-0566,In Person,Works,0.3552,3,2,5.24,0
EST-0567,In Person,Doesn't Work,0.8483,6,5,5.42,0
EST-0568,In Person,Works,0.7315,2,1,4.33,0
EST-0569,Hibrid,Doesn't Work,0.5873,4,3,4.96,0
EST-0570,In Person,Doesn't Work,0.689,4,3,5.12,0
EST-0571,In Person,Doesn't Work,0.8097,2,0,3.18,0
EST-0572,Hibrid,Works,0.4343,4,3,5.95,1
EST-0573,In Person,Doesn't Work,0.7313,2,1,5.35,0
EST-0574,In Person,Doesn't Work,0.8762,1,1,5.44,0
EST-0575,Hibrid,Works,0.7504,4,4,4.74,0
EST-0576,Hibrid,Works,0.7402,2,1,5.73,0
EST-0577,In Person,Works,0.7344,2,2,4.45,0
EST-0578,Hibrid,Doesn't Work,0.9248,4,3,4.59,0
EST-0579,In Person,Doesn't Work,0.7253,1,1,5.77,0
EST-0580,In Person,Doesn't Work,0.8186,4,2,5.52,0
EST-0581,In Person,Works,0.4069,2,0,4.05,0
EST-0582,Hibrid,Works,0.389,1,1,6.62,0
EST-0583,In Person,Doesn't Work,0.6022,5,4,4.52,0
EST-0584,In Person,Doesn't Work,0.9048,3,1,5.08,1
EST-0585,In Person,Doesn't Work,0.8839,4,3,6.77,0
EST-0586,In Person,Works,0.485,6,3,3.35,1
EST-0587,In Person,Doesn't Work,0.6539,2,1,3.66,0
EST-0588,In Person,Works,0.4625,1,1,5.6,0
EST-0589,In Person,Works,0.5914,1,0,2.97,0
EST-0590,In Person,Doesn't Work,0.862,5,3,3.36,0
EST-0591,In Person,Doesn't Work,0.8974,4,4,6.26,0
EST-0592,In Person,Doesn't Work,0.6926,6,2,4.83,0
EST-0593,In Person,Works,0.7779,1,1,7.0,0
EST-0594,In Person,Works,0.6106,5,3,4.27,0
EST-0595,Hibrid,Doesn't Work,0.7925,2,2,4.91,0
EST-0596,In Person,Doesn't Work,0.7985,5,3,5.69,0
EST-0597,Hibrid,Works,0.6319,4,4,5.62,0
EST-0598,In Person,Works,0.7401,5,4,4.97,0
EST-0599,In Person,Works,0.547,1,1,4.62,0
EST-0600,In Person,Doesn't Work,0.7779,1,1,5.41,0
EST-0601,In Person,Works,0.7274,3,2,6.44,0
EST-0602,In Person,Works,0.7939,5,5,5.89,0
EST-0603,In Person,Works,0.4764,2,2,4.63,0
EST-0604,In Person,Works,0.6465,6,4,4.31,0
EST-0605,In Person,Works,0.3152,6,3,4.58,1
EST-0606,In Person,Doesn't Work,0.8438,4,4,4.05,0
EST-0607,In Person,Works,0.7986,3,3,5.33,0
EST-0608,In Person,Doesn't Work,0.7726,5,2,5.93,0
EST-0609,In Person,Works,0.668,2,0,5.14,0
EST-0610,Hibrid,Doesn't Work,0.9633,3,2,6.75,0
EST-0611,In Person,Doesn't Work,0.7545,1,1,6.8,0
EST-0612,Hibrid,Doesn't Work,0.7871,4,4,5.48,0
EST-0613,In Person,Doesn't Work,0.8758,5,5,5.83,0
EST-0614,In Person,Works,0.3377,6,2,4.39,0
EST-0615,Hibrid,Works,0.3669,2,1,4.88,0
EST-0616,Hibrid,Doesn't Work,0.9655,1,0,6.35,0
EST-0617,In Person,Doesn't Work,0.6512,3,1,4.68,0
EST-0618,In Person,Doesn't Work,0.816,4,3,6.54,0
EST-0619,Hibrid,Doesn't Work,0.6768,3,3,6.2,0
EST-0620,Hibrid,Doesn't Work,0.4395,6,2,5.88,0
EST-0621,In Person,Doesn't Work,0.7315,3,1,4.14,0
EST-0622,In Person,Works,0.7355,2,2,4.92,0
EST-0623,In Person,Doesn't Work,0.8288,1,0,5.78,0
EST-0624,In Person,Doesn't Work,0.5509,5,3,4.31,0
EST-0625,In Person,Works,0.5188,2,2,5.03,0
EST-0626,In Person,Doesn't Work,0.9413,6,5,5.88,0
EST-0627,In Person,Works,0.5518,4,2,4.69,0
EST-0628,Hibrid,Doesn't Work,0.5698,6,5,4.44,0
EST-0629,In Person,Works,0.686,6,4,3.58,0
EST-0630,In Person,Doesn't Work,0.8883,3,0,4.85,0
EST-0631,Hibrid,Doesn't Work,0.7019,4,3,5.9,0
EST-0632,Hibrid,Works,0.6749,2,1,4.41,0
EST-0633,In Person,Doesn't Work,0.9184,6,6,5.22,0
EST-0634,In Person,Doesn't Work,0.8753,2,2,5.0,0
EST-0635,In Person,Doesn't Work,0.826,5,4,4.6,0
EST-0636,In Person,Works,0.6176,3,3,5.95,0
EST-0637,In Person,Works,0.5809,2,2,6.68,0
EST-0638,In Person,Works,0.6663,4,4,6.1,0
EST-0639,In Person,Doesn't Work,0.7459,6,4,4.57,0
EST-0640,In Person,Doesn't Work,0.8043,5,5,6.99,0
EST-0641,In Person,Doesn't Work,0.8162,4,3,6.1,0
EST-0642,In Person,Doesn't Work,0.8756,3,3,4.79,0
EST-0643,In Person,Doesn't Work,0.7958,5,4,4.5,0
EST-0644,In Person,Doesn't Work,0.5524,5,5,6.21,0
EST-0645,In Person,Works,0.3587,5,3,5.36,1
EST-0646,Hibrid,Doesn't Work,0.7419,6,6,4.43,0
EST-0647,In Person,Works,0.4536,1,0,5.07,0
EST-0648,Hibrid,Doesn't Work,0.5162,5,2,4.83,1
EST-0649,Hibrid,Doesn't Work,0.6477,4,3,4.46,0
EST-0650,In Person,Doesn't Work,0.6805,4,4,6.11,0
EST-0651,In Person,Works,0.7872,3,2,5.9,1
EST-0652,In Person,Works,0.8266,2,1,5.95,0
EST-0653,In Person,Works,0.55,2,2,5.94,0
EST-0654,Hibrid,Works,0.7651,1,1,4.38,0
EST-0655,In Person,Works,0.6568,6,3,4.66,1
EST-0656,In Person,Doesn't Work,0.6923,6,4,6.51,0
EST-0657,In Person,Doesn't Work,0.8652,5,3,5.3,0
EST-0658,In Person,Works,0.5887,1,0,3.88,0
EST-0659,In Person,Works,0.7126,3,1,5.4,0
EST-0660,In Person,Works,0.907,3,3,5.79,0
EST-0661,In Person,Doesn't Work,0.9434,2,2,5.67,0
EST-0662,Hibrid,Doesn't Work,0.6559,3,1,4.36,0
EST-0663,In Person,Doesn't Work,0.7158,5,4,6.9,0
EST-0664,In Person,Doesn't Work,0.8649,4,0,3.85,1
EST-0665,In Person,Doesn't Work,0.8838,3,3,5.3,0
EST-0666,In Person,Doesn't Work,0.7788,1,1,6.76,0
EST-0667,In Person,Works,0.6616,5,4,5.73,0
EST-0668,In Person,Doesn't Work,0.7614,3,3,5.95,0
EST-0669,In Person,Works,0.5022,6,4,4.25,1
EST-0670,In Person,Doesn't Work,0.8911,3,2,5.51,0
EST-0671,In Person,Works,0.9113,6,6,5.17,0
EST-0672,Hibrid,Doesn't Work,0.7407,5,3,4.18,0
EST-0673,In Person,Doesn't Work,0.722,3,2,5.98,0
EST-0674,In Person,Doesn't Work,0.8013,6,5,4.46,0
EST-0675,In Person,Doesn't Work,0.8878,4,4,4.57,0
EST-0676,Hibrid,Works,0.7359,2,1,5.59,0
EST-0677,In Person,Works,0.5724,5,2,5.07,0
EST-0678,In Person,Works,0.3435,6,4,5.35,0
EST-0679,Hibrid,Works,0.49,1,0,5.02,0
EST-0680,Hibrid,Doesn't Work,0.9257,3,2,6.47,0
EST-0681,Hibrid,Works,0.711,6,4,4.37,0
EST-0682,In Person,Doesn't Work,0.7724,6,3,4.78,0
EST-0683,In Person,Doesn't Work,0.5953,6,3,5.63,0
EST-0684,Hibrid,Doesn't Work,0.8997,2,1,4.31,0
EST-0685,Hibrid,Works,0.6827,1,1,4.27,0
EST-0686,In Person,Doesn't Work,0.7329,1,0,5.32,0
EST-0687,In Person,Doesn't Work,0.987,4,4,5.98,0
EST-0688,In Person,Doesn't Work,0.8542,2,1,6.1,0
EST-0689,Hibrid,Doesn't Work,0.7779,6,6,5.28,0
EST-0690,In Person,Doesn't Work,0.8048,5,4,4.6,0
EST-0691,In Person,Works,0.6567,2,2,5.45,0
EST-0692,In Person,Doesn't Work,0.7141,2,2,5.71,0
EST-0693,In Person,Works,0.673,1,1,5.44,0
EST-0694,In Person,Works,0.6169,4,2,6.31,0
EST-0695,In Person,Works,0.3468,4,0,4.29,1
EST-0696,In Person,Works,0.4594,2,1,4.57,0
EST-0697,In Person,Works,0.6535,2,2,6.19,0
EST-0698,In Person,Works,0.3848,3,1,3.87,0
EST-0699,In Person,Doesn't Work,0.9052,6,5,4.71,0
EST-0700,In Person,Doesn't Work,0.9173,1,1,4.54,0
EST-0701,In Person,Doesn't Work,0.8689,6,4,3.79,0
EST-0702,In Person,Works,0.6177,5,3,5.03,0
EST-0703,In Person,Works,0.4244,5,4,5.87,0
EST-0704,In Person,Doesn't Work,0.8932,4,4,5.34,0
EST-0705,In Person,Works,0.3647,1,1,5.92,0
EST-0706,Hibrid,Works,0.7558,6,3,6.27,0
EST-0707,In Person,Doesn't Work,0.8023,4,3,4.85,0
EST-0708,Hibrid,Works,0.3351,5,3,7.0,0
EST-0709,In Person,Doesn't Work,0.8709,2,2,4.89,0
EST-0710,Hibrid,Doesn't Work,0.8532,1,1,5.01,0
EST-0711,Hibrid,Works,0.5775,1,1,4.56,0
EST-0712,In Person,Works,0.6206,6,6,5.69,0
EST-0713,In Person,Doesn't Work,0.9094,3,3,4.94,0
EST-0714,In Person,Doesn't Work,0.3975,6,3,5.13,0
EST-0715,In Person,Works,0.5879,2,2,5.02,0
EST-0716,Hibrid,Doesn't Work,0.8733,2,1,4.99,0
EST-0717,Hibrid,Works,0.5413,4,0,5.34,0
EST-0718,Hibrid,Doesn't Work,0.7203,1,1,4.79,0
EST-0719,In Person,Doesn't Work,0.9358,4,3,4.84,0
EST-0720,Hibrid,Works,0.5708,2,0,6.17,0
EST-0721,Hibrid,Works,0.4755,3,1,4.94,0
EST-0722,In Person,Works,0.8109,1,1,5.55,0
EST-0723,In Person,Works,0.7282,2,1,4.4,0
EST-0724,Hibrid,Works,0.5922,3,3,4.8,0
EST-0725,In Person,Doesn't Work,0.9666,6,5,4.09,0
EST-0726,Hibrid,Works,0.6508,5,5,6.87,0
EST-0727,In Person,Doesn't Work,0.6091,2,2,6.44,0
EST-0728,Hibrid,Doesn't Work,0.8511,5,4,5.74,0
EST-0729,In Person,Doesn't Work,0.7886,4,4,7.0,0
EST-0730,In Person,Doesn't Work,0.7371,5,3,5.89,0
EST-0731,In Person,Works,0.1621,2,1,5.05,1
EST-0732,Hibrid,Doesn't Work,0.8234,5,4,5.54,0
EST-0733,Hibrid,Doesn't Work,0.6863,1,1,6.42,0
EST-0734,In Person,Works,0.9678,2,2,4.41,0
EST-0735,In Person,Works,0.6873,3,2,4.14,0
EST-0736,Hibrid,Doesn't Work,0.8495,5,2,5.57,0
EST-0737,In Person,Works,0.5639,1,1,4.67,0
EST-0738,Hibrid,Works,0.8356,4,3,3.17,0
EST-0739,In Person,Works,0.8893,4,2,4.08,0
EST-0740,In Person,Works,0.4776,5,4,4.92,0
EST-0741,In Person,Doesn't Work,0.405,4,1,5.24,0
EST-0742,In Person,Doesn't Work,0.79,3,2,5.27,0
EST-0743,Hibrid,Doesn't Work,0.7563,2,2,7.0,0
EST-0744,In Person,Works,0.7712,1,1,5.01,0
EST-0745,In Person,Works,0.4684,6,4,5.45,0
EST-0746,Hibrid,Doesn't Work,0.7022,5,4,3.74,0
EST-0747,In Person,Works,0.6261,4,1,5.17,0
EST-0748,In Person,Works,0.7429,2,1,5.5,0
EST-0749,Hibrid,Doesn't Work,0.8605,6,4,3.96,0
EST-0750,In Person,Works,0.5618,3,1,4.36,0
EST-0751,In Person,Doesn't Work,0.7276,4,4,5.64,0
EST-0752,Hibrid,Works,0.8703,4,3,4.71,0
EST-0753,In Person,Works,0.6351,1,1,5.11,0
EST-0754,In Person,Works,0.5216,3,1,5.69,0
EST-0755,In Person,Doesn't Work,0.7512,3,3,7.0,0
EST-0756,Hibrid,Doesn't Work,0.7774,5,3,5.0,0
EST-0757,In Person,Doesn't Work,0.8395,6,4,6.79,0
EST-0758,In Person,Doesn't Work,0.6824,1,1,3.44,0
EST-0759,Hibrid,Works,0.6051,5,5,5.33,0
EST-0760,Hibrid,Doesn't Work,0.4983,1,1,4.9,0
EST-0761,In Person,Doesn't Work,0.712,3,2,4.72,0
EST-0762,In Person,Doesn't Work,0.9505,1,1,5.79,0
EST-0763,In Person,Works,0.47,1,0,5.01,0
EST-0764,Hibrid,Doesn't Work,0.2532,1,0,3.38,0
EST-0765,In Person,Works,0.4335,2,1,3.77,0
EST-0766,In Person,Works,0.5995,2,1,5.85,0
EST-0767,Hibrid,Doesn't Work,0.7729,6,5,3.73,0
EST-0768,Hibrid,Doesn't Work,0.5264,6,5,5.87,0
EST-0769,In Person,Works,0.8682,3,1,5.31,0
EST-0770,Hibrid,Works,0.3554,1,1,5.05,0
EST-0771,In Person,Doesn't Work,0.871,3,3,5.13,0
EST-0772,Hibrid,Doesn't Work,0.7213,2,2,5.66,0
EST-0773,In Person,Doesn't Work,0.5497,1,0,4.14,0
EST-0774,In Person,Doesn't Work,0.827,5,3,5.24,0
EST-0775,Hibrid,Doesn't Work,0.9162,2,1,4.17,0
EST-0776,In Person,Works,0.8034,2,2,5.35,0
EST-0777,In Person,Doesn't Work,0.7758,2,1,5.78,0
EST-0778,Hibrid,Works,0.4356,1,0,5.35,0
EST-0779,Hibrid,Works,0.9443,6,5,5.75,0
EST-0780,In Person,Doesn't Work,0.6874,4,2,4.57,0
EST-0781,Hibrid,Works,0.7383,5,5,5.14,0
EST-0782,In Person,Doesn't Work,0.8694,5,2,4.99,0
EST-0783,In Person,Doesn't Work,0.7445,3,3,4.93,0
EST-0784,In Person,Doesn't Work,0.5816,1,1,4.48,0
EST-0785,In Person,Doesn't Work,0.8644,6,4,4.81,0
EST-0786,Hibrid,Doesn't Work,0.8642,2,1,4.2,0
EST-0787,In Person,Doesn't Work,0.8802,4,2,3.68,0
EST-0788,In Person,Doesn't Work,0.4439,1,0,4.57,0
EST-0789,In Person,Works,0.8416,1,1,5.76,0
EST-0790,Hibrid,Doesn't Work,0.8851,6,4,5.41,0
EST-0791,In Person,Works,0.9004,4,4,4.07,0
EST-0792,In Person,Doesn't Work,0.8986,2,1,4.83,0
EST-0793,In Person,Doesn't Work,0.8944,1,1,5.82,0
EST-0794,In Person,Doesn't Work,0.91,5,3,4.8,0
EST-0795,In Person,Works,0.4734,5,1,3.92,1
EST-0796,Hibrid,Works,0.5265,4,3,7.0,0
EST-0797,Hibrid,Doesn't Work,0.854,2,1,5.05,0
EST-0798,In Person,Works,0.5003,4,2,4.15,0
EST-0799,Hibrid,Works,0.4936,1,0,5.35,0
EST-0800,Hibrid,Works,0.7054,5,3,5.12,0
EST-0801,Hibrid,Doesn't Work,0.7232,3,1,3.81,0
EST-0802,In Person,Doesn't Work,0.8453,4,2,5.28,0
EST-0803,In Person,Works,0.526,3,2,6.09,0
EST-0804,In Person,Works,0.7476,3,3,4.99,0
EST-0805,In Person,Works,0.6646,6,5,3.35,1
EST-0806,Hibrid,Doesn't Work,0.9614,5,5,5.51,0
EST-0807,Hibrid,Doesn't Work,0.76,4,4,5.74,0
EST-0808,Hibrid,Works,0.4768,3,1,4.67,0
EST-0809,In Person,Works,0.4448,1,1,4.68,0
EST-0810,In Person,Works,0.8488,6,6,7.0,0
EST-0811,Hibrid,Doesn't Work,0.991,2,2,4.61,0
EST-0812,Hibrid,Doesn't Work,0.8981,6,4,4.19,0
EST-0813,In Person,Works,0.5122,2,1,5.22,0
EST-0814,Hibrid,Works,0.6461,5,4,5.72,0
EST-0815,Hibrid,Works,0.4979,4,1,5.68,0
EST-0816,In Person,Doesn't Work,0.819,4,4,2.98,0
EST-0817,In Person,Works,0.575,2,0,4.61,0
EST-0818,In Person,Doesn't Work,0.6829,1,1,6.24,0
EST-0819,In Person,Doesn't Work,0.7711,2,2,5.49,0
EST-0820,In Person,Works,0.4745,2,1,4.09,0
EST-0821,Hibrid,Works,0.6051,4,2,6.32,0
EST-0822,In Person,Works,0.592,3,2,6.07,0
EST-0823,In Person,Doesn't Work,0.5859,1,0,4.93,0
EST-0824,In Person,Doesn't Work,0.7647,4,4,5.99,0
EST-0825,In Person,Doesn't Work,0.8292,6,6,7.0,0
EST-0826,Hibrid,Doesn't Work,0.6441,2,2,4.25,0
EST-0827,In Person,Works,0.908,3,3,5.7,0
EST-0828,In Person,Doesn't Work,0.9717,1,1,6.99,0
EST-0829,Hibrid,Doesn't Work,0.9533,5,3,5.79,1
EST-0830,In Person,Works,0.4897,1,1,4.52,0
EST-0831,In Person,Doesn't Work,0.8107,1,1,5.42,0
EST-0832,In Person,Doesn't Work,0.7784,2,2,6.43,0
EST-0833,In Person,Doesn't Work,0.7838,2,1,4.29,0
EST-0834,In Person,Doesn't Work,0.7382,5,1,5.17,0
EST-0835,In Person,Doesn't Work,0.7662,5,3,4.98,0
EST-0836,In Person,Doesn't Work,0.7578,5,1,4.63,0
EST-0837,Hibrid,Works,0.5817,4,2,5.42,0
EST-0838,In Person,Works,0.581,4,1,5.11,0
EST-0839,In Person,Doesn't Work,0.6498,2,1,5.78,0
EST-0840,In Person,Works,0.5456,2,2,6.23,0
EST-0841,In Person,Works,0.7961,5,4,4.05,0
EST-0842,Hibrid,Works,0.4246,4,2,5.26,0
EST-0843,In Person,Works,0.38,3,2,5.15,0
EST-0844,Hibrid,Doesn't Work,0.8711,4,4,5.38,0
EST-0845,In Person,Doesn't Work,0.8935,2,2,5.19,0
EST-0846,In Person,Works,0.659,4,1,4.32,1
EST-0847,In Person,Works,0.6763,2,2,6.51,0
EST-0848,Hibrid,Works,0.8166,5,4,6.89,0
EST-0849,In Person,Doesn't Work,0.8206,5,4,5.69,0
EST-0850,In Person,Doesn't Work,0.864,5,3,5.74,0
EST-0851,In Person,Works,0.4204,3,0,5.32,0
EST-0852,Hibrid,Doesn't Work,0.7024,3,2,6.07,0
EST-0853,In Person,Doesn't Work,0.8961,2,1,3.0,0
EST-0854,Hibrid,Doesn't Work,0.8374,1,1,5.2,1
EST-0855,In Person,Doesn't Work,0.6827,6,4,4.68,0
EST-0856,In Person,Works,0.4638,3,1,5.33,0
EST-0857,In Person,Doesn't Work,0.9015,2,1,6.18,0
EST-0858,In Person,Doesn't Work,0.4395,4,1,5.22,0
EST-0859,In Person,Doesn't Work,0.9193,5,5,4.88,0
EST-0860,In Person,Doesn't Work,0.7737,5,3,3.94,0
EST-0861,Hibrid,Doesn't Work,0.4594,5,3,5.1,0
EST-0862,In Person,Doesn't Work,0.737,2,2,5.81,0
EST-0863,In Person,Works,0.8178,5,4,4.44,0
EST-0864,In Person,Doesn't Work,0.8709,3,3,3.4,0
EST-0865,In Person,Doesn't Work,0.9153,4,3,7.0,0
EST-0866,In Person,Doesn't Work,0.7019,5,4,5.5,1
EST-0867,In Person,Works,0.6928,2,2,5.0,0
EST-0868,In Person,Works,0.5929,4,1,6.45,0
EST-0869,Hibrid,Works,0.6602,5,4,5.33,0
EST-0870,Hibrid,Doesn't Work,0.916,1,1,6.4,0
EST-0871,In Person,Doesn't Work,0.51,5,4,6.32,0
EST-0872,In Person,Works,0.71,6,5,4.05,0
EST-0873,In Person,Doesn't Work,0.8118,3,3,5.61,0
EST-0874,In Person,Doesn't Work,0.7486,4,4,6.75,0
EST-0875,In Person,Doesn't Work,0.8053,3,2,4.3,0
EST-0876,In Person,Doesn't Work,0.4467,1,0,4.31,0
EST-0877,Hibrid,Doesn't Work,0.7554,1,1,4.46,0
EST-0878,Hibrid,Doesn't Work,0.9519,6,5,6.0,0
EST-0879,In Person,Works,0.7253,3,2,5.46,0
EST-0880,In Person,Doesn't Work,0.9009,4,2,3.46,0
EST-0881,In Person,Doesn't Work,0.917,1,1,5.51,0
EST-0882,Hibrid,Doesn't Work,0.6903,1,1,4.97,0
EST-0883,In Person,Doesn't Work,0.8637,5,3,5.1,0
EST-0884,In Person,Doesn't Work,0.7604,6,6,4.45,0
EST-0885,Hibrid,Works,0.5172,2,2,5.51,0
EST-0886,In Person,Doesn't Work,0.8027,1,1,6.15,0
EST-0887,Hibrid,Works,0.5314,4,2,4.62,0
EST-0888,In Person,Works,0.4557,4,3,3.69,0
EST-0889,In Person,Doesn't Work,0.8881,5,3,4.93,0
EST-0890,In Person,Doesn't Work,0.7741,4,4,5.88,0
EST-0891,Hibrid,Works,0.6341,5,3,4.82,0
EST-0892,Hibrid,Works,0.6969,1,1,3.85,0
EST-0893,Hibrid,Doesn't Work,0.695,1,0,5.89,0
EST-0894,In Person,Works,0.6254,1,0,6.47,0
EST-0895,Hibrid,Doesn't Work,0.8777,1,0,5.22,0
EST-0896,In Person,Doesn't Work,0.8763,2,2,3.75,0
EST-0897,In Person,Doesn't Work,0.7246,6,3,4.91,0
EST-0898,Hibrid,Works,0.9088,6,6,5.34,0
EST-0899,In Person,Works,0.6944,2,2,4.47,0
EST-0900,In Person,Doesn't Work,0.847,1,0,5.73,0
EST-0901,In Person,Doesn't Work,0.8236,4,4,4.38,0
EST-0902,In Person,Works,0.8003,5,2,5.22,0
EST-0903,In Person,Doesn't Work,0.851,6,6,4.99,0
EST-0904,In Person,Works,0.7336,3,2,5.75,0
EST-0905,In Person,Doesn't Work,0.9082,5,5,5.19,0
EST-0906,Hibrid,Works,0.5954,3,2,5.02,0
EST-0907,Hibrid,Doesn't Work,0.918,3,3,4.48,0
EST-0908,In Person,Works,0.8543,1,1,5.23,0
EST-0909,In Person,Works,0.6137,3,2,4.97,0
EST-0910,In Person,Works,0.6983,1,0,4.42,0
EST-0911,In Person,Works,0.5287,2,2,5.34,0
EST-0912,In Person,Doesn't Work,0.4495,4,4,5.53,0
EST-0913,In Person,Doesn't Work,0.7452,3,3,4.9,0
EST-0914,In Person,Works,0.5019,4,3,6.04,0
EST-0915,In Person,Works,0.6695,1,0,5.44,0
EST-0916,In Person,Doesn't Work,0.7651,3,1,4.04,0
EST-0917,In Person,Doesn't Work,0.6845,5,5,6.4,0
EST-0918,Hibrid,Doesn't Work,0.5805,2,1,5.89,0
EST-0919,Hibrid,Works,0.2397,6,4,6.1,1
EST-0920,In Person,Works,0.8218,2,2,5.56,0
EST-0921,Hibrid,Doesn't Work,0.7389,3,2,3.1,0
EST-0922,In Person,Works,0.6042,2,1,6.18,0
EST-0923,Hibrid,Doesn't Work,0.9085,6,6,5.25,0
EST-0924,Hibrid,Works,0.7835,5,5,5.52,0
EST-0925,In Person,Doesn't Work,0.963,4,4,5.69,0
EST-0926,In Person,Works,0.6613,5,4,6.94,0
EST-0927,In Person,Doesn't Work,0.8791,2,2,4.71,0
EST-0928,In Person,Doesn't Work,0.4798,6,4,4.73,0
EST-0929,In Person,Doesn't Work,0.767,4,3,4.43,0
EST-0930,In Person,Works,0.5762,6,5,4.24,0
EST-0931,In Person,Doesn't Work,0.6812,5,4,4.25,0
EST-0932,Hibrid,Doesn't Work,0.8798,1,0,3.53,0
EST-0933,Hibrid,Doesn't Work,0.9557,1,1,5.58,0
EST-0934,In Person,Doesn't Work,0.606,4,3,6.99,0
EST-0935,In Person,Doesn't Work,0.8219,4,4,7.0,0
EST-0936,In Person,Doesn't Work,0.7616,4,2,4.43,0
EST-0937,In Person,Works,0.4055,1,0,4.96,1
EST-0938,Hibrid,Doesn't Work,0.7439,5,5,6.31,0
EST-0939,In Person,Doesn't Work,0.5817,4,2,5.59,0
EST-0940,In Person,Works,0.445,1,1,5.32,0
EST-0941,In Person,Works,0.6667,4,2,4.97,0
EST-0942,In Person,Works,0.4315,4,3,6.26,0
EST-0943,Hibrid,Doesn't Work,0.6087,6,1,4.23,0
EST-0944,Hibrid,Doesn't Work,0.7336,4,2,5.63,0
EST-0945,In Person,Doesn't Work,0.522,1,0,5.51,0
EST-0946,In Person,Doesn't Work,0.8578,5,4,5.24,0
EST-0947,In Person,Works,0.6535,1,1,5.91,0
EST-0948,Hibrid,Works,0.865,1,1,5.23,0
EST-0949,Hibrid,Works,0.206,4,3,4.72,0
EST-0950,In Person,Works,0.8106,6,3,5.28,0
EST-0951,In Person,Works,0.6496,3,2,3.77,0
EST-0952,Hibrid,Works,0.7538,3,2,5.75,0
EST-0953,In Person,Doesn't Work,0.6387,3,3,5.98,0
EST-0954,Hibrid,Doesn't Work,0.8318,1,1,4.61,0
EST-0955,In Person,Doesn't Work,0.9012,4,4,4.39,0
EST-0956,In Person,Doesn't Work,0.7585,3,2,4.64,0
EST-0957,In Person,Doesn't Work,0.9471,2,1,5.14,0
EST-0958,In Person,Works,0.4125,2,2,5.39,0
EST-0959,In Person,Doesn't Work,0.9322,2,2,6.65,0
EST-0960,Hibrid,Doesn't Work,0.9738,5,5,6.54,0
EST-0961,In Person,Works,0.7351,4,4,5.95,0
EST-0962,In Person,Works,0.9117,1,1,5.31,0
EST-0963,In Person,Doesn't Work,0.5453,1,0,4.37,0
EST-0964,In Person,Works,0.6382,4,4,6.71,0
EST-0965,In Person,Doesn't Work,0.6585,6,3,4.98,0
EST-0966,In Person,Doesn't Work,0.725,6,2,5.45,0
EST-0967,Hibrid,Works,0.8863,5,5,5.76,0
EST-0968,In Person,Works,0.5811,4,3,5.26,0
EST-0969,In Person,Works,0.3579,6,4,6.24,0
EST-0970,In Person,Works,0.6619,5,3,5.61,0
EST-0971,Hibrid,Works,0.6717,3,3,6.09,0
EST-0972,Hibrid,Doesn't Work,0.8062,3,3,5.6,0
EST-0973,Hibrid,Works,0.7957,5,4,5.17,0
EST-0974,In Person,Doesn't Work,0.9196,4,2,5.27,0
EST-0975,In Person,Works,0.5237,3,2,5.14,0
EST-0976,In Person,Doesn't Work,0.5769,4,2,4.04,0
EST-0977,In Person,Works,0.5006,6,1,4.3,1
EST-0978,In Person,Works,0.4946,3,1,5.87,0
EST-0979,In Person,Doesn't Work,0.7617,6,3,4.07,0
EST-0980,In Person,Doesn't Work,0.7986,4,3,4.84,0
EST-0981,In Person,Works,0.5092,4,0,4.25,1
EST-0982,Hibrid,Doesn't Work,0.7538,2,1,4.79,0
EST-0983,In Person,Doesn't Work,0.8375,5,3,5.06,0
EST-0984,In Person,Doesn't Work,0.8492,1,1,4.83,0
EST-0985,Hibrid,Works,0.6702,5,2,3.91,0
EST-0986,In Person,Works,0.6499,4,3,5.03,0
EST-0987,Hibrid,Doesn't Work,0.9664,4,4,4.34,0
EST-0988,In Person,Doesn't Work,0.6941,5,2,4.63,0
EST-0989,In Person,Works,0.8366,4,4,5.87,0
EST-0990,Hibrid,Doesn't Work,0.7888,6,5,6.06,0
EST-0991,Hibrid,Doesn't Work,0.8725,3,3,4.02,0
EST-0992,Hibrid,Works,0.608,4,2,6.61,0
EST-0993,In Person,Doesn't Work,0.9508,5,5,4.74,0
EST-0994,In Person,Works,0.6155,5,2,4.66,1
EST-0995,In Person,Doesn't Work,0.7019,5,3,4.4,0
EST-0996,In Person,Doesn't Work,0.6444,2,2,5.25,0
EST-0997,Hibrid,Doesn't Work,0.6166,6,5,6.98,0
EST-0998,In Person,Works,0.6534,6,4,3.81,0
EST-0999,Hibrid,Works,0.8845,2,2,3.35,0
EST-1000,In Person,Works,0.5335,6,4,5.44,0



```

```csv
id,mode,working_condition,attendance_percentage,academic_load,subjects_passed,previous_average,dropping_out_risk
EST-0001,In Person,Works,0.6632,2,1,5.75,0
EST-0002,Hibrid,Doesn't Work,0.7922,1,0,5.81,0
EST-0003,Hibrid,Doesn't Work,0.7864,4,4,5.26,0
EST-0004,In Person,Doesn't Work,0.8916,3,2,5.94,0
EST-0005,In Person,Doesn't Work,0.8921,6,6,5.37,0
EST-0006,In Person,Doesn't Work,0.942,1,1,4.42,0
EST-0007,In Person,Doesn't Work,0.8626,5,4,5.02,0
EST-0008,Hibrid,Doesn't Work,0.8874,4,4,5.32,0
EST-0009,In Person,Works,0.6772,3,3,6.04,0
EST-0010,Hibrid,Doesn't Work,0.9649,4,4,5.4,0
EST-0011,In Person,Works,0.7055,2,1,4.69,0
EST-0012,Hibrid,Doesn't Work,0.8336,3,2,6.07,0
EST-0013,Hibrid,Doesn't Work,0.7079,5,5,5.54,0
EST-0014,In Person,Works,0.8177,3,2,6.59,0
EST-0015,In Person,Doesn't Work,0.6614,5,4,5.28,0
EST-0016,In Person,Doesn't Work,0.7955,2,1,4.27,0
EST-0017,In Person,Works,0.8045,5,4,6.26,0
EST-0018,In Person,Doesn't Work,0.9141,4,3,4.68,0
EST-0019,In Person,Doesn't Work,0.7205,1,0,5.47,0
EST-0020,In Person,Works,0.5302,4,1,5.37,0
EST-0021,In Person,Doesn't Work,0.4945,6,6,5.24,0
EST-0022,In Person,Works,0.7273,4,2,5.05,0
EST-0023,In Person,Doesn't Work,0.8521,3,3,3.71,0
EST-0024,In Person,Doesn't Work,0.7984,2,0,4.3,0
EST-0025,In Person,Doesn't Work,0.8078,6,6,6.21,0
EST-0026,Hibrid,Doesn't Work,0.89,6,5,5.59,0
EST-0027,In Person,Doesn't Work,0.3586,6,5,6.92,0
EST-0028,In Person,Doesn't Work,0.7655,1,1,4.27,0
EST-0029,In Person,Works,0.8102,5,3,5.17,0
EST-0030,In Person,Doesn't Work,0.5755,3,1,6.25,0
EST-0031,In Person,Works,0.5988,2,1,4.69,0
EST-0032,In Person,Doesn't Work,0.8742,1,1,4.76,0
EST-0033,In Person,Works,0.5185,4,2,4.94,0
EST-0034,Hibrid,Works,0.7413,2,2,4.8,0
EST-0035,Hibrid,Works,0.362,6,5,5.68,0
EST-0036,Hibrid,Doesn't Work,0.8255,2,2,3.6,0
EST-0037,In Person,Works,0.7981,2,2,6.68,0
EST-0038,In Person,Doesn't Work,0.8259,4,4,4.34,0
EST-0039,Hibrid,Works,0.4866,5,1,3.29,0
EST-0040,In Person,Works,0.6333,3,3,5.19,0
EST-0041,In Person,Doesn't Work,0.6105,6,5,4.86,0
EST-0042,In Person,Doesn't Work,0.7907,5,2,5.04,0
EST-0043,In Person,Works,0.6565,1,1,5.48,0
EST-0044,Hibrid,Doesn't Work,0.9297,6,5,5.78,0
EST-0045,In Person,Doesn't Work,0.8349,3,3,6.72,0
EST-0046,Hibrid,Doesn't Work,0.7339,1,1,4.58,0
EST-0047,In Person,Works,0.4824,4,4,5.9,0
EST-0048,In Person,Works,0.8127,5,5,5.84,0
EST-0049,In Person,Doesn't Work,0.8152,6,6,2.82,0
EST-0050,In Person,Works,0.7893,2,2,4.45,0
EST-0051,Hibrid,Doesn't Work,0.6532,3,3,4.04,0
EST-0052,Hibrid,Doesn't Work,0.5443,6,4,5.6,0
EST-0053,Hibrid,Doesn't Work,0.9781,2,2,5.5,0
EST-0054,Hibrid,Doesn't Work,0.7921,6,5,5.52,0
EST-0055,In Person,Doesn't Work,0.66,1,1,4.92,0
EST-0056,Hibrid,Works,0.5308,2,1,5.02,0
EST-0057,In Person,Doesn't Work,0.7827,2,1,5.17,0
EST-0058,In Person,Works,0.6365,6,5,4.84,0
EST-0059,In Person,Doesn't Work,0.5398,4,2,5.9,0
EST-0060,In Person,Doesn't Work,0.7274,4,2,4.24,0
EST-0061,In Person,Doesn't Work,0.7359,5,4,4.5,0
EST-0062,In Person,Doesn't Work,0.8889,1,1,5.31,0
EST-0063,Hibrid,Doesn't Work,0.4821,4,2,4.87,0
EST-0064,In Person,Works,0.6924,6,5,4.2,0
EST-0065,In Person,Works,0.4693,4,4,5.68,0
EST-0066,In Person,Doesn't Work,0.5787,5,4,4.68,0
EST-0067,In Person,Works,0.7136,3,3,4.44,0
EST-0068,Hibrid,Works,0.3694,5,2,5.04,1
EST-0069,In Person,Works,0.7607,2,1,4.4,0
EST-0070,Hibrid,Works,0.7325,2,2,5.17,0
EST-0071,Hibrid,Works,0.7894,4,2,4.88,1
EST-0072,In Person,Works,0.5949,4,1,4.74,0
EST-0073,In Person,Works,0.5594,5,1,4.74,0
EST-0074,Hibrid,Works,0.673,5,3,6.1,0
EST-0075,Hibrid,Doesn't Work,0.6646,6,5,5.53,0
EST-0076,Hibrid,Works,0.611,1,1,3.66,0
EST-0077,Hibrid,Works,0.5933,6,4,5.99,0
EST-0078,In Person,Doesn't Work,0.6811,2,1,5.66,0
EST-0079,In Person,Works,0.6901,6,3,5.61,0
EST-0080,In Person,Doesn't Work,0.8916,5,2,6.0,0
EST-0081,Hibrid,Doesn't Work,0.9209,6,5,5.21,0
EST-0082,In Person,Works,0.7923,3,3,5.76,0
EST-0083,In Person,Works,0.6741,1,1,4.78,0
EST-0084,In Person,Doesn't Work,0.8107,5,5,5.96,0
EST-0085,In Person,Doesn't Work,0.7867,4,3,5.4,0
EST-0086,In Person,Doesn't Work,0.9772,6,5,3.96,0
EST-0087,Hibrid,Doesn't Work,0.8379,5,4,6.13,0
EST-0088,In Person,Works,0.5669,4,2,5.08,0
EST-0089,Hibrid,Doesn't Work,0.7893,4,2,4.96,0
EST-0090,In Person,Works,0.589,2,1,4.8,0
EST-0091,In Person,Doesn't Work,0.8482,2,1,4.87,0
EST-0092,Hibrid,Doesn't Work,0.7856,5,5,5.41,0
EST-0093,Hibrid,Doesn't Work,0.8681,5,3,4.31,0
EST-0094,In Person,Doesn't Work,0.8087,3,2,3.79,0
EST-0095,Hibrid,Works,0.6801,1,1,4.24,0
EST-0096,In Person,Works,0.3989,1,1,5.54,0
EST-0097,In Person,Works,0.7461,5,2,5.05,1
EST-0098,In Person,Works,0.5615,2,2,6.2,0
EST-0099,In Person,Doesn't Work,0.7055,6,4,5.14,0
EST-0100,In Person,Works,0.634,3,3,5.96,0
EST-0101,In Person,Works,0.6405,1,0,4.58,0
EST-0102,In Person,Doesn't Work,0.8662,5,3,4.81,0
EST-0103,In Person,Doesn't Work,0.7827,2,2,5.21,0
EST-0104,In Person,Doesn't Work,0.6145,2,1,5.07,0
EST-0105,Hibrid,Doesn't Work,0.8311,2,0,4.98,1
EST-0106,In Person,Works,0.599,6,3,4.42,0
EST-0107,In Person,Works,0.7209,4,1,3.77,0
EST-0108,Hibrid,Doesn't Work,0.9496,6,6,5.77,0
EST-0109,In Person,Doesn't Work,0.5756,2,2,5.67,0
EST-0110,In Person,Works,0.2522,5,2,4.65,1
EST-0111,In Person,Doesn't Work,0.941,3,2,5.54,0
EST-0112,In Person,Works,0.7066,6,6,5.05,0
EST-0113,Hibrid,Works,0.6036,5,4,6.2,0
EST-0114,Hibrid,Doesn't Work,0.7123,2,2,4.02,0
EST-0115,In Person,Doesn't Work,0.9156,3,3,5.09,0
EST-0116,Hibrid,Doesn't Work,0.8256,1,0,5.01,0
EST-0117,Hibrid,Works,0.7601,1,1,5.35,0
EST-0118,In Person,Doesn't Work,0.7927,5,4,6.58,0
EST-0119,Hibrid,Works,0.5925,3,1,5.09,1
EST-0120,In Person,Works,0.5831,5,4,5.87,0
EST-0121,Hibrid,Doesn't Work,0.6892,3,2,5.07,0
EST-0122,Hibrid,Doesn't Work,0.8536,3,1,6.16,0
EST-0123,In Person,Doesn't Work,0.7491,5,2,5.88,0
EST-0124,In Person,Doesn't Work,0.6528,3,3,4.9,0
EST-0125,In Person,Doesn't Work,0.7608,6,6,4.84,0
EST-0126,In Person,Doesn't Work,0.6234,2,2,5.64,0
EST-0127,Hibrid,Works,0.4292,3,2,4.61,0
EST-0128,Hibrid,Works,0.5256,6,3,5.37,1
EST-0129,In Person,Works,0.3074,3,2,4.88,0
EST-0130,In Person,Works,0.4433,6,4,5.53,0
EST-0131,In Person,Doesn't Work,0.6863,5,4,5.06,0
EST-0132,In Person,Doesn't Work,0.6178,1,1,4.58,0
EST-0133,In Person,Works,0.7437,4,3,6.19,1
EST-0134,In Person,Works,0.618,3,2,5.42,0
EST-0135,Hibrid,Doesn't Work,0.8952,4,3,5.15,0
EST-0136,In Person,Works,0.577,3,2,5.58,0
EST-0137,In Person,Doesn't Work,0.7772,5,4,3.7,0
EST-0138,Hibrid,Doesn't Work,0.6383,5,5,7.0,0
EST-0139,In Person,Doesn't Work,0.8568,6,5,5.22,0
EST-0140,Hibrid,Doesn't Work,0.8985,3,3,3.43,0
EST-0141,Hibrid,Doesn't Work,0.9444,5,5,6.28,0
EST-0142,In Person,Works,0.5747,5,5,4.52,0
EST-0143,In Person,Doesn't Work,0.4354,3,3,6.14,0
EST-0144,In Person,Doesn't Work,0.3738,6,2,5.31,0
EST-0145,In Person,Works,0.5529,6,4,3.79,1
EST-0146,In Person,Doesn't Work,0.7808,6,4,5.44,0
EST-0147,In Person,Doesn't Work,0.9322,6,5,5.37,0
EST-0148,In Person,Doesn't Work,0.7277,1,1,6.24,0
EST-0149,In Person,Doesn't Work,0.9211,3,1,5.67,0
EST-0150,In Person,Doesn't Work,0.8663,1,1,4.08,0
EST-0151,Hibrid,Doesn't Work,0.7119,5,3,6.1,0
EST-0152,In Person,Doesn't Work,0.8718,1,0,4.83,0
EST-0153,In Person,Doesn't Work,0.8203,6,4,6.75,0
EST-0154,In Person,Doesn't Work,0.9132,5,5,4.88,0
EST-0155,Hibrid,Doesn't Work,0.8034,1,1,5.01,0
EST-0156,In Person,Doesn't Work,0.8539,1,1,4.46,0
EST-0157,Hibrid,Works,0.6138,3,2,4.75,0
EST-0158,Hibrid,Doesn't Work,0.9119,1,1,4.51,0
EST-0159,In Person,Doesn't Work,0.8145,5,5,6.26,0
EST-0160,Hibrid,Works,0.6466,6,3,4.0,0
EST-0161,In Person,Doesn't Work,0.6615,3,2,5.92,0
EST-0162,In Person,Works,0.5657,1,1,5.95,0
EST-0163,In Person,Works,0.8503,3,2,6.44,0
EST-0164,In Person,Doesn't Work,0.7467,4,2,5.24,0
EST-0165,In Person,Works,0.5779,3,3,4.28,0
EST-0166,Hibrid,Doesn't Work,0.9046,6,5,5.05,0
EST-0167,In Person,Doesn't Work,0.6413,4,4,5.74,0
EST-0168,In Person,Doesn't Work,0.7354,4,3,5.28,0
EST-0169,In Person,Works,0.7373,3,1,5.53,0
EST-0170,In Person,Works,0.7923,5,3,6.43,0
EST-0171,Hibrid,Doesn't Work,0.9149,6,6,5.72,0
EST-0172,In Person,Doesn't Work,0.8636,5,5,5.38,0
EST-0173,In Person,Doesn't Work,0.81,6,5,3.54,0
EST-0174,In Person,Works,0.6701,6,4,4.49,0
EST-0175,In Person,Doesn't Work,0.5313,5,4,4.84,0
EST-0176,In Person,Doesn't Work,0.7128,2,1,3.42,0
EST-0177,Hibrid,Doesn't Work,0.7259,4,4,4.2,0
EST-0178,In Person,Works,0.408,3,2,4.96,0
EST-0179,Hibrid,Doesn't Work,0.6216,5,4,5.1,0
EST-0180,In Person,Doesn't Work,0.4627,5,1,5.66,1
EST-0181,In Person,Doesn't Work,0.6327,3,1,5.02,0
EST-0182,In Person,Works,0.7558,4,3,5.42,0
EST-0183,Hibrid,Doesn't Work,0.8645,3,3,4.82,0
EST-0184,Hibrid,Doesn't Work,0.528,6,4,5.24,0
EST-0185,In Person,Works,0.7112,2,2,5.61,0
EST-0186,Hibrid,Doesn't Work,0.6327,6,4,6.53,0
EST-0187,Hibrid,Doesn't Work,0.5863,2,1,4.74,0
EST-0188,In Person,Works,0.7187,3,2,5.33,0
EST-0189,In Person,Works,0.5847,6,4,5.98,0
EST-0190,In Person,Doesn't Work,0.8917,4,3,3.9,0
EST-0191,In Person,Works,0.5839,6,3,7.0,0
EST-0192,Hibrid,Doesn't Work,0.9233,5,3,6.26,0
EST-0193,Hibrid,Works,0.8821,2,1,3.66,0
EST-0194,In Person,Doesn't Work,0.7324,2,2,5.53,0
EST-0195,In Person,Works,0.5594,4,2,4.69,0
EST-0196,In Person,Doesn't Work,0.8476,2,2,4.84,0
EST-0197,Hibrid,Doesn't Work,0.7256,2,1,5.52,0
EST-0198,Hibrid,Doesn't Work,0.938,4,4,6.15,0
EST-0199,Hibrid,Doesn't Work,0.6927,6,5,4.78,0
EST-0200,Hibrid,Works,0.6814,3,3,4.79,0
EST-0201,In Person,Doesn't Work,0.9053,6,5,5.53,0
EST-0202,In Person,Works,0.5193,5,2,5.71,1
EST-0203,In Person,Works,0.5165,5,2,4.29,0
EST-0204,Hibrid,Works,0.3786,2,1,6.75,0
EST-0205,In Person,Doesn't Work,0.6459,5,4,4.62,0
EST-0206,In Person,Doesn't Work,0.7209,3,2,6.07,0
EST-0207,In Person,Doesn't Work,0.8235,2,2,5.61,0
EST-0208,Hibrid,Works,0.5565,5,3,5.75,1
EST-0209,In Person,Works,0.7196,1,1,4.64,0
EST-0210,In Person,Doesn't Work,0.7186,1,1,3.65,0
EST-0211,In Person,Works,0.6469,1,1,5.18,0
EST-0212,Hibrid,Doesn't Work,0.8115,1,1,5.18,0
EST-0213,Hibrid,Works,0.5778,6,1,4.34,1
EST-0214,In Person,Doesn't Work,0.8758,1,1,6.03,0
EST-0215,Hibrid,Doesn't Work,0.8416,2,2,5.15,0
EST-0216,In Person,Works,0.4965,4,2,4.47,0
EST-0217,In Person,Works,0.6797,6,3,5.73,0
EST-0218,Hibrid,Doesn't Work,0.8818,4,4,5.75,0
EST-0219,In Person,Doesn't Work,0.8559,2,1,4.56,0
EST-0220,Hibrid,Works,0.7131,2,1,6.31,0
EST-0221,Hibrid,Doesn't Work,0.7685,3,3,7.0,0
EST-0222,In Person,Works,0.7146,6,4,4.22,0
EST-0223,In Person,Works,0.5737,4,3,5.03,0
EST-0224,In Person,Doesn't Work,0.3413,1,0,5.5,0
EST-0225,In Person,Works,0.3626,2,2,7.0,0
EST-0226,In Person,Doesn't Work,0.6709,1,1,4.54,0
EST-0227,Hibrid,Works,0.699,4,2,4.35,1
EST-0228,In Person,Works,0.5646,6,3,4.35,1
EST-0229,Hibrid,Doesn't Work,0.9166,5,5,6.3,0
EST-0230,In Person,Works,0.678,5,3,4.56,0
EST-0231,Hibrid,Doesn't Work,0.8689,2,2,4.43,0
EST-0232,In Person,Works,0.5461,3,2,5.41,0
EST-0233,In Person,Works,0.8283,1,1,5.48,0
EST-0234,In Person,Works,0.6484,2,1,6.16,0
EST-0235,In Person,Doesn't Work,0.6092,3,3,5.28,0
EST-0236,Hibrid,Works,0.3587,1,0,5.42,0
EST-0237,In Person,Doesn't Work,0.8574,4,3,5.11,0
EST-0238,In Person,Doesn't Work,0.9722,5,5,4.03,0
EST-0239,In Person,Doesn't Work,0.6865,3,2,5.02,0
EST-0240,In Person,Works,0.5907,1,1,5.26,1
EST-0241,Hibrid,Doesn't Work,0.5173,2,0,4.8,0
EST-0242,Hibrid,Doesn't Work,0.5834,1,0,3.79,0
EST-0243,Hibrid,Doesn't Work,0.7762,2,2,4.18,0
EST-0244,In Person,Doesn't Work,0.7341,5,4,5.16,0
EST-0245,In Person,Doesn't Work,0.8182,5,5,6.04,0
EST-0246,Hibrid,Doesn't Work,0.5884,1,1,6.71,0
EST-0247,In Person,Works,0.6522,1,1,5.03,0
EST-0248,Hibrid,Works,0.7315,2,2,4.55,0
EST-0249,Hibrid,Doesn't Work,0.6621,2,1,4.86,0
EST-0250,Hibrid,Doesn't Work,0.6837,3,3,4.26,1
EST-0251,In Person,Doesn't Work,0.4112,3,1,5.91,0
EST-0252,In Person,Doesn't Work,0.9123,3,3,7.0,0
EST-0253,Hibrid,Works,0.4596,1,1,4.58,0
EST-0254,In Person,Works,0.5707,6,2,4.13,0
EST-0255,In Person,Doesn't Work,0.742,6,5,6.11,0
EST-0256,In Person,Works,0.867,6,6,5.49,0
EST-0257,Hibrid,Works,0.4523,6,3,5.5,0
EST-0258,Hibrid,Doesn't Work,0.7764,3,3,6.63,0
EST-0259,In Person,Works,0.3887,1,0,3.97,1
EST-0260,In Person,Doesn't Work,0.9058,6,3,3.95,0
EST-0261,In Person,Works,0.669,5,1,5.05,1
EST-0262,Hibrid,Doesn't Work,0.677,4,2,5.26,0
EST-0263,In Person,Works,0.7322,5,3,4.88,0
EST-0264,In Person,Doesn't Work,0.7564,1,1,7.0,0
EST-0265,Hibrid,Works,0.5761,1,1,4.92,0
EST-0266,Hibrid,Works,0.7358,4,2,4.7,0
EST-0267,Hibrid,Works,0.7158,2,2,5.31,0
EST-0268,Hibrid,Doesn't Work,0.6583,2,2,6.07,0
EST-0269,In Person,Doesn't Work,0.826,2,2,4.55,0
EST-0270,In Person,Works,0.6333,3,2,5.07,0
EST-0271,Hibrid,Doesn't Work,0.8308,2,2,6.1,0
EST-0272,Hibrid,Doesn't Work,0.5113,3,2,6.15,0
EST-0273,Hibrid,Doesn't Work,0.8443,3,3,5.4,0
EST-0274,Hibrid,Works,0.5523,2,2,7.0,0
EST-0275,In Person,Doesn't Work,0.843,2,2,6.98,0
EST-0276,In Person,Works,0.7486,4,3,3.93,0
EST-0277,Hibrid,Works,0.7785,6,2,3.68,1
EST-0278,In Person,Doesn't Work,0.7803,4,4,5.92,0
EST-0279,Hibrid,Works,0.4701,3,3,6.43,0
EST-0280,Hibrid,Doesn't Work,0.7211,4,3,5.0,0
EST-0281,Hibrid,Doesn't Work,0.5523,3,2,4.75,0
EST-0282,In Person,Doesn't Work,0.7881,1,1,4.46,0
EST-0283,In Person,Doesn't Work,0.7501,3,3,4.15,0
EST-0284,In Person,Works,0.8828,3,1,6.22,0
EST-0285,In Person,Works,0.7309,4,2,4.86,0
EST-0286,In Person,Doesn't Work,0.8276,6,5,5.47,0
EST-0287,In Person,Works,0.5268,1,0,3.56,0
EST-0288,In Person,Works,0.4205,5,4,6.1,0
EST-0289,In Person,Works,0.8914,5,5,5.73,0
EST-0290,In Person,Works,0.776,6,4,3.99,0
EST-0291,In Person,Works,0.4921,5,2,5.43,0
EST-0292,In Person,Works,0.6734,1,1,5.2,0
EST-0293,Hibrid,Doesn't Work,0.5184,2,0,4.79,0
EST-0294,In Person,Doesn't Work,0.7301,1,1,2.59,0
EST-0295,In Person,Doesn't Work,0.6625,4,3,4.61,0
EST-0296,In Person,Works,0.5974,3,3,5.47,0
EST-0297,Hibrid,Works,0.666,3,3,4.71,0
EST-0298,In Person,Works,0.4683,4,1,5.0,0
EST-0299,In Person,Doesn't Work,0.6538,4,3,6.55,0
EST-0300,In Person,Doesn't Work,0.9034,3,3,5.13,0
EST-0301,In Person,Works,0.6171,6,1,3.31,1
EST-0302,In Person,Doesn't Work,0.5485,2,0,5.03,1
EST-0303,In Person,Works,0.8353,2,1,5.29,0
EST-0304,In Person,Doesn't Work,0.8473,2,0,4.46,0
EST-0305,Hibrid,Doesn't Work,0.799,4,4,5.86,0
EST-0306,Hibrid,Doesn't Work,0.8817,3,1,3.88,0
EST-0307,In Person,Doesn't Work,0.6838,6,4,5.08,0
EST-0308,In Person,Doesn't Work,0.866,3,1,3.87,0
EST-0309,Hibrid,Works,0.3253,3,2,4.93,0
EST-0310,In Person,Doesn't Work,0.6869,6,5,5.84,0
EST-0311,In Person,Works,0.5543,1,0,5.05,0
EST-0312,In Person,Doesn't Work,0.6833,6,5,5.0,0
EST-0313,In Person,Doesn't Work,0.9368,1,1,5.22,0
EST-0314,Hibrid,Doesn't Work,0.8628,5,5,5.27,0
EST-0315,Hibrid,Doesn't Work,0.953,1,0,5.82,0
EST-0316,Hibrid,Doesn't Work,0.8721,2,2,5.83,0
EST-0317,In Person,Doesn't Work,0.6194,2,2,6.48,0
EST-0318,In Person,Doesn't Work,0.98,6,6,5.35,0
EST-0319,In Person,Works,0.7143,2,0,3.71,0
EST-0320,In Person,Works,0.4044,2,1,5.97,0
EST-0321,In Person,Works,0.5836,5,3,6.41,0
EST-0322,Hibrid,Works,0.8425,2,1,7.0,0
EST-0323,Hibrid,Doesn't Work,0.857,4,4,6.51,0
EST-0324,In Person,Works,0.6242,4,3,4.76,0
EST-0325,Hibrid,Doesn't Work,0.8574,4,3,5.37,0
EST-0326,Hibrid,Works,0.5493,1,0,4.96,0
EST-0327,In Person,Works,0.6731,6,3,4.83,1
EST-0328,In Person,Doesn't Work,0.8108,6,5,5.1,0
EST-0329,In Person,Doesn't Work,0.8425,3,2,6.36,0
EST-0330,In Person,Doesn't Work,0.5547,5,3,3.77,0
EST-0331,In Person,Doesn't Work,0.963,4,3,4.3,0
EST-0332,Hibrid,Doesn't Work,0.9691,4,3,6.3,0
EST-0333,In Person,Works,0.7983,2,2,6.51,0
EST-0334,In Person,Doesn't Work,0.8626,4,2,4.61,0
EST-0335,In Person,Doesn't Work,0.6309,2,1,6.69,0
EST-0336,In Person,Doesn't Work,0.8098,2,1,6.08,0
EST-0337,Hibrid,Works,0.8343,3,3,4.96,0
EST-0338,Hibrid,Doesn't Work,0.5339,1,0,3.01,0
EST-0339,In Person,Works,0.5624,2,0,3.7,0
EST-0340,In Person,Works,0.6595,2,2,4.61,0
EST-0341,In Person,Works,0.7506,6,6,5.88,0
EST-0342,In Person,Doesn't Work,0.7793,3,1,5.89,0
EST-0343,In Person,Doesn't Work,0.9335,2,2,4.78,0
EST-0344,In Person,Doesn't Work,0.902,5,2,4.99,1
EST-0345,In Person,Doesn't Work,0.7056,1,1,5.17,0
EST-0346,In Person,Works,0.7467,5,4,5.0,0
EST-0347,In Person,Works,0.7154,4,2,5.35,0
EST-0348,In Person,Works,0.6996,1,1,5.17,1
EST-0349,In Person,Doesn't Work,0.8524,4,3,3.69,0
EST-0350,In Person,Works,0.723,6,4,5.01,0
EST-0351,In Person,Doesn't Work,0.6887,2,1,5.18,0
EST-0352,Hibrid,Doesn't Work,0.905,3,3,6.09,0
EST-0353,Hibrid,Works,0.3647,5,2,4.45,1
EST-0354,In Person,Doesn't Work,0.8458,4,4,5.38,0
EST-0355,In Person,Doesn't Work,0.7672,2,2,5.71,0
EST-0356,In Person,Doesn't Work,0.5419,5,4,5.07,0
EST-0357,In Person,Doesn't Work,0.8107,5,4,3.62,0
EST-0358,In Person,Works,0.7568,6,5,6.11,0
EST-0359,Hibrid,Works,0.689,6,6,5.87,0
EST-0360,In Person,Doesn't Work,0.5506,4,4,5.41,0
EST-0361,In Person,Doesn't Work,0.9188,2,2,5.24,0
EST-0362,In Person,Works,0.9397,3,2,3.6,0
EST-0363,In Person,Doesn't Work,0.808,4,2,5.9,0
EST-0364,In Person,Doesn't Work,0.8396,5,4,6.54,0
EST-0365,Hibrid,Doesn't Work,0.5337,5,5,4.2,0
EST-0366,In Person,Works,0.6592,1,1,6.41,0
EST-0367,Hibrid,Doesn't Work,0.9795,2,2,6.22,0
EST-0368,Hibrid,Doesn't Work,0.5383,1,0,4.17,0
EST-0369,In Person,Works,0.9745,6,3,4.85,0
EST-0370,In Person,Works,0.7851,3,2,3.94,0
EST-0371,In Person,Doesn't Work,0.7212,4,2,5.14,0
EST-0372,In Person,Works,0.7261,5,4,5.6,0
EST-0373,In Person,Doesn't Work,0.8787,2,2,4.01,0
EST-0374,Hibrid,Doesn't Work,0.815,5,3,5.55,0
EST-0375,In Person,Works,0.7778,6,6,5.78,0
EST-0376,In Person,Doesn't Work,0.7896,1,1,5.79,0
EST-0377,Hibrid,Works,0.6461,1,1,5.61,0
EST-0378,In Person,Doesn't Work,0.787,4,2,4.09,0
EST-0379,Hibrid,Works,0.7195,2,2,5.09,0
EST-0380,In Person,Doesn't Work,0.7831,6,3,5.52,0
EST-0381,In Person,Works,0.8276,2,2,6.51,0
EST-0382,Hibrid,Works,0.474,4,3,5.2,0
EST-0383,In Person,Doesn't Work,0.8909,6,5,5.58,0
EST-0384,Hibrid,Works,0.569,6,4,4.36,0
EST-0385,Hibrid,Doesn't Work,0.8044,1,1,4.16,0
EST-0386,Hibrid,Works,0.1237,1,0,4.05,0
EST-0387,In Person,Doesn't Work,0.9324,1,1,4.82,0
EST-0388,In Person,Works,0.6571,5,5,7.0,0
EST-0389,Hibrid,Doesn't Work,0.5919,3,1,4.84,0
EST-0390,Hibrid,Doesn't Work,0.7057,1,1,5.53,0
EST-0391,Hibrid,Works,0.8542,3,2,5.7,0
EST-0392,In Person,Works,0.504,3,2,4.22,1
EST-0393,In Person,Works,0.5726,2,1,4.81,0
EST-0394,Hibrid,Doesn't Work,0.966,4,4,5.76,0
EST-0395,In Person,Works,0.6828,5,2,5.59,0
EST-0396,Hibrid,Doesn't Work,0.8663,3,2,3.41,0
EST-0397,Hibrid,Works,0.8137,5,4,5.73,0
EST-0398,In Person,Works,0.598,2,2,4.7,0
EST-0399,Hibrid,Doesn't Work,0.8451,1,1,4.73,0
EST-0400,Hibrid,Doesn't Work,0.6889,6,4,5.56,0
EST-0401,In Person,Works,0.7394,3,1,6.57,0
EST-0402,Hibrid,Works,0.4499,4,3,5.02,1
EST-0403,In Person,Works,0.8174,6,6,6.26,0
EST-0404,Hibrid,Doesn't Work,0.8417,2,2,3.9,0
EST-0405,In Person,Doesn't Work,0.721,5,4,5.32,0
EST-0406,Hibrid,Doesn't Work,0.535,1,1,4.71,0
EST-0407,In Person,Doesn't Work,0.9671,5,4,5.45,0
EST-0408,In Person,Doesn't Work,0.6389,4,2,4.93,0
EST-0409,Hibrid,Doesn't Work,0.5922,3,1,3.99,0
EST-0410,In Person,Doesn't Work,0.8685,4,2,4.4,0
EST-0411,In Person,Works,0.6995,5,4,3.72,0
EST-0412,Hibrid,Works,0.5061,5,2,4.02,1
EST-0413,Hibrid,Works,0.7998,6,6,5.63,0
EST-0414,In Person,Works,0.3323,4,3,4.97,0
EST-0415,In Person,Doesn't Work,0.8537,4,4,5.41,0
EST-0416,In Person,Doesn't Work,0.5034,4,3,6.83,0
EST-0417,In Person,Works,0.8043,1,1,4.93,0
EST-0418,In Person,Works,0.6388,5,4,5.77,0
EST-0419,Hibrid,Doesn't Work,0.9951,2,2,5.23,0
EST-0420,Hibrid,Doesn't Work,0.8712,1,1,7.0,0
EST-0421,Hibrid,Works,0.5369,1,1,2.48,0
EST-0422,Hibrid,Doesn't Work,0.8071,1,0,4.55,0
EST-0423,In Person,Works,0.9024,6,6,5.79,0
EST-0424,In Person,Works,0.6646,2,1,4.52,0
EST-0425,In Person,Doesn't Work,0.8444,4,2,5.71,0
EST-0426,In Person,Doesn't Work,0.7831,6,6,5.15,0
EST-0427,In Person,Works,0.5152,5,2,5.98,0
EST-0428,Hibrid,Doesn't Work,0.8971,3,3,4.18,0
EST-0429,In Person,Doesn't Work,0.7214,6,4,5.36,0
EST-0430,In Person,Doesn't Work,0.7277,3,2,5.62,0
EST-0431,In Person,Doesn't Work,0.8734,5,4,5.32,0
EST-0432,Hibrid,Doesn't Work,0.9137,2,1,5.33,0
EST-0433,In Person,Doesn't Work,0.8946,3,3,6.28,0
EST-0434,In Person,Doesn't Work,0.8433,1,1,3.69,0
EST-0435,In Person,Works,0.6264,6,4,5.31,0
EST-0436,Hibrid,Works,0.618,3,1,4.11,0
EST-0437,In Person,Works,0.6211,5,4,5.88,0
EST-0438,Hibrid,Doesn't Work,0.4539,1,0,4.45,0
EST-0439,Hibrid,Works,0.5693,3,1,2.97,0
EST-0440,In Person,Doesn't Work,0.8082,5,5,5.9,0
EST-0441,In Person,Doesn't Work,0.7229,2,1,4.96,0
EST-0442,Hibrid,Works,0.7323,5,5,5.52,0
EST-0443,In Person,Doesn't Work,0.6491,2,2,6.48,0
EST-0444,In Person,Works,0.4192,1,1,5.53,0
EST-0445,Hibrid,Doesn't Work,0.3184,5,2,5.35,0
EST-0446,Hibrid,Works,0.83,4,3,5.16,0
EST-0447,Hibrid,Doesn't Work,0.9757,6,5,4.17,0
EST-0448,Hibrid,Doesn't Work,0.6171,4,3,5.01,0
EST-0449,In Person,Doesn't Work,0.9754,4,4,6.98,0
EST-0450,In Person,Doesn't Work,0.5712,4,3,4.86,1
EST-0451,Hibrid,Doesn't Work,0.6678,6,4,5.06,0
EST-0452,In Person,Doesn't Work,0.7113,4,2,6.33,0
EST-0453,In Person,Doesn't Work,0.902,2,1,3.81,0
EST-0454,Hibrid,Doesn't Work,0.7351,5,5,5.76,0
EST-0455,In Person,Works,0.4311,1,0,3.77,1
EST-0456,In Person,Doesn't Work,0.9418,4,2,5.56,0
EST-0457,In Person,Doesn't Work,0.8545,4,3,5.18,0
EST-0458,In Person,Doesn't Work,0.8379,1,1,5.75,0
EST-0459,In Person,Works,0.6731,1,0,4.29,0
EST-0460,In Person,Doesn't Work,0.9281,4,4,4.72,0
EST-0461,In Person,Works,0.7951,2,2,5.8,0
EST-0462,In Person,Doesn't Work,0.8166,1,0,4.74,0
EST-0463,Hibrid,Works,0.6844,5,3,5.88,0
EST-0464,In Person,Doesn't Work,0.7421,5,5,5.39,0
EST-0465,Hibrid,Doesn't Work,0.689,2,2,4.8,0
EST-0466,In Person,Works,0.6802,6,5,4.92,0
EST-0467,In Person,Doesn't Work,0.5255,3,1,5.2,0
EST-0468,Hibrid,Works,0.6172,2,2,5.0,0
EST-0469,In Person,Doesn't Work,0.9579,4,4,5.86,0
EST-0470,Hibrid,Doesn't Work,0.8275,3,1,4.87,1
EST-0471,In Person,Doesn't Work,0.5761,4,3,4.68,0
EST-0472,Hibrid,Doesn't Work,0.625,5,5,6.02,0
EST-0473,In Person,Doesn't Work,0.8656,1,1,4.82,0
EST-0474,Hibrid,Works,0.8172,4,4,4.71,0
EST-0475,In Person,Works,0.6193,5,3,2.96,1
EST-0476,Hibrid,Doesn't Work,0.8327,2,0,4.26,0
EST-0477,In Person,Doesn't Work,0.8691,4,2,4.45,0
EST-0478,In Person,Doesn't Work,0.94,5,4,4.71,0
EST-0479,Hibrid,Doesn't Work,0.789,1,1,4.55,0
EST-0480,In Person,Works,0.7107,4,3,6.08,0
EST-0481,In Person,Doesn't Work,0.8978,6,5,4.5,0
EST-0482,Hibrid,Works,0.7941,5,5,4.57,0
EST-0483,In Person,Doesn't Work,0.8464,4,2,5.34,1
EST-0484,In Person,Works,0.7826,1,1,4.21,0
EST-0485,In Person,Works,0.8902,4,4,6.61,0
EST-0486,Hibrid,Works,0.7696,4,3,5.92,0
EST-0487,In Person,Works,0.7792,6,4,4.15,0
EST-0488,In Person,Doesn't Work,0.742,5,5,6.11,0
EST-0489,Hibrid,Works,0.3855,3,3,6.3,0
EST-0490,Hibrid,Works,0.7079,4,3,4.17,0
EST-0491,In Person,Works,0.4856,3,1,4.15,0
EST-0492,In Person,Works,0.7354,2,2,4.65,0
EST-0493,In Person,Works,0.3476,4,1,3.93,0
EST-0494,In Person,Works,0.3969,3,2,4.5,0
EST-0495,In Person,Doesn't Work,0.8823,2,1,4.84,0
EST-0496,In Person,Doesn't Work,0.9557,6,6,5.65,0
EST-0497,In Person,Doesn't Work,0.7395,5,5,5.27,0
EST-0498,In Person,Doesn't Work,0.8744,4,3,5.07,0
EST-0499,Hibrid,Works,0.6362,5,2,4.77,1
EST-0500,Hibrid,Doesn't Work,0.7448,2,2,4.02,0
EST-0501,Hibrid,Doesn't Work,0.6864,5,4,5.73,0
EST-0502,In Person,Doesn't Work,0.7276,5,5,5.71,0
EST-0503,In Person,Works,0.7474,1,1,4.64,0
EST-0504,Hibrid,Works,0.6839,2,1,4.21,0
EST-0505,Hibrid,Works,0.8021,5,5,6.6,0
EST-0506,In Person,Works,0.7006,3,1,5.42,0
EST-0507,Hibrid,Doesn't Work,0.3756,2,1,5.56,0
EST-0508,Hibrid,Doesn't Work,0.7191,1,1,5.6,0
EST-0509,Hibrid,Doesn't Work,0.7976,4,3,4.12,0
EST-0510,Hibrid,Doesn't Work,0.7484,6,5,6.55,0
EST-0511,In Person,Doesn't Work,0.7628,5,5,4.84,0
EST-0512,In Person,Doesn't Work,0.5628,3,1,5.94,0
EST-0513,Hibrid,Doesn't Work,0.8391,1,0,6.51,0
EST-0514,Hibrid,Works,0.8037,2,2,5.27,0
EST-0515,In Person,Works,0.5131,6,5,4.14,0
EST-0516,In Person,Works,0.6448,6,4,2.96,0
EST-0517,In Person,Doesn't Work,0.7627,5,3,5.17,0
EST-0518,Hibrid,Doesn't Work,0.9018,1,1,5.43,0
EST-0519,Hibrid,Works,0.8654,1,0,4.68,0
EST-0520,In Person,Doesn't Work,0.9673,3,2,5.32,0
EST-0521,In Person,Works,0.7835,4,3,5.59,0
EST-0522,In Person,Doesn't Work,0.7077,4,3,6.09,0
EST-0523,Hibrid,Works,0.9339,4,3,4.99,0
EST-0524,Hibrid,Works,0.798,6,5,5.16,0
EST-0525,Hibrid,Doesn't Work,0.8066,6,5,5.04,0
EST-0526,In Person,Works,0.7729,2,1,5.23,0
EST-0527,In Person,Works,0.5396,6,3,4.74,0
EST-0528,In Person,Works,0.6046,6,3,4.73,1
EST-0529,In Person,Doesn't Work,0.6207,1,1,4.79,0
EST-0530,Hibrid,Doesn't Work,0.6178,2,1,5.34,0
EST-0531,Hibrid,Works,0.7975,1,1,4.44,0
EST-0532,Hibrid,Doesn't Work,0.8807,3,2,4.79,0
EST-0533,Hibrid,Doesn't Work,0.7345,3,2,6.83,0
EST-0534,In Person,Works,0.5402,2,1,6.01,0
EST-0535,Hibrid,Works,0.8112,2,2,4.34,0
EST-0536,Hibrid,Works,0.5896,6,6,5.98,0
EST-0537,Hibrid,Works,0.5967,3,2,5.76,0
EST-0538,In Person,Doesn't Work,0.7787,5,4,5.11,0
EST-0539,In Person,Works,0.4672,4,0,4.56,0
EST-0540,In Person,Doesn't Work,0.7692,6,5,4.46,0
EST-0541,Hibrid,Doesn't Work,0.937,4,4,5.4,0
EST-0542,In Person,Works,0.4406,3,1,4.44,0
EST-0543,In Person,Doesn't Work,0.6345,4,4,6.19,0
EST-0544,Hibrid,Works,0.7965,5,4,6.44,0
EST-0545,In Person,Doesn't Work,0.7204,5,4,5.16,0
EST-0546,In Person,Doesn't Work,0.8181,1,1,6.22,0
EST-0547,In Person,Doesn't Work,0.7018,2,1,4.83,0
EST-0548,Hibrid,Doesn't Work,0.5685,4,2,4.56,0
EST-0549,In Person,Works,0.8135,5,3,6.19,0
EST-0550,Hibrid,Doesn't Work,0.7607,2,2,6.62,0
EST-0551,In Person,Doesn't Work,0.9247,2,2,6.97,0
EST-0552,Hibrid,Doesn't Work,0.635,1,0,5.5,0
EST-0553,In Person,Doesn't Work,0.9225,2,1,5.98,0
EST-0554,In Person,Doesn't Work,0.8227,2,2,5.48,0
EST-0555,Hibrid,Works,0.6098,6,6,6.46,0
EST-0556,In Person,Works,0.7699,3,3,5.59,0
EST-0557,In Person,Doesn't Work,0.708,1,1,5.92,0
EST-0558,In Person,Works,0.7748,1,0,4.18,0
EST-0559,Hibrid,Doesn't Work,0.8578,4,3,4.99,0
EST-0560,In Person,Doesn't Work,0.7136,2,2,5.93,0
EST-0561,Hibrid,Works,0.3132,4,1,5.85,1
EST-0562,In Person,Works,0.4449,6,4,6.01,0
EST-0563,In Person,Doesn't Work,0.7754,5,4,5.31,0
EST-0564,In Person,Works,0.6099,2,0,4.61,0
EST-0565,In Person,Doesn't Work,0.9333,5,3,5.15,0
EST-0566,In Person,Works,0.3552,3,2,5.24,0
EST-0567,In Person,Doesn't Work,0.8483,6,5,5.42,0
EST-0568,In Person,Works,0.7315,2,1,4.33,0
EST-0569,Hibrid,Doesn't Work,0.5873,4,3,4.96,0
EST-0570,In Person,Doesn't Work,0.689,4,3,5.12,0
EST-0571,In Person,Doesn't Work,0.8097,2,0,3.18,0
EST-0572,Hibrid,Works,0.4343,4,3,5.95,1
EST-0573,In Person,Doesn't Work,0.7313,2,1,5.35,0
EST-0574,In Person,Doesn't Work,0.8762,1,1,5.44,0
EST-0575,Hibrid,Works,0.7504,4,4,4.74,0
EST-0576,Hibrid,Works,0.7402,2,1,5.73,0
EST-0577,In Person,Works,0.7344,2,2,4.45,0
EST-0578,Hibrid,Doesn't Work,0.9248,4,3,4.59,0
EST-0579,In Person,Doesn't Work,0.7253,1,1,5.77,0
EST-0580,In Person,Doesn't Work,0.8186,4,2,5.52,0
EST-0581,In Person,Works,0.4069,2,0,4.05,0
EST-0582,Hibrid,Works,0.389,1,1,6.62,0
EST-0583,In Person,Doesn't Work,0.6022,5,4,4.52,0
EST-0584,In Person,Doesn't Work,0.9048,3,1,5.08,1
EST-0585,In Person,Doesn't Work,0.8839,4,3,6.77,0
EST-0586,In Person,Works,0.485,6,3,3.35,1
EST-0587,In Person,Doesn't Work,0.6539,2,1,3.66,0
EST-0588,In Person,Works,0.4625,1,1,5.6,0
EST-0589,In Person,Works,0.5914,1,0,2.97,0
EST-0590,In Person,Doesn't Work,0.862,5,3,3.36,0
EST-0591,In Person,Doesn't Work,0.8974,4,4,6.26,0
EST-0592,In Person,Doesn't Work,0.6926,6,2,4.83,0
EST-0593,In Person,Works,0.7779,1,1,7.0,0
EST-0594,In Person,Works,0.6106,5,3,4.27,0
EST-0595,Hibrid,Doesn't Work,0.7925,2,2,4.91,0
EST-0596,In Person,Doesn't Work,0.7985,5,3,5.69,0
EST-0597,Hibrid,Works,0.6319,4,4,5.62,0
EST-0598,In Person,Works,0.7401,5,4,4.97,0
EST-0599,In Person,Works,0.547,1,1,4.62,0
EST-0600,In Person,Doesn't Work,0.7779,1,1,5.41,0
EST-0601,In Person,Works,0.7274,3,2,6.44,0
EST-0602,In Person,Works,0.7939,5,5,5.89,0
EST-0603,In Person,Works,0.4764,2,2,4.63,0
EST-0604,In Person,Works,0.6465,6,4,4.31,0
EST-0605,In Person,Works,0.3152,6,3,4.58,1
EST-0606,In Person,Doesn't Work,0.8438,4,4,4.05,0
EST-0607,In Person,Works,0.7986,3,3,5.33,0
EST-0608,In Person,Doesn't Work,0.7726,5,2,5.93,0
EST-0609,In Person,Works,0.668,2,0,5.14,0
EST-0610,Hibrid,Doesn't Work,0.9633,3,2,6.75,0
EST-0611,In Person,Doesn't Work,0.7545,1,1,6.8,0
EST-0612,Hibrid,Doesn't Work,0.7871,4,4,5.48,0
EST-0613,In Person,Doesn't Work,0.8758,5,5,5.83,0
EST-0614,In Person,Works,0.3377,6,2,4.39,0
EST-0615,Hibrid,Works,0.3669,2,1,4.88,0
EST-0616,Hibrid,Doesn't Work,0.9655,1,0,6.35,0
EST-0617,In Person,Doesn't Work,0.6512,3,1,4.68,0
EST-0618,In Person,Doesn't Work,0.816,4,3,6.54,0
EST-0619,Hibrid,Doesn't Work,0.6768,3,3,6.2,0
EST-0620,Hibrid,Doesn't Work,0.4395,6,2,5.88,0
EST-0621,In Person,Doesn't Work,0.7315,3,1,4.14,0
EST-0622,In Person,Works,0.7355,2,2,4.92,0
EST-0623,In Person,Doesn't Work,0.8288,1,0,5.78,0
EST-0624,In Person,Doesn't Work,0.5509,5,3,4.31,0
EST-0625,In Person,Works,0.5188,2,2,5.03,0
EST-0626,In Person,Doesn't Work,0.9413,6,5,5.88,0
EST-0627,In Person,Works,0.5518,4,2,4.69,0
EST-0628,Hibrid,Doesn't Work,0.5698,6,5,4.44,0
EST-0629,In Person,Works,0.686,6,4,3.58,0
EST-0630,In Person,Doesn't Work,0.8883,3,0,4.85,0
EST-0631,Hibrid,Doesn't Work,0.7019,4,3,5.9,0
EST-0632,Hibrid,Works,0.6749,2,1,4.41,0
EST-0633,In Person,Doesn't Work,0.9184,6,6,5.22,0
EST-0634,In Person,Doesn't Work,0.8753,2,2,5.0,0
EST-0635,In Person,Doesn't Work,0.826,5,4,4.6,0
EST-0636,In Person,Works,0.6176,3,3,5.95,0
EST-0637,In Person,Works,0.5809,2,2,6.68,0
EST-0638,In Person,Works,0.6663,4,4,6.1,0
EST-0639,In Person,Doesn't Work,0.7459,6,4,4.57,0
EST-0640,In Person,Doesn't Work,0.8043,5,5,6.99,0
EST-0641,In Person,Doesn't Work,0.8162,4,3,6.1,0
EST-0642,In Person,Doesn't Work,0.8756,3,3,4.79,0
EST-0643,In Person,Doesn't Work,0.7958,5,4,4.5,0
EST-0644,In Person,Doesn't Work,0.5524,5,5,6.21,0
EST-0645,In Person,Works,0.3587,5,3,5.36,1
EST-0646,Hibrid,Doesn't Work,0.7419,6,6,4.43,0
EST-0647,In Person,Works,0.4536,1,0,5.07,0
EST-0648,Hibrid,Doesn't Work,0.5162,5,2,4.83,1
EST-0649,Hibrid,Doesn't Work,0.6477,4,3,4.46,0
EST-0650,In Person,Doesn't Work,0.6805,4,4,6.11,0
EST-0651,In Person,Works,0.7872,3,2,5.9,1
EST-0652,In Person,Works,0.8266,2,1,5.95,0
EST-0653,In Person,Works,0.55,2,2,5.94,0
EST-0654,Hibrid,Works,0.7651,1,1,4.38,0
EST-0655,In Person,Works,0.6568,6,3,4.66,1
EST-0656,In Person,Doesn't Work,0.6923,6,4,6.51,0
EST-0657,In Person,Doesn't Work,0.8652,5,3,5.3,0
EST-0658,In Person,Works,0.5887,1,0,3.88,0
EST-0659,In Person,Works,0.7126,3,1,5.4,0
EST-0660,In Person,Works,0.907,3,3,5.79,0
EST-0661,In Person,Doesn't Work,0.9434,2,2,5.67,0
EST-0662,Hibrid,Doesn't Work,0.6559,3,1,4.36,0
EST-0663,In Person,Doesn't Work,0.7158,5,4,6.9,0
EST-0664,In Person,Doesn't Work,0.8649,4,0,3.85,1
EST-0665,In Person,Doesn't Work,0.8838,3,3,5.3,0
EST-0666,In Person,Doesn't Work,0.7788,1,1,6.76,0
EST-0667,In Person,Works,0.6616,5,4,5.73,0
EST-0668,In Person,Doesn't Work,0.7614,3,3,5.95,0
EST-0669,In Person,Works,0.5022,6,4,4.25,1
EST-0670,In Person,Doesn't Work,0.8911,3,2,5.51,0
EST-0671,In Person,Works,0.9113,6,6,5.17,0
EST-0672,Hibrid,Doesn't Work,0.7407,5,3,4.18,0
EST-0673,In Person,Doesn't Work,0.722,3,2,5.98,0
EST-0674,In Person,Doesn't Work,0.8013,6,5,4.46,0
EST-0675,In Person,Doesn't Work,0.8878,4,4,4.57,0
EST-0676,Hibrid,Works,0.7359,2,1,5.59,0
EST-0677,In Person,Works,0.5724,5,2,5.07,0
EST-0678,In Person,Works,0.3435,6,4,5.35,0
EST-0679,Hibrid,Works,0.49,1,0,5.02,0
EST-0680,Hibrid,Doesn't Work,0.9257,3,2,6.47,0
EST-0681,Hibrid,Works,0.711,6,4,4.37,0
EST-0682,In Person,Doesn't Work,0.7724,6,3,4.78,0
EST-0683,In Person,Doesn't Work,0.5953,6,3,5.63,0
EST-0684,Hibrid,Doesn't Work,0.8997,2,1,4.31,0
EST-0685,Hibrid,Works,0.6827,1,1,4.27,0
EST-0686,In Person,Doesn't Work,0.7329,1,0,5.32,0
EST-0687,In Person,Doesn't Work,0.987,4,4,5.98,0
EST-0688,In Person,Doesn't Work,0.8542,2,1,6.1,0
EST-0689,Hibrid,Doesn't Work,0.7779,6,6,5.28,0
EST-0690,In Person,Doesn't Work,0.8048,5,4,4.6,0
EST-0691,In Person,Works,0.6567,2,2,5.45,0
EST-0692,In Person,Doesn't Work,0.7141,2,2,5.71,0
EST-0693,In Person,Works,0.673,1,1,5.44,0
EST-0694,In Person,Works,0.6169,4,2,6.31,0
EST-0695,In Person,Works,0.3468,4,0,4.29,1
EST-0696,In Person,Works,0.4594,2,1,4.57,0
EST-0697,In Person,Works,0.6535,2,2,6.19,0
EST-0698,In Person,Works,0.3848,3,1,3.87,0
EST-0699,In Person,Doesn't Work,0.9052,6,5,4.71,0
EST-0700,In Person,Doesn't Work,0.9173,1,1,4.54,0
EST-0701,In Person,Doesn't Work,0.8689,6,4,3.79,0
EST-0702,In Person,Works,0.6177,5,3,5.03,0
EST-0703,In Person,Works,0.4244,5,4,5.87,0
EST-0704,In Person,Doesn't Work,0.8932,4,4,5.34,0
EST-0705,In Person,Works,0.3647,1,1,5.92,0
EST-0706,Hibrid,Works,0.7558,6,3,6.27,0
EST-0707,In Person,Doesn't Work,0.8023,4,3,4.85,0
EST-0708,Hibrid,Works,0.3351,5,3,7.0,0
EST-0709,In Person,Doesn't Work,0.8709,2,2,4.89,0
EST-0710,Hibrid,Doesn't Work,0.8532,1,1,5.01,0
EST-0711,Hibrid,Works,0.5775,1,1,4.56,0
EST-0712,In Person,Works,0.6206,6,6,5.69,0
EST-0713,In Person,Doesn't Work,0.9094,3,3,4.94,0
EST-0714,In Person,Doesn't Work,0.3975,6,3,5.13,0
EST-0715,In Person,Works,0.5879,2,2,5.02,0
EST-0716,Hibrid,Doesn't Work,0.8733,2,1,4.99,0
EST-0717,Hibrid,Works,0.5413,4,0,5.34,0
EST-0718,Hibrid,Doesn't Work,0.7203,1,1,4.79,0
EST-0719,In Person,Doesn't Work,0.9358,4,3,4.84,0
EST-0720,Hibrid,Works,0.5708,2,0,6.17,0
EST-0721,Hibrid,Works,0.4755,3,1,4.94,0
EST-0722,In Person,Works,0.8109,1,1,5.55,0
EST-0723,In Person,Works,0.7282,2,1,4.4,0
EST-0724,Hibrid,Works,0.5922,3,3,4.8,0
EST-0725,In Person,Doesn't Work,0.9666,6,5,4.09,0
EST-0726,Hibrid,Works,0.6508,5,5,6.87,0
EST-0727,In Person,Doesn't Work,0.6091,2,2,6.44,0
EST-0728,Hibrid,Doesn't Work,0.8511,5,4,5.74,0
EST-0729,In Person,Doesn't Work,0.7886,4,4,7.0,0
EST-0730,In Person,Doesn't Work,0.7371,5,3,5.89,0
EST-0731,In Person,Works,0.1621,2,1,5.05,1
EST-0732,Hibrid,Doesn't Work,0.8234,5,4,5.54,0
EST-0733,Hibrid,Doesn't Work,0.6863,1,1,6.42,0
EST-0734,In Person,Works,0.9678,2,2,4.41,0
EST-0735,In Person,Works,0.6873,3,2,4.14,0
EST-0736,Hibrid,Doesn't Work,0.8495,5,2,5.57,0
EST-0737,In Person,Works,0.5639,1,1,4.67,0
EST-0738,Hibrid,Works,0.8356,4,3,3.17,0
EST-0739,In Person,Works,0.8893,4,2,4.08,0
EST-0740,In Person,Works,0.4776,5,4,4.92,0
EST-0741,In Person,Doesn't Work,0.405,4,1,5.24,0
EST-0742,In Person,Doesn't Work,0.79,3,2,5.27,0
EST-0743,Hibrid,Doesn't Work,0.7563,2,2,7.0,0
EST-0744,In Person,Works,0.7712,1,1,5.01,0
EST-0745,In Person,Works,0.4684,6,4,5.45,0
EST-0746,Hibrid,Doesn't Work,0.7022,5,4,3.74,0
EST-0747,In Person,Works,0.6261,4,1,5.17,0
EST-0748,In Person,Works,0.7429,2,1,5.5,0
EST-0749,Hibrid,Doesn't Work,0.8605,6,4,3.96,0
EST-0750,In Person,Works,0.5618,3,1,4.36,0
EST-0751,In Person,Doesn't Work,0.7276,4,4,5.64,0
EST-0752,Hibrid,Works,0.8703,4,3,4.71,0
EST-0753,In Person,Works,0.6351,1,1,5.11,0
EST-0754,In Person,Works,0.5216,3,1,5.69,0
EST-0755,In Person,Doesn't Work,0.7512,3,3,7.0,0
EST-0756,Hibrid,Doesn't Work,0.7774,5,3,5.0,0
EST-0757,In Person,Doesn't Work,0.8395,6,4,6.79,0
EST-0758,In Person,Doesn't Work,0.6824,1,1,3.44,0
EST-0759,Hibrid,Works,0.6051,5,5,5.33,0
EST-0760,Hibrid,Doesn't Work,0.4983,1,1,4.9,0
EST-0761,In Person,Doesn't Work,0.712,3,2,4.72,0
EST-0762,In Person,Doesn't Work,0.9505,1,1,5.79,0
EST-0763,In Person,Works,0.47,1,0,5.01,0
EST-0764,Hibrid,Doesn't Work,0.2532,1,0,3.38,0
EST-0765,In Person,Works,0.4335,2,1,3.77,0
EST-0766,In Person,Works,0.5995,2,1,5.85,0
EST-0767,Hibrid,Doesn't Work,0.7729,6,5,3.73,0
EST-0768,Hibrid,Doesn't Work,0.5264,6,5,5.87,0
EST-0769,In Person,Works,0.8682,3,1,5.31,0
EST-0770,Hibrid,Works,0.3554,1,1,5.05,0
EST-0771,In Person,Doesn't Work,0.871,3,3,5.13,0
EST-0772,Hibrid,Doesn't Work,0.7213,2,2,5.66,0
EST-0773,In Person,Doesn't Work,0.5497,1,0,4.14,0
EST-0774,In Person,Doesn't Work,0.827,5,3,5.24,0
EST-0775,Hibrid,Doesn't Work,0.9162,2,1,4.17,0
EST-0776,In Person,Works,0.8034,2,2,5.35,0
EST-0777,In Person,Doesn't Work,0.7758,2,1,5.78,0
EST-0778,Hibrid,Works,0.4356,1,0,5.35,0
EST-0779,Hibrid,Works,0.9443,6,5,5.75,0
EST-0780,In Person,Doesn't Work,0.6874,4,2,4.57,0
EST-0781,Hibrid,Works,0.7383,5,5,5.14,0
EST-0782,In Person,Doesn't Work,0.8694,5,2,4.99,0
EST-0783,In Person,Doesn't Work,0.7445,3,3,4.93,0
EST-0784,In Person,Doesn't Work,0.5816,1,1,4.48,0
EST-0785,In Person,Doesn't Work,0.8644,6,4,4.81,0
EST-0786,Hibrid,Doesn't Work,0.8642,2,1,4.2,0
EST-0787,In Person,Doesn't Work,0.8802,4,2,3.68,0
EST-0788,In Person,Doesn't Work,0.4439,1,0,4.57,0
EST-0789,In Person,Works,0.8416,1,1,5.76,0
EST-0790,Hibrid,Doesn't Work,0.8851,6,4,5.41,0
EST-0791,In Person,Works,0.9004,4,4,4.07,0
EST-0792,In Person,Doesn't Work,0.8986,2,1,4.83,0
EST-0793,In Person,Doesn't Work,0.8944,1,1,5.82,0
EST-0794,In Person,Doesn't Work,0.91,5,3,4.8,0
EST-0795,In Person,Works,0.4734,5,1,3.92,1
EST-0796,Hibrid,Works,0.5265,4,3,7.0,0
EST-0797,Hibrid,Doesn't Work,0.854,2,1,5.05,0
EST-0798,In Person,Works,0.5003,4,2,4.15,0
EST-0799,Hibrid,Works,0.4936,1,0,5.35,0
EST-0800,Hibrid,Works,0.7054,5,3,5.12,0
EST-0801,Hibrid,Doesn't Work,0.7232,3,1,3.81,0
EST-0802,In Person,Doesn't Work,0.8453,4,2,5.28,0
EST-0803,In Person,Works,0.526,3,2,6.09,0
EST-0804,In Person,Works,0.7476,3,3,4.99,0
EST-0805,In Person,Works,0.6646,6,5,3.35,1
EST-0806,Hibrid,Doesn't Work,0.9614,5,5,5.51,0
EST-0807,Hibrid,Doesn't Work,0.76,4,4,5.74,0
EST-0808,Hibrid,Works,0.4768,3,1,4.67,0
EST-0809,In Person,Works,0.4448,1,1,4.68,0
EST-0810,In Person,Works,0.8488,6,6,7.0,0
EST-0811,Hibrid,Doesn't Work,0.991,2,2,4.61,0
EST-0812,Hibrid,Doesn't Work,0.8981,6,4,4.19,0
EST-0813,In Person,Works,0.5122,2,1,5.22,0
EST-0814,Hibrid,Works,0.6461,5,4,5.72,0
EST-0815,Hibrid,Works,0.4979,4,1,5.68,0
EST-0816,In Person,Doesn't Work,0.819,4,4,2.98,0
EST-0817,In Person,Works,0.575,2,0,4.61,0
EST-0818,In Person,Doesn't Work,0.6829,1,1,6.24,0
EST-0819,In Person,Doesn't Work,0.7711,2,2,5.49,0
EST-0820,In Person,Works,0.4745,2,1,4.09,0
EST-0821,Hibrid,Works,0.6051,4,2,6.32,0
EST-0822,In Person,Works,0.592,3,2,6.07,0
EST-0823,In Person,Doesn't Work,0.5859,1,0,4.93,0
EST-0824,In Person,Doesn't Work,0.7647,4,4,5.99,0
EST-0825,In Person,Doesn't Work,0.8292,6,6,7.0,0
EST-0826,Hibrid,Doesn't Work,0.6441,2,2,4.25,0
EST-0827,In Person,Works,0.908,3,3,5.7,0
EST-0828,In Person,Doesn't Work,0.9717,1,1,6.99,0
EST-0829,Hibrid,Doesn't Work,0.9533,5,3,5.79,1
EST-0830,In Person,Works,0.4897,1,1,4.52,0
EST-0831,In Person,Doesn't Work,0.8107,1,1,5.42,0
EST-0832,In Person,Doesn't Work,0.7784,2,2,6.43,0
EST-0833,In Person,Doesn't Work,0.7838,2,1,4.29,0
EST-0834,In Person,Doesn't Work,0.7382,5,1,5.17,0
EST-0835,In Person,Doesn't Work,0.7662,5,3,4.98,0
EST-0836,In Person,Doesn't Work,0.7578,5,1,4.63,0
EST-0837,Hibrid,Works,0.5817,4,2,5.42,0
EST-0838,In Person,Works,0.581,4,1,5.11,0
EST-0839,In Person,Doesn't Work,0.6498,2,1,5.78,0
EST-0840,In Person,Works,0.5456,2,2,6.23,0
EST-0841,In Person,Works,0.7961,5,4,4.05,0
EST-0842,Hibrid,Works,0.4246,4,2,5.26,0
EST-0843,In Person,Works,0.38,3,2,5.15,0
EST-0844,Hibrid,Doesn't Work,0.8711,4,4,5.38,0
EST-0845,In Person,Doesn't Work,0.8935,2,2,5.19,0
EST-0846,In Person,Works,0.659,4,1,4.32,1
EST-0847,In Person,Works,0.6763,2,2,6.51,0
EST-0848,Hibrid,Works,0.8166,5,4,6.89,0
EST-0849,In Person,Doesn't Work,0.8206,5,4,5.69,0
EST-0850,In Person,Doesn't Work,0.864,5,3,5.74,0
EST-0851,In Person,Works,0.4204,3,0,5.32,0
EST-0852,Hibrid,Doesn't Work,0.7024,3,2,6.07,0
EST-0853,In Person,Doesn't Work,0.8961,2,1,3.0,0
EST-0854,Hibrid,Doesn't Work,0.8374,1,1,5.2,1
EST-0855,In Person,Doesn't Work,0.6827,6,4,4.68,0
EST-0856,In Person,Works,0.4638,3,1,5.33,0
EST-0857,In Person,Doesn't Work,0.9015,2,1,6.18,0
EST-0858,In Person,Doesn't Work,0.4395,4,1,5.22,0
EST-0859,In Person,Doesn't Work,0.9193,5,5,4.88,0
EST-0860,In Person,Doesn't Work,0.7737,5,3,3.94,0
EST-0861,Hibrid,Doesn't Work,0.4594,5,3,5.1,0
EST-0862,In Person,Doesn't Work,0.737,2,2,5.81,0
EST-0863,In Person,Works,0.8178,5,4,4.44,0
EST-0864,In Person,Doesn't Work,0.8709,3,3,3.4,0
EST-0865,In Person,Doesn't Work,0.9153,4,3,7.0,0
EST-0866,In Person,Doesn't Work,0.7019,5,4,5.5,1
EST-0867,In Person,Works,0.6928,2,2,5.0,0
EST-0868,In Person,Works,0.5929,4,1,6.45,0
EST-0869,Hibrid,Works,0.6602,5,4,5.33,0
EST-0870,Hibrid,Doesn't Work,0.916,1,1,6.4,0
EST-0871,In Person,Doesn't Work,0.51,5,4,6.32,0
EST-0872,In Person,Works,0.71,6,5,4.05,0
EST-0873,In Person,Doesn't Work,0.8118,3,3,5.61,0
EST-0874,In Person,Doesn't Work,0.7486,4,4,6.75,0
EST-0875,In Person,Doesn't Work,0.8053,3,2,4.3,0
EST-0876,In Person,Doesn't Work,0.4467,1,0,4.31,0
EST-0877,Hibrid,Doesn't Work,0.7554,1,1,4.46,0
EST-0878,Hibrid,Doesn't Work,0.9519,6,5,6.0,0
EST-0879,In Person,Works,0.7253,3,2,5.46,0
EST-0880,In Person,Doesn't Work,0.9009,4,2,3.46,0
EST-0881,In Person,Doesn't Work,0.917,1,1,5.51,0
EST-0882,Hibrid,Doesn't Work,0.6903,1,1,4.97,0
EST-0883,In Person,Doesn't Work,0.8637,5,3,5.1,0
EST-0884,In Person,Doesn't Work,0.7604,6,6,4.45,0
EST-0885,Hibrid,Works,0.5172,2,2,5.51,0
EST-0886,In Person,Doesn't Work,0.8027,1,1,6.15,0
EST-0887,Hibrid,Works,0.5314,4,2,4.62,0
EST-0888,In Person,Works,0.4557,4,3,3.69,0
EST-0889,In Person,Doesn't Work,0.8881,5,3,4.93,0
EST-0890,In Person,Doesn't Work,0.7741,4,4,5.88,0
EST-0891,Hibrid,Works,0.6341,5,3,4.82,0
EST-0892,Hibrid,Works,0.6969,1,1,3.85,0
EST-0893,Hibrid,Doesn't Work,0.695,1,0,5.89,0
EST-0894,In Person,Works,0.6254,1,0,6.47,0
EST-0895,Hibrid,Doesn't Work,0.8777,1,0,5.22,0
EST-0896,In Person,Doesn't Work,0.8763,2,2,3.75,0
EST-0897,In Person,Doesn't Work,0.7246,6,3,4.91,0
EST-0898,Hibrid,Works,0.9088,6,6,5.34,0
EST-0899,In Person,Works,0.6944,2,2,4.47,0
EST-0900,In Person,Doesn't Work,0.847,1,0,5.73,0
EST-0901,In Person,Doesn't Work,0.8236,4,4,4.38,0
EST-0902,In Person,Works,0.8003,5,2,5.22,0
EST-0903,In Person,Doesn't Work,0.851,6,6,4.99,0
EST-0904,In Person,Works,0.7336,3,2,5.75,0
EST-0905,In Person,Doesn't Work,0.9082,5,5,5.19,0
EST-0906,Hibrid,Works,0.5954,3,2,5.02,0
EST-0907,Hibrid,Doesn't Work,0.918,3,3,4.48,0
EST-0908,In Person,Works,0.8543,1,1,5.23,0
EST-0909,In Person,Works,0.6137,3,2,4.97,0
EST-0910,In Person,Works,0.6983,1,0,4.42,0
EST-0911,In Person,Works,0.5287,2,2,5.34,0
EST-0912,In Person,Doesn't Work,0.4495,4,4,5.53,0
EST-0913,In Person,Doesn't Work,0.7452,3,3,4.9,0
EST-0914,In Person,Works,0.5019,4,3,6.04,0
EST-0915,In Person,Works,0.6695,1,0,5.44,0
EST-0916,In Person,Doesn't Work,0.7651,3,1,4.04,0
EST-0917,In Person,Doesn't Work,0.6845,5,5,6.4,0
EST-0918,Hibrid,Doesn't Work,0.5805,2,1,5.89,0
EST-0919,Hibrid,Works,0.2397,6,4,6.1,1
EST-0920,In Person,Works,0.8218,2,2,5.56,0
EST-0921,Hibrid,Doesn't Work,0.7389,3,2,3.1,0
EST-0922,In Person,Works,0.6042,2,1,6.18,0
EST-0923,Hibrid,Doesn't Work,0.9085,6,6,5.25,0
EST-0924,Hibrid,Works,0.7835,5,5,5.52,0
EST-0925,In Person,Doesn't Work,0.963,4,4,5.69,0
EST-0926,In Person,Works,0.6613,5,4,6.94,0
EST-0927,In Person,Doesn't Work,0.8791,2,2,4.71,0
EST-0928,In Person,Doesn't Work,0.4798,6,4,4.73,0
EST-0929,In Person,Doesn't Work,0.767,4,3,4.43,0
EST-0930,In Person,Works,0.5762,6,5,4.24,0
EST-0931,In Person,Doesn't Work,0.6812,5,4,4.25,0
EST-0932,Hibrid,Doesn't Work,0.8798,1,0,3.53,0
EST-0933,Hibrid,Doesn't Work,0.9557,1,1,5.58,0
EST-0934,In Person,Doesn't Work,0.606,4,3,6.99,0
EST-0935,In Person,Doesn't Work,0.8219,4,4,7.0,0
EST-0936,In Person,Doesn't Work,0.7616,4,2,4.43,0
EST-0937,In Person,Works,0.4055,1,0,4.96,1
EST-0938,Hibrid,Doesn't Work,0.7439,5,5,6.31,0
EST-0939,In Person,Doesn't Work,0.5817,4,2,5.59,0
EST-0940,In Person,Works,0.445,1,1,5.32,0
EST-0941,In Person,Works,0.6667,4,2,4.97,0
EST-0942,In Person,Works,0.4315,4,3,6.26,0
EST-0943,Hibrid,Doesn't Work,0.6087,6,1,4.23,0
EST-0944,Hibrid,Doesn't Work,0.7336,4,2,5.63,0
EST-0945,In Person,Doesn't Work,0.522,1,0,5.51,0
EST-0946,In Person,Doesn't Work,0.8578,5,4,5.24,0
EST-0947,In Person,Works,0.6535,1,1,5.91,0
EST-0948,Hibrid,Works,0.865,1,1,5.23,0
EST-0949,Hibrid,Works,0.206,4,3,4.72,0
EST-0950,In Person,Works,0.8106,6,3,5.28,0
EST-0951,In Person,Works,0.6496,3,2,3.77,0
EST-0952,Hibrid,Works,0.7538,3,2,5.75,0
EST-0953,In Person,Doesn't Work,0.6387,3,3,5.98,0
EST-0954,Hibrid,Doesn't Work,0.8318,1,1,4.61,0
EST-0955,In Person,Doesn't Work,0.9012,4,4,4.39,0
EST-0956,In Person,Doesn't Work,0.7585,3,2,4.64,0
EST-0957,In Person,Doesn't Work,0.9471,2,1,5.14,0
EST-0958,In Person,Works,0.4125,2,2,5.39,0
EST-0959,In Person,Doesn't Work,0.9322,2,2,6.65,0
EST-0960,Hibrid,Doesn't Work,0.9738,5,5,6.54,0
EST-0961,In Person,Works,0.7351,4,4,5.95,0
EST-0962,In Person,Works,0.9117,1,1,5.31,0
EST-0963,In Person,Doesn't Work,0.5453,1,0,4.37,0
EST-0964,In Person,Works,0.6382,4,4,6.71,0
EST-0965,In Person,Doesn't Work,0.6585,6,3,4.98,0
EST-0966,In Person,Doesn't Work,0.725,6,2,5.45,0
EST-0967,Hibrid,Works,0.8863,5,5,5.76,0
EST-0968,In Person,Works,0.5811,4,3,5.26,0
EST-0969,In Person,Works,0.3579,6,4,6.24,0
EST-0970,In Person,Works,0.6619,5,3,5.61,0
EST-0971,Hibrid,Works,0.6717,3,3,6.09,0
EST-0972,Hibrid,Doesn't Work,0.8062,3,3,5.6,0
EST-0973,Hibrid,Works,0.7957,5,4,5.17,0
EST-0974,In Person,Doesn't Work,0.9196,4,2,5.27,0
EST-0975,In Person,Works,0.5237,3,2,5.14,0
EST-0976,In Person,Doesn't Work,0.5769,4,2,4.04,0
EST-0977,In Person,Works,0.5006,6,1,4.3,1
EST-0978,In Person,Works,0.4946,3,1,5.87,0
EST-0979,In Person,Doesn't Work,0.7617,6,3,4.07,0
EST-0980,In Person,Doesn't Work,0.7986,4,3,4.84,0
EST-0981,In Person,Works,0.5092,4,0,4.25,1
EST-0982,Hibrid,Doesn't Work,0.7538,2,1,4.79,0
EST-0983,In Person,Doesn't Work,0.8375,5,3,5.06,0
EST-0984,In Person,Doesn't Work,0.8492,1,1,4.83,0
EST-0985,Hibrid,Works,0.6702,5,2,3.91,0
EST-0986,In Person,Works,0.6499,4,3,5.03,0
EST-0987,Hibrid,Doesn't Work,0.9664,4,4,4.34,0
EST-0988,In Person,Doesn't Work,0.6941,5,2,4.63,0
EST-0989,In Person,Works,0.8366,4,4,5.87,0
EST-0990,Hibrid,Doesn't Work,0.7888,6,5,6.06,0
EST-0991,Hibrid,Doesn't Work,0.8725,3,3,4.02,0
EST-0992,Hibrid,Works,0.608,4,2,6.61,0
EST-0993,In Person,Doesn't Work,0.9508,5,5,4.74,0
EST-0994,In Person,Works,0.6155,5,2,4.66,1
EST-0995,In Person,Doesn't Work,0.7019,5,3,4.4,0
EST-0996,In Person,Doesn't Work,0.6444,2,2,5.25,0
EST-0997,Hibrid,Doesn't Work,0.6166,6,5,6.98,0
EST-0998,In Person,Works,0.6534,6,4,3.81,0
EST-0999,Hibrid,Works,0.8845,2,2,3.35,0
EST-1000,In Person,Works,0.5335,6,4,5.44,0

```

| Nombre de Variable      | Descripción Operativa                                                          | Tipo de Dato       | Rango / Valores         | Rol en el Modelo                     |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------ | ----------------------- | ------------------------------------ |
| `id`                    | Identificador sintético único generado para cada estudiante                    | Texto (String)     | "EST-0001" a "EST-1000" | Identificador (Sin poder predictivo) |
| `mode`                  | Modalidad en la que el estudiante asiste a sus clases                          | Categórica         | "In Person", "Hibrid"   | Feature (Predictora)                 |
| `working_condition`     | Condición laboral actual que indica si el estudiante mantiene empleo           | Categórica         | "Works", "Doesn't Work" | Feature (Predictora)                 |
| `attendance_percentage` | Proporción histórica de asistencia a clases durante el periodo                 | Numérico (Float)   | 0.0000 a 1.0000         | Feature (Predictora)                 |
| `academic_load`         | Número total de asignaturas inscritas por el estudiante                        | Numérico (Integer) | 1 a 6                   | Feature (Predictora)                 |
| `subjects_passed`       | Cantidad de asignaturas aprobadas (menor o igual a la carga académica)         | Numérico (Integer) | 0 a `academic_load`     | Feature (Predictora)                 |
| `previous_average`      | Promedio de calificaciones históricas acumuladas antes del periodo actual      | Numérico (Float)   | 1.00 a 7.00             | Feature (Predictora)                 |
| `dropping_out_risk`     | Etiqueta de propensión a desertar (1 = En riesgo de abandonar, 0 = Sin riesgo) | Binaria (Integer)  | 0, 1                    | Target (Variable Objetivo)           |

## Prompt Utilizado 2

Actúa como un generador determinista de datos sintéticos educativos para un caso docente ficticio sobre riesgo de no continuidad estudiantil en educación superior.

Toma en cuenta las siguientes especificaciones obligatorias:

1. REQUISITO FUNDAMENTAL DE SALIDA:
   - Genera DIRECTAMENTE las filas del dataset sintético en texto plano con formato CSV (separado por comas), listo para ser copiado o guardado en un archivo .csv.

2. CAMBIOS EN LA COMPOSICIÓN (DATA DRIFT):
   - Mantén las mismas columnas y definiciones que la cohorte base, pero altera la composición demográfica y académica para simular un cambio en la población.
   - Aumenta la proporción de estudiantes que trabajan (`working_condition` = "Works") para que representen aproximadamente el 75% de la muestra.
   - Aumenta la proporción de estudiantes en modalidad híbrida (`mode` = "Hibrid") para que representen aproximadamente el 65% de la muestra.
   - Ajusta de manera coherente las variables dependientes (ej. una caída general en `attendance_percentage` o un cambio en `academic_load` debido a la mayor proporción de alumnos trabajadores).

3. REPRODUCIBILIDAD Y DETERMINISMO:
   - Asume una lógica de generación fija e invariable asociada a la SEMILLA = 2026, asegurando que todos los registros generados sean consistentes y reproducibles, manteniendo un comportamiento estrictamente determinista en tus probabilidades.

4. ESTRUCTURA Y COLUMNAS OBLIGATORIAS:
   Genera exactamente 1000 filas de datos con las siguientes columnas exactas en la primera línea (header):
   id, mode, working_condition, attendance_percentage, academic_load, subjects_passed, previous_average, dropping_out_risk

   Especificaciones por variable:
   - `id`: Identificador sintético único ("EST-0001" a "EST-1000").
   - `mode`: Categórica ("In Person" o "Hibrid").
   - `working_condition`: Categórica ("Works" o "Doesn't Work").
   - `attendance_percentage`: Numérico continuo entre 0.0 y 1.0 (con 4 decimales).
   - `academic_load`: Numérico entero entre 1 y 6 asignaturas.
   - `subjects_passed`: Numérico entero entre 0 y el valor de `academic_load`.
   - `previous_average`: Numérico continuo entre 1.0 y 7.0 (con 2 decimales).
   - `dropping_out_risk`: Variable binaria (0 = sin riesgo, 1 = en riesgo / requiere apoyo de orientación).

5. LÓGICA DE LA VARIABLE OBJETIVO (SIN FILTRACIÓN / TARGET LEAKAGE):
   - La asignación de `dropping_out_risk` debe ser probabilística y realista: mayor propensión a riesgo (1) cuando baja la asistencia, cuando aumentan las asignaturas reprobadas o si el estudiante trabaja.
   - La regla que conecta las variables predictoras con `dropping_out_risk` debe ser exactamente la misma que en un escenario normal, solo cambiará la tasa final de riesgo debido a que ahora hay más estudiantes con características de riesgo (trabajadores/híbridos).
   - La relación NO debe ser determinista ni depender al 100% de una sola variable.

6. FORMATO DE LA RESPUESTA ESPERADA:
   - Entregable 1: El dataset (1000 filas exactas) en un bloque de código en formato CSV listo para descargar/copiar.
   - Entregable 2: Una tabla en formato Markdown con el Diccionario de Datos estructurado de la siguiente forma:
     | Nombre de Variable | Descripción Operativa | Tipo de Dato | Rango / Valores | Rol en el Modelo |

## Respuesta

```python
import numpy as np
import pandas as pd

np.random.seed(2026)
N = 1000

# 1. id
ids = [f"EST-{str(i).zfill(4)}" for i in range(1, N+1)]

# 2. mode & working_condition
modes = np.random.choice(["Hibrid", "In Person"], p=[0.65, 0.35], size=N)
work_conds = np.random.choice(["Works", "Doesn't Work"], p=[0.75, 0.25], size=N)

# Initialize arrays
academic_load = np.zeros(N, dtype=int)
attendance_percentage = np.zeros(N, dtype=float)
previous_average = np.zeros(N, dtype=float)
subjects_passed = np.zeros(N, dtype=int)
dropping_out_risk = np.zeros(N, dtype=int)

for i in range(N):
    is_working = work_conds[i] == "Works"
    is_hibrid = modes[i] == "Hibrid"

    # academic_load: workers take fewer classes
    if is_working:
        academic_load[i] = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.35, 0.25, 0.08, 0.02])
    else:
        academic_load[i] = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.02, 0.03, 0.1, 0.2, 0.4, 0.25])

    # attendance_percentage
    if is_working:
        # workers have lower attendance, hybrid helps a bit
        base_mean = 0.70 if is_hibrid else 0.60
    else:
        base_mean = 0.85 if is_hibrid else 0.90

    # sample attendance from beta distribution
    a = base_mean * 10
    b = (1 - base_mean) * 10
    att = np.random.beta(a, b)
    attendance_percentage[i] = np.clip(att, 0.0, 1.0)

    # previous_average: correlated slightly with attendance + noise
    base_avg = 4.0 + (attendance_percentage[i] * 2.0) + np.random.normal(0, 0.5)
    previous_average[i] = np.clip(base_avg, 1.0, 7.0)

    # subjects_passed: binomial based on attendance and previous avg
    pass_prob = 0.3 + (0.4 * attendance_percentage[i]) + (0.3 * (previous_average[i] - 1)/6)
    pass_prob = np.clip(pass_prob, 0.05, 0.95)
    subjects_passed[i] = np.random.binomial(academic_load[i], pass_prob)

    # dropping_out_risk
    # Higher risk if attendance is low, fail rate is high, and working
    fail_rate = (academic_load[i] - subjects_passed[i]) / academic_load[i]

    # Logit components
    logit = -3.0 # base risk
    logit += 2.5 * fail_rate
    logit += 2.0 * (1.0 - attendance_percentage[i])
    logit += 0.8 * (1 if is_working else 0)
    logit -= 0.5 * ((previous_average[i] - 4.0) / 3.0)

    prob_risk = 1.0 / (1.0 + np.exp(-logit))
    dropping_out_risk[i] = np.random.binomial(1, prob_risk)

# Build DataFrame
df = pd.DataFrame({
    'id': ids,
    'mode': modes,
    'working_condition': work_conds,
    'attendance_percentage': np.round(attendance_percentage, 4),
    'academic_load': academic_load,
    'subjects_passed': subjects_passed,
    'previous_average': np.round(previous_average, 2),
    'dropping_out_risk': dropping_out_risk
})

# Generate CSV string
csv_str = df.to_csv(index=False)
with open("data.csv", "w") as f:
    f.write(csv_str)
print(f"Stats:\nWork: {(df['working_condition'] == 'Works').mean()}\nMode: {(df['mode'] == 'Hibrid').mean()}")


```

```text
Stats:
Work: 0.749
Mode: 0.647


```

```python
with open("data.csv", "r") as f:
    print(f.read())

```

```text
id,mode,working_condition,attendance_percentage,academic_load,subjects_passed,previous_average,dropping_out_risk
EST-0001,Hibrid,Works,0.7706,3,3,5.51,0
EST-0002,Hibrid,Works,0.7964,4,4,5.34,0
EST-0003,In Person,Works,0.3909,5,3,5.73,0
EST-0004,Hibrid,Doesn't Work,0.9087,4,3,5.98,0
EST-0005,Hibrid,Doesn't Work,0.851,5,4,5.68,0
EST-0006,In Person,Doesn't Work,0.9534,4,3,5.3,0
EST-0007,Hibrid,Doesn't Work,0.7953,4,2,5.53,1
EST-0008,In Person,Works,0.4427,6,4,5.3,0
EST-0009,Hibrid,Works,0.4905,2,1,4.42,0
EST-0010,In Person,Works,0.8176,2,2,5.26,0
EST-0011,Hibrid,Works,0.6707,3,3,4.98,0
EST-0012,Hibrid,Works,0.7616,3,3,6.14,0
EST-0013,Hibrid,Works,0.6613,3,2,5.47,0
EST-0014,In Person,Works,0.7422,3,3,5.01,0
EST-0015,Hibrid,Works,0.7577,3,2,6.0,1
EST-0016,Hibrid,Works,0.7777,2,2,7.0,0
EST-0017,Hibrid,Works,0.6759,1,1,4.9,0
EST-0018,Hibrid,Works,0.7626,5,5,6.42,0
EST-0019,In Person,Doesn't Work,0.9976,5,4,5.35,0
EST-0020,In Person,Works,0.4977,2,1,4.75,1
EST-0021,Hibrid,Works,0.8826,3,3,5.62,0
EST-0022,Hibrid,Doesn't Work,0.8175,4,4,6.71,0
EST-0023,Hibrid,Works,0.5355,3,1,4.05,1
EST-0024,Hibrid,Works,0.6695,2,1,5.0,0
EST-0025,Hibrid,Doesn't Work,0.5846,4,4,5.14,0
EST-0026,In Person,Works,0.8325,2,2,6.16,0
EST-0027,Hibrid,Works,0.6219,4,4,4.65,0
EST-0028,Hibrid,Works,0.7854,2,2,5.39,0
EST-0029,Hibrid,Works,0.4329,4,4,4.84,0
EST-0030,In Person,Works,0.9165,3,3,6.05,0
EST-0031,In Person,Works,0.7878,4,4,5.02,1
EST-0032,In Person,Doesn't Work,0.9494,3,3,6.44,0
EST-0033,Hibrid,Works,0.7764,3,2,5.66,0
EST-0034,Hibrid,Works,0.8715,1,1,5.34,0
EST-0035,Hibrid,Works,0.8027,3,3,5.59,1
EST-0036,Hibrid,Doesn't Work,0.7345,5,3,5.28,0
EST-0037,Hibrid,Doesn't Work,0.9114,4,4,5.76,0
EST-0038,Hibrid,Works,0.7108,3,0,5.4,1
EST-0039,Hibrid,Works,0.7823,5,5,6.22,0
EST-0040,Hibrid,Works,0.6752,2,1,5.32,0
EST-0041,In Person,Works,0.7376,3,3,5.91,0
EST-0042,Hibrid,Doesn't Work,0.9166,1,0,6.17,0
EST-0043,Hibrid,Doesn't Work,0.8432,4,2,5.58,0
EST-0044,Hibrid,Doesn't Work,0.804,3,3,5.51,0
EST-0045,Hibrid,Doesn't Work,0.8015,4,3,5.44,1
EST-0046,Hibrid,Doesn't Work,0.7259,6,5,5.01,0
EST-0047,In Person,Doesn't Work,0.8839,5,4,5.63,0
EST-0048,In Person,Works,0.5814,2,2,4.89,0
EST-0049,In Person,Works,0.3992,4,3,4.56,0
EST-0050,Hibrid,Works,0.831,2,2,5.57,0
EST-0051,Hibrid,Works,0.8147,3,1,5.69,1
EST-0052,In Person,Works,0.6301,3,1,4.96,0
EST-0053,Hibrid,Works,0.7365,4,4,5.58,0
EST-0054,Hibrid,Works,0.8377,2,2,5.33,0
EST-0055,Hibrid,Works,0.8361,1,1,6.02,0
EST-0056,In Person,Works,0.6383,4,2,5.42,0
EST-0057,In Person,Doesn't Work,0.9534,6,6,6.76,0
EST-0058,Hibrid,Works,0.704,2,2,5.85,0
EST-0059,Hibrid,Works,0.7416,1,1,6.21,1
EST-0060,In Person,Works,0.4671,4,3,4.56,0
EST-0061,In Person,Works,0.5105,4,1,4.91,0
EST-0062,Hibrid,Works,0.6908,4,3,6.25,0
EST-0063,Hibrid,Works,0.5093,5,4,4.7,0
EST-0064,Hibrid,Works,0.8598,3,3,5.72,1
EST-0065,In Person,Works,0.2991,4,3,5.05,0
EST-0066,Hibrid,Works,0.6815,4,3,4.88,1
EST-0067,Hibrid,Doesn't Work,0.9512,6,5,6.08,0
EST-0068,In Person,Works,0.8933,4,4,6.75,0
EST-0069,Hibrid,Works,0.8222,4,4,5.51,0
EST-0070,Hibrid,Doesn't Work,0.9128,5,5,5.36,0
EST-0071,Hibrid,Doesn't Work,0.8863,4,4,5.16,0
EST-0072,Hibrid,Doesn't Work,0.8183,5,5,6.43,0
EST-0073,Hibrid,Works,0.6303,4,3,5.69,0
EST-0074,Hibrid,Works,0.7693,2,2,5.36,0
EST-0075,Hibrid,Works,0.5093,1,0,4.81,1
EST-0076,In Person,Doesn't Work,0.9316,5,4,5.53,0
EST-0077,Hibrid,Works,0.6063,5,3,4.86,0
EST-0078,Hibrid,Works,0.6461,4,2,5.39,0
EST-0079,In Person,Doesn't Work,0.9303,5,5,6.46,1
EST-0080,In Person,Works,0.6003,4,3,4.69,1
EST-0081,Hibrid,Doesn't Work,0.8882,4,2,6.33,1
EST-0082,In Person,Works,0.4828,3,3,4.48,0
EST-0083,In Person,Doesn't Work,0.9607,5,5,5.65,0
EST-0084,In Person,Works,0.3611,3,2,5.57,0
EST-0085,Hibrid,Doesn't Work,0.8741,6,5,5.72,0
EST-0086,Hibrid,Works,0.7163,1,1,5.78,0
EST-0087,Hibrid,Works,0.6665,4,4,5.35,0
EST-0088,In Person,Works,0.4431,3,2,5.11,0
EST-0089,In Person,Works,0.9142,4,4,4.95,0
EST-0090,Hibrid,Doesn't Work,0.9405,6,5,5.51,1
EST-0091,Hibrid,Works,0.9089,4,3,5.74,0
EST-0092,In Person,Doesn't Work,0.7928,3,3,6.71,0
EST-0093,In Person,Doesn't Work,0.8273,5,5,5.24,0
EST-0094,Hibrid,Works,0.4886,1,1,4.29,0
EST-0095,Hibrid,Doesn't Work,0.9909,5,5,5.59,0
EST-0096,In Person,Works,0.8129,2,2,6.07,0
EST-0097,In Person,Works,0.901,3,3,5.56,1
EST-0098,Hibrid,Works,0.5964,3,2,5.71,0
EST-0099,In Person,Works,0.7509,2,2,4.95,1
EST-0100,Hibrid,Works,0.8134,5,5,5.43,0
EST-0101,In Person,Works,0.8228,6,5,6.25,0
EST-0102,Hibrid,Works,0.8579,5,5,6.53,0
EST-0103,In Person,Works,0.5568,4,3,5.88,0
EST-0104,Hibrid,Doesn't Work,0.8495,5,4,5.45,0
EST-0105,Hibrid,Works,0.5824,4,0,5.12,1
EST-0106,Hibrid,Works,0.8326,4,4,6.72,0
EST-0107,Hibrid,Works,0.7461,3,3,4.38,0
EST-0108,In Person,Doesn't Work,0.9616,5,4,5.62,0
EST-0109,Hibrid,Works,0.6316,2,2,5.07,1
EST-0110,Hibrid,Works,0.7374,4,4,6.26,1
EST-0111,In Person,Doesn't Work,0.8628,3,3,5.44,0
EST-0112,In Person,Doesn't Work,0.8322,4,4,5.46,0
EST-0113,Hibrid,Works,0.7927,2,2,6.03,0
EST-0114,In Person,Works,0.8941,4,4,5.47,0
EST-0115,In Person,Works,0.6095,3,3,4.89,0
EST-0116,Hibrid,Works,0.6729,2,2,5.14,1
EST-0117,Hibrid,Works,0.6716,2,2,5.6,0
EST-0118,Hibrid,Doesn't Work,0.7499,4,3,5.53,0
EST-0119,In Person,Doesn't Work,0.9135,5,5,6.86,0
EST-0120,In Person,Works,0.6527,3,3,4.73,0
EST-0121,In Person,Works,0.5738,2,2,4.95,0
EST-0122,Hibrid,Works,0.671,4,3,5.66,0
EST-0123,Hibrid,Works,0.5328,5,2,4.12,1
EST-0124,In Person,Works,0.7154,3,3,4.93,0
EST-0125,In Person,Works,0.5839,6,5,3.76,1
EST-0126,In Person,Doesn't Work,0.7551,4,2,4.86,0
EST-0127,Hibrid,Works,0.5193,2,2,5.63,1
EST-0128,Hibrid,Works,0.641,4,4,6.22,1
EST-0129,Hibrid,Doesn't Work,0.8177,5,5,4.86,0
EST-0130,In Person,Works,0.6815,3,3,5.53,0
EST-0131,Hibrid,Works,0.5767,4,1,4.78,0
EST-0132,Hibrid,Doesn't Work,0.8978,4,4,5.96,0
EST-0133,In Person,Doesn't Work,0.957,5,5,6.14,0
EST-0134,In Person,Works,0.7516,2,2,4.93,0
EST-0135,Hibrid,Doesn't Work,0.9159,5,5,6.36,0
EST-0136,Hibrid,Works,0.7,2,2,5.55,0
EST-0137,In Person,Works,0.6578,4,3,4.54,0
EST-0138,Hibrid,Works,0.6017,4,3,5.4,0
EST-0139,Hibrid,Doesn't Work,0.7656,5,5,5.26,0
EST-0140,Hibrid,Doesn't Work,0.6504,4,3,5.02,0
EST-0141,In Person,Works,0.4675,4,3,4.7,0
EST-0142,Hibrid,Works,0.7687,2,2,5.51,0
EST-0143,Hibrid,Works,0.6651,3,3,4.86,0
EST-0144,Hibrid,Works,0.4328,4,2,4.92,0
EST-0145,In Person,Works,0.7132,3,3,6.06,0
EST-0146,Hibrid,Works,0.6744,4,3,4.94,0
EST-0147,Hibrid,Works,0.6933,4,3,5.41,0
EST-0148,In Person,Works,0.49,4,3,5.07,0
EST-0149,Hibrid,Works,0.7338,4,4,5.03,0
EST-0150,Hibrid,Doesn't Work,0.9322,6,5,6.2,0
EST-0151,In Person,Works,0.5244,5,4,4.86,0
EST-0152,Hibrid,Works,0.7596,3,2,5.91,0
EST-0153,In Person,Doesn't Work,0.9078,4,4,6.3,0
EST-0154,Hibrid,Doesn't Work,0.8646,5,4,6.13,0
EST-0155,In Person,Works,0.7278,2,2,6.08,1
EST-0156,In Person,Works,0.4921,5,3,5.2,0
EST-0157,In Person,Works,0.7458,3,3,5.77,0
EST-0158,Hibrid,Doesn't Work,0.9911,5,3,6.49,0
EST-0159,In Person,Works,0.5025,4,2,5.21,1
EST-0160,Hibrid,Works,0.7897,2,2,5.63,0
EST-0161,Hibrid,Works,0.4977,1,1,5.01,1
EST-0162,Hibrid,Works,0.6932,1,1,5.01,1
EST-0163,Hibrid,Works,0.926,4,4,6.42,0
EST-0164,In Person,Works,0.8004,5,4,5.81,0
EST-0165,In Person,Works,0.7279,4,3,4.48,0
EST-0166,Hibrid,Works,0.5014,5,4,4.81,1
EST-0167,In Person,Works,0.7928,4,3,4.83,0
EST-0168,Hibrid,Works,0.75,1,1,4.72,0
EST-0169,In Person,Works,0.4318,2,2,5.15,1
EST-0170,Hibrid,Doesn't Work,0.6009,6,5,6.04,0
EST-0171,Hibrid,Doesn't Work,0.945,5,4,5.97,0
EST-0172,In Person,Doesn't Work,0.8731,5,4,5.21,0
EST-0173,In Person,Works,0.5889,3,0,4.58,1
EST-0174,Hibrid,Works,0.6854,3,2,6.73,0
EST-0175,Hibrid,Works,0.8092,2,2,6.12,1
EST-0176,In Person,Works,0.6495,4,4,5.16,0
EST-0177,In Person,Works,0.2877,2,2,4.67,0
EST-0178,In Person,Works,0.4769,4,3,5.4,0
EST-0179,In Person,Doesn't Work,0.9724,5,3,5.83,1
EST-0180,Hibrid,Works,0.8795,3,2,5.73,0
EST-0181,Hibrid,Works,0.5948,3,2,5.93,1
EST-0182,Hibrid,Works,0.5959,2,2,4.56,0
EST-0183,Hibrid,Works,0.6515,3,3,5.11,0
EST-0184,In Person,Works,0.4113,1,1,4.36,0
EST-0185,In Person,Works,0.5379,4,3,4.75,1
EST-0186,Hibrid,Works,0.6324,1,1,6.14,1
EST-0187,Hibrid,Works,0.9141,5,5,5.7,0
EST-0188,Hibrid,Works,0.8599,5,4,5.58,1
EST-0189,In Person,Works,0.7083,4,3,4.91,1
EST-0190,In Person,Doesn't Work,0.9577,5,4,6.28,0
EST-0191,In Person,Works,0.4818,3,2,5.31,0
EST-0192,Hibrid,Doesn't Work,0.9201,5,5,5.97,0
EST-0193,Hibrid,Works,0.8599,2,1,5.21,1
EST-0194,Hibrid,Works,0.94,3,3,5.85,0
EST-0195,In Person,Doesn't Work,0.6664,6,5,5.35,1
EST-0196,Hibrid,Doesn't Work,0.8922,5,3,5.37,0
EST-0197,Hibrid,Works,0.5514,2,2,5.08,0
EST-0198,Hibrid,Works,0.9672,3,2,5.01,1
EST-0199,Hibrid,Works,0.8954,5,5,5.44,0
EST-0200,Hibrid,Works,0.7362,2,1,5.87,1
EST-0201,Hibrid,Works,0.6905,2,2,5.24,0
EST-0202,In Person,Works,0.768,5,4,5.55,1
EST-0203,Hibrid,Works,0.7005,4,2,6.41,1
EST-0204,Hibrid,Works,0.863,4,4,6.17,0
EST-0205,Hibrid,Works,0.8088,3,3,5.74,1
EST-0206,In Person,Works,0.7372,1,1,5.81,0
EST-0207,In Person,Works,0.567,2,2,5.61,1
EST-0208,Hibrid,Doesn't Work,0.6201,6,4,4.94,0
EST-0209,In Person,Works,0.6542,5,3,5.07,0
EST-0210,In Person,Works,0.6673,3,2,4.92,0
EST-0211,Hibrid,Doesn't Work,0.8371,4,3,5.97,0
EST-0212,Hibrid,Works,0.8411,3,3,5.83,0
EST-0213,Hibrid,Works,0.7424,3,3,5.02,0
EST-0214,Hibrid,Works,0.6471,3,2,4.37,0
EST-0215,Hibrid,Works,0.7783,4,2,6.07,0
EST-0216,Hibrid,Works,0.6516,3,1,5.58,1
EST-0217,Hibrid,Doesn't Work,0.8682,6,6,5.11,0
EST-0218,Hibrid,Works,0.6591,3,0,5.44,1
EST-0219,Hibrid,Doesn't Work,0.731,4,3,5.79,1
EST-0220,Hibrid,Doesn't Work,0.849,5,5,5.97,0
EST-0221,Hibrid,Works,0.7533,3,3,5.24,0
EST-0222,In Person,Works,0.6894,4,4,5.96,0
EST-0223,In Person,Works,0.7471,6,4,4.8,1
EST-0224,In Person,Doesn't Work,0.9101,6,5,6.35,0
EST-0225,Hibrid,Works,0.8679,1,1,5.74,0
EST-0226,Hibrid,Works,0.5961,3,2,4.9,0
EST-0227,Hibrid,Works,0.8582,3,3,5.87,1
EST-0228,In Person,Works,0.548,4,3,5.88,1
EST-0229,Hibrid,Works,0.5484,6,5,6.27,1
EST-0230,Hibrid,Works,0.8572,2,1,6.22,1
EST-0231,In Person,Doesn't Work,0.9666,3,3,4.51,0
EST-0232,Hibrid,Doesn't Work,0.8118,6,4,5.74,0
EST-0233,Hibrid,Doesn't Work,0.8835,6,6,6.29,0
EST-0234,Hibrid,Doesn't Work,0.9808,5,5,6.24,0
EST-0235,Hibrid,Works,0.5139,2,2,5.01,0
EST-0236,Hibrid,Works,0.8554,4,3,4.75,0
EST-0237,Hibrid,Works,0.7437,1,1,6.03,1
EST-0238,Hibrid,Works,0.8734,3,3,5.98,0
EST-0239,Hibrid,Works,0.3715,1,1,4.72,0
EST-0240,In Person,Doesn't Work,0.9172,5,5,5.53,0
EST-0241,Hibrid,Works,0.8023,4,4,6.21,0
EST-0242,Hibrid,Works,0.6939,3,3,5.96,0
EST-0243,In Person,Works,0.7019,1,1,5.51,0
EST-0244,Hibrid,Doesn't Work,0.9748,3,2,5.52,0
EST-0245,Hibrid,Works,0.6947,2,1,6.08,1
EST-0246,Hibrid,Works,0.7618,3,3,5.0,0
EST-0247,Hibrid,Doesn't Work,0.9337,5,5,5.92,0
EST-0248,In Person,Works,0.5161,3,3,4.97,0
EST-0249,Hibrid,Works,0.6265,2,0,4.76,1
EST-0250,In Person,Works,0.6989,4,3,4.58,1
EST-0251,Hibrid,Works,0.5634,3,2,5.43,0
EST-0252,Hibrid,Doesn't Work,0.929,6,6,5.77,0
EST-0253,In Person,Works,0.7121,1,1,5.27,0
EST-0254,In Person,Works,0.7548,4,4,5.65,0
EST-0255,In Person,Works,0.5468,3,2,5.07,1
EST-0256,Hibrid,Doesn't Work,0.8542,5,4,6.08,0
EST-0257,Hibrid,Doesn't Work,0.9406,6,6,6.37,0
EST-0258,In Person,Works,0.3949,5,4,4.08,1
EST-0259,Hibrid,Doesn't Work,0.9138,3,3,5.02,0
EST-0260,In Person,Works,0.6242,3,0,4.37,1
EST-0261,Hibrid,Works,0.3817,2,1,4.78,1
EST-0262,In Person,Doesn't Work,0.7243,6,3,5.37,0
EST-0263,Hibrid,Works,0.9333,4,4,5.5,0
EST-0264,In Person,Works,0.592,3,3,4.62,0
EST-0265,Hibrid,Works,0.7801,4,3,5.13,0
EST-0266,Hibrid,Works,0.5997,3,3,5.79,0
EST-0267,Hibrid,Works,0.4522,3,1,4.61,0
EST-0268,Hibrid,Works,0.7788,5,4,5.42,0
EST-0269,Hibrid,Works,0.7336,4,2,5.45,1
EST-0270,Hibrid,Works,0.6395,2,2,5.12,0
EST-0271,In Person,Works,0.5574,2,2,5.62,0
EST-0272,Hibrid,Works,0.8284,3,3,5.59,0
EST-0273,Hibrid,Works,0.7381,4,4,5.39,0
EST-0274,Hibrid,Works,0.6086,2,2,5.41,0
EST-0275,Hibrid,Doesn't Work,0.903,4,3,5.56,0
EST-0276,Hibrid,Works,0.7729,4,3,5.48,0
EST-0277,Hibrid,Works,0.4836,3,2,5.54,1
EST-0278,In Person,Works,0.6076,2,2,5.59,0
EST-0279,Hibrid,Works,0.6681,6,5,5.23,0
EST-0280,In Person,Works,0.572,2,2,5.36,0
EST-0281,Hibrid,Works,0.4416,4,1,5.28,1
EST-0282,Hibrid,Works,0.7471,3,2,5.91,0
EST-0283,Hibrid,Doesn't Work,0.7475,4,4,5.82,0
EST-0284,Hibrid,Works,0.6727,2,1,5.9,0
EST-0285,In Person,Works,0.8695,2,1,5.55,0
EST-0286,Hibrid,Works,0.9477,4,3,5.12,0
EST-0287,Hibrid,Works,0.4566,3,3,4.65,1
EST-0288,Hibrid,Doesn't Work,0.7376,5,5,5.35,0
EST-0289,Hibrid,Works,0.7579,1,1,4.9,0
EST-0290,Hibrid,Works,0.7933,3,1,5.64,1
EST-0291,Hibrid,Works,0.6842,2,2,5.71,0
EST-0292,Hibrid,Works,0.735,5,3,4.97,1
EST-0293,In Person,Doesn't Work,0.9391,1,1,4.48,0
EST-0294,In Person,Works,0.6553,2,1,5.58,0
EST-0295,Hibrid,Doesn't Work,0.8927,2,2,6.33,0
EST-0296,In Person,Works,0.5822,2,2,5.68,0
EST-0297,Hibrid,Works,0.709,5,5,5.79,0
EST-0298,Hibrid,Works,0.6697,2,1,4.12,1
EST-0299,In Person,Works,0.7036,1,1,6.48,0
EST-0300,Hibrid,Works,0.7294,1,1,5.24,0
EST-0301,Hibrid,Doesn't Work,0.8062,5,5,6.44,0
EST-0302,In Person,Works,0.6653,3,2,5.07,0
EST-0303,In Person,Works,0.467,5,5,4.54,0
EST-0304,In Person,Works,0.764,2,1,6.11,0
EST-0305,Hibrid,Works,0.5424,4,2,5.77,1
EST-0306,In Person,Works,0.6777,2,2,5.34,0
EST-0307,In Person,Works,0.8632,3,2,6.06,1
EST-0308,In Person,Works,0.7175,3,3,5.25,0
EST-0309,In Person,Doesn't Work,0.9437,6,5,6.71,0
EST-0310,Hibrid,Works,0.5115,2,1,4.96,1
EST-0311,In Person,Doesn't Work,0.933,5,5,6.37,0
EST-0312,Hibrid,Works,0.7674,2,1,6.0,0
EST-0313,Hibrid,Works,0.3811,4,2,4.72,0
EST-0314,Hibrid,Works,0.6758,2,1,4.57,0
EST-0315,Hibrid,Works,0.8446,2,1,5.16,1
EST-0316,Hibrid,Works,0.7807,4,4,5.53,0
EST-0317,Hibrid,Works,0.8204,4,4,5.48,0
EST-0318,Hibrid,Works,0.7905,4,3,4.94,1
EST-0319,Hibrid,Doesn't Work,0.8676,4,3,5.58,0
EST-0320,Hibrid,Works,0.6278,4,4,6.19,0
EST-0321,In Person,Works,0.8452,4,3,5.91,0
EST-0322,Hibrid,Works,0.7454,3,3,6.13,0
EST-0323,Hibrid,Works,0.5976,4,3,5.38,0
EST-0324,Hibrid,Works,0.6593,3,3,4.78,0
EST-0325,In Person,Works,0.82,2,2,6.54,0
EST-0326,Hibrid,Works,0.6411,4,3,4.64,0
EST-0327,In Person,Works,0.5701,1,1,4.59,0
EST-0328,Hibrid,Doesn't Work,0.8833,5,5,5.74,0
EST-0329,In Person,Works,0.341,5,4,5.58,0
EST-0330,Hibrid,Works,0.7268,2,1,5.66,0
EST-0331,In Person,Works,0.4944,2,1,5.41,1
EST-0332,In Person,Works,0.6599,4,3,5.02,0
EST-0333,In Person,Works,0.8876,4,4,6.42,0
EST-0334,In Person,Works,0.5624,3,3,6.29,0
EST-0335,Hibrid,Works,0.6866,3,2,5.49,0
EST-0336,In Person,Works,0.851,2,2,5.81,0
EST-0337,Hibrid,Works,0.7589,3,3,6.05,0
EST-0338,In Person,Works,0.9252,1,1,7.0,0
EST-0339,Hibrid,Doesn't Work,0.9211,5,5,6.15,0
EST-0340,Hibrid,Works,0.7501,4,3,5.93,0
EST-0341,Hibrid,Doesn't Work,0.7951,4,3,6.04,0
EST-0342,In Person,Works,0.7042,3,2,5.09,0
EST-0343,In Person,Works,0.6216,4,3,5.1,1
EST-0344,In Person,Works,0.9101,4,4,6.31,0
EST-0345,Hibrid,Doesn't Work,0.9588,5,5,5.56,0
EST-0346,Hibrid,Works,0.8541,4,4,6.1,0
EST-0347,Hibrid,Doesn't Work,0.7989,5,5,5.67,0
EST-0348,Hibrid,Works,0.8261,1,1,5.86,0
EST-0349,In Person,Doesn't Work,0.857,5,5,6.02,0
EST-0350,Hibrid,Works,0.7318,4,4,4.43,0
EST-0351,Hibrid,Works,0.767,2,2,4.28,0
EST-0352,Hibrid,Works,0.5993,3,3,5.14,1
EST-0353,Hibrid,Works,0.7878,2,1,5.62,0
EST-0354,Hibrid,Works,0.7329,3,3,6.07,0
EST-0355,Hibrid,Works,0.6328,2,0,4.71,1
EST-0356,Hibrid,Doesn't Work,0.8599,3,3,5.31,0
EST-0357,Hibrid,Works,0.8227,4,2,5.56,0
EST-0358,In Person,Works,0.8144,4,4,5.38,0
EST-0359,Hibrid,Works,0.7895,4,3,5.77,0
EST-0360,In Person,Works,0.7221,1,1,4.95,1
EST-0361,Hibrid,Works,0.8823,3,3,5.42,0
EST-0362,Hibrid,Works,0.6794,1,0,4.96,1
EST-0363,Hibrid,Works,0.5642,4,2,5.98,0
EST-0364,Hibrid,Doesn't Work,0.9887,5,5,6.37,0
EST-0365,Hibrid,Works,0.8092,4,4,4.99,0
EST-0366,Hibrid,Works,0.4842,3,2,4.73,1
EST-0367,Hibrid,Works,0.8921,2,1,5.75,1
EST-0368,Hibrid,Works,0.7081,2,2,5.88,0
EST-0369,Hibrid,Works,0.6632,3,2,5.6,0
EST-0370,In Person,Works,0.8181,1,1,5.41,0
EST-0371,In Person,Works,0.5818,3,3,5.21,0
EST-0372,Hibrid,Works,0.6168,1,0,5.29,0
EST-0373,Hibrid,Works,0.6147,4,4,6.32,1
EST-0374,Hibrid,Works,0.6921,4,2,5.7,0
EST-0375,Hibrid,Works,0.7624,3,3,5.59,0
EST-0376,In Person,Works,0.8273,3,3,5.83,0
EST-0377,Hibrid,Works,0.8735,4,3,5.49,1
EST-0378,In Person,Works,0.595,3,3,6.35,0
EST-0379,Hibrid,Doesn't Work,0.8019,5,3,5.36,0
EST-0380,Hibrid,Works,0.4694,2,2,4.37,0
EST-0381,Hibrid,Works,0.7016,4,3,4.49,1
EST-0382,In Person,Doesn't Work,0.9616,6,6,6.15,0
EST-0383,In Person,Works,0.6916,1,1,5.7,0
EST-0384,In Person,Works,0.6863,2,2,6.36,0
EST-0385,Hibrid,Works,0.7718,3,3,5.93,0
EST-0386,Hibrid,Works,0.5172,4,2,4.53,1
EST-0387,Hibrid,Doesn't Work,0.897,5,5,6.68,0
EST-0388,In Person,Works,0.8072,1,1,5.77,0
EST-0389,In Person,Works,0.5693,1,1,5.12,0
EST-0390,In Person,Works,0.538,1,0,4.11,1
EST-0391,In Person,Works,0.3327,2,1,4.55,0
EST-0392,Hibrid,Works,0.4243,4,2,5.22,0
EST-0393,Hibrid,Doesn't Work,0.4865,5,3,4.73,1
EST-0394,Hibrid,Doesn't Work,0.8417,4,4,5.06,0
EST-0395,Hibrid,Doesn't Work,0.888,5,2,6.02,0
EST-0396,Hibrid,Works,0.7107,4,4,5.78,0
EST-0397,Hibrid,Works,0.779,1,1,5.75,0
EST-0398,Hibrid,Works,0.6,4,4,5.19,0
EST-0399,In Person,Works,0.6631,3,2,4.88,1
EST-0400,In Person,Works,0.4826,3,3,4.36,0
EST-0401,In Person,Works,0.3136,3,2,4.7,0
EST-0402,Hibrid,Works,0.6485,4,4,5.6,0
EST-0403,In Person,Works,0.8003,4,2,4.78,0
EST-0404,In Person,Works,0.6627,1,1,5.58,0
EST-0405,Hibrid,Works,0.8469,4,4,5.65,0
EST-0406,Hibrid,Works,0.7416,4,3,4.66,0
EST-0407,Hibrid,Works,0.6782,3,3,5.2,0
EST-0408,Hibrid,Works,0.7513,4,3,5.99,0
EST-0409,Hibrid,Doesn't Work,0.9274,4,4,6.32,0
EST-0410,In Person,Works,0.7727,3,2,5.85,0
EST-0411,In Person,Works,0.8927,2,2,5.8,1
EST-0412,Hibrid,Works,0.3214,1,0,4.51,1
EST-0413,Hibrid,Works,0.5206,3,2,5.02,0
EST-0414,In Person,Doesn't Work,0.9834,4,3,5.77,0
EST-0415,Hibrid,Works,0.7996,3,1,4.6,0
EST-0416,Hibrid,Works,0.7273,2,1,5.4,1
EST-0417,Hibrid,Works,0.7937,3,3,5.12,0
EST-0418,In Person,Works,0.5422,1,0,4.88,1
EST-0419,In Person,Works,0.4477,3,2,5.17,1
EST-0420,Hibrid,Doesn't Work,0.9699,5,5,6.91,0
EST-0421,Hibrid,Works,0.7776,1,1,5.89,1
EST-0422,Hibrid,Works,0.8775,3,3,5.51,0
EST-0423,In Person,Doesn't Work,0.7447,4,4,5.94,0
EST-0424,In Person,Doesn't Work,0.6595,6,4,4.97,0
EST-0425,Hibrid,Doesn't Work,0.8991,6,5,6.26,1
EST-0426,Hibrid,Doesn't Work,0.6563,6,3,5.43,0
EST-0427,Hibrid,Works,0.8249,3,3,5.71,0
EST-0428,Hibrid,Works,0.9504,4,3,5.27,0
EST-0429,Hibrid,Works,0.4807,4,4,5.13,0
EST-0430,Hibrid,Works,0.4658,4,3,4.79,1
EST-0431,In Person,Works,0.6263,3,0,5.19,1
EST-0432,Hibrid,Works,0.8153,4,4,5.17,1
EST-0433,Hibrid,Works,0.7571,2,2,6.07,0
EST-0434,Hibrid,Works,0.8874,3,3,6.46,0
EST-0435,Hibrid,Works,0.8066,2,2,5.62,0
EST-0436,In Person,Works,0.5995,3,0,5.76,1
EST-0437,Hibrid,Doesn't Work,0.9184,5,4,5.38,0
EST-0438,In Person,Works,0.3808,3,3,4.16,0
EST-0439,Hibrid,Doesn't Work,0.952,5,5,6.36,0
EST-0440,Hibrid,Doesn't Work,0.5738,3,3,4.01,0
EST-0441,In Person,Works,0.7524,1,1,4.57,0
EST-0442,Hibrid,Doesn't Work,0.7046,3,2,5.57,0
EST-0443,In Person,Doesn't Work,0.8174,5,5,6.07,0
EST-0444,Hibrid,Doesn't Work,0.9305,4,4,5.32,0
EST-0445,Hibrid,Works,0.6652,1,1,5.7,0
EST-0446,Hibrid,Works,0.5337,4,3,4.94,0
EST-0447,Hibrid,Doesn't Work,0.6435,6,5,4.83,0
EST-0448,In Person,Doesn't Work,0.8714,6,6,4.97,0
EST-0449,In Person,Works,0.6311,3,3,5.4,0
EST-0450,In Person,Doesn't Work,0.8314,4,4,5.8,0
EST-0451,Hibrid,Works,0.926,3,3,5.4,0
EST-0452,In Person,Works,0.3508,3,3,3.46,0
EST-0453,In Person,Works,0.6609,2,2,5.04,1
EST-0454,In Person,Works,0.5994,2,2,4.85,1
EST-0455,Hibrid,Works,0.7617,4,4,4.2,0
EST-0456,Hibrid,Works,0.848,4,4,5.66,1
EST-0457,Hibrid,Doesn't Work,0.8194,4,4,5.79,0
EST-0458,Hibrid,Works,0.7071,2,1,5.45,0
EST-0459,In Person,Works,0.5134,6,5,5.28,0
EST-0460,In Person,Works,0.6289,3,2,4.97,0
EST-0461,Hibrid,Doesn't Work,0.9575,4,2,6.03,0
EST-0462,In Person,Doesn't Work,0.868,6,5,5.0,0
EST-0463,Hibrid,Works,0.8004,4,2,6.08,1
EST-0464,Hibrid,Works,0.7803,3,2,6.03,0
EST-0465,Hibrid,Works,0.3707,4,4,6.02,0
EST-0466,Hibrid,Works,0.596,2,1,5.14,0
EST-0467,Hibrid,Works,0.7546,3,2,5.45,0
EST-0468,Hibrid,Works,0.8467,2,2,5.32,0
EST-0469,Hibrid,Works,0.9351,6,6,6.07,0
EST-0470,Hibrid,Doesn't Work,0.9655,5,4,5.59,0
EST-0471,Hibrid,Works,0.499,4,3,5.25,0
EST-0472,Hibrid,Doesn't Work,0.9135,6,5,6.18,0
EST-0473,In Person,Works,0.5624,4,2,5.25,0
EST-0474,Hibrid,Works,0.6959,4,3,5.32,0
EST-0475,Hibrid,Works,0.8062,3,2,5.6,0
EST-0476,In Person,Works,0.4329,1,1,5.26,0
EST-0477,Hibrid,Doesn't Work,0.9356,1,1,5.82,0
EST-0478,In Person,Works,0.4062,3,3,4.93,1
EST-0479,In Person,Works,0.5343,4,4,5.24,0
EST-0480,Hibrid,Works,0.7115,1,0,4.98,1
EST-0481,Hibrid,Doesn't Work,0.9615,3,3,5.69,0
EST-0482,Hibrid,Doesn't Work,0.9907,2,2,6.6,0
EST-0483,Hibrid,Doesn't Work,0.6977,4,3,6.49,0
EST-0484,Hibrid,Works,0.7636,4,4,5.88,0
EST-0485,Hibrid,Doesn't Work,0.8323,5,2,6.14,0
EST-0486,Hibrid,Works,0.7779,4,4,4.83,1
EST-0487,In Person,Doesn't Work,0.9616,4,4,5.02,1
EST-0488,Hibrid,Doesn't Work,0.966,5,5,5.57,0
EST-0489,In Person,Doesn't Work,0.9261,5,5,6.47,0
EST-0490,In Person,Doesn't Work,0.9774,3,3,5.38,0
EST-0491,In Person,Works,0.6541,2,0,4.56,1
EST-0492,Hibrid,Works,0.6246,4,3,5.45,0
EST-0493,Hibrid,Works,0.6966,3,2,5.47,0
EST-0494,In Person,Works,0.4263,2,2,4.39,0
EST-0495,Hibrid,Doesn't Work,0.8807,6,3,5.34,0
EST-0496,Hibrid,Works,0.664,2,2,5.39,0
EST-0497,In Person,Works,0.4236,3,2,4.52,1
EST-0498,Hibrid,Doesn't Work,0.9353,6,5,6.26,0
EST-0499,Hibrid,Works,0.8317,4,3,5.47,0
EST-0500,In Person,Works,0.4839,3,2,5.4,0
EST-0501,In Person,Works,0.5038,1,1,4.71,0
EST-0502,In Person,Works,0.6412,4,2,5.08,0
EST-0503,In Person,Doesn't Work,0.9659,4,4,6.19,0
EST-0504,In Person,Works,0.835,3,3,4.67,0
EST-0505,In Person,Doesn't Work,0.9164,4,4,6.26,0
EST-0506,Hibrid,Doesn't Work,0.9853,6,6,6.53,0
EST-0507,Hibrid,Works,0.8061,6,5,5.27,0
EST-0508,Hibrid,Works,0.9163,5,5,6.09,0
EST-0509,Hibrid,Works,0.7768,5,5,5.26,0
EST-0510,In Person,Works,0.5252,2,2,6.32,0
EST-0511,In Person,Works,0.573,3,3,5.76,0
EST-0512,In Person,Works,0.7582,4,3,5.71,0
EST-0513,Hibrid,Works,0.8672,1,1,5.43,0
EST-0514,Hibrid,Works,0.4992,2,2,4.3,0
EST-0515,Hibrid,Works,0.7415,4,2,5.52,0
EST-0516,In Person,Doesn't Work,0.9551,4,4,6.09,0
EST-0517,Hibrid,Doesn't Work,0.802,5,5,6.25,0
EST-0518,Hibrid,Works,0.9059,3,3,5.61,0
EST-0519,In Person,Doesn't Work,0.906,1,0,4.95,0
EST-0520,Hibrid,Works,0.8296,3,3,5.64,0
EST-0521,In Person,Works,0.3505,4,2,5.5,1
EST-0522,Hibrid,Doesn't Work,0.9714,5,5,5.69,0
EST-0523,In Person,Doesn't Work,0.9945,6,5,5.38,0
EST-0524,Hibrid,Works,0.7453,5,5,5.86,0
EST-0525,Hibrid,Works,0.8169,1,1,5.0,0
EST-0526,In Person,Works,0.7516,2,2,5.63,0
EST-0527,Hibrid,Works,0.8741,4,4,5.93,0
EST-0528,Hibrid,Works,0.5701,1,1,5.83,0
EST-0529,In Person,Works,0.6108,3,2,6.24,1
EST-0530,Hibrid,Works,0.3576,3,2,4.82,1
EST-0531,Hibrid,Doesn't Work,0.9859,5,5,6.13,0
EST-0532,Hibrid,Doesn't Work,0.646,5,5,5.53,0
EST-0533,Hibrid,Works,0.7629,5,3,5.54,1
EST-0534,Hibrid,Works,0.7221,3,3,4.85,0
EST-0535,Hibrid,Works,0.7821,5,5,5.42,0
EST-0536,Hibrid,Works,0.7048,4,3,5.45,0
EST-0537,Hibrid,Works,0.7872,4,4,5.75,1
EST-0538,In Person,Works,0.7205,2,2,5.19,0
EST-0539,In Person,Doesn't Work,0.9524,4,4,5.23,0
EST-0540,In Person,Works,0.5823,3,2,5.63,0
EST-0541,In Person,Works,0.6526,5,5,6.14,0
EST-0542,Hibrid,Doesn't Work,0.4897,6,4,5.27,0
EST-0543,In Person,Doesn't Work,0.9456,5,5,6.09,0
EST-0544,Hibrid,Works,0.5488,4,3,5.21,1
EST-0545,Hibrid,Works,0.7328,5,5,6.27,1
EST-0546,In Person,Works,0.6969,3,3,5.37,0
EST-0547,Hibrid,Works,0.7653,3,2,5.88,1
EST-0548,Hibrid,Works,0.6547,3,2,4.94,0
EST-0549,Hibrid,Works,0.7296,4,2,5.83,0
EST-0550,In Person,Works,0.6308,1,1,4.53,0
EST-0551,Hibrid,Doesn't Work,0.8653,2,1,5.91,0
EST-0552,Hibrid,Works,0.884,3,3,5.62,0
EST-0553,In Person,Works,0.3108,1,1,5.4,0
EST-0554,In Person,Works,0.6959,5,3,5.03,0
EST-0555,In Person,Works,0.7527,3,1,5.36,0
EST-0556,In Person,Works,0.4355,3,1,5.59,1
EST-0557,In Person,Works,0.6327,4,1,4.74,0
EST-0558,Hibrid,Works,0.8807,4,4,6.23,0
EST-0559,Hibrid,Doesn't Work,0.8643,5,5,6.35,0
EST-0560,In Person,Works,0.6243,2,2,5.65,0
EST-0561,Hibrid,Doesn't Work,0.9435,6,6,6.32,0
EST-0562,In Person,Works,0.5908,3,3,5.47,1
EST-0563,In Person,Doesn't Work,0.9996,6,6,5.65,0
EST-0564,In Person,Works,0.4042,1,1,5.14,1
EST-0565,Hibrid,Works,0.5784,2,2,5.29,0
EST-0566,In Person,Works,0.5695,1,0,5.19,0
EST-0567,Hibrid,Works,0.8138,3,2,6.47,0
EST-0568,In Person,Works,0.6851,5,5,5.58,0
EST-0569,Hibrid,Works,0.7335,2,2,4.89,0
EST-0570,Hibrid,Doesn't Work,0.7403,4,2,6.65,0
EST-0571,Hibrid,Works,0.5444,3,1,5.15,0
EST-0572,Hibrid,Works,0.4365,4,4,5.35,1
EST-0573,In Person,Works,0.7949,4,4,5.92,1
EST-0574,In Person,Works,0.7167,4,2,4.74,1
EST-0575,Hibrid,Works,0.8937,4,4,6.31,0
EST-0576,Hibrid,Works,0.9549,4,4,6.09,1
EST-0577,Hibrid,Works,0.8495,3,2,5.67,0
EST-0578,Hibrid,Works,0.5654,5,3,4.97,1
EST-0579,Hibrid,Doesn't Work,0.9103,4,4,4.98,0
EST-0580,Hibrid,Doesn't Work,0.9077,3,3,5.57,0
EST-0581,Hibrid,Works,0.6216,3,2,5.57,1
EST-0582,Hibrid,Works,0.5028,1,1,4.5,0
EST-0583,In Person,Works,0.7988,2,2,5.9,0
EST-0584,Hibrid,Works,0.8938,3,3,6.12,0
EST-0585,In Person,Doesn't Work,0.6659,6,4,5.06,1
EST-0586,Hibrid,Works,0.889,3,3,5.25,0
EST-0587,Hibrid,Works,0.6827,5,5,5.73,0
EST-0588,In Person,Works,0.582,4,2,5.98,1
EST-0589,Hibrid,Doesn't Work,0.9538,4,3,6.68,0
EST-0590,In Person,Doesn't Work,0.9924,4,4,5.75,0
EST-0591,Hibrid,Works,0.5235,4,4,5.52,0
EST-0592,Hibrid,Works,0.7402,1,1,6.21,0
EST-0593,Hibrid,Works,0.7608,3,3,5.51,0
EST-0594,Hibrid,Doesn't Work,0.8221,5,4,6.08,1
EST-0595,In Person,Works,0.7189,2,1,5.66,1
EST-0596,Hibrid,Doesn't Work,0.7668,5,3,4.9,0
EST-0597,Hibrid,Works,0.6976,4,4,5.47,1
EST-0598,In Person,Doesn't Work,0.9898,5,4,5.78,0
EST-0599,Hibrid,Doesn't Work,0.6935,4,3,4.12,0
EST-0600,In Person,Works,0.7211,2,2,6.59,0
EST-0601,Hibrid,Doesn't Work,0.9446,5,5,5.24,0
EST-0602,In Person,Works,0.6064,3,3,5.07,1
EST-0603,Hibrid,Doesn't Work,0.7896,5,4,5.62,0
EST-0604,Hibrid,Works,0.6572,3,3,5.03,0
EST-0605,Hibrid,Works,0.474,3,3,5.26,0
EST-0606,In Person,Works,0.5331,3,2,4.38,0
EST-0607,Hibrid,Works,0.9189,4,2,6.11,0
EST-0608,Hibrid,Works,0.7286,2,1,5.44,1
EST-0609,Hibrid,Works,0.6385,4,4,4.77,0
EST-0610,Hibrid,Doesn't Work,0.7237,1,1,5.84,0
EST-0611,Hibrid,Works,0.5966,4,3,4.24,0
EST-0612,Hibrid,Works,0.6074,2,2,5.52,0
EST-0613,Hibrid,Works,0.5775,2,1,4.02,0
EST-0614,Hibrid,Works,0.8348,2,2,5.83,0
EST-0615,Hibrid,Works,0.7318,5,4,5.24,0
EST-0616,In Person,Works,0.5707,3,1,5.04,1
EST-0617,Hibrid,Doesn't Work,0.8088,4,2,5.0,0
EST-0618,In Person,Works,0.4706,6,5,4.99,0
EST-0619,Hibrid,Doesn't Work,0.9336,5,5,5.92,0
EST-0620,In Person,Works,0.4591,3,3,5.58,0
EST-0621,In Person,Works,0.3439,3,1,4.52,0
EST-0622,In Person,Works,0.604,2,0,4.82,1
EST-0623,Hibrid,Works,0.7522,3,3,5.44,1
EST-0624,Hibrid,Works,0.6731,1,0,4.33,1
EST-0625,In Person,Works,0.4235,2,0,4.32,1
EST-0626,Hibrid,Works,0.6758,2,2,5.9,0
EST-0627,In Person,Works,0.6457,3,2,5.58,0
EST-0628,In Person,Works,0.655,3,2,5.59,0
EST-0629,Hibrid,Works,0.5925,3,1,5.79,1
EST-0630,In Person,Works,0.4472,1,0,4.13,1
EST-0631,Hibrid,Works,0.6288,3,3,4.24,0
EST-0632,Hibrid,Works,0.7266,3,3,5.31,0
EST-0633,Hibrid,Doesn't Work,0.958,6,4,5.94,0
EST-0634,Hibrid,Works,0.6084,4,3,4.93,0
EST-0635,Hibrid,Doesn't Work,0.9169,5,5,5.32,0
EST-0636,Hibrid,Works,0.904,5,5,6.14,0
EST-0637,Hibrid,Works,0.5101,4,2,5.42,0
EST-0638,In Person,Doesn't Work,0.9785,5,5,6.21,0
EST-0639,Hibrid,Works,0.9467,3,3,6.1,1
EST-0640,In Person,Works,0.3886,3,1,5.16,1
EST-0641,Hibrid,Works,0.8252,5,5,5.72,0
EST-0642,Hibrid,Works,0.7381,5,4,5.32,0
EST-0643,In Person,Works,0.5926,4,3,5.18,0
EST-0644,Hibrid,Works,0.842,4,4,5.05,0
EST-0645,In Person,Works,0.7643,3,3,5.42,1
EST-0646,Hibrid,Works,0.7811,5,5,5.52,1
EST-0647,In Person,Doesn't Work,0.9504,5,5,7.0,0
EST-0648,Hibrid,Works,0.7506,5,4,5.86,0
EST-0649,Hibrid,Doesn't Work,0.9603,4,4,6.39,0
EST-0650,Hibrid,Works,0.8865,3,3,6.31,0
EST-0651,Hibrid,Works,0.9494,3,3,6.71,0
EST-0652,In Person,Works,0.6561,2,2,5.24,0
EST-0653,In Person,Works,0.7956,3,3,5.92,0
EST-0654,In Person,Works,0.4708,3,2,5.84,1
EST-0655,Hibrid,Works,0.7477,2,2,5.08,0
EST-0656,In Person,Works,0.6857,3,3,4.97,0
EST-0657,Hibrid,Doesn't Work,0.7909,4,3,5.32,0
EST-0658,Hibrid,Works,0.6951,4,4,5.29,0
EST-0659,Hibrid,Works,0.9489,3,3,4.99,0
EST-0660,Hibrid,Works,0.762,1,1,4.91,0
EST-0661,Hibrid,Works,0.4113,3,2,4.72,0
EST-0662,In Person,Works,0.6498,4,4,5.47,0
EST-0663,In Person,Works,0.6217,3,2,6.18,1
EST-0664,Hibrid,Doesn't Work,0.6978,4,4,5.05,1
EST-0665,In Person,Doesn't Work,0.9588,6,6,5.84,1
EST-0666,Hibrid,Doesn't Work,0.9248,5,4,5.41,0
EST-0667,Hibrid,Works,0.7108,6,4,5.61,0
EST-0668,In Person,Works,0.8317,3,3,5.62,0
EST-0669,Hibrid,Works,0.5044,5,3,4.56,1
EST-0670,Hibrid,Works,0.5364,3,3,5.66,0
EST-0671,Hibrid,Works,0.5018,5,4,4.45,0
EST-0672,Hibrid,Doesn't Work,0.9156,6,4,5.41,0
EST-0673,In Person,Works,0.6984,4,3,5.69,0
EST-0674,Hibrid,Doesn't Work,0.7769,5,3,4.82,1
EST-0675,Hibrid,Works,0.5536,4,4,4.99,0
EST-0676,Hibrid,Works,0.6947,4,4,5.19,0
EST-0677,Hibrid,Works,0.5797,6,6,6.34,0
EST-0678,Hibrid,Works,0.7672,3,3,5.5,0
EST-0679,Hibrid,Works,0.385,4,3,5.22,1
EST-0680,In Person,Works,0.7068,1,1,6.38,0
EST-0681,Hibrid,Works,0.8685,3,3,5.6,0
EST-0682,Hibrid,Works,0.7585,4,4,6.01,1
EST-0683,Hibrid,Works,0.7535,2,2,5.36,0
EST-0684,Hibrid,Works,0.6515,2,2,5.07,0
EST-0685,Hibrid,Works,0.5261,5,4,4.38,1
EST-0686,Hibrid,Works,0.7281,3,1,5.59,0
EST-0687,Hibrid,Works,0.7721,5,5,5.33,0
EST-0688,Hibrid,Works,0.6957,6,5,5.48,0
EST-0689,Hibrid,Works,0.3946,2,2,5.07,0
EST-0690,Hibrid,Works,0.8518,2,2,6.27,0
EST-0691,Hibrid,Doesn't Work,0.7005,6,6,4.97,1
EST-0692,Hibrid,Doesn't Work,0.9924,6,5,6.33,0
EST-0693,Hibrid,Doesn't Work,0.9645,5,4,5.23,0
EST-0694,Hibrid,Works,0.6549,2,2,5.48,0
EST-0695,In Person,Works,0.6356,6,6,5.73,0
EST-0696,Hibrid,Works,0.4646,2,1,4.76,1
EST-0697,Hibrid,Works,0.8148,4,4,5.64,0
EST-0698,Hibrid,Works,0.6318,6,6,5.11,0
EST-0699,Hibrid,Works,0.7418,3,3,5.29,0
EST-0700,Hibrid,Doesn't Work,0.7843,5,5,5.08,1
EST-0701,Hibrid,Works,0.6956,3,2,5.27,0
EST-0702,In Person,Works,0.6928,3,3,5.9,0
EST-0703,Hibrid,Works,0.6491,2,2,6.03,0
EST-0704,In Person,Doesn't Work,0.9167,5,3,5.81,0
EST-0705,Hibrid,Works,0.7797,4,4,5.8,0
EST-0706,Hibrid,Works,0.6845,3,2,4.64,1
EST-0707,In Person,Works,0.4152,2,2,3.77,0
EST-0708,Hibrid,Doesn't Work,0.7375,2,2,6.01,0
EST-0709,Hibrid,Works,0.8323,2,1,5.84,0
EST-0710,Hibrid,Works,0.6983,3,3,5.35,0
EST-0711,In Person,Doesn't Work,0.9979,3,3,6.32,0
EST-0712,Hibrid,Doesn't Work,0.7904,4,4,6.3,0
EST-0713,In Person,Works,0.3492,5,4,4.41,0
EST-0714,Hibrid,Works,0.7173,2,1,5.84,0
EST-0715,Hibrid,Works,0.4598,3,3,4.89,1
EST-0716,Hibrid,Works,0.885,3,3,6.07,0
EST-0717,Hibrid,Works,0.7689,3,3,5.03,0
EST-0718,Hibrid,Works,0.472,5,3,4.73,1
EST-0719,Hibrid,Works,0.772,3,2,4.72,0
EST-0720,Hibrid,Doesn't Work,0.8481,5,4,6.37,0
EST-0721,Hibrid,Works,0.8553,4,4,6.62,0
EST-0722,In Person,Doesn't Work,0.9769,5,4,4.05,0
EST-0723,Hibrid,Works,0.7071,3,2,5.19,0
EST-0724,In Person,Works,0.6419,3,2,5.39,1
EST-0725,Hibrid,Works,0.6378,2,2,4.82,0
EST-0726,Hibrid,Doesn't Work,0.6546,3,3,5.18,0
EST-0727,Hibrid,Works,0.5736,2,1,4.82,1
EST-0728,In Person,Doesn't Work,0.9096,6,6,6.35,0
EST-0729,In Person,Works,0.6006,4,2,4.94,0
EST-0730,In Person,Works,0.7069,3,3,5.59,0
EST-0731,Hibrid,Doesn't Work,0.792,6,5,6.21,1
EST-0732,Hibrid,Works,0.8151,4,4,5.95,0
EST-0733,Hibrid,Doesn't Work,0.8701,6,5,6.0,0
EST-0734,In Person,Works,0.6257,4,1,5.71,1
EST-0735,In Person,Works,0.6646,3,3,5.5,0
EST-0736,Hibrid,Works,0.6096,3,3,4.43,0
EST-0737,Hibrid,Works,0.7843,4,4,6.31,0
EST-0738,Hibrid,Works,0.5715,3,3,4.45,0
EST-0739,Hibrid,Doesn't Work,0.8102,4,2,5.41,0
EST-0740,Hibrid,Doesn't Work,0.947,6,6,5.27,0
EST-0741,Hibrid,Works,0.59,3,2,5.43,0
EST-0742,In Person,Works,0.6028,1,1,5.24,0
EST-0743,In Person,Works,0.8619,2,2,5.73,1
EST-0744,In Person,Doesn't Work,0.8725,4,4,5.59,0
EST-0745,Hibrid,Works,0.7608,5,5,6.07,0
EST-0746,Hibrid,Works,0.6014,3,3,4.82,0
EST-0747,Hibrid,Works,0.8288,4,4,5.61,0
EST-0748,In Person,Works,0.6334,3,3,5.37,0
EST-0749,In Person,Works,0.5891,4,3,6.19,0
EST-0750,Hibrid,Works,0.8372,4,4,5.96,0
EST-0751,Hibrid,Works,0.5524,5,3,4.22,0
EST-0752,Hibrid,Works,0.8018,3,3,6.05,0
EST-0753,Hibrid,Works,0.6205,1,1,5.44,1
EST-0754,In Person,Works,0.4217,4,4,6.06,0
EST-0755,Hibrid,Doesn't Work,0.8386,4,4,5.21,0
EST-0756,In Person,Works,0.7537,3,1,5.07,0
EST-0757,In Person,Works,0.5687,3,2,4.25,0
EST-0758,Hibrid,Works,0.6414,3,3,5.52,1
EST-0759,In Person,Works,0.5642,2,2,5.93,0
EST-0760,Hibrid,Works,0.7565,4,3,5.99,1
EST-0761,Hibrid,Works,0.8988,2,2,6.74,0
EST-0762,Hibrid,Works,0.6807,4,3,5.96,0
EST-0763,In Person,Works,0.6676,2,1,5.49,0
EST-0764,In Person,Works,0.3941,3,3,4.49,1
EST-0765,In Person,Works,0.5427,4,4,5.03,0
EST-0766,Hibrid,Works,0.9033,4,4,6.32,0
EST-0767,Hibrid,Works,0.4499,1,1,5.31,1
EST-0768,Hibrid,Works,0.7394,4,3,6.25,0
EST-0769,Hibrid,Works,0.8828,3,3,5.85,0
EST-0770,Hibrid,Works,0.776,1,1,5.82,0
EST-0771,Hibrid,Doesn't Work,0.8724,6,6,6.47,0
EST-0772,In Person,Works,0.5775,2,2,6.16,0
EST-0773,Hibrid,Doesn't Work,0.8776,6,5,6.0,0
EST-0774,Hibrid,Doesn't Work,0.8952,4,4,5.97,0
EST-0775,In Person,Doesn't Work,0.7109,6,5,5.35,0
EST-0776,In Person,Works,0.666,3,3,5.58,0
EST-0777,Hibrid,Works,0.8106,2,2,5.54,0
EST-0778,Hibrid,Works,0.7101,3,3,4.67,0
EST-0779,Hibrid,Works,0.4834,4,3,5.16,1
EST-0780,In Person,Doesn't Work,0.938,3,3,6.21,0
EST-0781,Hibrid,Doesn't Work,0.795,5,4,5.49,0
EST-0782,Hibrid,Works,0.9127,5,4,5.7,0
EST-0783,In Person,Works,0.7015,2,2,6.85,0
EST-0784,Hibrid,Works,0.6056,2,2,5.54,0
EST-0785,In Person,Works,0.6724,4,4,4.95,0
EST-0786,Hibrid,Works,0.893,2,1,5.38,1
EST-0787,Hibrid,Works,0.5506,1,1,5.28,0
EST-0788,Hibrid,Works,0.8151,3,2,5.33,0
EST-0789,Hibrid,Doesn't Work,0.9821,6,5,5.75,0
EST-0790,Hibrid,Works,0.7628,4,3,5.53,0
EST-0791,Hibrid,Works,0.8601,3,3,6.04,0
EST-0792,In Person,Works,0.5264,4,2,5.63,1
EST-0793,In Person,Doesn't Work,0.9524,6,6,6.13,0
EST-0794,In Person,Works,0.6693,4,4,5.58,1
EST-0795,Hibrid,Works,0.7926,5,3,6.15,0
EST-0796,Hibrid,Works,0.7423,3,3,5.18,0
EST-0797,In Person,Works,0.6056,2,2,6.17,0
EST-0798,In Person,Works,0.4346,2,1,5.41,1
EST-0799,Hibrid,Works,0.7232,3,2,4.8,0
EST-0800,In Person,Works,0.7424,1,1,5.04,0
EST-0801,In Person,Works,0.2323,2,1,4.39,1
EST-0802,Hibrid,Doesn't Work,0.8179,5,4,4.4,0
EST-0803,In Person,Works,0.8385,4,3,5.65,0
EST-0804,Hibrid,Works,0.4731,2,2,6.06,0
EST-0805,Hibrid,Works,0.5841,2,1,5.84,1
EST-0806,Hibrid,Doesn't Work,0.8584,5,4,6.28,0
EST-0807,Hibrid,Works,0.729,4,3,7.0,0
EST-0808,In Person,Works,0.6907,3,2,5.92,0
EST-0809,In Person,Works,0.7579,4,2,5.71,0
EST-0810,In Person,Works,0.7449,3,3,5.88,0
EST-0811,In Person,Works,0.5311,4,2,5.01,1
EST-0812,In Person,Works,0.581,3,1,4.37,0
EST-0813,Hibrid,Doesn't Work,0.7816,4,3,5.08,0
EST-0814,Hibrid,Works,0.5996,3,3,5.68,0
EST-0815,In Person,Works,0.7639,4,3,5.54,0
EST-0816,Hibrid,Doesn't Work,0.7393,3,3,5.62,0
EST-0817,Hibrid,Doesn't Work,0.9453,6,5,5.43,0
EST-0818,In Person,Doesn't Work,0.9809,5,4,6.0,0
EST-0819,Hibrid,Works,0.5944,5,2,4.66,1
EST-0820,Hibrid,Doesn't Work,0.9346,3,3,5.99,0
EST-0821,Hibrid,Doesn't Work,0.5791,5,2,5.18,0
EST-0822,Hibrid,Works,0.6033,2,2,5.93,0
EST-0823,Hibrid,Works,0.6378,3,3,5.56,0
EST-0824,In Person,Doesn't Work,0.9889,5,5,6.64,0
EST-0825,In Person,Works,0.686,3,2,5.05,0
EST-0826,Hibrid,Doesn't Work,0.8752,5,5,6.06,0
EST-0827,Hibrid,Doesn't Work,0.7707,5,5,6.09,0
EST-0828,Hibrid,Works,0.5835,4,2,4.94,0
EST-0829,Hibrid,Doesn't Work,0.7686,6,6,5.52,0
EST-0830,In Person,Works,0.6592,3,3,5.45,0
EST-0831,Hibrid,Doesn't Work,0.9221,6,5,6.27,0
EST-0832,Hibrid,Works,0.532,5,4,6.46,0
EST-0833,Hibrid,Works,0.8577,2,2,5.92,0
EST-0834,Hibrid,Works,0.801,4,4,6.12,0
EST-0835,In Person,Works,0.4813,2,1,5.15,0
EST-0836,Hibrid,Works,0.6592,3,3,4.34,0
EST-0837,In Person,Doesn't Work,0.8472,6,5,5.62,0
EST-0838,Hibrid,Doesn't Work,0.8859,6,6,5.39,0
EST-0839,Hibrid,Works,0.7919,2,2,6.16,0
EST-0840,Hibrid,Works,0.5717,6,5,5.06,0
EST-0841,Hibrid,Doesn't Work,0.8611,5,5,5.8,0
EST-0842,Hibrid,Works,0.8213,4,4,5.84,0
EST-0843,In Person,Works,0.5129,4,2,5.67,1
EST-0844,Hibrid,Works,0.8878,2,2,5.61,0
EST-0845,Hibrid,Works,0.7178,2,2,5.24,0
EST-0846,Hibrid,Works,0.8407,5,5,5.75,1
EST-0847,Hibrid,Doesn't Work,0.5533,6,5,5.64,0
EST-0848,In Person,Works,0.5349,3,2,5.45,0
EST-0849,Hibrid,Works,0.8911,2,2,5.34,0
EST-0850,Hibrid,Works,0.7058,3,2,6.54,0
EST-0851,Hibrid,Works,0.6448,2,2,5.76,0
EST-0852,In Person,Works,0.5388,1,0,5.06,1
EST-0853,In Person,Works,0.492,1,1,5.6,0
EST-0854,Hibrid,Works,0.554,4,3,5.02,1
EST-0855,Hibrid,Works,0.8938,2,2,5.73,0
EST-0856,Hibrid,Works,0.7268,1,1,5.46,0
EST-0857,Hibrid,Works,0.586,2,0,5.19,0
EST-0858,Hibrid,Works,0.7236,1,1,5.09,0
EST-0859,Hibrid,Works,0.4479,4,4,4.73,0
EST-0860,Hibrid,Works,0.9036,3,3,6.12,0
EST-0861,In Person,Doesn't Work,0.9451,6,6,5.68,0
EST-0862,Hibrid,Works,0.3602,2,0,5.36,0
EST-0863,In Person,Doesn't Work,0.9568,6,6,6.56,0
EST-0864,In Person,Doesn't Work,0.8734,6,6,5.88,0
EST-0865,Hibrid,Doesn't Work,0.6044,6,4,5.83,0
EST-0866,In Person,Works,0.8023,4,2,5.04,0
EST-0867,Hibrid,Works,0.908,3,2,6.33,0
EST-0868,In Person,Works,0.6712,3,3,5.24,0
EST-0869,Hibrid,Works,0.6709,4,4,5.65,0
EST-0870,Hibrid,Works,0.7891,4,4,6.79,0
EST-0871,Hibrid,Works,0.6103,1,0,4.83,1
EST-0872,Hibrid,Works,0.8099,2,2,5.64,0
EST-0873,In Person,Works,0.4086,3,3,4.87,1
EST-0874,Hibrid,Works,0.8223,3,3,4.94,0
EST-0875,Hibrid,Works,0.8327,1,0,6.18,1
EST-0876,Hibrid,Works,0.8913,3,2,6.04,0
EST-0877,In Person,Works,0.5417,6,5,6.12,0
EST-0878,Hibrid,Works,0.531,1,1,5.09,0
EST-0879,In Person,Doesn't Work,0.9574,5,5,6.3,0
EST-0880,Hibrid,Works,0.8669,3,3,4.86,0
EST-0881,Hibrid,Works,0.8084,2,1,5.38,1
EST-0882,Hibrid,Works,0.6611,3,3,6.06,1
EST-0883,Hibrid,Works,0.6766,4,2,5.18,1
EST-0884,In Person,Works,0.5256,3,2,5.29,0
EST-0885,Hibrid,Works,0.8785,5,4,5.15,0
EST-0886,Hibrid,Doesn't Work,0.8805,5,5,5.46,0
EST-0887,Hibrid,Works,0.7117,2,1,4.79,1
EST-0888,Hibrid,Doesn't Work,0.9935,4,4,5.61,0
EST-0889,In Person,Works,0.6466,3,3,4.16,1
EST-0890,In Person,Works,0.6148,3,3,5.14,0
EST-0891,Hibrid,Doesn't Work,0.9275,5,5,6.25,1
EST-0892,Hibrid,Works,0.8784,4,4,5.75,0
EST-0893,In Person,Works,0.5299,4,3,4.94,0
EST-0894,In Person,Works,0.48,3,3,3.74,1
EST-0895,In Person,Works,0.5862,4,0,6.39,0
EST-0896,Hibrid,Works,0.6618,3,2,5.82,0
EST-0897,Hibrid,Doesn't Work,0.9945,6,6,6.78,0
EST-0898,Hibrid,Works,0.8802,3,3,6.71,0
EST-0899,Hibrid,Doesn't Work,0.8936,4,3,5.9,0
EST-0900,In Person,Works,0.8497,4,4,6.14,0
EST-0901,Hibrid,Doesn't Work,0.8704,5,4,5.95,0
EST-0902,Hibrid,Works,0.7399,6,6,5.26,0
EST-0903,Hibrid,Works,0.5546,4,3,6.01,0
EST-0904,Hibrid,Works,0.5705,3,2,6.09,1
EST-0905,In Person,Doesn't Work,0.9063,5,4,5.44,0
EST-0906,Hibrid,Works,0.6065,1,0,5.5,0
EST-0907,In Person,Works,0.5064,2,2,5.6,1
EST-0908,Hibrid,Works,0.8584,1,1,5.48,1
EST-0909,In Person,Works,0.521,1,1,4.96,0
EST-0910,In Person,Works,0.5633,4,3,5.53,0
EST-0911,In Person,Works,0.7267,1,1,4.79,0
EST-0912,Hibrid,Works,0.8001,6,5,6.05,0
EST-0913,In Person,Doesn't Work,0.9234,3,3,6.17,0
EST-0914,Hibrid,Works,0.6234,3,0,4.61,1
EST-0915,In Person,Works,0.7624,4,4,5.04,0
EST-0916,In Person,Doesn't Work,0.9807,4,4,6.43,0
EST-0917,Hibrid,Works,0.5841,3,3,4.75,0
EST-0918,Hibrid,Works,0.7326,3,3,5.19,0
EST-0919,In Person,Works,0.3268,1,0,4.49,1
EST-0920,Hibrid,Works,0.7658,4,4,5.93,0
EST-0921,In Person,Works,0.5415,2,1,4.42,0
EST-0922,In Person,Works,0.3292,3,3,4.37,0
EST-0923,Hibrid,Works,0.7508,1,1,5.14,0
EST-0924,In Person,Works,0.5282,2,2,4.79,0
EST-0925,In Person,Works,0.5456,3,2,4.67,1
EST-0926,Hibrid,Doesn't Work,0.8859,6,5,5.79,0
EST-0927,In Person,Works,0.5961,5,2,5.19,1
EST-0928,In Person,Works,0.5805,3,2,5.6,0
EST-0929,Hibrid,Works,0.5969,2,2,4.97,0
EST-0930,Hibrid,Works,0.5508,3,2,5.35,1
EST-0931,In Person,Works,0.5363,1,1,4.4,0
EST-0932,Hibrid,Doesn't Work,0.711,3,2,5.06,1
EST-0933,In Person,Works,0.5409,2,2,3.52,0
EST-0934,Hibrid,Works,0.8066,3,3,5.61,0
EST-0935,In Person,Works,0.4723,1,0,5.36,0
EST-0936,Hibrid,Works,0.693,3,2,5.44,0
EST-0937,Hibrid,Works,0.8249,1,1,5.37,0
EST-0938,In Person,Works,0.9225,3,3,5.34,0
EST-0939,Hibrid,Doesn't Work,0.8831,6,6,5.11,0
EST-0940,Hibrid,Doesn't Work,0.9762,4,4,5.68,0
EST-0941,Hibrid,Works,0.9396,1,1,5.35,0
EST-0942,Hibrid,Works,0.67,4,3,5.14,1
EST-0943,In Person,Doesn't Work,0.7426,4,2,4.21,0
EST-0944,In Person,Works,0.7311,5,5,6.82,0
EST-0945,Hibrid,Works,0.735,3,3,5.8,0
EST-0946,Hibrid,Doesn't Work,0.894,4,4,5.7,0
EST-0947,Hibrid,Works,0.9485,6,5,6.18,0
EST-0948,In Person,Works,0.5062,3,1,5.75,0
EST-0949,In Person,Works,0.573,5,5,4.93,1
EST-0950,In Person,Works,0.6945,3,2,4.81,0
EST-0951,In Person,Doesn't Work,0.9994,4,4,5.94,0
EST-0952,Hibrid,Doesn't Work,0.6438,5,3,4.32,0
EST-0953,Hibrid,Works,0.7533,4,4,5.9,0
EST-0954,In Person,Doesn't Work,0.6375,5,3,4.72,0
EST-0955,Hibrid,Works,0.6251,1,1,4.55,1
EST-0956,Hibrid,Works,0.4162,5,3,4.83,1
EST-0957,Hibrid,Doesn't Work,0.8378,5,4,5.61,0
EST-0958,In Person,Doesn't Work,0.9683,6,6,5.66,0
EST-0959,Hibrid,Doesn't Work,0.6581,5,5,4.88,0
EST-0960,Hibrid,Works,0.8161,3,3,5.23,0
EST-0961,Hibrid,Doesn't Work,0.9698,5,4,5.46,0
EST-0962,Hibrid,Works,0.5602,3,3,5.25,0
EST-0963,In Person,Doesn't Work,0.9905,5,5,6.6,0
EST-0964,Hibrid,Doesn't Work,0.7627,4,4,5.64,1
EST-0965,Hibrid,Doesn't Work,0.9542,6,5,5.67,1
EST-0966,Hibrid,Works,0.8789,3,3,5.96,0
EST-0967,Hibrid,Works,0.7359,2,2,5.11,0
EST-0968,In Person,Doesn't Work,0.9634,5,5,6.37,0
EST-0969,In Person,Works,0.4048,3,2,4.8,0
EST-0970,Hibrid,Works,0.8049,4,3,5.98,0
EST-0971,Hibrid,Works,0.9435,3,3,6.09,0
EST-0972,In Person,Doesn't Work,0.9573,5,4,5.24,1
EST-0973,Hibrid,Works,0.9312,2,2,5.35,0
EST-0974,In Person,Works,0.4472,4,3,4.32,0
EST-0975,Hibrid,Works,0.6443,2,1,4.53,1
EST-0976,In Person,Works,0.5228,4,1,5.28,0
EST-0977,Hibrid,Works,0.6001,3,2,5.97,0
EST-0978,Hibrid,Works,0.4112,1,0,6.09,1
EST-0979,Hibrid,Works,0.5852,1,1,5.68,0
EST-0980,In Person,Works,0.5359,3,3,4.78,0
EST-0981,Hibrid,Doesn't Work,0.6413,5,5,5.49,0
EST-0982,Hibrid,Doesn't Work,0.781,4,4,5.56,0
EST-0983,Hibrid,Works,0.509,3,1,5.4,1
EST-0984,In Person,Works,0.7603,3,2,5.69,0
EST-0985,In Person,Works,0.3631,4,4,5.01,0
EST-0986,Hibrid,Works,0.7005,3,1,4.39,0
EST-0987,Hibrid,Doesn't Work,0.9207,5,4,5.56,0
EST-0988,Hibrid,Works,0.352,2,2,3.56,0
EST-0989,Hibrid,Works,0.6254,2,2,4.73,1
EST-0990,Hibrid,Works,0.5153,1,0,5.59,1
EST-0991,In Person,Works,0.8594,3,3,5.82,1
EST-0992,Hibrid,Works,0.7123,2,2,5.88,0
EST-0993,Hibrid,Works,0.9484,3,2,6.56,0
EST-0994,Hibrid,Works,0.6964,2,2,4.43,0
EST-0995,Hibrid,Works,0.6778,3,2,5.33,0
EST-0996,Hibrid,Works,0.8779,3,2,6.08,0
EST-0997,Hibrid,Doesn't Work,0.8372,4,4,5.61,0
EST-0998,Hibrid,Works,0.6077,4,4,5.62,0
EST-0999,In Person,Works,0.3939,1,1,4.63,0
EST-1000,Hibrid,Doesn't Work,0.8549,5,4,5.09,0



```

Tu archivo CSV con el dataset sintético está listo

### Entregable 1: Dataset Sintético (1000 filas)

```csv
id,mode,working_condition,attendance_percentage,academic_load,subjects_passed,previous_average,dropping_out_risk
EST-0001,Hibrid,Works,0.7706,3,3,5.51,0
EST-0002,Hibrid,Works,0.7964,4,4,5.34,0
EST-0003,In Person,Works,0.3909,5,3,5.73,0
EST-0004,Hibrid,Doesn't Work,0.9087,4,3,5.98,0
EST-0005,Hibrid,Doesn't Work,0.851,5,4,5.68,0
EST-0006,In Person,Doesn't Work,0.9534,4,3,5.3,0
EST-0007,Hibrid,Doesn't Work,0.7953,4,2,5.53,1
EST-0008,In Person,Works,0.4427,6,4,5.3,0
EST-0009,Hibrid,Works,0.4905,2,1,4.42,0
EST-0010,In Person,Works,0.8176,2,2,5.26,0
EST-0011,Hibrid,Works,0.6707,3,3,4.98,0
EST-0012,Hibrid,Works,0.7616,3,3,6.14,0
EST-0013,Hibrid,Works,0.6613,3,2,5.47,0
EST-0014,In Person,Works,0.7422,3,3,5.01,0
EST-0015,Hibrid,Works,0.7577,3,2,6.0,1
EST-0016,Hibrid,Works,0.7777,2,2,7.0,0
EST-0017,Hibrid,Works,0.6759,1,1,4.9,0
EST-0018,Hibrid,Works,0.7626,5,5,6.42,0
EST-0019,In Person,Doesn't Work,0.9976,5,4,5.35,0
EST-0020,In Person,Works,0.4977,2,1,4.75,1
EST-0021,Hibrid,Works,0.8826,3,3,5.62,0
EST-0022,Hibrid,Doesn't Work,0.8175,4,4,6.71,0
EST-0023,Hibrid,Works,0.5355,3,1,4.05,1
EST-0024,Hibrid,Works,0.6695,2,1,5.0,0
EST-0025,Hibrid,Doesn't Work,0.5846,4,4,5.14,0
EST-0026,In Person,Works,0.8325,2,2,6.16,0
EST-0027,Hibrid,Works,0.6219,4,4,4.65,0
EST-0028,Hibrid,Works,0.7854,2,2,5.39,0
EST-0029,Hibrid,Works,0.4329,4,4,4.84,0
EST-0030,In Person,Works,0.9165,3,3,6.05,0
EST-0031,In Person,Works,0.7878,4,4,5.02,1
EST-0032,In Person,Doesn't Work,0.9494,3,3,6.44,0
EST-0033,Hibrid,Works,0.7764,3,2,5.66,0
EST-0034,Hibrid,Works,0.8715,1,1,5.34,0
EST-0035,Hibrid,Works,0.8027,3,3,5.59,1
EST-0036,Hibrid,Doesn't Work,0.7345,5,3,5.28,0
EST-0037,Hibrid,Doesn't Work,0.9114,4,4,5.76,0
EST-0038,Hibrid,Works,0.7108,3,0,5.4,1
EST-0039,Hibrid,Works,0.7823,5,5,6.22,0
EST-0040,Hibrid,Works,0.6752,2,1,5.32,0
EST-0041,In Person,Works,0.7376,3,3,5.91,0
EST-0042,Hibrid,Doesn't Work,0.9166,1,0,6.17,0
EST-0043,Hibrid,Doesn't Work,0.8432,4,2,5.58,0
EST-0044,Hibrid,Doesn't Work,0.804,3,3,5.51,0
EST-0045,Hibrid,Doesn't Work,0.8015,4,3,5.44,1
EST-0046,Hibrid,Doesn't Work,0.7259,6,5,5.01,0
EST-0047,In Person,Doesn't Work,0.8839,5,4,5.63,0
EST-0048,In Person,Works,0.5814,2,2,4.89,0
EST-0049,In Person,Works,0.3992,4,3,4.56,0
EST-0050,Hibrid,Works,0.831,2,2,5.57,0
EST-0051,Hibrid,Works,0.8147,3,1,5.69,1
EST-0052,In Person,Works,0.6301,3,1,4.96,0
EST-0053,Hibrid,Works,0.7365,4,4,5.58,0
EST-0054,Hibrid,Works,0.8377,2,2,5.33,0
EST-0055,Hibrid,Works,0.8361,1,1,6.02,0
EST-0056,In Person,Works,0.6383,4,2,5.42,0
EST-0057,In Person,Doesn't Work,0.9534,6,6,6.76,0
EST-0058,Hibrid,Works,0.704,2,2,5.85,0
EST-0059,Hibrid,Works,0.7416,1,1,6.21,1
EST-0060,In Person,Works,0.4671,4,3,4.56,0
EST-0061,In Person,Works,0.5105,4,1,4.91,0
EST-0062,Hibrid,Works,0.6908,4,3,6.25,0
EST-0063,Hibrid,Works,0.5093,5,4,4.7,0
EST-0064,Hibrid,Works,0.8598,3,3,5.72,1
EST-0065,In Person,Works,0.2991,4,3,5.05,0
EST-0066,Hibrid,Works,0.6815,4,3,4.88,1
EST-0067,Hibrid,Doesn't Work,0.9512,6,5,6.08,0
EST-0068,In Person,Works,0.8933,4,4,6.75,0
EST-0069,Hibrid,Works,0.8222,4,4,5.51,0
EST-0070,Hibrid,Doesn't Work,0.9128,5,5,5.36,0
EST-0071,Hibrid,Doesn't Work,0.8863,4,4,5.16,0
EST-0072,Hibrid,Doesn't Work,0.8183,5,5,6.43,0
EST-0073,Hibrid,Works,0.6303,4,3,5.69,0
EST-0074,Hibrid,Works,0.7693,2,2,5.36,0
EST-0075,Hibrid,Works,0.5093,1,0,4.81,1
EST-0076,In Person,Doesn't Work,0.9316,5,4,5.53,0
EST-0077,Hibrid,Works,0.6063,5,3,4.86,0
EST-0078,Hibrid,Works,0.6461,4,2,5.39,0
EST-0079,In Person,Doesn't Work,0.9303,5,5,6.46,1
EST-0080,In Person,Works,0.6003,4,3,4.69,1
EST-0081,Hibrid,Doesn't Work,0.8882,4,2,6.33,1
EST-0082,In Person,Works,0.4828,3,3,4.48,0
EST-0083,In Person,Doesn't Work,0.9607,5,5,5.65,0
EST-0084,In Person,Works,0.3611,3,2,5.57,0
EST-0085,Hibrid,Doesn't Work,0.8741,6,5,5.72,0
EST-0086,Hibrid,Works,0.7163,1,1,5.78,0
EST-0087,Hibrid,Works,0.6665,4,4,5.35,0
EST-0088,In Person,Works,0.4431,3,2,5.11,0
EST-0089,In Person,Works,0.9142,4,4,4.95,0
EST-0090,Hibrid,Doesn't Work,0.9405,6,5,5.51,1
EST-0091,Hibrid,Works,0.9089,4,3,5.74,0
EST-0092,In Person,Doesn't Work,0.7928,3,3,6.71,0
EST-0093,In Person,Doesn't Work,0.8273,5,5,5.24,0
EST-0094,Hibrid,Works,0.4886,1,1,4.29,0
EST-0095,Hibrid,Doesn't Work,0.9909,5,5,5.59,0
EST-0096,In Person,Works,0.8129,2,2,6.07,0
EST-0097,In Person,Works,0.901,3,3,5.56,1
EST-0098,Hibrid,Works,0.5964,3,2,5.71,0
EST-0099,In Person,Works,0.7509,2,2,4.95,1
EST-0100,Hibrid,Works,0.8134,5,5,5.43,0
EST-0101,In Person,Works,0.8228,6,5,6.25,0
EST-0102,Hibrid,Works,0.8579,5,5,6.53,0
EST-0103,In Person,Works,0.5568,4,3,5.88,0
EST-0104,Hibrid,Doesn't Work,0.8495,5,4,5.45,0
EST-0105,Hibrid,Works,0.5824,4,0,5.12,1
EST-0106,Hibrid,Works,0.8326,4,4,6.72,0
EST-0107,Hibrid,Works,0.7461,3,3,4.38,0
EST-0108,In Person,Doesn't Work,0.9616,5,4,5.62,0
EST-0109,Hibrid,Works,0.6316,2,2,5.07,1
EST-0110,Hibrid,Works,0.7374,4,4,6.26,1
EST-0111,In Person,Doesn't Work,0.8628,3,3,5.44,0
EST-0112,In Person,Doesn't Work,0.8322,4,4,5.46,0
EST-0113,Hibrid,Works,0.7927,2,2,6.03,0
EST-0114,In Person,Works,0.8941,4,4,5.47,0
EST-0115,In Person,Works,0.6095,3,3,4.89,0
EST-0116,Hibrid,Works,0.6729,2,2,5.14,1
EST-0117,Hibrid,Works,0.6716,2,2,5.6,0
EST-0118,Hibrid,Doesn't Work,0.7499,4,3,5.53,0
EST-0119,In Person,Doesn't Work,0.9135,5,5,6.86,0
EST-0120,In Person,Works,0.6527,3,3,4.73,0
EST-0121,In Person,Works,0.5738,2,2,4.95,0
EST-0122,Hibrid,Works,0.671,4,3,5.66,0
EST-0123,Hibrid,Works,0.5328,5,2,4.12,1
EST-0124,In Person,Works,0.7154,3,3,4.93,0
EST-0125,In Person,Works,0.5839,6,5,3.76,1
EST-0126,In Person,Doesn't Work,0.7551,4,2,4.86,0
EST-0127,Hibrid,Works,0.5193,2,2,5.63,1
EST-0128,Hibrid,Works,0.641,4,4,6.22,1
EST-0129,Hibrid,Doesn't Work,0.8177,5,5,4.86,0
EST-0130,In Person,Works,0.6815,3,3,5.53,0
EST-0131,Hibrid,Works,0.5767,4,1,4.78,0
EST-0132,Hibrid,Doesn't Work,0.8978,4,4,5.96,0
EST-0133,In Person,Doesn't Work,0.957,5,5,6.14,0
EST-0134,In Person,Works,0.7516,2,2,4.93,0
EST-0135,Hibrid,Doesn't Work,0.9159,5,5,6.36,0
EST-0136,Hibrid,Works,0.7,2,2,5.55,0
EST-0137,In Person,Works,0.6578,4,3,4.54,0
EST-0138,Hibrid,Works,0.6017,4,3,5.4,0
EST-0139,Hibrid,Doesn't Work,0.7656,5,5,5.26,0
EST-0140,Hibrid,Doesn't Work,0.6504,4,3,5.02,0
EST-0141,In Person,Works,0.4675,4,3,4.7,0
EST-0142,Hibrid,Works,0.7687,2,2,5.51,0
EST-0143,Hibrid,Works,0.6651,3,3,4.86,0
EST-0144,Hibrid,Works,0.4328,4,2,4.92,0
EST-0145,In Person,Works,0.7132,3,3,6.06,0
EST-0146,Hibrid,Works,0.6744,4,3,4.94,0
EST-0147,Hibrid,Works,0.6933,4,3,5.41,0
EST-0148,In Person,Works,0.49,4,3,5.07,0
EST-0149,Hibrid,Works,0.7338,4,4,5.03,0
EST-0150,Hibrid,Doesn't Work,0.9322,6,5,6.2,0
EST-0151,In Person,Works,0.5244,5,4,4.86,0
EST-0152,Hibrid,Works,0.7596,3,2,5.91,0
EST-0153,In Person,Doesn't Work,0.9078,4,4,6.3,0
EST-0154,Hibrid,Doesn't Work,0.8646,5,4,6.13,0
EST-0155,In Person,Works,0.7278,2,2,6.08,1
EST-0156,In Person,Works,0.4921,5,3,5.2,0
EST-0157,In Person,Works,0.7458,3,3,5.77,0
EST-0158,Hibrid,Doesn't Work,0.9911,5,3,6.49,0
EST-0159,In Person,Works,0.5025,4,2,5.21,1
EST-0160,Hibrid,Works,0.7897,2,2,5.63,0
EST-0161,Hibrid,Works,0.4977,1,1,5.01,1
EST-0162,Hibrid,Works,0.6932,1,1,5.01,1
EST-0163,Hibrid,Works,0.926,4,4,6.42,0
EST-0164,In Person,Works,0.8004,5,4,5.81,0
EST-0165,In Person,Works,0.7279,4,3,4.48,0
EST-0166,Hibrid,Works,0.5014,5,4,4.81,1
EST-0167,In Person,Works,0.7928,4,3,4.83,0
EST-0168,Hibrid,Works,0.75,1,1,4.72,0
EST-0169,In Person,Works,0.4318,2,2,5.15,1
EST-0170,Hibrid,Doesn't Work,0.6009,6,5,6.04,0
EST-0171,Hibrid,Doesn't Work,0.945,5,4,5.97,0
EST-0172,In Person,Doesn't Work,0.8731,5,4,5.21,0
EST-0173,In Person,Works,0.5889,3,0,4.58,1
EST-0174,Hibrid,Works,0.6854,3,2,6.73,0
EST-0175,Hibrid,Works,0.8092,2,2,6.12,1
EST-0176,In Person,Works,0.6495,4,4,5.16,0
EST-0177,In Person,Works,0.2877,2,2,4.67,0
EST-0178,In Person,Works,0.4769,4,3,5.4,0
EST-0179,In Person,Doesn't Work,0.9724,5,3,5.83,1
EST-0180,Hibrid,Works,0.8795,3,2,5.73,0
EST-0181,Hibrid,Works,0.5948,3,2,5.93,1
EST-0182,Hibrid,Works,0.5959,2,2,4.56,0
EST-0183,Hibrid,Works,0.6515,3,3,5.11,0
EST-0184,In Person,Works,0.4113,1,1,4.36,0
EST-0185,In Person,Works,0.5379,4,3,4.75,1
EST-0186,Hibrid,Works,0.6324,1,1,6.14,1
EST-0187,Hibrid,Works,0.9141,5,5,5.7,0
EST-0188,Hibrid,Works,0.8599,5,4,5.58,1
EST-0189,In Person,Works,0.7083,4,3,4.91,1
EST-0190,In Person,Doesn't Work,0.9577,5,4,6.28,0
EST-0191,In Person,Works,0.4818,3,2,5.31,0
EST-0192,Hibrid,Doesn't Work,0.9201,5,5,5.97,0
EST-0193,Hibrid,Works,0.8599,2,1,5.21,1
EST-0194,Hibrid,Works,0.94,3,3,5.85,0
EST-0195,In Person,Doesn't Work,0.6664,6,5,5.35,1
EST-0196,Hibrid,Doesn't Work,0.8922,5,3,5.37,0
EST-0197,Hibrid,Works,0.5514,2,2,5.08,0
EST-0198,Hibrid,Works,0.9672,3,2,5.01,1
EST-0199,Hibrid,Works,0.8954,5,5,5.44,0
EST-0200,Hibrid,Works,0.7362,2,1,5.87,1
EST-0201,Hibrid,Works,0.6905,2,2,5.24,0
EST-0202,In Person,Works,0.768,5,4,5.55,1
EST-0203,Hibrid,Works,0.7005,4,2,6.41,1
EST-0204,Hibrid,Works,0.863,4,4,6.17,0
EST-0205,Hibrid,Works,0.8088,3,3,5.74,1
EST-0206,In Person,Works,0.7372,1,1,5.81,0
EST-0207,In Person,Works,0.567,2,2,5.61,1
EST-0208,Hibrid,Doesn't Work,0.6201,6,4,4.94,0
EST-0209,In Person,Works,0.6542,5,3,5.07,0
EST-0210,In Person,Works,0.6673,3,2,4.92,0
EST-0211,Hibrid,Doesn't Work,0.8371,4,3,5.97,0
EST-0212,Hibrid,Works,0.8411,3,3,5.83,0
EST-0213,Hibrid,Works,0.7424,3,3,5.02,0
EST-0214,Hibrid,Works,0.6471,3,2,4.37,0
EST-0215,Hibrid,Works,0.7783,4,2,6.07,0
EST-0216,Hibrid,Works,0.6516,3,1,5.58,1
EST-0217,Hibrid,Doesn't Work,0.8682,6,6,5.11,0
EST-0218,Hibrid,Works,0.6591,3,0,5.44,1
EST-0219,Hibrid,Doesn't Work,0.731,4,3,5.79,1
EST-0220,Hibrid,Doesn't Work,0.849,5,5,5.97,0
EST-0221,Hibrid,Works,0.7533,3,3,5.24,0
EST-0222,In Person,Works,0.6894,4,4,5.96,0
EST-0223,In Person,Works,0.7471,6,4,4.8,1
EST-0224,In Person,Doesn't Work,0.9101,6,5,6.35,0
EST-0225,Hibrid,Works,0.8679,1,1,5.74,0
EST-0226,Hibrid,Works,0.5961,3,2,4.9,0
EST-0227,Hibrid,Works,0.8582,3,3,5.87,1
EST-0228,In Person,Works,0.548,4,3,5.88,1
EST-0229,Hibrid,Works,0.5484,6,5,6.27,1
EST-0230,Hibrid,Works,0.8572,2,1,6.22,1
EST-0231,In Person,Doesn't Work,0.9666,3,3,4.51,0
EST-0232,Hibrid,Doesn't Work,0.8118,6,4,5.74,0
EST-0233,Hibrid,Doesn't Work,0.8835,6,6,6.29,0
EST-0234,Hibrid,Doesn't Work,0.9808,5,5,6.24,0
EST-0235,Hibrid,Works,0.5139,2,2,5.01,0
EST-0236,Hibrid,Works,0.8554,4,3,4.75,0
EST-0237,Hibrid,Works,0.7437,1,1,6.03,1
EST-0238,Hibrid,Works,0.8734,3,3,5.98,0
EST-0239,Hibrid,Works,0.3715,1,1,4.72,0
EST-0240,In Person,Doesn't Work,0.9172,5,5,5.53,0
EST-0241,Hibrid,Works,0.8023,4,4,6.21,0
EST-0242,Hibrid,Works,0.6939,3,3,5.96,0
EST-0243,In Person,Works,0.7019,1,1,5.51,0
EST-0244,Hibrid,Doesn't Work,0.9748,3,2,5.52,0
EST-0245,Hibrid,Works,0.6947,2,1,6.08,1
EST-0246,Hibrid,Works,0.7618,3,3,5.0,0
EST-0247,Hibrid,Doesn't Work,0.9337,5,5,5.92,0
EST-0248,In Person,Works,0.5161,3,3,4.97,0
EST-0249,Hibrid,Works,0.6265,2,0,4.76,1
EST-0250,In Person,Works,0.6989,4,3,4.58,1
EST-0251,Hibrid,Works,0.5634,3,2,5.43,0
EST-0252,Hibrid,Doesn't Work,0.929,6,6,5.77,0
EST-0253,In Person,Works,0.7121,1,1,5.27,0
EST-0254,In Person,Works,0.7548,4,4,5.65,0
EST-0255,In Person,Works,0.5468,3,2,5.07,1
EST-0256,Hibrid,Doesn't Work,0.8542,5,4,6.08,0
EST-0257,Hibrid,Doesn't Work,0.9406,6,6,6.37,0
EST-0258,In Person,Works,0.3949,5,4,4.08,1
EST-0259,Hibrid,Doesn't Work,0.9138,3,3,5.02,0
EST-0260,In Person,Works,0.6242,3,0,4.37,1
EST-0261,Hibrid,Works,0.3817,2,1,4.78,1
EST-0262,In Person,Doesn't Work,0.7243,6,3,5.37,0
EST-0263,Hibrid,Works,0.9333,4,4,5.5,0
EST-0264,In Person,Works,0.592,3,3,4.62,0
EST-0265,Hibrid,Works,0.7801,4,3,5.13,0
EST-0266,Hibrid,Works,0.5997,3,3,5.79,0
EST-0267,Hibrid,Works,0.4522,3,1,4.61,0
EST-0268,Hibrid,Works,0.7788,5,4,5.42,0
EST-0269,Hibrid,Works,0.7336,4,2,5.45,1
EST-0270,Hibrid,Works,0.6395,2,2,5.12,0
EST-0271,In Person,Works,0.5574,2,2,5.62,0
EST-0272,Hibrid,Works,0.8284,3,3,5.59,0
EST-0273,Hibrid,Works,0.7381,4,4,5.39,0
EST-0274,Hibrid,Works,0.6086,2,2,5.41,0
EST-0275,Hibrid,Doesn't Work,0.903,4,3,5.56,0
EST-0276,Hibrid,Works,0.7729,4,3,5.48,0
EST-0277,Hibrid,Works,0.4836,3,2,5.54,1
EST-0278,In Person,Works,0.6076,2,2,5.59,0
EST-0279,Hibrid,Works,0.6681,6,5,5.23,0
EST-0280,In Person,Works,0.572,2,2,5.36,0
EST-0281,Hibrid,Works,0.4416,4,1,5.28,1
EST-0282,Hibrid,Works,0.7471,3,2,5.91,0
EST-0283,Hibrid,Doesn't Work,0.7475,4,4,5.82,0
EST-0284,Hibrid,Works,0.6727,2,1,5.9,0
EST-0285,In Person,Works,0.8695,2,1,5.55,0
EST-0286,Hibrid,Works,0.9477,4,3,5.12,0
EST-0287,Hibrid,Works,0.4566,3,3,4.65,1
EST-0288,Hibrid,Doesn't Work,0.7376,5,5,5.35,0
EST-0289,Hibrid,Works,0.7579,1,1,4.9,0
EST-0290,Hibrid,Works,0.7933,3,1,5.64,1
EST-0291,Hibrid,Works,0.6842,2,2,5.71,0
EST-0292,Hibrid,Works,0.735,5,3,4.97,1
EST-0293,In Person,Doesn't Work,0.9391,1,1,4.48,0
EST-0294,In Person,Works,0.6553,2,1,5.58,0
EST-0295,Hibrid,Doesn't Work,0.8927,2,2,6.33,0
EST-0296,In Person,Works,0.5822,2,2,5.68,0
EST-0297,Hibrid,Works,0.709,5,5,5.79,0
EST-0298,Hibrid,Works,0.6697,2,1,4.12,1
EST-0299,In Person,Works,0.7036,1,1,6.48,0
EST-0300,Hibrid,Works,0.7294,1,1,5.24,0
EST-0301,Hibrid,Doesn't Work,0.8062,5,5,6.44,0
EST-0302,In Person,Works,0.6653,3,2,5.07,0
EST-0303,In Person,Works,0.467,5,5,4.54,0
EST-0304,In Person,Works,0.764,2,1,6.11,0
EST-0305,Hibrid,Works,0.5424,4,2,5.77,1
EST-0306,In Person,Works,0.6777,2,2,5.34,0
EST-0307,In Person,Works,0.8632,3,2,6.06,1
EST-0308,In Person,Works,0.7175,3,3,5.25,0
EST-0309,In Person,Doesn't Work,0.9437,6,5,6.71,0
EST-0310,Hibrid,Works,0.5115,2,1,4.96,1
EST-0311,In Person,Doesn't Work,0.933,5,5,6.37,0
EST-0312,Hibrid,Works,0.7674,2,1,6.0,0
EST-0313,Hibrid,Works,0.3811,4,2,4.72,0
EST-0314,Hibrid,Works,0.6758,2,1,4.57,0
EST-0315,Hibrid,Works,0.8446,2,1,5.16,1
EST-0316,Hibrid,Works,0.7807,4,4,5.53,0
EST-0317,Hibrid,Works,0.8204,4,4,5.48,0
EST-0318,Hibrid,Works,0.7905,4,3,4.94,1
EST-0319,Hibrid,Doesn't Work,0.8676,4,3,5.58,0
EST-0320,Hibrid,Works,0.6278,4,4,6.19,0
EST-0321,In Person,Works,0.8452,4,3,5.91,0
EST-0322,Hibrid,Works,0.7454,3,3,6.13,0
EST-0323,Hibrid,Works,0.5976,4,3,5.38,0
EST-0324,Hibrid,Works,0.6593,3,3,4.78,0
EST-0325,In Person,Works,0.82,2,2,6.54,0
EST-0326,Hibrid,Works,0.6411,4,3,4.64,0
EST-0327,In Person,Works,0.5701,1,1,4.59,0
EST-0328,Hibrid,Doesn't Work,0.8833,5,5,5.74,0
EST-0329,In Person,Works,0.341,5,4,5.58,0
EST-0330,Hibrid,Works,0.7268,2,1,5.66,0
EST-0331,In Person,Works,0.4944,2,1,5.41,1
EST-0332,In Person,Works,0.6599,4,3,5.02,0
EST-0333,In Person,Works,0.8876,4,4,6.42,0
EST-0334,In Person,Works,0.5624,3,3,6.29,0
EST-0335,Hibrid,Works,0.6866,3,2,5.49,0
EST-0336,In Person,Works,0.851,2,2,5.81,0
EST-0337,Hibrid,Works,0.7589,3,3,6.05,0
EST-0338,In Person,Works,0.9252,1,1,7.0,0
EST-0339,Hibrid,Doesn't Work,0.9211,5,5,6.15,0
EST-0340,Hibrid,Works,0.7501,4,3,5.93,0
EST-0341,Hibrid,Doesn't Work,0.7951,4,3,6.04,0
EST-0342,In Person,Works,0.7042,3,2,5.09,0
EST-0343,In Person,Works,0.6216,4,3,5.1,1
EST-0344,In Person,Works,0.9101,4,4,6.31,0
EST-0345,Hibrid,Doesn't Work,0.9588,5,5,5.56,0
EST-0346,Hibrid,Works,0.8541,4,4,6.1,0
EST-0347,Hibrid,Doesn't Work,0.7989,5,5,5.67,0
EST-0348,Hibrid,Works,0.8261,1,1,5.86,0
EST-0349,In Person,Doesn't Work,0.857,5,5,6.02,0
EST-0350,Hibrid,Works,0.7318,4,4,4.43,0
EST-0351,Hibrid,Works,0.767,2,2,4.28,0
EST-0352,Hibrid,Works,0.5993,3,3,5.14,1
EST-0353,Hibrid,Works,0.7878,2,1,5.62,0
EST-0354,Hibrid,Works,0.7329,3,3,6.07,0
EST-0355,Hibrid,Works,0.6328,2,0,4.71,1
EST-0356,Hibrid,Doesn't Work,0.8599,3,3,5.31,0
EST-0357,Hibrid,Works,0.8227,4,2,5.56,0
EST-0358,In Person,Works,0.8144,4,4,5.38,0
EST-0359,Hibrid,Works,0.7895,4,3,5.77,0
EST-0360,In Person,Works,0.7221,1,1,4.95,1
EST-0361,Hibrid,Works,0.8823,3,3,5.42,0
EST-0362,Hibrid,Works,0.6794,1,0,4.96,1
EST-0363,Hibrid,Works,0.5642,4,2,5.98,0
EST-0364,Hibrid,Doesn't Work,0.9887,5,5,6.37,0
EST-0365,Hibrid,Works,0.8092,4,4,4.99,0
EST-0366,Hibrid,Works,0.4842,3,2,4.73,1
EST-0367,Hibrid,Works,0.8921,2,1,5.75,1
EST-0368,Hibrid,Works,0.7081,2,2,5.88,0
EST-0369,Hibrid,Works,0.6632,3,2,5.6,0
EST-0370,In Person,Works,0.8181,1,1,5.41,0
EST-0371,In Person,Works,0.5818,3,3,5.21,0
EST-0372,Hibrid,Works,0.6168,1,0,5.29,0
EST-0373,Hibrid,Works,0.6147,4,4,6.32,1
EST-0374,Hibrid,Works,0.6921,4,2,5.7,0
EST-0375,Hibrid,Works,0.7624,3,3,5.59,0
EST-0376,In Person,Works,0.8273,3,3,5.83,0
EST-0377,Hibrid,Works,0.8735,4,3,5.49,1
EST-0378,In Person,Works,0.595,3,3,6.35,0
EST-0379,Hibrid,Doesn't Work,0.8019,5,3,5.36,0
EST-0380,Hibrid,Works,0.4694,2,2,4.37,0
EST-0381,Hibrid,Works,0.7016,4,3,4.49,1
EST-0382,In Person,Doesn't Work,0.9616,6,6,6.15,0
EST-0383,In Person,Works,0.6916,1,1,5.7,0
EST-0384,In Person,Works,0.6863,2,2,6.36,0
EST-0385,Hibrid,Works,0.7718,3,3,5.93,0
EST-0386,Hibrid,Works,0.5172,4,2,4.53,1
EST-0387,Hibrid,Doesn't Work,0.897,5,5,6.68,0
EST-0388,In Person,Works,0.8072,1,1,5.77,0
EST-0389,In Person,Works,0.5693,1,1,5.12,0
EST-0390,In Person,Works,0.538,1,0,4.11,1
EST-0391,In Person,Works,0.3327,2,1,4.55,0
EST-0392,Hibrid,Works,0.4243,4,2,5.22,0
EST-0393,Hibrid,Doesn't Work,0.4865,5,3,4.73,1
EST-0394,Hibrid,Doesn't Work,0.8417,4,4,5.06,0
EST-0395,Hibrid,Doesn't Work,0.888,5,2,6.02,0
EST-0396,Hibrid,Works,0.7107,4,4,5.78,0
EST-0397,Hibrid,Works,0.779,1,1,5.75,0
EST-0398,Hibrid,Works,0.6,4,4,5.19,0
EST-0399,In Person,Works,0.6631,3,2,4.88,1
EST-0400,In Person,Works,0.4826,3,3,4.36,0
EST-0401,In Person,Works,0.3136,3,2,4.7,0
EST-0402,Hibrid,Works,0.6485,4,4,5.6,0
EST-0403,In Person,Works,0.8003,4,2,4.78,0
EST-0404,In Person,Works,0.6627,1,1,5.58,0
EST-0405,Hibrid,Works,0.8469,4,4,5.65,0
EST-0406,Hibrid,Works,0.7416,4,3,4.66,0
EST-0407,Hibrid,Works,0.6782,3,3,5.2,0
EST-0408,Hibrid,Works,0.7513,4,3,5.99,0
EST-0409,Hibrid,Doesn't Work,0.9274,4,4,6.32,0
EST-0410,In Person,Works,0.7727,3,2,5.85,0
EST-0411,In Person,Works,0.8927,2,2,5.8,1
EST-0412,Hibrid,Works,0.3214,1,0,4.51,1
EST-0413,Hibrid,Works,0.5206,3,2,5.02,0
EST-0414,In Person,Doesn't Work,0.9834,4,3,5.77,0
EST-0415,Hibrid,Works,0.7996,3,1,4.6,0
EST-0416,Hibrid,Works,0.7273,2,1,5.4,1
EST-0417,Hibrid,Works,0.7937,3,3,5.12,0
EST-0418,In Person,Works,0.5422,1,0,4.88,1
EST-0419,In Person,Works,0.4477,3,2,5.17,1
EST-0420,Hibrid,Doesn't Work,0.9699,5,5,6.91,0
EST-0421,Hibrid,Works,0.7776,1,1,5.89,1
EST-0422,Hibrid,Works,0.8775,3,3,5.51,0
EST-0423,In Person,Doesn't Work,0.7447,4,4,5.94,0
EST-0424,In Person,Doesn't Work,0.6595,6,4,4.97,0
EST-0425,Hibrid,Doesn't Work,0.8991,6,5,6.26,1
EST-0426,Hibrid,Doesn't Work,0.6563,6,3,5.43,0
EST-0427,Hibrid,Works,0.8249,3,3,5.71,0
EST-0428,Hibrid,Works,0.9504,4,3,5.27,0
EST-0429,Hibrid,Works,0.4807,4,4,5.13,0
EST-0430,Hibrid,Works,0.4658,4,3,4.79,1
EST-0431,In Person,Works,0.6263,3,0,5.19,1
EST-0432,Hibrid,Works,0.8153,4,4,5.17,1
EST-0433,Hibrid,Works,0.7571,2,2,6.07,0
EST-0434,Hibrid,Works,0.8874,3,3,6.46,0
EST-0435,Hibrid,Works,0.8066,2,2,5.62,0
EST-0436,In Person,Works,0.5995,3,0,5.76,1
EST-0437,Hibrid,Doesn't Work,0.9184,5,4,5.38,0
EST-0438,In Person,Works,0.3808,3,3,4.16,0
EST-0439,Hibrid,Doesn't Work,0.952,5,5,6.36,0
EST-0440,Hibrid,Doesn't Work,0.5738,3,3,4.01,0
EST-0441,In Person,Works,0.7524,1,1,4.57,0
EST-0442,Hibrid,Doesn't Work,0.7046,3,2,5.57,0
EST-0443,In Person,Doesn't Work,0.8174,5,5,6.07,0
EST-0444,Hibrid,Doesn't Work,0.9305,4,4,5.32,0
EST-0445,Hibrid,Works,0.6652,1,1,5.7,0
EST-0446,Hibrid,Works,0.5337,4,3,4.94,0
EST-0447,Hibrid,Doesn't Work,0.6435,6,5,4.83,0
EST-0448,In Person,Doesn't Work,0.8714,6,6,4.97,0
EST-0449,In Person,Works,0.6311,3,3,5.4,0
EST-0450,In Person,Doesn't Work,0.8314,4,4,5.8,0
EST-0451,Hibrid,Works,0.926,3,3,5.4,0
EST-0452,In Person,Works,0.3508,3,3,3.46,0
EST-0453,In Person,Works,0.6609,2,2,5.04,1
EST-0454,In Person,Works,0.5994,2,2,4.85,1
EST-0455,Hibrid,Works,0.7617,4,4,4.2,0
EST-0456,Hibrid,Works,0.848,4,4,5.66,1
EST-0457,Hibrid,Doesn't Work,0.8194,4,4,5.79,0
EST-0458,Hibrid,Works,0.7071,2,1,5.45,0
EST-0459,In Person,Works,0.5134,6,5,5.28,0
EST-0460,In Person,Works,0.6289,3,2,4.97,0
EST-0461,Hibrid,Doesn't Work,0.9575,4,2,6.03,0
EST-0462,In Person,Doesn't Work,0.868,6,5,5.0,0
EST-0463,Hibrid,Works,0.8004,4,2,6.08,1
EST-0464,Hibrid,Works,0.7803,3,2,6.03,0
EST-0465,Hibrid,Works,0.3707,4,4,6.02,0
EST-0466,Hibrid,Works,0.596,2,1,5.14,0
EST-0467,Hibrid,Works,0.7546,3,2,5.45,0
EST-0468,Hibrid,Works,0.8467,2,2,5.32,0
EST-0469,Hibrid,Works,0.9351,6,6,6.07,0
EST-0470,Hibrid,Doesn't Work,0.9655,5,4,5.59,0
EST-0471,Hibrid,Works,0.499,4,3,5.25,0
EST-0472,Hibrid,Doesn't Work,0.9135,6,5,6.18,0
EST-0473,In Person,Works,0.5624,4,2,5.25,0
EST-0474,Hibrid,Works,0.6959,4,3,5.32,0
EST-0475,Hibrid,Works,0.8062,3,2,5.6,0
EST-0476,In Person,Works,0.4329,1,1,5.26,0
EST-0477,Hibrid,Doesn't Work,0.9356,1,1,5.82,0
EST-0478,In Person,Works,0.4062,3,3,4.93,1
EST-0479,In Person,Works,0.5343,4,4,5.24,0
EST-0480,Hibrid,Works,0.7115,1,0,4.98,1
EST-0481,Hibrid,Doesn't Work,0.9615,3,3,5.69,0
EST-0482,Hibrid,Doesn't Work,0.9907,2,2,6.6,0
EST-0483,Hibrid,Doesn't Work,0.6977,4,3,6.49,0
EST-0484,Hibrid,Works,0.7636,4,4,5.88,0
EST-0485,Hibrid,Doesn't Work,0.8323,5,2,6.14,0
EST-0486,Hibrid,Works,0.7779,4,4,4.83,1
EST-0487,In Person,Doesn't Work,0.9616,4,4,5.02,1
EST-0488,Hibrid,Doesn't Work,0.966,5,5,5.57,0
EST-0489,In Person,Doesn't Work,0.9261,5,5,6.47,0
EST-0490,In Person,Doesn't Work,0.9774,3,3,5.38,0
EST-0491,In Person,Works,0.6541,2,0,4.56,1
EST-0492,Hibrid,Works,0.6246,4,3,5.45,0
EST-0493,Hibrid,Works,0.6966,3,2,5.47,0
EST-0494,In Person,Works,0.4263,2,2,4.39,0
EST-0495,Hibrid,Doesn't Work,0.8807,6,3,5.34,0
EST-0496,Hibrid,Works,0.664,2,2,5.39,0
EST-0497,In Person,Works,0.4236,3,2,4.52,1
EST-0498,Hibrid,Doesn't Work,0.9353,6,5,6.26,0
EST-0499,Hibrid,Works,0.8317,4,3,5.47,0
EST-0500,In Person,Works,0.4839,3,2,5.4,0
EST-0501,In Person,Works,0.5038,1,1,4.71,0
EST-0502,In Person,Works,0.6412,4,2,5.08,0
EST-0503,In Person,Doesn't Work,0.9659,4,4,6.19,0
EST-0504,In Person,Works,0.835,3,3,4.67,0
EST-0505,In Person,Doesn't Work,0.9164,4,4,6.26,0
EST-0506,Hibrid,Doesn't Work,0.9853,6,6,6.53,0
EST-0507,Hibrid,Works,0.8061,6,5,5.27,0
EST-0508,Hibrid,Works,0.9163,5,5,6.09,0
EST-0509,Hibrid,Works,0.7768,5,5,5.26,0
EST-0510,In Person,Works,0.5252,2,2,6.32,0
EST-0511,In Person,Works,0.573,3,3,5.76,0
EST-0512,In Person,Works,0.7582,4,3,5.71,0
EST-0513,Hibrid,Works,0.8672,1,1,5.43,0
EST-0514,Hibrid,Works,0.4992,2,2,4.3,0
EST-0515,Hibrid,Works,0.7415,4,2,5.52,0
EST-0516,In Person,Doesn't Work,0.9551,4,4,6.09,0
EST-0517,Hibrid,Doesn't Work,0.802,5,5,6.25,0
EST-0518,Hibrid,Works,0.9059,3,3,5.61,0
EST-0519,In Person,Doesn't Work,0.906,1,0,4.95,0
EST-0520,Hibrid,Works,0.8296,3,3,5.64,0
EST-0521,In Person,Works,0.3505,4,2,5.5,1
EST-0522,Hibrid,Doesn't Work,0.9714,5,5,5.69,0
EST-0523,In Person,Doesn't Work,0.9945,6,5,5.38,0
EST-0524,Hibrid,Works,0.7453,5,5,5.86,0
EST-0525,Hibrid,Works,0.8169,1,1,5.0,0
EST-0526,In Person,Works,0.7516,2,2,5.63,0
EST-0527,Hibrid,Works,0.8741,4,4,5.93,0
EST-0528,Hibrid,Works,0.5701,1,1,5.83,0
EST-0529,In Person,Works,0.6108,3,2,6.24,1
EST-0530,Hibrid,Works,0.3576,3,2,4.82,1
EST-0531,Hibrid,Doesn't Work,0.9859,5,5,6.13,0
EST-0532,Hibrid,Doesn't Work,0.646,5,5,5.53,0
EST-0533,Hibrid,Works,0.7629,5,3,5.54,1
EST-0534,Hibrid,Works,0.7221,3,3,4.85,0
EST-0535,Hibrid,Works,0.7821,5,5,5.42,0
EST-0536,Hibrid,Works,0.7048,4,3,5.45,0
EST-0537,Hibrid,Works,0.7872,4,4,5.75,1
EST-0538,In Person,Works,0.7205,2,2,5.19,0
EST-0539,In Person,Doesn't Work,0.9524,4,4,5.23,0
EST-0540,In Person,Works,0.5823,3,2,5.63,0
EST-0541,In Person,Works,0.6526,5,5,6.14,0
EST-0542,Hibrid,Doesn't Work,0.4897,6,4,5.27,0
EST-0543,In Person,Doesn't Work,0.9456,5,5,6.09,0
EST-0544,Hibrid,Works,0.5488,4,3,5.21,1
EST-0545,Hibrid,Works,0.7328,5,5,6.27,1
EST-0546,In Person,Works,0.6969,3,3,5.37,0
EST-0547,Hibrid,Works,0.7653,3,2,5.88,1
EST-0548,Hibrid,Works,0.6547,3,2,4.94,0
EST-0549,Hibrid,Works,0.7296,4,2,5.83,0
EST-0550,In Person,Works,0.6308,1,1,4.53,0
EST-0551,Hibrid,Doesn't Work,0.8653,2,1,5.91,0
EST-0552,Hibrid,Works,0.884,3,3,5.62,0
EST-0553,In Person,Works,0.3108,1,1,5.4,0
EST-0554,In Person,Works,0.6959,5,3,5.03,0
EST-0555,In Person,Works,0.7527,3,1,5.36,0
EST-0556,In Person,Works,0.4355,3,1,5.59,1
EST-0557,In Person,Works,0.6327,4,1,4.74,0
EST-0558,Hibrid,Works,0.8807,4,4,6.23,0
EST-0559,Hibrid,Doesn't Work,0.8643,5,5,6.35,0
EST-0560,In Person,Works,0.6243,2,2,5.65,0
EST-0561,Hibrid,Doesn't Work,0.9435,6,6,6.32,0
EST-0562,In Person,Works,0.5908,3,3,5.47,1
EST-0563,In Person,Doesn't Work,0.9996,6,6,5.65,0
EST-0564,In Person,Works,0.4042,1,1,5.14,1
EST-0565,Hibrid,Works,0.5784,2,2,5.29,0
EST-0566,In Person,Works,0.5695,1,0,5.19,0
EST-0567,Hibrid,Works,0.8138,3,2,6.47,0
EST-0568,In Person,Works,0.6851,5,5,5.58,0
EST-0569,Hibrid,Works,0.7335,2,2,4.89,0
EST-0570,Hibrid,Doesn't Work,0.7403,4,2,6.65,0
EST-0571,Hibrid,Works,0.5444,3,1,5.15,0
EST-0572,Hibrid,Works,0.4365,4,4,5.35,1
EST-0573,In Person,Works,0.7949,4,4,5.92,1
EST-0574,In Person,Works,0.7167,4,2,4.74,1
EST-0575,Hibrid,Works,0.8937,4,4,6.31,0
EST-0576,Hibrid,Works,0.9549,4,4,6.09,1
EST-0577,Hibrid,Works,0.8495,3,2,5.67,0
EST-0578,Hibrid,Works,0.5654,5,3,4.97,1
EST-0579,Hibrid,Doesn't Work,0.9103,4,4,4.98,0
EST-0580,Hibrid,Doesn't Work,0.9077,3,3,5.57,0
EST-0581,Hibrid,Works,0.6216,3,2,5.57,1
EST-0582,Hibrid,Works,0.5028,1,1,4.5,0
EST-0583,In Person,Works,0.7988,2,2,5.9,0
EST-0584,Hibrid,Works,0.8938,3,3,6.12,0
EST-0585,In Person,Doesn't Work,0.6659,6,4,5.06,1
EST-0586,Hibrid,Works,0.889,3,3,5.25,0
EST-0587,Hibrid,Works,0.6827,5,5,5.73,0
EST-0588,In Person,Works,0.582,4,2,5.98,1
EST-0589,Hibrid,Doesn't Work,0.9538,4,3,6.68,0
EST-0590,In Person,Doesn't Work,0.9924,4,4,5.75,0
EST-0591,Hibrid,Works,0.5235,4,4,5.52,0
EST-0592,Hibrid,Works,0.7402,1,1,6.21,0
EST-0593,Hibrid,Works,0.7608,3,3,5.51,0
EST-0594,Hibrid,Doesn't Work,0.8221,5,4,6.08,1
EST-0595,In Person,Works,0.7189,2,1,5.66,1
EST-0596,Hibrid,Doesn't Work,0.7668,5,3,4.9,0
EST-0597,Hibrid,Works,0.6976,4,4,5.47,1
EST-0598,In Person,Doesn't Work,0.9898,5,4,5.78,0
EST-0599,Hibrid,Doesn't Work,0.6935,4,3,4.12,0
EST-0600,In Person,Works,0.7211,2,2,6.59,0
EST-0601,Hibrid,Doesn't Work,0.9446,5,5,5.24,0
EST-0602,In Person,Works,0.6064,3,3,5.07,1
EST-0603,Hibrid,Doesn't Work,0.7896,5,4,5.62,0
EST-0604,Hibrid,Works,0.6572,3,3,5.03,0
EST-0605,Hibrid,Works,0.474,3,3,5.26,0
EST-0606,In Person,Works,0.5331,3,2,4.38,0
EST-0607,Hibrid,Works,0.9189,4,2,6.11,0
EST-0608,Hibrid,Works,0.7286,2,1,5.44,1
EST-0609,Hibrid,Works,0.6385,4,4,4.77,0
EST-0610,Hibrid,Doesn't Work,0.7237,1,1,5.84,0
EST-0611,Hibrid,Works,0.5966,4,3,4.24,0
EST-0612,Hibrid,Works,0.6074,2,2,5.52,0
EST-0613,Hibrid,Works,0.5775,2,1,4.02,0
EST-0614,Hibrid,Works,0.8348,2,2,5.83,0
EST-0615,Hibrid,Works,0.7318,5,4,5.24,0
EST-0616,In Person,Works,0.5707,3,1,5.04,1
EST-0617,Hibrid,Doesn't Work,0.8088,4,2,5.0,0
EST-0618,In Person,Works,0.4706,6,5,4.99,0
EST-0619,Hibrid,Doesn't Work,0.9336,5,5,5.92,0
EST-0620,In Person,Works,0.4591,3,3,5.58,0
EST-0621,In Person,Works,0.3439,3,1,4.52,0
EST-0622,In Person,Works,0.604,2,0,4.82,1
EST-0623,Hibrid,Works,0.7522,3,3,5.44,1
EST-0624,Hibrid,Works,0.6731,1,0,4.33,1
EST-0625,In Person,Works,0.4235,2,0,4.32,1
EST-0626,Hibrid,Works,0.6758,2,2,5.9,0
EST-0627,In Person,Works,0.6457,3,2,5.58,0
EST-0628,In Person,Works,0.655,3,2,5.59,0
EST-0629,Hibrid,Works,0.5925,3,1,5.79,1
EST-0630,In Person,Works,0.4472,1,0,4.13,1
EST-0631,Hibrid,Works,0.6288,3,3,4.24,0
EST-0632,Hibrid,Works,0.7266,3,3,5.31,0
EST-0633,Hibrid,Doesn't Work,0.958,6,4,5.94,0
EST-0634,Hibrid,Works,0.6084,4,3,4.93,0
EST-0635,Hibrid,Doesn't Work,0.9169,5,5,5.32,0
EST-0636,Hibrid,Works,0.904,5,5,6.14,0
EST-0637,Hibrid,Works,0.5101,4,2,5.42,0
EST-0638,In Person,Doesn't Work,0.9785,5,5,6.21,0
EST-0639,Hibrid,Works,0.9467,3,3,6.1,1
EST-0640,In Person,Works,0.3886,3,1,5.16,1
EST-0641,Hibrid,Works,0.8252,5,5,5.72,0
EST-0642,Hibrid,Works,0.7381,5,4,5.32,0
EST-0643,In Person,Works,0.5926,4,3,5.18,0
EST-0644,Hibrid,Works,0.842,4,4,5.05,0
EST-0645,In Person,Works,0.7643,3,3,5.42,1
EST-0646,Hibrid,Works,0.7811,5,5,5.52,1
EST-0647,In Person,Doesn't Work,0.9504,5,5,7.0,0
EST-0648,Hibrid,Works,0.7506,5,4,5.86,0
EST-0649,Hibrid,Doesn't Work,0.9603,4,4,6.39,0
EST-0650,Hibrid,Works,0.8865,3,3,6.31,0
EST-0651,Hibrid,Works,0.9494,3,3,6.71,0
EST-0652,In Person,Works,0.6561,2,2,5.24,0
EST-0653,In Person,Works,0.7956,3,3,5.92,0
EST-0654,In Person,Works,0.4708,3,2,5.84,1
EST-0655,Hibrid,Works,0.7477,2,2,5.08,0
EST-0656,In Person,Works,0.6857,3,3,4.97,0
EST-0657,Hibrid,Doesn't Work,0.7909,4,3,5.32,0
EST-0658,Hibrid,Works,0.6951,4,4,5.29,0
EST-0659,Hibrid,Works,0.9489,3,3,4.99,0
EST-0660,Hibrid,Works,0.762,1,1,4.91,0
EST-0661,Hibrid,Works,0.4113,3,2,4.72,0
EST-0662,In Person,Works,0.6498,4,4,5.47,0
EST-0663,In Person,Works,0.6217,3,2,6.18,1
EST-0664,Hibrid,Doesn't Work,0.6978,4,4,5.05,1
EST-0665,In Person,Doesn't Work,0.9588,6,6,5.84,1
EST-0666,Hibrid,Doesn't Work,0.9248,5,4,5.41,0
EST-0667,Hibrid,Works,0.7108,6,4,5.61,0
EST-0668,In Person,Works,0.8317,3,3,5.62,0
EST-0669,Hibrid,Works,0.5044,5,3,4.56,1
EST-0670,Hibrid,Works,0.5364,3,3,5.66,0
EST-0671,Hibrid,Works,0.5018,5,4,4.45,0
EST-0672,Hibrid,Doesn't Work,0.9156,6,4,5.41,0
EST-0673,In Person,Works,0.6984,4,3,5.69,0
EST-0674,Hibrid,Doesn't Work,0.7769,5,3,4.82,1
EST-0675,Hibrid,Works,0.5536,4,4,4.99,0
EST-0676,Hibrid,Works,0.6947,4,4,5.19,0
EST-0677,Hibrid,Works,0.5797,6,6,6.34,0
EST-0678,Hibrid,Works,0.7672,3,3,5.5,0
EST-0679,Hibrid,Works,0.385,4,3,5.22,1
EST-0680,In Person,Works,0.7068,1,1,6.38,0
EST-0681,Hibrid,Works,0.8685,3,3,5.6,0
EST-0682,Hibrid,Works,0.7585,4,4,6.01,1
EST-0683,Hibrid,Works,0.7535,2,2,5.36,0
EST-0684,Hibrid,Works,0.6515,2,2,5.07,0
EST-0685,Hibrid,Works,0.5261,5,4,4.38,1
EST-0686,Hibrid,Works,0.7281,3,1,5.59,0
EST-0687,Hibrid,Works,0.7721,5,5,5.33,0
EST-0688,Hibrid,Works,0.6957,6,5,5.48,0
EST-0689,Hibrid,Works,0.3946,2,2,5.07,0
EST-0690,Hibrid,Works,0.8518,2,2,6.27,0
EST-0691,Hibrid,Doesn't Work,0.7005,6,6,4.97,1
EST-0692,Hibrid,Doesn't Work,0.9924,6,5,6.33,0
EST-0693,Hibrid,Doesn't Work,0.9645,5,4,5.23,0
EST-0694,Hibrid,Works,0.6549,2,2,5.48,0
EST-0695,In Person,Works,0.6356,6,6,5.73,0
EST-0696,Hibrid,Works,0.4646,2,1,4.76,1
EST-0697,Hibrid,Works,0.8148,4,4,5.64,0
EST-0698,Hibrid,Works,0.6318,6,6,5.11,0
EST-0699,Hibrid,Works,0.7418,3,3,5.29,0
EST-0700,Hibrid,Doesn't Work,0.7843,5,5,5.08,1
EST-0701,Hibrid,Works,0.6956,3,2,5.27,0
EST-0702,In Person,Works,0.6928,3,3,5.9,0
EST-0703,Hibrid,Works,0.6491,2,2,6.03,0
EST-0704,In Person,Doesn't Work,0.9167,5,3,5.81,0
EST-0705,Hibrid,Works,0.7797,4,4,5.8,0
EST-0706,Hibrid,Works,0.6845,3,2,4.64,1
EST-0707,In Person,Works,0.4152,2,2,3.77,0
EST-0708,Hibrid,Doesn't Work,0.7375,2,2,6.01,0
EST-0709,Hibrid,Works,0.8323,2,1,5.84,0
EST-0710,Hibrid,Works,0.6983,3,3,5.35,0
EST-0711,In Person,Doesn't Work,0.9979,3,3,6.32,0
EST-0712,Hibrid,Doesn't Work,0.7904,4,4,6.3,0
EST-0713,In Person,Works,0.3492,5,4,4.41,0
EST-0714,Hibrid,Works,0.7173,2,1,5.84,0
EST-0715,Hibrid,Works,0.4598,3,3,4.89,1
EST-0716,Hibrid,Works,0.885,3,3,6.07,0
EST-0717,Hibrid,Works,0.7689,3,3,5.03,0
EST-0718,Hibrid,Works,0.472,5,3,4.73,1
EST-0719,Hibrid,Works,0.772,3,2,4.72,0
EST-0720,Hibrid,Doesn't Work,0.8481,5,4,6.37,0
EST-0721,Hibrid,Works,0.8553,4,4,6.62,0
EST-0722,In Person,Doesn't Work,0.9769,5,4,4.05,0
EST-0723,Hibrid,Works,0.7071,3,2,5.19,0
EST-0724,In Person,Works,0.6419,3,2,5.39,1
EST-0725,Hibrid,Works,0.6378,2,2,4.82,0
EST-0726,Hibrid,Doesn't Work,0.6546,3,3,5.18,0
EST-0727,Hibrid,Works,0.5736,2,1,4.82,1
EST-0728,In Person,Doesn't Work,0.9096,6,6,6.35,0
EST-0729,In Person,Works,0.6006,4,2,4.94,0
EST-0730,In Person,Works,0.7069,3,3,5.59,0
EST-0731,Hibrid,Doesn't Work,0.792,6,5,6.21,1
EST-0732,Hibrid,Works,0.8151,4,4,5.95,0
EST-0733,Hibrid,Doesn't Work,0.8701,6,5,6.0,0
EST-0734,In Person,Works,0.6257,4,1,5.71,1
EST-0735,In Person,Works,0.6646,3,3,5.5,0
EST-0736,Hibrid,Works,0.6096,3,3,4.43,0
EST-0737,Hibrid,Works,0.7843,4,4,6.31,0
EST-0738,Hibrid,Works,0.5715,3,3,4.45,0
EST-0739,Hibrid,Doesn't Work,0.8102,4,2,5.41,0
EST-0740,Hibrid,Doesn't Work,0.947,6,6,5.27,0
EST-0741,Hibrid,Works,0.59,3,2,5.43,0
EST-0742,In Person,Works,0.6028,1,1,5.24,0
EST-0743,In Person,Works,0.8619,2,2,5.73,1
EST-0744,In Person,Doesn't Work,0.8725,4,4,5.59,0
EST-0745,Hibrid,Works,0.7608,5,5,6.07,0
EST-0746,Hibrid,Works,0.6014,3,3,4.82,0
EST-0747,Hibrid,Works,0.8288,4,4,5.61,0
EST-0748,In Person,Works,0.6334,3,3,5.37,0
EST-0749,In Person,Works,0.5891,4,3,6.19,0
EST-0750,Hibrid,Works,0.8372,4,4,5.96,0
EST-0751,Hibrid,Works,0.5524,5,3,4.22,0
EST-0752,Hibrid,Works,0.8018,3,3,6.05,0
EST-0753,Hibrid,Works,0.6205,1,1,5.44,1
EST-0754,In Person,Works,0.4217,4,4,6.06,0
EST-0755,Hibrid,Doesn't Work,0.8386,4,4,5.21,0
EST-0756,In Person,Works,0.7537,3,1,5.07,0
EST-0757,In Person,Works,0.5687,3,2,4.25,0
EST-0758,Hibrid,Works,0.6414,3,3,5.52,1
EST-0759,In Person,Works,0.5642,2,2,5.93,0
EST-0760,Hibrid,Works,0.7565,4,3,5.99,1
EST-0761,Hibrid,Works,0.8988,2,2,6.74,0
EST-0762,Hibrid,Works,0.6807,4,3,5.96,0
EST-0763,In Person,Works,0.6676,2,1,5.49,0
EST-0764,In Person,Works,0.3941,3,3,4.49,1
EST-0765,In Person,Works,0.5427,4,4,5.03,0
EST-0766,Hibrid,Works,0.9033,4,4,6.32,0
EST-0767,Hibrid,Works,0.4499,1,1,5.31,1
EST-0768,Hibrid,Works,0.7394,4,3,6.25,0
EST-0769,Hibrid,Works,0.8828,3,3,5.85,0
EST-0770,Hibrid,Works,0.776,1,1,5.82,0
EST-0771,Hibrid,Doesn't Work,0.8724,6,6,6.47,0
EST-0772,In Person,Works,0.5775,2,2,6.16,0
EST-0773,Hibrid,Doesn't Work,0.8776,6,5,6.0,0
EST-0774,Hibrid,Doesn't Work,0.8952,4,4,5.97,0
EST-0775,In Person,Doesn't Work,0.7109,6,5,5.35,0
EST-0776,In Person,Works,0.666,3,3,5.58,0
EST-0777,Hibrid,Works,0.8106,2,2,5.54,0
EST-0778,Hibrid,Works,0.7101,3,3,4.67,0
EST-0779,Hibrid,Works,0.4834,4,3,5.16,1
EST-0780,In Person,Doesn't Work,0.938,3,3,6.21,0
EST-0781,Hibrid,Doesn't Work,0.795,5,4,5.49,0
EST-0782,Hibrid,Works,0.9127,5,4,5.7,0
EST-0783,In Person,Works,0.7015,2,2,6.85,0
EST-0784,Hibrid,Works,0.6056,2,2,5.54,0
EST-0785,In Person,Works,0.6724,4,4,4.95,0
EST-0786,Hibrid,Works,0.893,2,1,5.38,1
EST-0787,Hibrid,Works,0.5506,1,1,5.28,0
EST-0788,Hibrid,Works,0.8151,3,2,5.33,0
EST-0789,Hibrid,Doesn't Work,0.9821,6,5,5.75,0
EST-0790,Hibrid,Works,0.7628,4,3,5.53,0
EST-0791,Hibrid,Works,0.8601,3,3,6.04,0
EST-0792,In Person,Works,0.5264,4,2,5.63,1
EST-0793,In Person,Doesn't Work,0.9524,6,6,6.13,0
EST-0794,In Person,Works,0.6693,4,4,5.58,1
EST-0795,Hibrid,Works,0.7926,5,3,6.15,0
EST-0796,Hibrid,Works,0.7423,3,3,5.18,0
EST-0797,In Person,Works,0.6056,2,2,6.17,0
EST-0798,In Person,Works,0.4346,2,1,5.41,1
EST-0799,Hibrid,Works,0.7232,3,2,4.8,0
EST-0800,In Person,Works,0.7424,1,1,5.04,0
EST-0801,In Person,Works,0.2323,2,1,4.39,1
EST-0802,Hibrid,Doesn't Work,0.8179,5,4,4.4,0
EST-0803,In Person,Works,0.8385,4,3,5.65,0
EST-0804,Hibrid,Works,0.4731,2,2,6.06,0
EST-0805,Hibrid,Works,0.5841,2,1,5.84,1
EST-0806,Hibrid,Doesn't Work,0.8584,5,4,6.28,0
EST-0807,Hibrid,Works,0.729,4,3,7.0,0
EST-0808,In Person,Works,0.6907,3,2,5.92,0
EST-0809,In Person,Works,0.7579,4,2,5.71,0
EST-0810,In Person,Works,0.7449,3,3,5.88,0
EST-0811,In Person,Works,0.5311,4,2,5.01,1
EST-0812,In Person,Works,0.581,3,1,4.37,0
EST-0813,Hibrid,Doesn't Work,0.7816,4,3,5.08,0
EST-0814,Hibrid,Works,0.5996,3,3,5.68,0
EST-0815,In Person,Works,0.7639,4,3,5.54,0
EST-0816,Hibrid,Doesn't Work,0.7393,3,3,5.62,0
EST-0817,Hibrid,Doesn't Work,0.9453,6,5,5.43,0
EST-0818,In Person,Doesn't Work,0.9809,5,4,6.0,0
EST-0819,Hibrid,Works,0.5944,5,2,4.66,1
EST-0820,Hibrid,Doesn't Work,0.9346,3,3,5.99,0
EST-0821,Hibrid,Doesn't Work,0.5791,5,2,5.18,0
EST-0822,Hibrid,Works,0.6033,2,2,5.93,0
EST-0823,Hibrid,Works,0.6378,3,3,5.56,0
EST-0824,In Person,Doesn't Work,0.9889,5,5,6.64,0
EST-0825,In Person,Works,0.686,3,2,5.05,0
EST-0826,Hibrid,Doesn't Work,0.8752,5,5,6.06,0
EST-0827,Hibrid,Doesn't Work,0.7707,5,5,6.09,0
EST-0828,Hibrid,Works,0.5835,4,2,4.94,0
EST-0829,Hibrid,Doesn't Work,0.7686,6,6,5.52,0
EST-0830,In Person,Works,0.6592,3,3,5.45,0
EST-0831,Hibrid,Doesn't Work,0.9221,6,5,6.27,0
EST-0832,Hibrid,Works,0.532,5,4,6.46,0
EST-0833,Hibrid,Works,0.8577,2,2,5.92,0
EST-0834,Hibrid,Works,0.801,4,4,6.12,0
EST-0835,In Person,Works,0.4813,2,1,5.15,0
EST-0836,Hibrid,Works,0.6592,3,3,4.34,0
EST-0837,In Person,Doesn't Work,0.8472,6,5,5.62,0
EST-0838,Hibrid,Doesn't Work,0.8859,6,6,5.39,0
EST-0839,Hibrid,Works,0.7919,2,2,6.16,0
EST-0840,Hibrid,Works,0.5717,6,5,5.06,0
EST-0841,Hibrid,Doesn't Work,0.8611,5,5,5.8,0
EST-0842,Hibrid,Works,0.8213,4,4,5.84,0
EST-0843,In Person,Works,0.5129,4,2,5.67,1
EST-0844,Hibrid,Works,0.8878,2,2,5.61,0
EST-0845,Hibrid,Works,0.7178,2,2,5.24,0
EST-0846,Hibrid,Works,0.8407,5,5,5.75,1
EST-0847,Hibrid,Doesn't Work,0.5533,6,5,5.64,0
EST-0848,In Person,Works,0.5349,3,2,5.45,0
EST-0849,Hibrid,Works,0.8911,2,2,5.34,0
EST-0850,Hibrid,Works,0.7058,3,2,6.54,0
EST-0851,Hibrid,Works,0.6448,2,2,5.76,0
EST-0852,In Person,Works,0.5388,1,0,5.06,1
EST-0853,In Person,Works,0.492,1,1,5.6,0
EST-0854,Hibrid,Works,0.554,4,3,5.02,1
EST-0855,Hibrid,Works,0.8938,2,2,5.73,0
EST-0856,Hibrid,Works,0.7268,1,1,5.46,0
EST-0857,Hibrid,Works,0.586,2,0,5.19,0
EST-0858,Hibrid,Works,0.7236,1,1,5.09,0
EST-0859,Hibrid,Works,0.4479,4,4,4.73,0
EST-0860,Hibrid,Works,0.9036,3,3,6.12,0
EST-0861,In Person,Doesn't Work,0.9451,6,6,5.68,0
EST-0862,Hibrid,Works,0.3602,2,0,5.36,0
EST-0863,In Person,Doesn't Work,0.9568,6,6,6.56,0
EST-0864,In Person,Doesn't Work,0.8734,6,6,5.88,0
EST-0865,Hibrid,Doesn't Work,0.6044,6,4,5.83,0
EST-0866,In Person,Works,0.8023,4,2,5.04,0
EST-0867,Hibrid,Works,0.908,3,2,6.33,0
EST-0868,In Person,Works,0.6712,3,3,5.24,0
EST-0869,Hibrid,Works,0.6709,4,4,5.65,0
EST-0870,Hibrid,Works,0.7891,4,4,6.79,0
EST-0871,Hibrid,Works,0.6103,1,0,4.83,1
EST-0872,Hibrid,Works,0.8099,2,2,5.64,0
EST-0873,In Person,Works,0.4086,3,3,4.87,1
EST-0874,Hibrid,Works,0.8223,3,3,4.94,0
EST-0875,Hibrid,Works,0.8327,1,0,6.18,1
EST-0876,Hibrid,Works,0.8913,3,2,6.04,0
EST-0877,In Person,Works,0.5417,6,5,6.12,0
EST-0878,Hibrid,Works,0.531,1,1,5.09,0
EST-0879,In Person,Doesn't Work,0.9574,5,5,6.3,0
EST-0880,Hibrid,Works,0.8669,3,3,4.86,0
EST-0881,Hibrid,Works,0.8084,2,1,5.38,1
EST-0882,Hibrid,Works,0.6611,3,3,6.06,1
EST-0883,Hibrid,Works,0.6766,4,2,5.18,1
EST-0884,In Person,Works,0.5256,3,2,5.29,0
EST-0885,Hibrid,Works,0.8785,5,4,5.15,0
EST-0886,Hibrid,Doesn't Work,0.8805,5,5,5.46,0
EST-0887,Hibrid,Works,0.7117,2,1,4.79,1
EST-0888,Hibrid,Doesn't Work,0.9935,4,4,5.61,0
EST-0889,In Person,Works,0.6466,3,3,4.16,1
EST-0890,In Person,Works,0.6148,3,3,5.14,0
EST-0891,Hibrid,Doesn't Work,0.9275,5,5,6.25,1
EST-0892,Hibrid,Works,0.8784,4,4,5.75,0
EST-0893,In Person,Works,0.5299,4,3,4.94,0
EST-0894,In Person,Works,0.48,3,3,3.74,1
EST-0895,In Person,Works,0.5862,4,0,6.39,0
EST-0896,Hibrid,Works,0.6618,3,2,5.82,0
EST-0897,Hibrid,Doesn't Work,0.9945,6,6,6.78,0
EST-0898,Hibrid,Works,0.8802,3,3,6.71,0
EST-0899,Hibrid,Doesn't Work,0.8936,4,3,5.9,0
EST-0900,In Person,Works,0.8497,4,4,6.14,0
EST-0901,Hibrid,Doesn't Work,0.8704,5,4,5.95,0
EST-0902,Hibrid,Works,0.7399,6,6,5.26,0
EST-0903,Hibrid,Works,0.5546,4,3,6.01,0
EST-0904,Hibrid,Works,0.5705,3,2,6.09,1
EST-0905,In Person,Doesn't Work,0.9063,5,4,5.44,0
EST-0906,Hibrid,Works,0.6065,1,0,5.5,0
EST-0907,In Person,Works,0.5064,2,2,5.6,1
EST-0908,Hibrid,Works,0.8584,1,1,5.48,1
EST-0909,In Person,Works,0.521,1,1,4.96,0
EST-0910,In Person,Works,0.5633,4,3,5.53,0
EST-0911,In Person,Works,0.7267,1,1,4.79,0
EST-0912,Hibrid,Works,0.8001,6,5,6.05,0
EST-0913,In Person,Doesn't Work,0.9234,3,3,6.17,0
EST-0914,Hibrid,Works,0.6234,3,0,4.61,1
EST-0915,In Person,Works,0.7624,4,4,5.04,0
EST-0916,In Person,Doesn't Work,0.9807,4,4,6.43,0
EST-0917,Hibrid,Works,0.5841,3,3,4.75,0
EST-0918,Hibrid,Works,0.7326,3,3,5.19,0
EST-0919,In Person,Works,0.3268,1,0,4.49,1
EST-0920,Hibrid,Works,0.7658,4,4,5.93,0
EST-0921,In Person,Works,0.5415,2,1,4.42,0
EST-0922,In Person,Works,0.3292,3,3,4.37,0
EST-0923,Hibrid,Works,0.7508,1,1,5.14,0
EST-0924,In Person,Works,0.5282,2,2,4.79,0
EST-0925,In Person,Works,0.5456,3,2,4.67,1
EST-0926,Hibrid,Doesn't Work,0.8859,6,5,5.79,0
EST-0927,In Person,Works,0.5961,5,2,5.19,1
EST-0928,In Person,Works,0.5805,3,2,5.6,0
EST-0929,Hibrid,Works,0.5969,2,2,4.97,0
EST-0930,Hibrid,Works,0.5508,3,2,5.35,1
EST-0931,In Person,Works,0.5363,1,1,4.4,0
EST-0932,Hibrid,Doesn't Work,0.711,3,2,5.06,1
EST-0933,In Person,Works,0.5409,2,2,3.52,0
EST-0934,Hibrid,Works,0.8066,3,3,5.61,0
EST-0935,In Person,Works,0.4723,1,0,5.36,0
EST-0936,Hibrid,Works,0.693,3,2,5.44,0
EST-0937,Hibrid,Works,0.8249,1,1,5.37,0
EST-0938,In Person,Works,0.9225,3,3,5.34,0
EST-0939,Hibrid,Doesn't Work,0.8831,6,6,5.11,0
EST-0940,Hibrid,Doesn't Work,0.9762,4,4,5.68,0
EST-0941,Hibrid,Works,0.9396,1,1,5.35,0
EST-0942,Hibrid,Works,0.67,4,3,5.14,1
EST-0943,In Person,Doesn't Work,0.7426,4,2,4.21,0
EST-0944,In Person,Works,0.7311,5,5,6.82,0
EST-0945,Hibrid,Works,0.735,3,3,5.8,0
EST-0946,Hibrid,Doesn't Work,0.894,4,4,5.7,0
EST-0947,Hibrid,Works,0.9485,6,5,6.18,0
EST-0948,In Person,Works,0.5062,3,1,5.75,0
EST-0949,In Person,Works,0.573,5,5,4.93,1
EST-0950,In Person,Works,0.6945,3,2,4.81,0
EST-0951,In Person,Doesn't Work,0.9994,4,4,5.94,0
EST-0952,Hibrid,Doesn't Work,0.6438,5,3,4.32,0
EST-0953,Hibrid,Works,0.7533,4,4,5.9,0
EST-0954,In Person,Doesn't Work,0.6375,5,3,4.72,0
EST-0955,Hibrid,Works,0.6251,1,1,4.55,1
EST-0956,Hibrid,Works,0.4162,5,3,4.83,1
EST-0957,Hibrid,Doesn't Work,0.8378,5,4,5.61,0
EST-0958,In Person,Doesn't Work,0.9683,6,6,5.66,0
EST-0959,Hibrid,Doesn't Work,0.6581,5,5,4.88,0
EST-0960,Hibrid,Works,0.8161,3,3,5.23,0
EST-0961,Hibrid,Doesn't Work,0.9698,5,4,5.46,0
EST-0962,Hibrid,Works,0.5602,3,3,5.25,0
EST-0963,In Person,Doesn't Work,0.9905,5,5,6.6,0
EST-0964,Hibrid,Doesn't Work,0.7627,4,4,5.64,1
EST-0965,Hibrid,Doesn't Work,0.9542,6,5,5.67,1
EST-0966,Hibrid,Works,0.8789,3,3,5.96,0
EST-0967,Hibrid,Works,0.7359,2,2,5.11,0
EST-0968,In Person,Doesn't Work,0.9634,5,5,6.37,0
EST-0969,In Person,Works,0.4048,3,2,4.8,0
EST-0970,Hibrid,Works,0.8049,4,3,5.98,0
EST-0971,Hibrid,Works,0.9435,3,3,6.09,0
EST-0972,In Person,Doesn't Work,0.9573,5,4,5.24,1
EST-0973,Hibrid,Works,0.9312,2,2,5.35,0
EST-0974,In Person,Works,0.4472,4,3,4.32,0
EST-0975,Hibrid,Works,0.6443,2,1,4.53,1
EST-0976,In Person,Works,0.5228,4,1,5.28,0
EST-0977,Hibrid,Works,0.6001,3,2,5.97,0
EST-0978,Hibrid,Works,0.4112,1,0,6.09,1
EST-0979,Hibrid,Works,0.5852,1,1,5.68,0
EST-0980,In Person,Works,0.5359,3,3,4.78,0
EST-0981,Hibrid,Doesn't Work,0.6413,5,5,5.49,0
EST-0982,Hibrid,Doesn't Work,0.781,4,4,5.56,0
EST-0983,Hibrid,Works,0.509,3,1,5.4,1
EST-0984,In Person,Works,0.7603,3,2,5.69,0
EST-0985,In Person,Works,0.3631,4,4,5.01,0
EST-0986,Hibrid,Works,0.7005,3,1,4.39,0
EST-0987,Hibrid,Doesn't Work,0.9207,5,4,5.56,0
EST-0988,Hibrid,Works,0.352,2,2,3.56,0
EST-0989,Hibrid,Works,0.6254,2,2,4.73,1
EST-0990,Hibrid,Works,0.5153,1,0,5.59,1
EST-0991,In Person,Works,0.8594,3,3,5.82,1
EST-0992,Hibrid,Works,0.7123,2,2,5.88,0
EST-0993,Hibrid,Works,0.9484,3,2,6.56,0
EST-0994,Hibrid,Works,0.6964,2,2,4.43,0
EST-0995,Hibrid,Works,0.6778,3,2,5.33,0
EST-0996,Hibrid,Works,0.8779,3,2,6.08,0
EST-0997,Hibrid,Doesn't Work,0.8372,4,4,5.61,0
EST-0998,Hibrid,Works,0.6077,4,4,5.62,0
EST-0999,In Person,Works,0.3939,1,1,4.63,0
EST-1000,Hibrid,Doesn't Work,0.8549,5,4,5.09,0

```

### Entregable 2: Diccionario de Datos

| Nombre de Variable        | Descripción Operativa                                                                                                                                                          | Tipo de Dato      | Rango / Valores               | Rol en el Modelo                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ----------------------------- | ------------------------------------------------------ |
| **id**                    | Identificador sintético secuencial y único del estudiante.                                                                                                                     | String            | "EST-0001" a "EST-1000"       | Identificador único (no predictivo).                   |
| **mode**                  | Modalidad de estudio actual del alumno.                                                                                                                                        | Categórico        | "In Person", "Hibrid"         | Predictor demográfico y académico.                     |
| **working_condition**     | Estado laboral del estudiante (simula ~75% de la cohorte trabajando para evidenciar el _data drift_).                                                                          | Categórico        | "Works", "Doesn't Work"       | Predictor socioeconómico (impacta asistencia y carga). |
| **attendance_percentage** | Porcentaje de asistencia a clases. Afectado negativamente por la condición laboral.                                                                                            | Numérico continuo | 0.0000 a 1.0000               | Predictor académico de alto impacto.                   |
| **academic_load**         | Número de asignaturas inscritas. Los estudiantes trabajadores tienden a inscribir menos ramos.                                                                                 | Numérico entero   | 1 a 6                         | Predictor académico (divisor de tasa de reprobación).  |
| **subjects_passed**       | Asignaturas aprobadas. Depende probabilísticamente de la asistencia y del promedio histórico.                                                                                  | Numérico entero   | 0 a `academic_load`           | Predictor de rendimiento actual.                       |
| **previous_average**      | Promedio de calificaciones de periodos anteriores (escala 1.0 a 7.0).                                                                                                          | Numérico continuo | 1.00 a 7.00                   | Predictor de trayectoria académica.                    |
| **dropping_out_risk**     | Clasificación de riesgo de no continuidad. Calculado probabilísticamente sin _target leakage_ (mayor riesgo con baja asistencia, alta reprobación y condición laboral activa). | Binario           | 0 (sin riesgo), 1 (en riesgo) | Variable objetivo (Target).                            |
