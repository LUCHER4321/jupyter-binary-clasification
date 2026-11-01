# Laboratorio A

## Notebook Ejecutado

Acceder al [Notebook de Jupyter](/Laboratorio%20A.ipynb)

## Registro de los Prompts Principales

Acceder al [Registro de Prompts](/Registro%20de%20Prompts.md)

## Resumen de Hasta 300 Palabras

### Cambios entre Cohortes

Se evidenció un claro _data drift_ poblacional. En la nueva cohorte, la proporción de estudiantes que trabajan aumentó drásticamente del 45% al 75%, mientras que la modalidad híbrida pasó de representar el 35% al 65% de los casos.

### Distribución de las Predicciones

Dado que trabajar tiene un coeficiente positivo fuerte en el modelo logístico (0.96) y reduce la asistencia promedio, la distribución de los puntajes predichos se desplazó hacia la derecha. El modelo estima hoy un volumen significativamente mayor de estudiantes en riesgo.

### Umbral y Acción

Se selecciona un umbral operativo conservador (ej. 0.65). Superar este corte activa exclusivamente una revisión humana para priorizar el contacto y ofrecer apoyo. Respetando las reglas del gobierno de datos institucionales, la alerta jamás desencadena sanciones automáticas.

### Errores y Limitaciones Presentes

El modelo es predictivo, no causal. Persiste el riesgo de falsos negativos (estudiantes vulnerables que quedan fuera del radar y retrasan su ayuda) y de falsos positivos, los cuales consumen recursos limitados de orientación y corren el riesgo de estigmatizar al estudiante si la alerta se confunde con un veredicto.

### Mitigación y Condiciones

El drift detectado no implica automáticamente que el modelo haya fallado. La medida de mitigación proporcional es aumentar temporalmente la capacidad de revisión humana y ajustar el umbral según la disponibilidad de los orientadores. Solo se justificará reentrenar el modelo si, al recolectar nuevas etiquetas reales al final del semestre, se comprueba una degradación en el desempeño predictivo (_concept drift_).
