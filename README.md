# IA - Ejercicios de Fine-Tuning

Este repositorio contiene una base de datos de ejemplo para resolver los ejercicios del PDF **Ejercicios Fine-Tuning**.

## Estructura

- `datasets/01_correos_electronicos.csv`
- `datasets/02_banca.csv`
- `datasets/03_soporte_tecnico.csv`
- `datasets/04_universidad.csv`
- `datasets/05_comentarios_estudiantes.csv`
- `datasets/06_tienda_en_linea.csv`
- `datasets/07_aseguradora.csv`
- `datasets/08_compania_telefonica.csv`
- `datasets/09_resenas_productos.csv`
- `datasets/10_noticias.csv`

Cada archivo tiene dos columnas:

- `label`: la clase o categoría
- `text`: el mensaje a clasificar

## Uso

Puedes usar estos CSVs para:
- entrenar un modelo de clasificación
- convertirlos a JSONL para fine-tuning
- probar un pipeline con BERT o con otro modelo de texto

## Nota

Los ejemplos fueron redactados como datos sintéticos de práctica, siguiendo las etiquetas mostradas en el PDF.
