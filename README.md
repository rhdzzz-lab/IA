# IA - Ejercicios de Fine-Tuning

Este repositorio contiene datasets sintéticos para practicar clasificación de texto y fine-tuning con BERT.

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
- `scripts/build_jsonl.py`
- `scripts/train_bert.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- `requirements.txt`

Cada CSV tiene dos columnas:

- `label`: la clase o categoría
- `text`: el mensaje a clasificar

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Convertir CSV a JSONL

```bash
python scripts/build_jsonl.py
```

Esto genera los archivos JSONL dentro de `jsonl/`.

## Entrenar un modelo

```bash
python scripts/train_bert.py --data-file jsonl/01_correos_electronicos.jsonl --output-dir models/correos-bert
```

El script ajusta automáticamente la configuración compatible con la versión instalada de `transformers`.

## Evaluar el modelo

```bash
python scripts/evaluate.py --model-dir models/correos-bert --data-file jsonl/01_correos_electronicos.jsonl
```

## Predecir un texto nuevo

```bash
python scripts/predict.py --model-dir models/correos-bert --text "Necesito saber por qué mi pago fue rechazado."
```

## Nota

Los ejemplos fueron redactados como datos sintéticos de práctica, siguiendo las etiquetas del PDF original.
