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
- `scripts/build_jsonl.py`
- `scripts/train_bert.py`
- `requirements.txt`

Cada archivo CSV tiene dos columnas:

- `label`: la clase o categoría
- `text`: el mensaje a clasificar

## Convertir a JSONL

El script `scripts/build_jsonl.py` convierte todos los CSV dentro de `datasets/` a archivos JSONL dentro de `jsonl/`.

```bash
python scripts/build_jsonl.py
```

Cada salida tendrá el formato:

```json
{"label": "spam", "text": "..."}
```

## Entrenar BERT

El script `scripts/train_bert.py` entrena un clasificador BERT con un archivo `.csv` o `.jsonl`.

Instala dependencias:

```bash
pip install -r requirements.txt
```

Ejemplo con uno de los JSONL generados:

```bash
python scripts/train_bert.py --data-file jsonl/01_correos_electronicos.jsonl --output-dir models/correos-bert
```

Ejemplo con CSV:

```bash
python scripts/train_bert.py --data-file datasets/02_banca.csv --output-dir models/banca-bert
```

## Nota

Los ejemplos fueron redactados como datos sintéticos de práctica, siguiendo las etiquetas mostradas en el PDF.
